"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json
from tracemalloc import start

import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from diffugen.rounding import denoised_fn_round, get_weights
from diffugen.text_datasets import load_data_text

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import time
from diffugen.utils import dist_util, logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_model_emb,
    load_tokenizer
)

def create_argparser():
    defaults = dict(model_path='', step=0, out_dir='', top_p=0)
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main(verbose=False, output_step=100):
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size  # 修改batch_size
    args.__dict__.update(training_args)

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.to(dist_util.dev())
    model.eval()

    tokenizer = load_tokenizer(args)
    model_emb, tokenizer = load_model_emb(args, tokenizer)

    model_emb.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_emb_copy = get_weights(model_emb, args)

    set_seed(args.seed2)

    print("### Sampling...on", args.split)

    ## load data
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        deterministic=True,
        data_args=args,
        split=args.split,
        loaded_vocab=tokenizer,
        model_emb=model_emb.cpu(), # using the same embedding wight with tranining data
        loop=False
    )  # 4693个句子（e2e）

    start_t = time.time()
    
    # batch, cond = next(data_valid)
    # print(batch.shape)

    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}.json")
    # fout = open(out_path, 'a')

    all_test_data = []

    try:
        while True:
            batch, cond = next(data_valid)
            # print(batch.shape)
            all_test_data.append(cond)

    except StopIteration:
        print('### End of reading iteration...')
    
    from tqdm import tqdm

    for cond in tqdm(all_test_data):  # for each batch

        input_ids_x = cond.pop('input_id_x').to(dist_util.dev())  # input_id_x torch.Size([10, 128])
        input_ids_y = cond.pop('input_id_y').to(dist_util.dev())  # input_id_y
        x_start = model.get_embeds(input_ids_x)
        input_ids_mask = cond.pop('input_mask')  # input_mask
        input_ids_mask_ori = input_ids_mask

        noise = th.randn_like(x_start)
        input_ids_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(dist_util.dev())
        x_noised = th.where(input_ids_mask==0, x_start, noise)  # mask为0的地方固定，1的地方为随机噪声

        model_kwargs = {}

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:  # if 'step' less than diffusion training steps, like 1000, use ddim sampling
            args.use_ddim = True
            step_gap = args.diffusion_steps//args.step

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        # sample_shape = (batch.shape[0], args.seq_len, args.hidden_dim)
        sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)  # (b, 128, 128) =？ BUG

        samples = sample_fn(
            model,
            sample_shape,
            noise=x_noised,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model_emb_copy.cuda()),
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            clamp_step=args.clamp_step,  # 0
            clamp_first=True,
            mask=input_ids_mask,
            x_start=x_start,
            gap=step_gap
        )  # 1个batch所有时间步的采样结果，按时间步倒排 e.g. [[bsz, 128, 128](t==1999), ...]

        model_emb_copy.cpu()
        # print(samples[0].shape) # samples for each step
        if verbose:
            # for t in range(args.step, 0, -output_step):
            for t in range(0, args.step, output_step):
                # 时间步t的生成结果（一个batch）
                sample = samples[t]
                word_lst_output, word_lst_ref, word_lst_input = parse_result(sample, model, tokenizer,
                                                                             input_ids_x, input_ids_y,
                                                                             input_ids_mask_ori)
                out_path_ = out_path.split('.json')[0] + f'_t_{args.step-t}.json'
                fout = open(out_path_, 'a')
                for (trg, ref, src) in zip(word_lst_output, word_lst_ref, word_lst_input):
                    print(json.dumps({"result": trg, "reference": ref, "constraint": src}), file=fout)
                fout.close()
        else:
            sample = samples[-1]  # 在t==0这一时间步的输出
            word_lst_output, word_lst_ref, word_lst_input = parse_result(sample, model, tokenizer, input_ids_x, input_ids_y, input_ids_mask_ori)
            # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            # dist.all_gather(gathered_samples, sample)
            # all_sentence = [sample.cpu().numpy() for sample in gathered_samples]
            # # print('sampling takes {:.2f}s .....'.format(time.time() - start_t))
            #
            # word_lst_output = []  # 生成结果
            # word_lst_ref = []  # 参考
            # word_lst_input = []  # 输入
            #
            # arr = np.concatenate(all_sentence, axis=0)
            # x_t = th.tensor(arr).cuda()
            # # print('decoding for seq2seq', )
            # # print(arr.shape)
            #
            # reshaped_x_t = x_t
            # logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
            #
            # cands = th.topk(logits, k=1, dim=-1)
            # sample = cands.indices
            # # tokenizer = load_tokenizer(args)
            #
            # for seq, input_mask, x in zip(cands.indices, input_ids_mask_ori, input_ids_x):
            #     # todo: 替换seq中固定的部分，即mask==0的部分
            #     # seq (128,1)  input_mask (128)  x (128)
            #     seq = th.where(input_mask.unsqueeze(-1).to(dist_util.dev()) == 0, x.unsqueeze(-1), seq)
            #     # len_x = args.seq_len - sum(input_mask).tolist()
            #     # tokens = tokenizer.decode_token(seq[len_x:])  # 后半部分是生成的
            #     tokens = tokenizer.decode_token(seq)  # 后半部分是生成的
            #     word_lst_output.append(tokens)
            #
            # # for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
            # #     len_x = args.seq_len - sum(input_mask).tolist()
            # #     word_lst_input.append(tokenizer.decode_token(seq[:len_x]))
            # #     word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))
            # for x, y in zip(input_ids_x, input_ids_y):
            #     word_lst_input.append(tokenizer.decode_token(x))
            #     word_lst_ref.append(tokenizer.decode_token(y))

            fout = open(out_path, 'a')
            for (trg, ref, src) in zip(word_lst_output, word_lst_ref, word_lst_input):
                print(json.dumps({"result": trg, "reference": ref, "constraint": src}), file=fout)
            fout.close()

    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')


def parse_result(sample, model, tokenizer, input_ids_x, input_ids_y, input_ids_mask_ori):
    gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_samples, sample)
    all_sentence = [sample.cpu().numpy() for sample in gathered_samples]

    # print('sampling takes {:.2f}s .....'.format(time.time() - start_t))

    word_lst_output = []  # 生成结果
    word_lst_ref = []  # 参考
    word_lst_input = []  # 输入

    arr = np.concatenate(all_sentence, axis=0)
    x_t = th.tensor(arr).cuda()
    # print('decoding for seq2seq', )
    # print(arr.shape)

    reshaped_x_t = x_t
    logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab

    cands = th.topk(logits, k=1, dim=-1)
    sample = cands.indices
    # tokenizer = load_tokenizer(args)

    for seq, input_mask, x in zip(cands.indices, input_ids_mask_ori, input_ids_x):
        # todo: 替换seq中固定的部分，即mask==0的部分
        # seq (128,1)  input_mask (128)  x (128)
        seq = th.where(input_mask.unsqueeze(-1).to(dist_util.dev()) == 0, x.unsqueeze(-1), seq)
        # len_x = args.seq_len - sum(input_mask).tolist()
        # tokens = tokenizer.decode_token(seq[len_x:])  # 后半部分是生成的
        tokens = tokenizer.decode_token(seq)  # 后半部分是生成的
        word_lst_output.append(tokens)

    # for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
    #     len_x = args.seq_len - sum(input_mask).tolist()
    #     word_lst_input.append(tokenizer.decode_token(seq[:len_x]))
    #     word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))
    for x, y in zip(input_ids_x, input_ids_y):
        word_lst_input.append(tokenizer.decode_token(x))
        word_lst_ref.append(tokenizer.decode_token(y))

    return word_lst_output, word_lst_ref, word_lst_input


if __name__ == "__main__":
    main(verbose=False, output_step=200)
