"""
Train a diffusion model on images.
"""

import argparse
import numpy as np
import json, torch, os, sys
sys.path.append('../')
from diffugen.utils import dist_util, logger
from diffugen.text_datasets import load_data_text
from diffugen.step_sample import create_named_schedule_sampler
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    load_model_emb,
    load_tokenizer
)
from train_util_with_meta import TrainLoop
from transformers import set_seed
import wandb


### custom your wandb setting here ###
# os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"

def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)  # update latest args according to argparse
    return parser

def main():
    args = create_argparser().parse_args()
    set_seed(args.seed) 
    dist_util.setup_dist()
    logger.configure()
    logger.log("### Creating data loader...")

    # 加载tokenizer
    tokenizer = load_tokenizer(args)
    model_weight, tokenizer = load_model_emb(args, tokenizer)

    # 加载训练集数据
    data = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_args = args,
        loaded_vocab=tokenizer,
        model_emb=model_weight  # use model's weights as init
    )
    # next(data)  # todo: 为什么要过掉第一批数据？

    # 加载验证集数据
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_args=args,
        split='valid',
        deterministic=True,
        loaded_vocab=tokenizer,
        model_emb=model_weight  # using the same embedding wight with tranining data
    )

    data_meta = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_args=args,
        split='meta',
        deterministic=True,
        loaded_vocab=tokenizer,
        model_emb=model_weight  # using the same embedding wight with tranining data
    )

    print('#'*30, 'size of vocab', args.vocab_size)

    # 加载transformer和扩散模型SpacedDiffusion
    logger.log("### Creating model and diffusion...")
    # print('#'*30, 'CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )
    meta_model, meta_diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )
    print('#'*30, 'cuda', dist_util.dev())
    model.to(dist_util.dev())  # DEBUG **
    meta_model.to(dist_util.dev())  # DEBUG **
    # model.cuda()  # DEBUG **

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f'### The parameter count is {pytorch_total_params}')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)  # Create a ScheduleSampler from a library of pre-defined samplers.

    logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
        try:
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "diffugen"),  # 'diffugen'
                name=args.checkpoint_path,  # 'diffusion_models/diffuseq_CommensenseConversation_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test-CommensenseConversation'
            )
        except:
            print("Wandb init error!!! Now try it again...")
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "diffugen"),  # 'diffugen'
                name=args.checkpoint_path,
                # 'diffusion_models/diffuseq_CommensenseConversation_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test-CommensenseConversation'
            )
        wandb.config.update(args.__dict__, allow_val_change=True)

    logger.log("### Training...")

    TrainLoop(
        model=model,
        meta_model=meta_model,
        diffusion=diffusion,
        meta_diffusion=meta_diffusion,
        data=data,
        meta_data=data_meta,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,  # 平滑训练损失
        meta_interval=args.meta_interval,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval
    ).run_loop()


if __name__ == "__main__":
    main()
