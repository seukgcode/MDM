# encoding: utf-8
# import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch
import json
import psutil
import datasets
from datasets import Dataset as Dataset2

def load_data_text(
    batch_size, 
    seq_len, 
    deterministic=False, 
    data_args=None, 
    model_emb=None,
    split='train', 
    loaded_vocab=None,
    loop=True,
):
    """
    For a dataset, create a generator over (seqs, kwargs) pairs.

    Each seq is an (bsz, len, h) float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for some meta information.

    :param batch_size: the batch size of each returned pair.
    :param seq_len: the max sequence length (one-side).
    :param deterministic: if True, yield results in a deterministic order.
    :param data_args: including dataset directory, num of dataset, basic settings, etc.
    :param model_emb: loaded word embeddings.
    :param loaded_vocab: loaded word vocabs.
    :param loop: loop to get batch data or not.
    """

    print('#'*30, '\nLoading text data...')

    training_data = get_corpus(data_args, seq_len, split=split, loaded_vocab=loaded_vocab)

    dataset = TextDataset2(
        training_data,
        data_args,
        model_emb=model_emb
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,  # 20,
        # drop_last=True,
        shuffle=not deterministic,
        num_workers=0,
    )
    if loop:
        return infinite_loader(data_loader)
    else:
        # print(data_loader)
        return iter(data_loader)

def infinite_loader(data_loader):  # 循环读数据
    while True:
        yield from data_loader

def helper_tokenize(sentence_lst, vocab_dict, seq_len):
    # pipeline式数据处理，1. encode_token 2.
    # Process.memory_info is expressed in bytes, so convert to megabytes
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    raw_datasets = Dataset2.from_dict(sentence_lst)
    print(raw_datasets)  # Dataset({features: ['src', 'trg'],  num_rows: 19})
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def tokenize_function(examples):
        input_id_x = vocab_dict.encode_token(examples['src'])
        input_id_y = vocab_dict.encode_token(examples['trg'])
        result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y}

        return result_dict

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['src', 'trg'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    print('### tokenized_datasets', tokenized_datasets)  # tokenized_datasets Dataset({features: ['input_id_x', 'input_id_y'], num_rows: 19})
    print('### tokenized_datasets...example', tokenized_datasets['input_id_x'][0])
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def merge_and_mask(group_lst):
        lst = []
        mask = []
        for i in range(len(group_lst['input_id_x'])):  # 遍历所有句子
            end_token = group_lst['input_id_x'][i][-1]  # [SEP]
            src = group_lst['input_id_x'][i][:-1]
            trg = group_lst['input_id_y'][i][:-1]
            while len(src) + len(trg) > seq_len - 3:  # 截断 最大序列输入长度 seq_len = 128
                if len(src)>len(trg):
                    src.pop()
                elif len(src)<len(trg):
                    trg.pop()
                else:
                    src.pop()
                    trg.pop()
            src.append(end_token)
            trg.append(end_token)

            lst.append(src + [vocab_dict.sep_token_id] + trg)  # [CLS]+[src]+[SEP] + [SEP] + [CLS]+[trg]+[SEP]
            mask.append([0]*(len(src)+1))
        group_lst['input_ids'] = lst
        group_lst['input_mask'] = mask
        return group_lst

    tokenized_datasets = tokenized_datasets.map(
        merge_and_mask,
        batched=True,
        num_proc=1,
        desc=f"merge and mask",
    )
    
    def pad_function(group_lst):
        max_length = seq_len
        group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict.pad_token_id, max_length)
        group_lst['input_mask'] = _collate_batch_helper(group_lst['input_mask'], 1, max_length)  # [PAD] 的位置 mask 都是 1
        return group_lst

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        num_proc=1,
        desc=f"padding",
    )

    print(lm_datasets, 'padded dataset')
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = lm_datasets
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return raw_datasets


def get_corpus(data_args, seq_len, split='train', loaded_vocab=None):

    print('#'*30, '\nLoading dataset {} from {}...'.format(data_args.dataset, data_args.data_dir))

    sentence_lst = {'src':[], 'trg': []}
    
    if split == 'train':
        print('### Loading form the TRAIN set...')
        path = f'{data_args.data_dir}/train.jsonl'
    elif split == 'valid':
        print('### Loading form the VALID set...')
        path = f'{data_args.data_dir}/valid.jsonl'
    elif split == 'meta':
        print('### Loading form the TEST set...')
        path = f'{data_args.data_dir}/meta.jsonl'
    elif split == 'test':
        print('### Loading form the TEST set...')
        path = f'{data_args.data_dir}/test.jsonl'
    else:
        assert False, "invalid split for dataset"

    with open(path, 'r') as f_reader:
        for row in f_reader:
            sentence_lst['src'].append(json.loads(row)['src'].strip())
            sentence_lst['trg'].append(json.loads(row)['trg'].strip())

    print('### Data samples...\n', sentence_lst['src'][:2], sentence_lst['trg'][:2])
        
    # get tokenizer.
    vocab_dict = loaded_vocab

    train_dataset = helper_tokenize_4_mlm(sentence_lst, vocab_dict, seq_len)
    return train_dataset


class TextDataset(Dataset):
    def __init__(self, text_datasets, data_args, model_emb=None):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.data_args = data_args
        self.model_emb = model_emb

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with torch.no_grad():

            input_ids = self.text_datasets['train'][idx]['input_ids']  # 训练和验证集都实际都用了 'train' 这个 key，但数据是不同的
            hidden_state = self.model_emb(torch.tensor(input_ids))

            # obtain the input vectors, only used when word embedding is fixed (not trained end-to-end)
            arr = np.array(hidden_state, dtype=np.float32)  # (128, 128)

            out_kwargs = {}
            out_kwargs['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            out_kwargs['input_mask'] = np.array(self.text_datasets['train'][idx]['input_mask'])  # 此处的mask只在q_sample中使用

            return arr, out_kwargs


def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result


def helper_tokenize_4_mlm(sentence_lst, vocab_dict, seq_len):
    # pipeline式数据处理，1. encode_token  2. mask  3. pad
    # Process.memory_info is expressed in bytes, so convert to megabytes
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    raw_datasets = Dataset2.from_dict(sentence_lst)
    print(raw_datasets)  # Dataset({features: ['src', 'trg'],  num_rows: 19})
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    # 1. convert tokens to ids
    def tokenize_function(examples):
        input_id_x = vocab_dict.encode_token(examples['src'])
        input_id_y = vocab_dict.encode_token(examples['trg'])
        result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y}

        return result_dict

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['src', 'trg'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    print('### tokenized_datasets',
          tokenized_datasets)  # tokenized_datasets Dataset({features: ['input_id_x', 'input_id_y'], num_rows: 19})
    print('### tokenized_datasets...example', tokenized_datasets['input_id_x'][0])
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    # 2. get input mask
    def get_input_mask(group_lst):
        # group_lst{} = {'input_id_x': input_id_x, 'input_id_y': input_id_y}
        # todo: 查找input_id_x里的mask, 构造input_id_mask
        # todo: tokenize输入为字符串，不是字符串列表
        # todo: encode会自动添加[CLS]和[SEP] 修改训练数据生成代码，让max_seq_len-2
        input_mask = []
        src_list = []
        trg_list = []
        for i in range(len(group_lst['input_id_x'])):
            end_token_src = group_lst['input_id_x'][i][-1]  # [SEP]
            end_token_trg = group_lst['input_id_y'][i][-1]  # [SEP]
            # tokenize会导致src和trg不等长
            src = group_lst['input_id_x'][i][:-1]
            trg = group_lst['input_id_y'][i][:-1]
            j = 0
            if len(src) > len(trg):  # 有BUG
                continue
            while len(src) != len(trg):
                assert len(src) < len(trg)
                if j == len(src) - 1 or (src[j] != trg[j] and src[j] != vocab_dict.mask_token_id):  # unmatched
                    src.insert(j, vocab_dict.mask_token_id)
                j += 1
            # 截断 最大序列输入长度 seq_len = 128
            while len(src) > seq_len - 1:
                src.pop()
                trg.pop()
            src.append(end_token_src)
            trg.append(end_token_trg)
            src_mask = ((np.array(src) == vocab_dict.mask_token_id) * 1).tolist()  # 1代表扩散，0代表固定
            assert len(src) == len(trg) == len(src_mask)
            src_list.append(src)
            trg_list.append(trg)
            input_mask.append(src_mask)
        group_lst['input_id_x'] = src_list
        group_lst['input_id_y'] = trg_list
        group_lst['input_mask'] = input_mask
        return group_lst

    tokenized_datasets = tokenized_datasets.map(
        get_input_mask,
        batched=True,
        num_proc=1,
        desc=f"get input mask",
    )

    def pad_function(group_lst):
        input_mask = []
        src_list = []
        trg_list = []
        for i in range(len(group_lst['input_id_x'])):
            src = group_lst['input_id_x'][i]
            trg = group_lst['input_id_y'][i]
            src_mask = group_lst['input_mask'][i]
            pad_num = max(0, seq_len - len(src))
            src += [vocab_dict.pad_token_id] * pad_num
            trg += [vocab_dict.pad_token_id] * pad_num
            src_mask += [0] * pad_num
            assert len(src) == len(trg) == len(src_mask) == seq_len
            src_list.append(src)
            trg_list.append(trg)
            input_mask.append(src_mask)
        group_lst['input_id_x'] = src_list
        group_lst['input_id_y'] = trg_list
        group_lst['input_mask'] = input_mask
        return group_lst

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        num_proc=1,
        desc=f"padding",
    )
    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = lm_datasets
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return raw_datasets


class TextDataset2(Dataset):
    def __init__(self, text_datasets, data_args, model_emb=None):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.data_args = data_args
        self.model_emb = model_emb

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with torch.no_grad():

            input_id_x = self.text_datasets['train'][idx]['input_id_x']  # 训练和验证集都实际都用了 'train' 这个 key，但数据是不同的
            input_id_y = self.text_datasets['train'][idx]['input_id_y']  # 训练和验证集都实际都用了 'train' 这个 key，但数据是不同的
            hidden_state_x = self.model_emb(torch.tensor(input_id_x))
            hidden_state_y = self.model_emb(torch.tensor(input_id_y))

            # obtain the input vectors, only used when word embedding is fixed (not trained end-to-end)
            arr_x = np.array(hidden_state_x, dtype=np.float32)
            arr_y = np.array(hidden_state_y, dtype=np.float32)

            out_kwargs = {}
            out_kwargs['input_id_x'] = np.array(self.text_datasets['train'][idx]['input_id_x'])
            out_kwargs['input_id_y'] = np.array(self.text_datasets['train'][idx]['input_id_y'])
            out_kwargs['input_mask'] = np.array(self.text_datasets['train'][idx]['input_mask'])  # 此处的mask只在q_sample中使用

            return (arr_x, arr_y), out_kwargs