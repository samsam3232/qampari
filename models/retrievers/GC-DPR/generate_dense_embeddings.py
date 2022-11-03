#!/usr/bin/env python3
# Copyright GC-DPR authors.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import os
import pathlib
import argparse
import logging
import pickle
from typing import List, Tuple
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset

from dpr.models import init_biencoder_components
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params
from dpr.utils.data_utils import Tensorizer,reformat_data_answer, read_data_from_json_files
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint,move_to_device
import more_itertools
import glob

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


class CtxDataset(Dataset):
    def __init__(self, ctx_rows: List[Tuple[object, str, str]], tensorizer: Tensorizer, insert_title: bool = True):
        self.rows = ctx_rows
        self.tensorizer = tensorizer
        self.insert_title = insert_title

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        ctx = self.rows[item]

        return self.tensorizer.text_to_tensor(ctx[1], title=ctx[2] if self.insert_title else None)


def no_op_collate(xx: List[object]):
    return xx


def gen_ctx_vectors(ctx_rows: List[Tuple[object, str, str]], model: nn.Module, tensorizer: Tensorizer,
                    insert_title: bool = True, fp16: bool = False) -> List[Tuple[object, np.array]]:
    bsz = args.batch_size
    total = 0
    results = []

    dataset = CtxDataset(ctx_rows, tensorizer, insert_title)
    loader = DataLoader(
        dataset, shuffle=False, num_workers=2, collate_fn=no_op_collate, drop_last=False, batch_size=bsz)

    for batch_id, batch_token_tensors in enumerate(tqdm(loader)):
        ctx_ids_batch = move_to_device(torch.stack(batch_token_tensors, dim=0),args.device)
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch),args.device)
        ctx_attn_mask = move_to_device(tensorizer.get_attn_mask(ctx_ids_batch),args.device).int()
        with torch.no_grad():
            if fp16:
                with autocast():
                    out = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)[1]
            else:
                out= model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)[1]

        out = out.float().cpu()

        batch_start = batch_id*bsz
        ctx_ids = [r[0] for r in ctx_rows[batch_start:batch_start + bsz]]

        assert len(ctx_ids) == out.size(0)

        total += len(ctx_ids)

        results.extend([
            (ctx_ids[i], out[i].view(-1).numpy())
            for i in range(out.size(0))
        ])

    return results


def main(args):
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)
    print_args(args)
    
    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

    encoder = encoder.ctx_model

    encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                            args.local_rank,
                                            args.fp16,
                                            args.fp16_opt_level)
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info('Loading saved model state ...')
    logger.debug('saved model keys =%s', saved_state.model_dict.keys())

    prefix_len = len('ctx_model.')
    ctx_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                 key.startswith('ctx_model.')}
    model_to_load.load_state_dict(ctx_state)
    ctx_file = args.ctx_file.replace('"',"")
    logger.info('reading data from file=%s', ctx_file)
    file_list = glob.glob(ctx_file)
    file_list = more_itertools.distribute(args.num_shards, file_list)[args.shard_id]
    rows = []
    import json
    for file_path in file_list:
        with open(file_path) as f:
            for line in f:
                datum = json.loads(line)
                # for i,proof_el in enumerate(datum['positive_ctxs']): 
                rows.append([datum['id'],datum['contents'],datum['meta']['title']])

    data = gen_ctx_vectors(rows, encoder, tensorizer, True, fp16=args.fp16)

    file = args.out_file + '_' + str(args.shard_id) + '.pkl'
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info('Writing results to %s' % file)
    with open(file, mode='wb') as f:
        pickle.dump(data, f)

    logger.info('Total passages processed %d. Written to %s', len(data), file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--ctx_file', type=str, default=None, help='Path to passages set .tsv file')
    parser.add_argument('--out_file', required=True, type=str, default=None,
                        help='output file path to write results to')
    parser.add_argument('--shard_id', type=int, default=0, help="Number(0-based) of data shard to process")
    parser.add_argument('--num_shards', type=int, default=1, help="Total amount of data shards")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for the passage encoder forward pass")
    args = parser.parse_args()

    assert args.model_file, 'Please specify --model_file checkpoint to init model weights'

    setup_args_gpu(args)

    
    main(args)
