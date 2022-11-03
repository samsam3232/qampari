#!/usr/bin/env python3
# Copyright GC-DPR authors.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for general purpose data processing
"""
import more_itertools
import json
import logging
import math
import os
import pickle
import tqdm.auto as tqdm
import random
from typing import List, Iterator, Callable
import torch
from torch import Tensor as T
from torch.utils.data import IterableDataset

logger = logging.getLogger()



def reformat_data_question(data):
    url_prefix = len("https://en.wikipedia.org/wiki/")
    new_data = []
    for datum in data:
        ans_list = []
        for answer in datum['answers']:
            if isinstance(answer['proof'],list):
                for url,proof in zip(answer['found_in_url'],answer['proof']):
                    tmp = dict(text=proof.replace("  "," "),title=url[url_prefix:].replace("_"," "))
                    ans_list.append(tmp)

            else:
                title = answer['found_in_url'][url_prefix:].replace("_"," ")
                tmp = dict(text=answer['proof'].replace("  "," "),title=title)
                ans_list.append(tmp)
        new_data.append(dict(question=datum['question_text'],
                    answers=[answer['answer_text'] for answer in datum['answers']] ,
                    positive_ctxs=ans_list,
                    aid=f"{datum['qid']}"
                ))
    return new_data

def reformat_data_answer(data):
    url_prefix = len("https://en.wikipedia.org/wiki/")
    new_data = []
    for datum in data:
        answer_id = 0

        for answer in datum['answers']:
            if isinstance(answer['proof'],list):
                for url,proof in zip(answer['found_in_url'],answer['proof']):
                    title =title=url[url_prefix:].replace("_"," ")
                    tmp = dict(question=datum['question_text'],
                            answers=[answer['answer_text']],
                            positive_ctxs=[dict(text=proof.replace("  "," "),title=title)],
                            aid=f"{datum['qid']}__{answer_id}"
                            )
                    new_data.append(tmp)
                    answer_id +=1

            else:
                title = answer['found_in_url'][url_prefix:].replace("_"," ")
                tmp = dict(question=datum['question_text'],
                    answers=[answer['answer_text']],
                    positive_ctxs=[dict(text=answer['proof'].replace("  "," "),title=title)],
                    aid=f"{datum['qid']}__{answer_id}"
                )
                new_data.append(tmp)
                answer_id +=1
    return new_data

def read_serialized_data_from_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "rb") as reader:
            logger.info('Reading file %s', path)
            data = pickle.load(reader)
            results.extend(data)
            logger.info('Aggregated data size: {}'.format(len(results)))
    logger.info('Total data size: {}'.format(len(results)))
    return results

def read_data_from_json_files(paths: List[str], upsample_rates: List = None) -> List:
    data_list= []
    for path in tqdm.tqdm(paths):
        if path.endswith(".jsonl"):
            with open(path) as f:
                for line in f:
                    data_list.extend(json.loads(line))
        elif path.endswith(".json"):
            with open(path) as f:
                data_list.extend(json.load(f))
        else:
            assert False,f"Unsupported file format {path}"
    return data_list
    #     for line in f:
    #         line =  json.loads(line)
    #         if len(line['question_text'])>0:
    #             data_list.append(line)
    # if format_type == 'answer':
    #     data_list = reformat_data_answer(data_list)
    # elif format_type == 'question':
    #     data_list = reformat_data_question(data_list)
    # else:
    #     raise ValueError('format_type must be answer or question')
    # return data_list

# def read_data_from_json_files(paths: List[str], format_type, upsample_rates: List = None) -> List:
#     data_list= []
#     with open(paths[0]) as f:
#         for line in f:
#             line =  json.loads(line)
#             if len(line['question_text'])>0:
#                 data_list.append(line)
#     if format_type == 'answer':
#         data_list = reformat_data_answer(data_list)
#     elif format_type == 'question':
#         data_list = reformat_data_question(data_list)
#     else:
#         raise ValueError('format_type must be answer or question')
#     return data_list



    # results = []
    # if upsample_rates is None:
    #     upsample_rates = [1] * len(paths)

    # assert len(upsample_rates) == len(paths), 'up-sample rates parameter doesn\'t match input files amount'

    # for i, path in enumerate(paths):
    #     with open(path, 'r', encoding="utf-8") as f:
    #         logger.info('Reading file %s' % path)
    #         data = json.load(f)
    #         upsample_factor = int(upsample_rates[i])
    #         data = data * upsample_factor
    #         results.extend(data)
    #         logger.info('Aggregated data size: {}'.format(len(results)))

class ShardedDataIterator(object):
    """
    General purpose data iterator to be used for Pytorch's DDP mode where every node should handle its own part of
    the data.
    Instead of cutting data shards by their min size, it sets the amount of iterations by the maximum shard size.
    It fills the extra sample by just taking first samples in a shard.
    It can also optionally enforce identical batch size for all iterations (might be useful for DP mode).
    """

    def __init__(self, data: list, shard_id: int = 0, num_shards: int = 1, batch_size: int = 1, shuffle=True,
                 shuffle_seed: int = 0, offset: int = 0,
                 strict_batch_size: bool = False
                 ):

        self.data = data
        total_size = len(data)

        self.shards_num = max(num_shards, 1)
        self.shard_id = max(shard_id, 0)

        samples_per_shard = math.ceil(total_size / self.shards_num)

        self.shard_start_idx = self.shard_id * samples_per_shard

        self.shard_end_idx = min(self.shard_start_idx + samples_per_shard, total_size)

        if strict_batch_size:
            self.max_iterations = math.ceil(samples_per_shard / batch_size)
        else:
            self.max_iterations = int(samples_per_shard / batch_size)

        logger.debug(
            'samples_per_shard=%d, shard_start_idx=%d, shard_end_idx=%d, max_iterations=%d', samples_per_shard,
            self.shard_start_idx,
            self.shard_end_idx,
            self.max_iterations)

        self.iteration = offset  # to track in-shard iteration status
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.strict_batch_size = strict_batch_size

    def total_data_len(self) -> int:
        return len(self.data)

    def iterate_data(self, epoch: int = 0) -> Iterator[List]:
        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(self.data)

        # if resuming iteration somewhere in the middle of epoch, one needs to adjust max_iterations

        max_iterations = self.max_iterations - self.iteration

        shard_samples = self.data[self.shard_start_idx:self.shard_end_idx]
        for i in range(self.iteration * self.batch_size, len(shard_samples), self.batch_size):
            items = shard_samples[i:i + self.batch_size]
            if self.strict_batch_size and len(items) < self.batch_size:
                logger.debug('Extending batch to max size')
                items.extend(shard_samples[0:self.batch_size - len(items)])
            self.iteration += 1
            yield items

        # some shards may done iterating while the others are at the last batch. Just return the first batch
        while self.iteration < max_iterations:
            logger.debug('Fulfilling non complete shard='.format(self.shard_id))
            self.iteration += 1
            batch = shard_samples[0:self.batch_size]
            yield batch

        logger.debug('Finished iterating, iteration={}, shard={}'.format(self.iteration, self.shard_id))
        # reset the iteration status
        self.iteration = 0

    def get_iteration(self) -> int:
        return self.iteration

    def apply(self, visitor_func: Callable):
        for sample in self.data:
            visitor_func(sample)



class ShardedDistinctDataIterableDataset(IterableDataset):
    def __init__(self, data: list, shard_id: int = 0, num_shards: int = 1, batch_size: int = 1, shuffle=True,
                 shuffle_seed: int = 0, offset: int = 0,
                 strict_batch_size: bool = False, process_fn=None):
                 
        

        self.data = data
        total_size = len(data)

        self.shards_num = max(num_shards, 1)
        self.shard_id = max(shard_id, 0)

        samples_per_shard = math.ceil(total_size / self.shards_num)

        self.shard_start_idx = self.shard_id * samples_per_shard

        self.shard_end_idx = min(self.shard_start_idx + samples_per_shard, total_size)

        if strict_batch_size:
            self.max_iterations = math.ceil(samples_per_shard / batch_size)
        else:
            self.max_iterations = int(samples_per_shard / batch_size)

        logger.info(
            'samples_per_shard=%d, shard_start_idx=%d, shard_end_idx=%d, max_iterations=%d', samples_per_shard,
            self.shard_start_idx,
            self.shard_end_idx,
            self.max_iterations)

        self.iteration = offset  # to track in-shard iteration status
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.strict_batch_size = strict_batch_size


        # ShardedDataIterator.__init__(self, *args, **kwargs)

        self.epoch = None
        self.curr_idx = None
        self.idx_gen = None
        self.shard_samples = None
        self._max_iterations = None
        self._ended = False
        self.process_fn = process_fn
        
    def get_iteration(self) -> int:
        return self.iteration

    def apply(self, visitor_func: Callable):
        for sample in self.data:
            visitor_func(sample)

    def total_data_len(self) -> int:
        return len(self.data)

    def set_epoch(self, epoch: int):
        self.epoch = epoch


    def get_epoch_shuffle(self,answer_list):
        rand = random.Random(self.shuffle_seed + self.epoch)  
        tmp = more_itertools.bucket(answer_list,key=lambda x:x['aid'].split("__")[0])
        q_dict = {key:list(tmp[key]) for key in list(tmp)}
        
        for _,datum in q_dict.items():        
            rand.shuffle(datum)


        all_batch = []
        total = sum(len(x) for x in q_dict.values())
        while True:
            if total<self.batch_size:
                    return all_batch
            batch_list = []
            counter_keys = set(q_dict.keys())
            if len(counter_keys)>self.batch_size:
                chosen_questions = rand.sample(counter_keys,k=self.batch_size)
            else:
                return all_batch
            for chosen_question in chosen_questions:
                if len(q_dict[chosen_question])>0:
                    batch_list.append(q_dict[chosen_question].pop())
                else:
                    q_dict.pop(chosen_question)
            shard_batch = list(more_itertools.distribute(self.shards_num, batch_list)[self.shard_id])

            # shard_batch = list(more_itertools.distribute(self.shards_num, batch_list)[self.shard_id])
            yield shard_batch
            total = total - len(batch_list)
        


    def __iter__(self):

        self.inner_iter = iter(self.get_epoch_shuffle(self.data))
        self._is_first_batch = True
        # if self.shuffle:
        #     # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
        #     epoch_rnd = random.Random(self.shuffle_seed + self.epoch)
        #     epoch_rnd.shuffle(self.data)

        # # if resuming iteration somewhere in the middle of epoch, one needs to adjust max_iterations
        self._max_iterations = self.max_iterations - self.iteration
        self.iteration = 0
        # assert not self.strict_batch_size
        self._ended = False

        return self




    def __next__(self):
        if not self._ended:
            try:
                items = next(self.inner_iter)
                if self._is_first_batch:
                    self._is_first_batch = False
                    self._first_batch = items
                # items = self.shard_samples[i:i + self.batch_size]
                self.iteration += 1
                if self.process_fn:
                    random.seed(self.shuffle_seed + self.epoch + self.iteration)
                    return self.process_fn(items)
                return items
            except StopIteration:
                self._ended = True

        # some shards may done iterating while the others are at the last batch. Just return the first batch
        if self.iteration < self._max_iterations:
            logger.debug('Fulfilling non complete shard='.format(self.shard_id))
            self.iteration += 1
            batch = self._first_batch
            if self.process_fn:
                random.seed(self.shuffle_seed + self.epoch + self.iteration)
                return self.process_fn(batch)
            return batch

        logger.info('Finished iterating, iteration={}, shard={}'.format(self.iteration, self.shard_id))
        # reset the iteration status
        raise StopIteration


class ShardedDataIterableDataset(ShardedDataIterator, IterableDataset):
    def __init__(self, *args, process_fn=None, **kwargs):
        ShardedDataIterator.__init__(self, *args, **kwargs)

        self.epoch = None
        self.curr_idx = None
        self.idx_gen = None
        self.shard_samples = None
        self._max_iterations = None
        self._ended = False
        self.process_fn = process_fn

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + self.epoch)
            epoch_rnd.shuffle(self.data)

        # if resuming iteration somewhere in the middle of epoch, one needs to adjust max_iterations
        self._max_iterations = self.max_iterations - self.iteration
        self.shard_samples = self.data[self.shard_start_idx:self.shard_end_idx]
        self.idx_gen = iter(range(self.iteration * self.batch_size, len(self.shard_samples), self.batch_size))
        self.iteration = 0
        # assert self.strict_batch_size
        self._ended = False

        return self

    def __next__(self):
        if not self._ended:
            try:
                i = next(self.idx_gen)
                items = self.shard_samples[i:i + self.batch_size]
                self.iteration += 1
                if self.process_fn:
                    random.seed(self.shuffle_seed + self.epoch + self.iteration)
                    return self.process_fn(items)
                return items
            except StopIteration:
                self._ended = True

        # some shards may done iterating while the others are at the last batch. Just return the first batch
        if self.iteration < self._max_iterations:
            logger.debug('Fulfilling non complete shard='.format(self.shard_id))
            self.iteration += 1
            batch = self.shard_samples[0:self.batch_size]
            if self.process_fn:
                random.seed(self.shuffle_seed + self.epoch + self.iteration)
                return self.process_fn(batch)
            return batch

        logger.info('Finished iterating, iteration={}, shard={}'.format(self.iteration, self.shard_id))
        # reset the iteration status
        raise StopIteration


def normalize_question(question: str) -> str:
    if question[-1] == '?':
        question = question[:-1]
    return question


class Tensorizer(object):
    """
    Component for all text to model input data conversions and related utility methods
    """

    # Note: title, if present, is supposed to be put before text (i.e. optional title + document body)
    def text_to_tensor(self, text: str, title: str = None, add_special_tokens: bool = True):
        raise NotImplementedError

    def get_pair_separator_ids(self) -> T:
        raise NotImplementedError

    def get_pad_id(self) -> int:
        raise NotImplementedError

    def get_attn_mask(self, tokens_tensor: T):
        raise NotImplementedError

    def is_sub_word_id(self, token_id: int):
        raise NotImplementedError

    def to_string(self, token_ids, skip_special_tokens=True):
        raise NotImplementedError

    def set_pad_to_max(self, pad: bool):
        raise NotImplementedError
