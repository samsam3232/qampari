import json
import os
from tqdm import tqdm
import argparse


def parse_wikidata_log(input_dir: str):

    treated_dirs = dict()
    cache_list = os.listdir(input_dir)
    for cache in cache_list:
        max_cache = -1
        successfully_treated = list()
        curr_dir = os.path.join(input_dir, cache)
        with open(os.path.join(curr_dir, 'app.log'), 'r') as f:
            data = f.read()
        data_lines = data.split('\n')
        for i in range(len(data_lines)):
            if 'now treating' in data_lines[i].lower() and 'error' not in data_lines[i+ 1].lower():
                successfully_treated.append(data_lines[i].lower().split('treating ')[-1])
            if 'cache' in data_lines[i] and int(data_lines[i].split('cache ')[-1]) > max_cache:
                max_cache = int(data_lines[i].lower().split('cache ')[-1])
        if len(successfully_treated) > 0:
            treated_dirs[cache] = dict()
            treated_dirs[cache]['max cache'] = max_cache
            treated_dirs[cache]['successes'] = successfully_treated

    return treated_dirs


def parse_typing_logs(input_dir: str):

    treated_dirs = dict()
    cache_list = os.listdir(input_dir)
    for cache in cache_list:
        max_cache = -1
        successfully_treated = list()
        curr_dir = os.path.join(input_dir, cache)
        with open(os.path.join(curr_dir, 'app.log'), 'r') as f:
            data = f.read()
        data_lines = data.split('\n')
        for i in range(len(data_lines)):
            if 'finished' in data_lines[i].lower():
                max_cache = data_lines[i].split(' ')[-1].split('_')[-1]
            if 'treating' in data_lines[i].lower() and i < (len(data_lines) - 1):
                successfully_treated.append(data_lines[i].lower().split('treating ')[-1].split(' ')[0])
        if len(successfully_treated) > 0:
            treated_dirs[cache] = dict()
            treated_dirs[cache]['max cache'] = max_cache
            treated_dirs[cache]['successes'] = successfully_treated

    return treated_dirs

def parse_table_logs(input_dir: str):

    raise NotImplementedError


def main(input_dir: str, log_type: str):

    parsing_funcs = {'wikidata': parse_wikidata_log, 'typing': parse_typing_logs, 'tables': parse_table_logs}
    treated_dirs = parsing_funcs[log_type](input_dir)
    with open(os.path.join(input_dir, 'treated_chunks.json'), 'w') as f:
        json.dump(treated_dirs, f)
    return 1



if __name__ == "__main__":

    parser = argparse.ArgumentParser('Log parser for failsafe')
    parser.add_argument('-i', '--input_dir', type=str, help = 'Path to where we kept the outputs of the run')
    parser.add_argument('-t', '--log_type', type=str, choices=['wikidata', 'tables', 'typing'], default='typing',
                        help = 'Job that failed that we will parse')
    args = parser.parse_args()
    main(**args)