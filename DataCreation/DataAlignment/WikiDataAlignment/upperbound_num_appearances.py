import os
import argparse
import multiprocessing as mp
from typing import DefaultDict
from DataCreation.DataAlignment.utils import read_parsed_wikipedia
import json
from tqdm import tqdm

def check_single_dir(inputs):

    input_dir, targets = inputs
    results = DefaultDict(lambda : 0)
    if 'indices' in input_dir:
        return results
    dir_list = os.listdir(input_dir)
    for dir in tqdm(dir_list):
        if 'index' in dir:
            continue
        all_pages = read_parsed_wikipedia(os.path.join(input_dir, dir))
        for page in all_pages:
            for target in targets:
                for alias in targets[target]:
                    if alias in page['text']:
                        results[target] += 1
                        break

    with open(os.path.join(input_dir, 'targets_counts.json'), 'w') as f:
        json.dump(results, f)

def collect_targets(input_args):

    input_dir, wikidata_ub = input_args
    dir_list = os.listdir(input_dir)
    targets = dict()
    for dir in dir_list:
        if 'app' in dir:
            continue
        sub_dir_list = os.listdir(os.path.join(input_dir, dir))
        for sub_dir in sub_dir_list:
            if 'newle'not in sub_dir:
                continue
            with open(os.path.join(input_dir, dir, sub_dir), 'r') as f:
                curr_data = json.load(f)

            for key in curr_data:
                if curr_data[key][-1] <= wikidata_ub and (key.split('###')[-2] not in targets):
                    targets[key.split('###')[-2]] = [key.split('###')[-1]] + curr_data[key][0]

    return targets


def main(target_input_dir, wikipedia_parsed_dir, wikipedia_ub, output_dir):

    processes = mp.Pool(50)
    target_dirs = os.listdir(target_input_dir)
    target_dirs_true = [i for i in target_dirs if (int(i.split('_')[-1]) >= 300)]

    targets_res = processes.map(collect_targets, [(os.path.join(target_input_dir, i), wikipedia_ub) for i in target_dirs_true])
    targets = dict()
    for specific_target in targets_res:
        targets.update(specific_target)

    processes.map(check_single_dir, [(os.path.join(wikipedia_parsed_dir, i), targets) for i in os.listdir(wikipedia_parsed_dir)])
    final_results = DefaultDict(lambda: 0)
    for dir in os.listdir(wikipedia_parsed_dir):
        with open(os.path.join(wikipedia_parsed_dir, dir, 'targets_counts.json'), 'r') as f:
            curr_count = json.load(f)

        for key in curr_count:
            final_results[key] += curr_count[key]

    processes.close()
    with open(os.path.join(output_dir, 'wikipedia_upper_bounds.json'), 'w') as f:
        json.dump(final_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Finding wikipedia upper bounds")
    parser.add_argument('-t', '--target_input_dir', type=str)
    parser.add_argument('-w', '--wikipedia_parsed_dir', type=str)
    parser.add_argument('-u', '--wikipedia_ub', type=int)
    parser.add_argument('-o', '--output_dir', type=str)
    args = parser.parse_args()
    main(**vars(args))
