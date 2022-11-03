import argparse
import json
import os
from collections import defaultdict
from tqdm import tqdm


def clean_file(quest_path: str, newlen_path: str, filtered_path: str):

    with open(quest_path, 'r') as f:
        questions = json.load(f)
    with open(newlen_path, 'r') as f:
        newlengths = json.load(f)
    with open(filtered_path, 'r') as f:
        filtered = json.load(f)

    final_questions = defaultdict(lambda: defaultdict(lambda : list()))
    for key in newlengths:
        base_quest = f'Who has {key.split("##")[-1]} as {key.split("##")[1]}'
        if base_quest not in filtered:
            continue
        for subquestion in filtered[base_quest]:
            final_questions[base_quest][subquestion].append(questions[base_quest][subquestion][0])
            all_answers = dict()
            for ans in filtered[base_quest][subquestion]:
                all_answers[ans] = questions[base_quest][subquestion][1][ans][:3] + [filtered[base_quest][subquestion][ans][1]]
            final_questions[base_quest][subquestion].append(all_answers)

    with open(quest_path.replace('_questions', '_newquestions'), 'w') as f:
        json.dump(final_questions, f)


def clean_cache(cache_path: str, newlen_path: str, filtered_path: str):

    prop_list = os.listdir(filtered_path)
    for prop in prop_list:
        curr_prop = prop.split('_')[0]
        if 'kept_sa' not in prop:
            continue
        clean_file(os.path.join(cache_path, f'{curr_prop}_questions.json'), os.path.join(newlen_path, f'{curr_prop}_newlengths.json'),
                   os.path.join(filtered_path, prop))


def clean_chunk(chunk_path: str, newlen_path: str):

    cache_list = os.listdir(chunk_path)
    for cache in cache_list:
        if 'app' in cache:
            continue
        if not os.path.exists(os.path.join(chunk_path, cache + '/filtering/')):
            continue
        clean_cache(os.path.join(chunk_path, cache), os.path.join(newlen_path, cache), os.path.join(chunk_path, cache + '/filtering/'))


def main(chunk_path: str, newlen_path: str):

    for i in tqdm(range(15)):
        clean_chunk(os.path.join(chunk_path, f'chunk_{i+1}'), os.path.join(newlen_path, f'chunk_{i+1}'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser('cleaning files')
    parser.add_argument('-c', '--chunk_path', type=str)
    parser.add_argument('-n', '--newlen_path', type=str)
    args = parser.parse_args()
    main(**vars(args))