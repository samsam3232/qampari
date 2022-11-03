import os
import simplejson as json
from tqdm import tqdm
import argparse

def filter(base_dir_questions, base_infos_wiki, output_txt):

    """
    Given the path to a chunk, outputs a file with a list of all the instances of the answers (i.e 'Albert Einstein' is
    an instance of human).
    """

    questions_caches = os.listdir(base_dir_questions)
    with open(base_infos_wiki, 'r') as f:
        curr_infos = json.load(f)
    print('Loaded infos')
    relevant_entities = set()

    for cache in tqdm(questions_caches):
        if 'app' in cache:
            continue
        curr_cache = os.path.join(base_dir_questions, cache)
        questions_list = os.listdir(curr_cache)
        for question in questions_list:
            with open(os.path.join(curr_cache, question), 'r') as f:
                questions_ans = list()
                for line in f:
                    questions_ans.append(json.loads(line))
            for quest in questions_ans:
                for ans in quest['answer_list']:
                    entity = ans['answer_wiki_id']
                    if entity in curr_infos and 'composition' in curr_infos[entity] and 'P31' in curr_infos[entity]['composition']:
                        for father in curr_infos[entity]['composition']['P31']:
                            relevant_entities.add(father)


    txt = '\n'.join(list(relevant_entities))
    with open(output_txt, 'w') as f:
        f.write(txt)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--base_dir_questions', type=str)
    parser.add_argument('-w', '--base_infos_wiki', type=str)
    parser.add_argument('-t', '--output_txt', type=str)
    args = parser.parse_args()
    filter(**vars(args))