import json
import os
import logging
from collections import defaultdict
from typing import DefaultDict, Dict, List
from DataCreation.DataAlignment.utils.alignment_utils import get_webpage_sentences, find_all_phrases
from DataCreation.complex_questions.WikiData.utils import rephrase_quest_composition, filter_answer
from DataCreation.DataAlignment.utils.properties_constants import COMP_PROPERTIES
from tqdm import tqdm
import argparse


def check_labels(labels: List, sentences: str):

    """
    Looks for the labels within the wikipedia page, returns the first label found.
    """

    for label in labels:
        currently_checked = find_all_phrases(sentences, label.lower())
        if len(currently_checked) > 0:
            return currently_checked, label.lower()
    return None, None


def align_comp_entity(mappings: Dict, basic_entity: str, comp_entity: str, list_indices):


    orig_url = mappings[basic_entity]['url'] if (basic_entity in mappings and 'url' in mappings[basic_entity]) else ""
    original_sentence = get_webpage_sentences([orig_url], list_indices)
    aliases_orig = mappings[basic_entity]['potential_labels'] if (basic_entity in mappings and 'potential_labels' in mappings[basic_entity]) else ['']

    comp_url = mappings[comp_entity]['url'] if (comp_entity in mappings and 'url' in  mappings[comp_entity]) else ""
    aliases_comp = mappings[comp_entity]['potential_labels'] if (comp_entity in mappings and 'potential_labels' in mappings[comp_entity]) else ['']

    answer_data = dict()
    answer_data['answer_url'] = comp_url
    answer_data['answer_text'] = aliases_comp[0]
    answer_data['aliases'] = aliases_comp
    answer_data['answer_wiki_id'] = comp_entity

    currently_checked, label = check_labels(aliases_comp, original_sentence.lower())
    if currently_checked is not None:
        answer_data['potential_proofs'] = [{'found_in_url': orig_url, 'proof': sent, 'found_label': label} for sent in
                                           currently_checked]
        return 1, answer_data

    comp_sentence = get_webpage_sentences([comp_url], list_indices)
    currently_checked, label = check_labels(aliases_orig, comp_sentence.lower())
    if currently_checked is not None:
        answer_data['potential_proofs'] = [{'found_in_url': comp_url, 'proof': sent, 'found_label': label} for sent in
                                           currently_checked]
        return 1, answer_data

    has_url = 1 if (len(orig_url) > 0 or len(comp_url) > 0) else 0
    return has_url, None


def find_possible_comps(answers: List, relevant_infos: Dict, wikidata_mappings: Dict, list_indices: Dict):

    potential_comps = defaultdict(lambda: list())
    num_url, num_answer = defaultdict(lambda : 0), defaultdict(lambda : 0)
    for ans in answers:
        curr_entity = ans['answer_wiki_id']
        curr_infos = relevant_infos[curr_entity]['composition'] if (curr_entity in relevant_infos and 'composition' in relevant_infos[curr_entity]) else {}
        for prop in curr_infos:
            if prop in COMP_PROPERTIES:
                for ent in curr_infos[prop]:
                    has_url, curr_answer = align_comp_entity(wikidata_mappings, curr_entity, ent, list_indices)
                    num_url[prop] += has_url
                    if curr_answer is not None:
                        curr_answer['orig_answer_url'] = ans['answer_url']
                        curr_answer['orig_answer_text'] = ans['answer_text']
                        curr_answer['orig_answer_aliases'] = ans['aliases']
                        curr_answer['orig_answer_potential_proofs'] = ans['potential_proofs']
                        curr_answer['orig_answer_wiki_id'] = ans['answer_wiki_id']
                        potential_comps[prop].append(curr_answer)
                        num_answer[prop] += 1

    return potential_comps, num_url, num_answer


def create_potential_questions(relevant_infos: Dict, wikidata_mappings: Dict, list_indices: Dict, questions_path: str,
                               output_path: str):

    """
    Given a path to the rephrased questions and all the relevant infos, will return a dict of all the possible questions
    with the composition properties.
    """

    all_questions = list()
    with open(questions_path, 'r') as f:
        for line in f:
            all_questions.append(json.loads(line))

    comp_questions = list()
    for sample in all_questions:

        if not filter_answer([sample['base_quest_num'], sample['num_has_wikipage'], sample['original_num_answer']], thresh = 0.7):
            continue

        curr_property = sample['qid'].split('__')[0]
        possible_comps, num_url, num_answer = find_possible_comps(sample['answer_list'], relevant_infos, wikidata_mappings, list_indices)
        for comp_prop in possible_comps:
            if len(possible_comps[comp_prop]) >= 5 and comp_prop in COMP_PROPERTIES and comp_prop != curr_property:
                if not filter_answer([num_answer[comp_prop], num_url[comp_prop], len(sample['answer_list'])], thresh = 0.7):
                    continue
                curr_comp_questions = dict()
                for key in sample:
                    if key == 'answer_list' or key == 'qid' or key == 'question_text':
                        continue
                    curr_comp_questions[key] = sample[key]
                curr_comp_questions['orig_question_text'] = sample['question_text']
                curr_comp_questions['qid'] = f'{comp_prop}__{sample["qid"]}'
                new_question = rephrase_quest_composition(sample['question_text'], curr_property, comp_prop, sample['type'])
                curr_comp_questions['answer_list'] = possible_comps[comp_prop]
                curr_comp_questions['question_text'] = new_question
                comp_questions.append(curr_comp_questions)

    with open(output_path, 'w') as f:
        for sample in comp_questions:
            f.write(json.dumps(sample) + '\n')


def main(relevant_infos_path: str, questions_path: str, output_dir: str, wikidata_mappings_path: str, list_indices_path: str):

    logging.basicConfig(filename=os.path.join(output_dir, "app.log"), filemode="w", level=logging.INFO)

    with open(wikidata_mappings_path, 'r') as f:
        wikidata_mappings = json.load(f)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print('Loaded mappings')

    with open(list_indices_path, 'r') as f:
        list_indices = json.load(f)

    with open(relevant_infos_path, 'r') as f:
        relevant_infos = json.load(f)

    print('Loaded indices')

    cache_list = os.listdir(questions_path)
    for cache in tqdm(cache_list):
        if 'app' in cache:
            continue
        if not os.path.exists(os.path.join(output_dir, cache)):
            os.mkdir(os.path.join(output_dir, cache))
        curr_cache = os.path.join(questions_path, cache)
        questions_dirs = os.listdir(curr_cache)
        for dir_name in questions_dirs:
            if 'specified_question' not in dir_name:
                continue
            create_potential_questions(relevant_infos, wikidata_mappings, list_indices, os.path.join(curr_cache, dir_name),
                                       os.path.join(output_dir, cache, f'original_comp_questions.json'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--relevant_infos_path', type=str)
    parser.add_argument('-q', '--questions_path', type=str)
    parser.add_argument('-o', '--output_dir', type=str)
    parser.add_argument('-w', '--wikidata_mappings_path', type=str)
    parser.add_argument('-l', '--list_indices_path', type=str)
    args = parser.parse_args()
    main(**vars(args))
