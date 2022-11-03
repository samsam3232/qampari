import json
import os
import logging
from tqdm import tqdm
from typing import DefaultDict, List, Dict
import argparse
from DataCreation.DataAlignment.utils.alignment_utils import get_webpage_sentences, find_all_phrases
from DataCreation.DataAlignment.utils.properties_constants import SELECTED_PRORPERTIES


def check_labels(labels: List, sentences: str):

    """
    Looks for the labels within the wikipedia page, returns the first label found.
    """

    for label in labels:
        currently_checked = find_all_phrases(sentences, label.lower())
        if len(currently_checked) > 0:
            return currently_checked, label.lower()
    return None, None


def align_entities(original_entity: str, entities: List, list_indices: Dict, relevant_infos: Dict, all_mappings: Dict, curr_question: Dict):

    """
    Given the original entity (the one in the question) and all the answers, tries to align each answer with the entity
    (i.e to find e1 in the wikipedia page of e2 or the inverse).
    """

    orig_url = all_mappings[original_entity]['url'] if (original_entity in all_mappings and 'url' in all_mappings[original_entity]) else ""
    original_sentence = get_webpage_sentences([orig_url], list_indices)
    aliases = all_mappings[original_entity]['potential_labels'] if (original_entity in all_mappings and 'potential_labels' in all_mappings[original_entity]) else ['']
    curr_question['entities'] = [{'entity_text': aliases[0], 'aliases': aliases, 'entity_url': orig_url,
                                  'entity_id': original_entity}]
    curr_question['answer_list'] = list()
    in_list_indices, has_ref = 0, dict()
    for entity in entities:
        try:
            answer_data = dict()
            answer_data['answer_url'] = ''
            curr_sentences = ""
            if entity in relevant_infos and 'url' in relevant_infos[entity] and relevant_infos[entity]['url'] in list_indices:
                curr_sentences = get_webpage_sentences([relevant_infos[entity]['url']], list_indices)
                answer_data['answer_url'] = relevant_infos[entity]['url']
                in_list_indices += 1
            curr_labels = relevant_infos[entity]['potential_labels'] if entity in relevant_infos and 'potential_labels' in relevant_infos[entity] else ['']
            answer_data['answer_text'] = curr_labels[0]
            answer_data['aliases'] = curr_labels
            answer_data['answer_wiki_id'] = entity
            currently_checked, label = check_labels(aliases, curr_sentences.lower())
            if currently_checked is not None:
                answer_data['potential_proofs'] = [{'found_in_url': relevant_infos[entity]['url'], 'proof': sent, 'found_label': label}
                                                   for sent in currently_checked]
                curr_question['answer_list'].append(answer_data)
                continue
            currently_checked, label = check_labels(curr_labels, original_sentence.lower())
            if currently_checked is not None:
                answer_data['potential_proofs'] = [{'found_in_url': orig_url, 'proof': sent, 'found_label': label}
                                                   for sent in currently_checked]
                curr_question['answer_list'].append(answer_data)
        except Exception:
            logging.error("\n Exception occurred", exc_info=True)
            continue

    curr_question['num_has_wikipage'] = in_list_indices

    return curr_question


def load_files(input_path: str, infos_path: str, list_indices_path: str, all_mappings_path: str):

    """
    Load the relevant data from the files.
    """

    with open(input_path, 'r') as f:
        data = json.load(f)
    print('Loaded questions')

    with open(infos_path, 'r') as f:
        relevant_infos = json.load(f)
    print('Loaded infos')

    with open(list_indices_path, 'r') as f:
        list_indices = json.load(f)
    print('Loaded indices')

    with open(all_mappings_path, 'r') as f:
        all_mappings = json.load(f)
    print('Loaded mappings')

    return data, relevant_infos, list_indices, all_mappings


def align_queries(input_path: str, infos_path: str, output: str, list_indices_path: str, all_mappings_path: str):

    logging.basicConfig(filename=os.path.join(output, "app.log"), filemode="w", level=logging.INFO)

    data, relevant_infos, list_indices, all_mappings = load_files(input_path, infos_path, list_indices_path, all_mappings_path)
    cache_num, total_answers, all_questions, question_num = 1, 0, list(), 0

    for key in tqdm(data):
        if len(data[key]) > 3000:
            continue
        logging.info(f'Now treating {key}')
        curr_question = dict()
        prop, entity = key.split("##")
        curr_question['qid'] = f'{prop}__{entity}__wikidata_simple_unspecified__{question_num}'
        curr_question['original_num_answer'] = len(data[key])
        question_num += 1

        try:
           curr_question = align_entities(entity, data[key][:3000], list_indices, relevant_infos, all_mappings, curr_question)
        except Exception:
            logging.error("\n Exception occurred", exc_info=True)
            continue

        question = f"Who has {curr_question['entities'][0]['entity_text']} as {SELECTED_PRORPERTIES[prop]}"
        curr_question['base_question'] = question
        curr_question['base_quest_num'] = len(curr_question['answer_list'])
        all_questions.append(curr_question)

        total_answers += len(curr_question['answer_list'])
        if total_answers >= 1000:
            logging.info(f'Saving cache {cache_num}')
            if not os.path.exists(os.path.join(output, f'cache_{cache_num}')):
                os.mkdir(os.path.join(output, f'cache_{cache_num}'))
            with open(os.path.join(output, f'cache_{cache_num}/original_simple_questions.jsonl'), 'w') as f:
                for sample in all_questions:
                    f.write(json.dumps(sample) + '\n')
            total_answers, all_questions = 0, list()
            cache_num += 1
        logging.info('Completed job')

    if len(all_questions) != 0:
        logging.info(f'Saving cache {cache_num}')
        if not os.path.exists(os.path.join(output, f'cache_{cache_num}')):
            os.mkdir(os.path.join(output, f'cache_{cache_num}'))
        with open(os.path.join(output, f'cache_{cache_num}/original_simple_questions.jsonl'), 'w') as f:
            for sample in all_questions:
                f.write(json.dumps(sample) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument('--infos_path', type=str)
    parser.add_argument('--list_indices_path', type=str)
    parser.add_argument("--all_mappings_path", type=str)
    args = parser.parse_args()
    align_queries(**vars(args))