import argparse
from typing import Dict, List
from collections import defaultdict
from copy import deepcopy
import json
import os
from DataCreation.complex_questions.WikiData.utils import reformulate_questions_intersec, filter_answer
from tqdm import tqdm


def create_dictionary(questions: List):

    """
    Creates a dictionary of wikiids and the questions they answer to.
    """

    meta_data, answers_data, answer_to_quest = dict(), defaultdict(lambda: dict()), defaultdict(lambda: list())
    wiki_ids = dict()
    for sample in questions:
        curr_ids = list()
        curr_meta = deepcopy(sample)
        curr_meta.pop('answer_list')
        meta_data[sample['question_text']] = curr_meta
        for answer in sample['answer_list']:
            curr_ids.append(answer['answer_wiki_id'])
            curr_ans_meta = deepcopy(answer)
            answers_data[answer['answer_wiki_id']][sample['question_text']] = curr_ans_meta
            answer_to_quest[answer['answer_wiki_id']].append(sample['question_text'])
        wiki_ids[sample['question_text']] = set(curr_ids)

    return meta_data, answers_data, answer_to_quest, wiki_ids


def create_intersec_questions_data(meta_data: Dict, answers_data: Dict, potential_questions: Dict, wiki_ids: Dict):

    intersec_questions_data = list()
    for i, pot_couple in enumerate(potential_questions):
        q1, q2 = pot_couple.split('##')
        if wiki_ids[q1] <= wiki_ids[q2] or wiki_ids[q2] <= wiki_ids[q1]:
            continue

        curr_data = dict()
        curr_data['original_num_answer'] = min(meta_data[q1]['original_num_answer'], meta_data[q2]['original_num_answer'])
        curr_data['entities'] = [meta_data[q1]['entities'], meta_data[q2]['entities']]
        curr_data['num_has_wikipage'] = min(meta_data[q1]['num_has_wikipage'], meta_data[q2]['num_has_wikipage'])
        curr_data['q1_qid'] = meta_data[q1]['qid']
        curr_data['q2_qid'] = meta_data[q2]['qid']
        first_couple = '__'.join(meta_data[q1]['qid'].split('__')[:2])
        second_couple = '__'.join(meta_data[q2]['qid'].split('__')[:2])
        curr_data['qid'] = f'{first_couple}__{second_couple}__wikidata_intersection__{i}'
        curr_data['q1_base_question'] = q1
        curr_data['q2_base_question'] = q2
        curr_data['q1_question_text'] = meta_data[q1]['question_text']
        curr_data['q2_question_text'] = meta_data[q2]['question_text']
        curr_data['type'] = [meta_data[q1]['type'], meta_data[q2]['type']]
        curr_data['base_quest_num'] = min(meta_data[q1]['base_quest_num'], meta_data[q2]['base_quest_num'])
        curr_data['answer_list'] = list()
        for ans in potential_questions[pot_couple]:
            new_ans = dict()
            new_ans['answer_text'] = answers_data[ans][q1]['answer_text']
            new_ans['answer_url'] = answers_data[ans][q1]['answer_url']
            new_ans['aliases'] = answers_data[ans][q1]['aliases']
            new_ans['answer_wikiid'] = ans
            new_ans['potential_proofs'] = [answers_data[ans][q1]['potential_proofs'], answers_data[ans][q2]['potential_proofs']]
            curr_data['answer_list'].append(new_ans)
        intersec_questions_data.append(curr_data)

    return intersec_questions_data


def find_potential_questions(answers_to_quest: Dict):

    all_pot_questions = defaultdict(lambda : list())
    for wiki_id in answers_to_quest:

        if len(answers_to_quest[wiki_id]) < 2:
            continue

        for i in range(len(answers_to_quest[wiki_id]) - 1):
            for j in range(i+1, len(answers_to_quest[wiki_id])):
                all_pot_questions[f'{answers_to_quest[wiki_id][i]}##{answers_to_quest[wiki_id][j]}'].append(wiki_id)

    potential_questions = dict()
    for i, potential_couple in enumerate(all_pot_questions):
        if len(all_pot_questions[potential_couple]) > 4:
            potential_questions[potential_couple] = all_pot_questions[potential_couple]

    return potential_questions


def backward_pass(meta_data: Dict, answers_data: Dict, wiki_ids: Dict, answer_to_quest: Dict, output_path: str = ""):

    """
    Takes a dictionary of wikiids and the questions they answer to, can output it to some file if they have more than
    5 answers in common.
    """

    potential_couples = find_potential_questions(answer_to_quest)
    intersec_questions = create_intersec_questions_data(meta_data, answers_data, potential_couples, wiki_ids)

    if len(output_path) > 0:
        with open(output_path, 'w') as f:
            for sample in intersec_questions:
                f.write(json.dumps(sample) + '\n')

    return intersec_questions


def load_questions(root_path: str, concatenate_path: str, thresh: float = 0.7):

    """
    With some root path load all the questions.
    """

    questions = list()
    chunk_dir = os.listdir(root_path)
    for chunk in tqdm(chunk_dir):
        curr_chunk_quests = os.path.join(root_path, chunk)
        cache_list = os.listdir(curr_chunk_quests)
        for cache in cache_list:
            if 'app' in cache:
                continue
            curr_cache_quests = os.path.join(curr_chunk_quests, cache)
            questions_list = os.listdir(curr_cache_quests)
            for quest in questions_list:
                if 'specified_quest' not in quest:
                    continue
                temporary_questions, filtered_questions = list(), list()
                with open(os.path.join(curr_cache_quests, quest), 'r') as f:
                    for line in f:
                        temporary_questions.append(json.loads(line))
                for sample in temporary_questions:
                    if filter_answer([sample['base_quest_num'], sample['num_has_wikipage'], sample['original_num_answer']],
                                     thresh):
                        filtered_questions.append(sample)
                questions += filtered_questions

    if len(concatenate_path) > 0:
        with open(concatenate_path, 'w') as f:
            for sample in questions:
                f.write(json.dumps(sample) + '\n')

    return questions


def main(root_path: str, output_path_backward: str, questions_concatenation_path: str, reformulated_path: str):


    if len(questions_concatenation_path) > 1 and os.path.exists(questions_concatenation_path):
        questions = list()
        with open(questions_concatenation_path, 'r') as f:
            for line in f:
                questions.append(json.loads(line))
    else:
        questions = load_questions(root_path, questions_concatenation_path)


    if len(output_path_backward) > 1 and os.path.exists(output_path_backward):
        intersection_quest = list()
        with open(output_path_backward, 'r') as f:
            for line in f:
                intersection_quest.append(json.loads(line))
    else:
        meta_data, answers_data, answer_to_quest, wiki_ids = create_dictionary(questions)
        intersection_quest = backward_pass(meta_data, answers_data, wiki_ids, answer_to_quest, output_path_backward)

    for sample in tqdm(intersection_quest):
        reformulated_quest = reformulate_questions_intersec(sample['q1_question_text'], sample['q2_question_text'],
                                                            sample['type'][-1])
        sample['question_text'] = reformulated_quest

    if len(reformulated_path) > 1:
        with open(reformulated_path, 'w') as f:
            for sample in intersection_quest:
                f.write(json.dumps(sample) + '\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Intersection dictionary')
    parser.add_argument('-r', '--root_path', type=str, help="Path to where the questions are kept.")
    parser.add_argument('-o', '--output_path_backward', type=str, default="", help="Where to keep the backward path dict.")
    parser.add_argument('-c', '--questions_concatenation_path', type=str, default="")
    parser.add_argument('-n', '--reformulated_path', type=str, default='')
    args = parser.parse_args()
    main(**vars(args))