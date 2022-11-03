import math
import os
import json
from tqdm import tqdm
import argparse


def retrieve_relevant_infos(relevant_infos: str, answers: list, output_dir: str):

    """
    Given output dir and a list of answers, retrieves all the informations regarding these answers.
    """

    curr_relevant_infos = dict()
    subdirs = os.listdir(relevant_infos)

    for subdir in subdirs:
        if 'rele' not in subdir:
            continue
        with open(os.path.join(relevant_infos, subdir), 'r') as f:
            curr_rele = json.load(f)
        to_del = list()
        for answer in tqdm(answers):
            if answer in curr_rele:
                curr_relevant_infos[answer] = curr_rele[answer]
                to_del.append(answer)
        answers = list(set(answers) - set(to_del))

    with open(os.path.join(output_dir, 'relevant_infos.json'), 'w') as f:
        json.dump(curr_relevant_infos, f)

    if not os.path.exists(os.path.join(output_dir, 'mappings_only.json')):
        mappings_only = dict()
        for key in curr_rele:
            mappings_only[key] = dict()
            if 'url' in curr_rele[key]:
                mappings_only[key]['url'] = curr_rele[key]['url']
            if 'potential_labels' in curr_rele[key]:
                mappings_only[key]['potential_labels'] = curr_rele[key]['potential_labels']

        with open(os.path.join(output_dir, 'mappings_only.json'), 'w') as f:
            json.dump(mappings_only, f)



def retrieve_all_answers(data: dict, keys: list) -> list:

    """
    Retrieves all the answers in a given chunk.
    """

    all_answers = list()
    for key in keys:
        all_answers += data[key]
    return list(set(all_answers))

def main(input_questions: str, output_dir: str, relevant_infos_path: str, num_questions: int = 100000):

    """
    Given a path to relevant infos dir and a file with all the potential questions, outputs chunks of 100000 questions
    each and all the relevant infos for the answers in said chunk.
    """

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    try:
        with open(input_questions, 'r') as f:
            data = json.load(f)
    except:
        data = None

    questions_couples = list(data.keys())
    for i in range(math.ceil(len(questions_couples) / num_questions)):
        if not os.path.exists(os.path.join(output_dir, f'chunk_{i}')):
            os.mkdir(os.path.join(output_dir, f'chunk_{i}'))
        if data is not None:
            answers = retrieve_all_answers(data, questions_couples[i*num_questions:(i+1)*100000])

            curr_couples = dict()
            for key in questions_couples[i * num_questions:(i + 1) * num_questions]:
                curr_couples[key] = data[key]
        else:
            with open(os.path.join(input_questions, f'chunk_{i}/question_couples.json'), 'r') as f:
                data = json.load(f)
            questions_couples = list(data.keys())
            answers = retrieve_all_answers(data, questions_couples[i * num_questions:(i + 1) * num_questions])
        retrieve_relevant_infos(relevant_infos_path, answers, os.path.join(output_dir, f'chunk_{i}'))

        curr_couples = dict()
        for key in questions_couples[i*num_questions:(i+1)*num_questions]:
            curr_couples[key] = data[key]

        with open(os.path.join(output_dir, f'chunk_{i}', 'question_couples.json'), 'r') as f:
            json.dump(curr_couples, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Chunking wikidata')
    parser.add_argument('-i', '--input_question', type=str, help='Path to where the input couples are kept')
    parser.add_argument('-o', '--output_dir', type = str, help = 'Path to where to keep the chunked wikidata')
    parser.add_argument('-n', '--num_questions', type=int, default=100000, help='How many questions to keep in each chunk.')
    parser.add_argument('-r', '--relevant_infos_path', type=str)
    args = parser.parse_args()
    main(**vars(args))