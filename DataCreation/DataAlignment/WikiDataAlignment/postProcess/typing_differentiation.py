import os
from tqdm import tqdm
import simplejson as json
import argparse
import logging
from collections import defaultdict
from typing import Dict, DefaultDict, List
from DataCreation.DataAlignment.utils.alignment_utils import get_question_subject
from DataCreation.DataAlignment.utils.properties_constants import SUBCLASSES, PROP_SENTENCES, REPHRASING, SUBJECT_MAPPING


def get_answer_superclass(wikidata_id: str, relevant_infos : Dict, superclasses_cache: Dict):

    """
    Given an answer to a question, returns a list with all the superclasses of this answers.
    """

    curr_superclasses = set()

    if (wikidata_id not in relevant_infos) or ('composition' not in relevant_infos[wikidata_id]) or ('P31' not in relevant_infos[wikidata_id]['composition']):
      return curr_superclasses
    for wikiid in relevant_infos[wikidata_id]['composition']['P31']:
        if wikiid not in superclasses_cache:
            superclasses = []
        else:
            superclasses = superclasses_cache[wikiid]
        curr_superclasses = curr_superclasses.union(superclasses)

    return curr_superclasses


def add_helping_info(all_quests, quest_form: str, prop: str, subject: str, subtype: str):

    """
    Adds the informations to the dictionary.
    """
    if prop not in all_quests[quest_form][0]:
        all_quests[quest_form][0].append(prop)
        all_quests[quest_form][0].append(subject)
        all_quests[quest_form][0].append(subtype)
    return all_quests


def reformulate_single_question(all_questions: dict, property: str, subject: str, answer: str, answers_inf):

    """
    Reforulates a question with a single option (questions that apply to one type of subject).
    """

    quest_form = PROP_SENTENCES[property][1].format(subject)
    all_questions[PROP_SENTENCES[property][1].format(subject)][1][answer] = answers_inf
    all_questions = add_helping_info(all_questions, quest_form, property, subject, 'human')

    return all_questions


def reformulate_subtyped_question(curr_superclasses: list, property: str, subject: str):

    """
    Reformulates a question with a single option (questions that apply to one type of subject).
    """

    all_questions = list()
    if len(curr_superclasses) > 0:
        for relevant_class in SUBCLASSES[property]:
            if relevant_class in curr_superclasses:
                if relevant_class in REPHRASING:
                    relevant_class = REPHRASING[relevant_class]
                question = PROP_SENTENCES[property][1].format(relevant_class, subject)
                all_questions.append({'question': question, 'type': relevant_class})

    return all_questions

def treat_question(question: Dict, relevant_infos: Dict, superclasses_cache: Dict):

    """
    Given a question and its answers, divides it in all correctly formulated questions.
    """

    questions = list()
    property = question['qid'].split('__')[0]
    subject = get_question_subject(question['base_question'])
    if PROP_SENTENCES[property][0] == 1:
        quest_form = PROP_SENTENCES[property][1].format(subject)
        question['question_text'] = quest_form
        question['type'] = SUBJECT_MAPPING[property]
        question['qid'] = question['qid'].replace('wikidata_simple_unspecified', 'wikidata_simple')
        return [question]
    else:
        new_questions = defaultdict(lambda: list())
        for answer in question['answer_list']:
            curr_superclasses = get_answer_superclass(answer['answer_wiki_id'], relevant_infos, superclasses_cache)
            all_questions = reformulate_subtyped_question(curr_superclasses, property, subject)
            for quest in all_questions:
                new_questions[quest['question']].append({'answer': answer, 'type': quest['type']})

        for new_quest in new_questions:
            if len(new_questions[new_quest]) > 4:
                curr_quest = dict()
                for key in question:
                    if key == 'answer_list':
                        continue
                    curr_quest[key] = question[key]
                curr_quest['qid'] = curr_quest['qid'].replace('wikidata_simple_unspecified', 'wikidata_simple')
                curr_quest['question_text'] = new_quest
                curr_quest['answer_list'] = list()
                for answer in new_questions[new_quest]:
                    curr_quest['answer_list'].append(answer['answer'])
                    curr_quest['type'] = answer['type']
                questions.append(curr_quest)

    return questions


def treat_file(input_path: str, relevant_infos: Dict, superclasses_cache: Dict):

    """
    Given a path to a file, the relevant infos and the superclasses of each instance, iterates over all the questions in
    the file and reformulates them and save them in a file.
    """

    data = list()
    with open(input_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    new_questions = list()
    for question in data:
        new_questions += treat_question(question, relevant_infos, superclasses_cache)

    return new_questions

def main(input_path: str, superclasses_path : str, relevant_infos_path: str):

    logging.basicConfig(filename=os.path.join(input_path, "app_specified.log"), filemode="w", level=logging.INFO)

    cache_list = os.listdir(input_path)

    with open(superclasses_path, 'r') as f:
        superclasses_cache = json.load(f)

    with open(relevant_infos_path, 'r') as f:
        relevant_infos = json.load(f)

    for cache_dir in tqdm(cache_list):
        if "cache" not in cache_dir:
            continue
        curr_dir = os.path.join(input_path,cache_dir)

        questions_list = os.listdir(curr_dir)
        for question_path in questions_list:
            if "original_simple" not in question_path:
                continue

            logging.info(f"Treating {cache_dir}")

            new_questions = treat_file(os.path.join(curr_dir, question_path), relevant_infos, superclasses_cache)
            with open(os.path.join(curr_dir, 'specified_questions.jsonl'), 'w') as f:
                for sample in new_questions:
                    f.write(json.dumps(sample) + '\n')

        logging.info(f"Finished cache {cache_dir}")

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Typing classification runner")
    parser.add_argument('-i', '--input_path', type=str, help="Path to where the results we want to classify are")
    parser.add_argument('-s', '--superclasses_path', type=str, help='Path to where the superclasses for a given wikidata'
                                                                    'id are kept')
    parser.add_argument('-r', '--relevant_infos_path', type=str, help='Path to where the mappings are kept')
    args = parser.parse_args()
    main(**vars(args))