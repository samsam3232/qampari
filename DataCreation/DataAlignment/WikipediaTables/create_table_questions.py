import os
import json
from tqdm import tqdm
from collections import defaultdict
import argparse
from typing import Dict
import numpy as np

def create_questions_strs(aligned, key, lst_name):

    base_str = "What are the {} such that their {} value is {}?"
    questions = dict()
    col2_poss = defaultdict(lambda : [])
    for aligned_txt in aligned[key]:
        c1, e1, c2, e2 = aligned_txt[1].split('##')
        col2_poss[e2].append(aligned_txt[1:])
    for poss in col2_poss:
        if len(col2_poss[poss]) >= 5 and len(poss) > 5:
            questions[base_str.format(lst_name, key[1].lower(), poss)] = col2_poss[poss]
    return questions


def get_entities(key):
  return key.split('##')


def create_questions_int(aligned, key, lst_name):

    base_str = "What are the {} such that their {} value is {} than {}?"
    questions = defaultdict(lambda : [])
    col2_poss = defaultdict(lambda : [])
    for aligned_txt in aligned[key]:
        c1, e1, c2, e2 = aligned_txt[1].split('##')
        col2_poss[e2].append(aligned_txt[1:])
    for i in ['lower', 'greater']:
        rand_val = list(col2_poss.keys())[np.random.randint(0, len(col2_poss))]
        for poss in col2_poss:
            if poss >= rand_val and i == 'greater':
                questions[base_str.format(lst_name, key[1].lower(), i, rand_val)].append(col2_poss[poss])
            elif poss < rand_val and i == 'lower':
                questions[base_str.format(lst_name, key[1].lower(), i, rand_val)].append(col2_poss[poss])
    return questions


def create_questions_year(aligned, key, lst_name):

    base_str = "What are the {} such that they were created or happened {} {}?"
    questions = defaultdict(lambda : [])
    col2_poss = defaultdict(lambda : [])
    for aligned_txt in aligned[key]:
        c1, e1, c2, e2 = aligned_txt[1].split('##')
        col2_poss[e2].append(aligned_txt[1:])
    for i in ['before', 'after']:
        rand_val = list(col2_poss.keys())[np.random.randint(0, len(col2_poss))]
        for poss in col2_poss:
            if poss >= rand_val and i == 'after':
                questions[base_str.format(lst_name, i, rand_val)].append(col2_poss[poss])
            elif poss < rand_val and i == 'before':
                questions[base_str.format(lst_name, i, rand_val)].append(col2_poss[poss])

    return questions


def is_date(entity):
    if entity.isnumeric() and len(entity) == 4:
        return True
    return False


def create_questions_complex_table(tables):

    questions_int = dict()
    questions_str = dict()
    questions_date = dict()
    for table in tqdm(tables):
        lst_name = ' '.join(table.split(', ')[0].split('/')[-1].split('_')[2:]).lower()
        for couples in tables[table]:
            base = f'{couples[0].lower()} of {lst_name}' if couples[0].lower() not in lst_name else lst_name
            is_int, is_dat = True, True
            for answer in tables[table][couples]:
                c1, e1, c2, e2 = answer[1].split('##')
                if not e2.isnumeric():
                    is_int = False
                    is_dat = False
                    break
                if not is_date(e2):
                    is_dat = False
            if is_dat or 'year' in couples[1].lower() or 'date' in couples[1].lower() or 'last' in couples[1].lower():
                questions_date.update({table: create_questions_year(tables[table], couples, base)})
            elif is_int:
                questions_int.update({table: create_questions_int(tables[table], couples, base)})
            else:
                questions_str.update({table: create_questions_strs(tables[table], couples, base)})

    return questions_str, questions_int, questions_date

def load_tables(base_path: str):

    """
    Loads all the tables and their alignment, delete tables that don't stand by our rules
    """

    dir_list = os.listdir(base_path)
    total_data = dict()
    for dir in tqdm(dir_list):
        curr_path = os.path.join(base_path, dir)
        curr_dir_lists = os.listdir(curr_path)
        for subdir in curr_dir_lists:
            to_del = list()
            with open(os.path.join(curr_path, subdir), 'r') as f:
                curr_data = json.load(f)
            for key in curr_data:
                if not first_heuristic_rule(key):
                    to_del.append(key)
                elif 'list_of' not in key.lower():
                    to_del.append(key)
                elif len(total_data[key][1]) > 0 :
                    to_del.append(key)
            for key in to_del:
                curr_data.pop(key)
            total_data.update(curr_data)

    return total_data


def filter_align(table_data: Dict):

    total_aligned = 0
    filtered_align = dict()
    for key in tqdm(table_data):
        filtered_align[key] = [table_data[key][0], dict()]
        for aligned in table_data[key][1]:
            if 'rank' not in aligned.lower() and 'score' not in aligned.lower():
                filtered_align[key][1][aligned] = table_data[key][1][aligned]
                total_aligned += 1
        if len(filtered_align[key][1]) == 0:
            filtered_align.pop(key)

    joined = dict()
    for key in filtered_align:
        if key.split(', ')[0] not in joined:
            joined[key.split(', ')[0]] = filtered_align[key]
        else:
            for aligned in filtered_align[key][0]:
                if aligned in joined[key.split(', ')[0]][0]:
                    joined[key.split(', ')[0]][0][aligned] += filtered_align[key][0][aligned]
                else:
                    joined[key.split(', ')[0]][0][aligned] = filtered_align[key][0][aligned]
            joined[key.split(', ')[0]][1].update(filtered_align[key][1])

    return filtered_align, joined


def aggregate_and_filter(filt_align: Dict):

    agrregate_per_columns = dict()
    appartenance = dict()
    for key in tqdm(filt_align):
        appartenance[key] = dict()
        agrregate_per_columns[key] = dict()
        for aligned in filt_align[key][1]:
            try:
                entities = get_entities(aligned)
            except:
                continue
            if len(entities) == 4 and ('appartenance' not in entities[0].lower()):
                col1, e1, col2, e2 = entities
                num_rows = filt_align[key][0][f'{col1}##{col2}'] if f'{col1}##{col2}' in filt_align[key][0] else 100
                if (col1, col2) not in agrregate_per_columns[key]:
                    agrregate_per_columns[key][(col1, col2)] = [[num_rows, aligned, filt_align[key][1][aligned]]]
                else:
                    agrregate_per_columns[key][(col1, col2)].append([num_rows, aligned, filt_align[key][1][aligned]])
            elif ('appartenance' in entities[0].lower()):
                potential = entities[1]
                num_rows = filt_align[key][0][f'{col1}'] if f'{col1}' in filt_align[key][0] else 100
                if potential not in appartenance[key]:
                    appartenance[key][potential] = [[num_rows, aligned, filt_align[key][1][aligned]]]
                else:
                    appartenance[key][potential].append([num_rows, aligned, filt_align[key][1][aligned]])

    keys_to_del = list()
    for key in tqdm(appartenance):
        to_delete = list()
        for cols in appartenance[key]:
            num_rows = float(appartenance[key][cols][0][0])
            if num_rows == 0:
                num_rows += 10
            if len(appartenance[key][cols]) < 5 or (len(appartenance[key][cols]) / num_rows) < 0.8:
                to_delete.append(cols)
        for col in to_delete:
            appartenance[key].pop(col)
        if len(appartenance[key]) == 0 or not first_heuristic_rule(key):
            keys_to_del.append(key)

    for key in keys_to_del:
        appartenance.pop(key)

    new_appartenance = dict()
    for key in appartenance:
        try:
            if ':' not in key.split('/')[-1] and key.split('/')[-1][-3] != '–' and key.split('/')[-1][-2] != '–' and \
                    key.split('/')[-1][-2] != '-' and (
                    key.split('/')[-1][-1] != ')' and key.split('/')[-1][-3] != '(') and (
                    len(key.split(', ')[-1]) != 3):
                new_appartenance[key] = appartenance[key]
        except Exception as e:
            print(e)

    new_aggreg = dict()
    for key in agrregate_per_columns:
        try:
            new_key = key.split('/')[-1].split(', ')[0]
            if ':' not in new_key[-1] and new_key[-3] != '–' and new_key[-2] != '–' and new_key[-2] != '-' and (
                    new_key[-1] != ')' and new_key[-3] != '('):
                new_aggreg[key] = agrregate_per_columns[key]
        except:
            print(key.split('/')[-1])

    keys_to_del = list()
    for key in tqdm(new_aggreg):
        to_delete = list()
        for cols in new_aggreg[key]:
            num_rows = float(new_aggreg[key][cols][0][0])
            if num_rows == 0:
                num_rows += 10
            if len(new_aggreg[key][cols]) < 5 or (len(new_aggreg[key][cols]) / num_rows) < 0.8:
                to_delete.append(cols)
        for col in to_delete:
            new_aggreg[key].pop(col)
        if len(new_aggreg[key]) == 0 or not first_heuristic_rule(key):
            keys_to_del.append(key)

    for key in keys_to_del:
        new_aggreg.pop(key)

    return appartenance, new_aggreg


def save_str_quest(question, table, answers, curr_id, partition):
    quest = dict()
    quest['qid'] = f'{curr_id}_wikitables_composition_{partition}'
    quest['question_text'] = question
    quest['entities'] = [
        {'entity_text': table.split('/List_of_')[-1].split(', ')[0].replace('_', ' '), 'entity_url': table}]
    quest['answers'] = list()
    for answer in answers:
        curr_answer = dict()
        curr_answer['answer_text'] = answer[0].split('##')[1]
        curr_answer['answer_url'] = answer[1][1]
        curr_answer['found_in_url'] = answer[1][1]
        curr_answer['proof'] = answer[1][2][0]
        quest['answers'].append(curr_answer)

    return quest


def save_int_quest(question, table, answers, curr_id, partition):
    quest = dict()
    quest['qid'] = f'{curr_id}_wikitables_composition_{partition}'
    quest['question_text'] = question
    quest['entities'] = [
        {'entity_text': table.split('/List_of_')[-1].split(', ')[0].replace('_', ' '), 'entity_url': table}]
    quest['answers'] = list()
    for answer in answers:
        curr_answer = dict()
        curr_answer['answer_text'] = answer[0][0].split('##')[1]
        curr_answer['answer_url'] = answer[0][1][1]
        curr_answer['found_in_url'] = answer[0][1][1]
        curr_answer['proof'] = answer[0][1][2][0]
        quest['answers'].append(curr_answer)

    return quest

def first_heuristic_rule(title):

    first_title = ' '.join(title.split(', ')[0].split('/')[-1].split('_')[:])
    second_title = title.split(', ')[-1].split('_')[0][1:]

    if len(second_title) == 1:
        return False

    if second_title.lower() in first_title.lower():
        return True
    return False


def main(base_path: str, output_path: str):

    tables = load_tables(base_path)
    filtered_align, joined = filter_align(tables)
    appartenance, new_aggreg = aggregate_and_filter(filtered_align)
    questions_str, questions_int, questions_year = create_questions_complex_table(new_aggreg)
    curr_id = 0
    all_quests = list()
    for key in questions_year:
        for subquestion in questions_year[key]:
            all_quests.append(save_int_quest(subquestion, key, questions_year[key][subquestion], curr_id, 'year_train'))
            curr_id += 1
    for key in questions_str:
        for subquestion in questions_str[key]:
            all_quests.append(save_str_quest(subquestion, key, questions_str[key][subquestion], curr_id, 'str_train'))
            curr_id += 1
    for key in questions_int:
        for subquestion in questions_int[key]:
            all_quests.append(save_str_quest(subquestion, key, questions_str[key][subquestion], curr_id, 'int_train'))

    with open(output_path, 'w') as f:
        for sample in all_quests:
            f.write(json.dumps(sample) + '\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Wikipedia tables')
    parser.add_argument('-b', '--base_path', type=str, help="Where the wikipedia table alignments are kept.")
    parser.add_argument('-o', '--output_path', type=str, help="Where to keep the results")
    args = parser.parse_args()
    main(**vars(args))