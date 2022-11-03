import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from copy import deepcopy
from collections import defaultdict
from DataCreation.mnli_filtering.mnli_utils import rephrase_mnli
from DataCreation.complex_questions.WikiData.utils import filter_answer
import os
import argparse
import json
from tqdm import tqdm

MODEL_NAME = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"

def retrieve_all_quests(root_path: str, is_comp: bool = False, thresh: float = 0.8):

    caches = os.listdir(root_path)
    all_samples, filtered_samp = list(), defaultdict(lambda : list())
    for cache in caches:
        if 'app' in cache:
            continue
        curr_cache = os.path.join(root_path, cache)
        questions_list = os.listdir(curr_cache)
        for quest in questions_list:
            if not is_comp and 'specified_quest' not in quest:
                continue
            with open(os.path.join(curr_cache, quest), 'r') as f:
                for line in f:
                    all_samples.append(json.loads(line))
            for sample in all_samples:
                if filter_answer([sample['base_quest_num'], sample['num_has_wikipage'], sample['original_num_answer']], thresh=thresh):
                    filtered_samp[cache].append(sample)
            all_samples = list()

    return filtered_samp


def retrieve_meta(filtered_questions: list):

    meta_data_kept, meta_data_deleted, full_answer_data = dict(), dict(), dict()
    for cache_num in filtered_questions:
        for sample in filtered_questions[cache_num]:
            copy_1 = deepcopy(sample)
            copy_1['answer_list'] = list()
            copy_1['cache_num'] = cache_num
            meta_data_kept[sample['question_text']] = copy_1
            copy_2 = deepcopy(sample)
            copy_2['answer_list'] = list()
            copy_2['cache_num'] = cache_num
            meta_data_deleted[sample['question_text']] = copy_2
            for answer_data in sample['answer_list']:
                new_data = deepcopy(answer_data)
                if 'potential_proofs' in new_data:
                    new_data.pop('potential_proofs')
                full_answer_data[f"{sample['question_text']}##{answer_data['answer_wiki_id']}"] = new_data

    return meta_data_kept, meta_data_deleted, full_answer_data


def retrieves_batches_reg(questions: list):

    filtered_quests = defaultdict(lambda: defaultdict(lambda: dict()))
    for cache in questions:
        for sample in questions[cache]:
            for answer in sample['answer_list']:
                rephrased = rephrase_mnli(sample['qid'].split('__')[0], answer['answer_text'], sample['entities'][0]['entity_text'])
                for i in range(len(answer['potential_proofs'])):
                    for key in answer['potential_proofs'][i]:
                        filtered_quests[i][rephrased][key] = answer['potential_proofs'][i][key]
                    filtered_quests[i][rephrased]['origin'] = f'{sample["question_text"]}##{answer["answer_wiki_id"]}'

    return filtered_quests


def retrieves_batches_comp(questions: dict):

    filtered_quests = defaultdict(lambda: defaultdict(lambda: dict()))
    for key in questions:
        for subquestion in questions[key][0]:
            for comp_quest in questions[key][0][subquestion]:
                curr_comp = questions[key][0][subquestion][comp_quest]
                for answers in curr_comp[1]:
                    rephrased = rephrase_mnli(curr_comp[0], curr_comp[1][answers][0], answers, True)
                    if len(rephrased) < 3:
                        continue
                    for i in range(len(curr_comp[1][answers][3])):
                        align = '. '.join(j.capitalize() for j in curr_comp[1][answers][3][i].split('. '))
                        align = align.replace('\n', ' ').replace('  ', ' ')
                        filtered_quests[i][rephrased]['aligned'] = align
                        filtered_quests[i][rephrased]['rephrased'] = rephrased
                        filtered_quests[i][rephrased]['orig_answer'] = answers
                        filtered_quests[i][rephrased]['property'] = curr_comp[0]
                        filtered_quests[i][rephrased]['comp_question'] = comp_quest
                        filtered_quests[i][rephrased]['comp_answer'] = curr_comp[1][answers][0]
                        filtered_quests[i][rephrased]['path'] = questions[key][1]
                        filtered_quests[i][rephrased]['base_quest'] = key

    return filtered_quests


def treat_single_batch(model, tokenizer, batch : list):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    premises = [i[0] for i in batch]
    hypothesis = [i[1] for i in batch]

    tokenized_input_seq_pair = tokenizer(premises, hypothesis, max_length=256, return_tensors='pt',
                                         return_token_type_ids=True, padding=True,
                                         truncation=True)

    input_ids = tokenized_input_seq_pair['input_ids'].long().to(device)
    token_type_ids = tokenized_input_seq_pair['token_type_ids'].long().to(device)
    attention_mask = tokenized_input_seq_pair['attention_mask'].long().to(device)

    with torch.no_grad():
        outputs = model(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=None)

    predicted_probability = torch.softmax(outputs[0], dim=1)
    return predicted_probability


def treat_samples(meta_data_all_kept: dict, meta_data_all_deleted: dict, answer_data: dict, filtered_questions: list, model, tokenizer):

    for key in filtered_questions:
        num_batches = math.ceil(len(filtered_questions[key]) / 400.)
        rephrased = list(filtered_questions[key].keys())
        for i in tqdm(range(num_batches)):
            curr_rephrased = rephrased[i*400:(i+1)*400]
            curr_quests = [[filtered_questions[key][j]['proof'], j] for j in curr_rephrased]
            predicted_probabilities = treat_single_batch(model, tokenizer, curr_quests)
            for k in range(len(predicted_probabilities)):
                cfilt = filtered_questions[key][curr_rephrased[k]]
                if predicted_probabilities[k, 0] > 0.0019540872:
                    new_answer_data = deepcopy(answer_data[cfilt['origin']])
                    proof_data = deepcopy(cfilt)
                    proof_data.pop('origin')
                    new_answer_data['proofs'] = proof_data
                    meta_data_all_kept[cfilt['origin'].split('##')[0]]['answer_list'].append(new_answer_data)
                    for num in filtered_questions:
                        if num > key and curr_rephrased[k] in filtered_questions[num]:
                            filtered_questions[num].pop(curr_rephrased[k])
                else:
                    new_answer_data = deepcopy(answer_data[cfilt['origin']])
                    proof_data = deepcopy(cfilt)
                    proof_data.pop('origin')
                    new_answer_data['proof'] = proof_data
                    meta_data_all_deleted[cfilt['origin'].split('##')[0]]['answer_list'].append(new_answer_data)

    return meta_data_all_kept, meta_data_all_deleted


def save_mnli_results(base_dir: str, kept_sample: dict, kept_samples = True, type_data = 'simple'):

    cached_samples = defaultdict(lambda : list())
    for sample in kept_sample:
        cache_num = kept_sample[sample].pop('cache_num')
        cached_samples[cache_num].append(kept_sample[sample])

    replace_with = f'{type_data}_kept_samples.json' if kept_samples else f'{type_data}_deleted_samples.json'
    for cache in cached_samples:
        with open(os.path.join(base_dir, cache + '/' + replace_with), 'w') as f:
            for sample in cached_samples[cache]:
                f.write(json.dumps(sample) + '\n')


def main(root_path: str, thresh : float, type_data = 'simple'):

    questions = retrieve_all_quests(root_path, type_data == 'comp', thresh)
    meta_data_kept, meta_data_deleted, answer_data = retrieve_meta(questions)
    if type_data == 'simple':
        batches = retrieves_batches_reg(questions)
    else:
        batches = retrieves_batches_comp(questions)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    if torch.cuda.is_available():
        model = model.cuda()
        model = nn.DataParallel(model)

    kept_samples, deleted_samples = treat_samples(meta_data_kept, meta_data_deleted, answer_data, batches, model, tokenizer)
    save_mnli_results(root_path, kept_samples, type_data = type_data)
    save_mnli_results(root_path, deleted_samples, False, type_data = type_data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root_path', type=str, help='Path to the root of the data.')
    parser.add_argument('-t', '--thresh', default=0.8, type=float, help='Threshhold for the data we want to filter.')
    parser.add_argument('-d', '--type_data', default = 'simple', type=str)
    args = parser.parse_args()
    main(**vars(args))
