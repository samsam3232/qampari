import json
import os
import argparse
from collections import defaultdict
import regex
import string
import pandas as pd
import numpy as np
from typing import Dict, List


def create_inverse_mappings(mapping):

    """
    Based on a mapping, inverses it and returns a dic of aliases as key and their basic name as value.
    :param mapping: dictionary of answer names as key and a list of their aliases as values
    """

    inverse_mapping = dict()
    for key in mapping:
        for alias in mapping[key]:
            inverse_mapping[alias] = key
    return inverse_mapping


def load_data(input_path: str):

    """
    Loads the data depending on the format of the file:  a jsonl or json
    :param input_path: path to  where the data is kept
    :return: the loaded data represented with a list
    """

    with open(input_path, 'r') as f:
        if 'jsonl' in input_path:
            samples = list()
            for line in f:
                samples.append(json.loads(line))
            return samples
        else:
            data = json.load(f)
            return data


def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def create_mapping(prediction: Dict):

    """
    Create a mapping from an answer name to its aliases
    """

    mapping = dict()
    for ans in prediction['answer_list']:
        mapping[ans['answer_text']] = ans['aliases']

    return mapping

def compute_metrics_qampari(predictions: List):

    """
    Computes the metrics for QAMPARI predictions.
    :param predictions: all the QAMPARI predictions
    :return: dictionary of the computed metrics
    """

    metrics = {'precision': list(), 'recall': list(), 'f1': list()}
    for pred in predictions:

        # checks the answers correctly predicted
        already_predicted = list()
        mapping = create_mapping(pred)
        inversed_mapping = create_inverse_mappings(mapping)
        predicted_answers = set(pred['predictions'])
        all_answers = len(pred['answer_list'])
        for predicted in predicted_answers:
            for alias in inversed_mapping:
                if exact_match_score(predicted, alias):
                    if inversed_mapping[alias] not in already_predicted:
                        already_predicted.append(inversed_mapping[alias])
                    break

        # computes the metrics
        curr_prec = (len(already_predicted) / len(predicted_answers))
        curr_rec = (len(already_predicted) / all_answers)
        if curr_rec == 0 or curr_prec == 0:
            metrics['f1'].append(0)
        else:
            metrics['f1'].append(((2 * curr_rec * curr_prec) / (curr_rec + curr_prec)))
        metrics['precision'].append(curr_prec)
        metrics['recall'].append(curr_rec)

    mean_prec = np.mean(np.array(metrics['precision']))
    mean_rec = np.mean(np.array(metrics['recall']))
    mean_f1 = np.mean(np.array(metrics['f1']))
    f1_above = np.sum(np.array(metrics['f1']) >= 0.5) / float(len(metrics['f1']))
    rec_above = np.sum(np.array(metrics['recall']) >= 0.8) / float(len(metrics['f1']))
    return {'f1': mean_f1, 'recall': mean_rec, 'precision': mean_prec, 'f1_above': f1_above, 'rec_above': rec_above}

def compute_metrics_nq(predictions: List):

    """
    Computes the EM metric for preictions from NQ.
    """

    has_em = list()
    for pred in predictions:
        em_val = 0
        all_answers = pred['answers']
        for alias in all_answers:
            if exact_match_score(alias, pred['prediction']):
                em_val = 1
                break
        has_em.append(em_val)

    return {'em': np.mean(np.array(has_em))}



def main(input_path: str, output_path: str = None, is_nq: bool = False):

    """
    Expect input to a prediction file. Expect a list of dicts of the following (minimum) format:
    {"answer_list": list of dict({"answer_text": str, "aliases": list of str}),
     "answers": list of str,
     "predictions": list of str}
    """

    data = load_data(input_path)
    metrics = compute_metrics_nq(data) if is_nq else compute_metrics_qampari(data)
    if output_path is not None:
        with open(output_path, 'w') as f:
            json.dump(metrics, f)

    for key, val in metrics.items():
        print(f'{key}:\t {val}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Reader evaluation')
    parser.add_argument('-i', '--input_path', default='none', type=str, help="Path to where the FiD results are kepts")
    parser.add_argument('--is_nq', default=False, action='store_true')
    parser.add_argument('-o', '--output_path', type=str, help="Path to where we keep the csv")
    args = parser.parse_args()
    main(**vars(args))