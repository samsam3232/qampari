import re
import string
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

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


def compute_nq_metric(final_preds, mappings):

    """
    Based on the final predictions and the mapping from an answer value to an alias, computes NQ-EM score for the overall
    predictions
    :param final_preds: List of the actual predictions made by the model (i.e prob of Answer: higher than prob of irrelevant)
    :param mappings: Mapping from an answer name to a list of its aliases.
    :return: EM score
    """

    has_em = 0
    for qid in final_preds:
        for elem in mappings[qid][list(mappings[qid].keys())[0]]:
            if exact_match_score(final_preds[qid][0].replace('Answer: ', ''), elem):
                has_em += 1
                break
    return has_em / float(len(mappings))


def compute_qampari_metrics(final_preds, inverse_mappings, mappings):

    """
    Based on the final predictions made by the model computes the ensemble of metrics for QAMPARI for the overall
    predictions
    :param final_preds: List of the actual predictions made by the model (i.e prob of Answer: higher than prob of irrelevant)
    :param inverse_mappings: Mapping from an alias to its original answer.
    :param mappings: Mapping from an answer name to a list of its aliases.
    :return: Ensemble of QAMPARI metrics.
    """

    prec, rec, f1 = list(), list(), list()
    for qid in mappings:
        if qid not in final_preds:
            prec.append(0)
            rec.append(0)
            f1.append(0)
            continue
        already_found = list()
        for ans in final_preds[qid]:
            for key in inverse_mappings[qid]:
                if exact_match_score(ans.replace('Answer: ', ''), key) and inverse_mappings[qid][key] not in already_found:
                    already_found.append(inverse_mappings[qid][key])
                    break
        if len(already_found) > 0:
            curr_prec = len(already_found) / len(final_preds[qid])
            curr_rec = len(already_found) / len(mappings[qid])
            f1.append((2 * curr_rec * curr_prec) / (curr_rec + curr_prec))
            prec.append(curr_prec)
            rec.append(curr_rec)
        else:
            prec.append(0)
            rec.append(0)
            f1.append(0)

    mean_prec = np.mean(np.array(prec))
    mean_rec = np.mean(np.array(rec))
    mean_f1 = np.mean(np.array(f1))
    f1_above = np.sum(np.array(f1) >= 0.5) / len(f1)
    rec_above = np.sum(np.array(rec) >= 0.8) / len(rec)

    return mean_prec, mean_rec, mean_f1, f1_above, rec_above


def compute_em_nq(preds, references):

    inverse_mappings = dict()
    mappings = dict()
    for i in range(len(references)):
        if np.random.random() < 0.01:
            logger.warning(f'Pred: {preds[i]["prediction_text"]}, Prob: {preds[i]["answer_prob"]}')
        inverse_mappings[references[i]['qid']] = create_inverse_mappings(json.loads(references[i]['full_answers']))
        mappings[references[i]['qid']] = json.loads(references[i]['full_answers'])

    num_answers = 0
    for qid in mappings:
        num_answers += len(mappings[qid])

    final_predictions = dict()
    for i in range(len(preds)):
        if preds[i]['prediction_text'].lower() != 'Irrelevant'.lower():
            if 'wiki' in preds[i]['qid']:
                if preds[i]['qid'] in final_predictions:
                    final_predictions[preds[i]['qid']].append(preds[i]['prediction_text'])
                else:
                    final_predictions[preds[i]['qid']] = [preds[i]['prediction_text']]
            else:
                if preds[i]['qid'] in final_predictions:
                    if preds[i]['answer_prob'] < final_predictions[preds[i]['qid']][-1]:
                        final_predictions[preds[i]['qid']] = [preds[i]["prediction_text"], preds[i]['answer_prob']]
                else:
                    final_predictions[preds[i]['qid']] = [preds[i]["prediction_text"], preds[i]['answer_prob']]

    wiki_mappings, wiki_inverse, wiki_preds = dict(), dict(), dict()
    nq_mappings, nq_inverse, nq_preds = dict(), dict(), dict()
    for qid in mappings:
        if 'wiki' in qid:
            wiki_mappings[qid] = mappings[qid]
            wiki_inverse[qid] = inverse_mappings[qid]
        else:
            nq_mappings[qid] = mappings[qid]
            nq_inverse[qid] = inverse_mappings[qid]
        if qid in final_predictions:
            if 'wiki' in qid:
                wiki_preds[qid] = final_predictions[qid]
            else:
                nq_preds[qid] = final_predictions[qid]


    em, rec, prec, f1, f1_above, rec_above = -1, -1, -1, -1, -1, -1
    if len(nq_mappings) > 0:
        em = compute_nq_metric(nq_preds, nq_mappings)
    if len(wiki_mappings) > 0:
        prec, rec, f1, f1_above, rec_above = compute_qampari_metrics(wiki_preds, wiki_inverse, wiki_mappings)
    return {"nq_exact_match": em, 'qamp_rec': rec, 'qamp_prec': prec, 'qamp_f1': f1, 'qamp_rec_above': rec_above,
            'qamp_f1_above': f1_above}