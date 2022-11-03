import json
import argparse
from reformating_utils import randomly_merge

def fix_train_ctxs(positive_ctxs, total_ctxs):

    """
    Fixes the train contexts. Merges the positive and the negative contexts while making sure the order of the answers
    is recorded for later loss computation
    :param positive_ctxs: the positive contexts for the model
    :param total_ctxs: all the contexts retrieved
    :return: the merged positive and negative contexts and the order in which the answers appear
    """

    new_positive_ctxs, new_negative_ctxs, added_id, answers_orders = list(), list(), list(), list()
    for ctx in positive_ctxs:
        new_positive_ctxs.append({'title': ctx['title'], 'text': ctx['text'], 'id': ctx['chunk_id']})
        added_id.append(ctx['chunk_id'])
        if '__'.join(ctx['pid'].split('__')[:-1]) not in answers_orders:
            answers_orders.append('__'.join(ctx['pid'].split('__')[:-1]))

    for ctx in total_ctxs:
        if ctx['id'] not in added_id:
            new_negative_ctxs.append({'title': ctx['title'], 'text': ctx['text'], 'id': ctx['id']})
        if len(new_negative_ctxs) == (200 - len(new_positive_ctxs)):
            break

    return randomly_merge(new_positive_ctxs, new_negative_ctxs), answers_orders


def main(input_path: str, output_path: str):

    """
    Reparses the qampari data to fit the FiD supported format.
    :param input_path: path to where the data is kept
    :param output_path: path to where to keep the reformatted data
    """

    with open(input_path, 'r') as f:
        data = json.load(f)

    new_data = list()
    for sample in data:

        if len(sample['answer_list']) > 100:
            continue

        new_samp = dict()
        aliases_mapping = dict()
        new_samp['id'] = sample['qid']
        new_samp['question_text'] = sample['question_text']
        all_answer = list()
        for ans in sample['answer_list']:
            aliases_mapping[ans['answer_text']] = ans['aliases']
            all_answer += ans['aliases']
        new_samp['answers'] = list(set(all_answer))
        new_samp['ans_mappings'] = aliases_mapping
        if 'train' in input_path:
            ctxs_curr, answer_order = fix_train_ctxs(sample['positive_ctxs'], sample['ctxs'])
            all_ans = list() # the answers in the target will appear in the correct order (the one of the contexts)
            for pid in answer_order:
                for ans in sample['answer_list']:
                    if ans['aid'] == pid:
                        all_ans.append(ans['answer_text'])
                        break
        else:
            ctxs_curr = sample['ctxs']
            all_ans = [ans['answer_text'] for ans in sample['answer_list']]
        new_samp['target'] = '#'.join(all_ans)
        new_samp['ctxs'] = ctxs_curr
        new_samp['positive_ctxs'] = sample['positive_ctxs']
        new_data.append(new_samp)

    with open(output_path, 'w') as f:
        for sample in new_data:
            f.write(json.dumps(sample) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Qampari data reformater to FiD format')
    parser.add_argument('-i', '--input_path', type=str, help="Path to where the basis of the data is")
    parser.add_argument('-o', '--output_path', type=str, help="Path to where we keep the new data")
    args = parser.parse_args()
    main(**vars(args))