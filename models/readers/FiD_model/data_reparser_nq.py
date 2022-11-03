import os
import json
import argparse

def main(input_path: str, output_path: str):

    """
    Reparses the NQ data to fit the FiD supported format.
    :param input_path: path to where the data is kept
    :param output_path: path to where to keep the reformatted data
    """

    with open(input_path, 'r') as f:
        data = json.load(f)

    new_data = list()
    for i, sample in enumerate(data):

        new_samp = dict()
        new_samp['id'] = f'nq_{i}'
        new_samp['question'] = sample['question']
        new_samp['answers'] = sample['answers']
        new_samp['target'] = sample['answers'][0]
        new_samp['ctxs'] = sample['ctxs']
        new_data.append(new_samp)

    with open(output_path, 'w') as f:
        for sample in new_data:
            f.write(json.dumps(sample) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Data reformater to FiD')
    parser.add_argument('-i', '--input_path', type=str, help="Path to where the basis of the data is")
    parser.add_argument('-o', '--output_path', type=str, help="Path to where we keep the new data")
    args = parser.parse_args()
    main(**vars(args))