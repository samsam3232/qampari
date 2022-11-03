from functools import lru_cache
from operator import itemgetter
import json
import numpy as np
from collections import defaultdict
import argparse
from tqdm import tqdm

def longest_common_substring(x: str, y: str) -> (int, int, int):
    # function to find the longest common substring

    # Memorizing with maximum size of the memory as 1
    @lru_cache(maxsize=1)
    # function to find the longest common prefix
    def longest_common_prefix(i: int, j: int) -> int:

        if 0 <= i < len(x) and 0 <= j < len(y) and x[i] == y[j]:
            return 1 + longest_common_prefix(i + 1, j + 1)
        else:
            return 0

    # diagonally computing the subproblems
    # to decrease memory dependency
    def digonal_computation():

        # upper right triangle of the 2D array
        for k in range(len(x)):
            yield from ((longest_common_prefix(i, j), i, j)
                        for i, j in zip(range(k, -1, -1),
                                        range(len(y) - 1, -1, -1)))

        # lower left triangle of the 2D array
        for k in range(len(y)):
            yield from ((longest_common_prefix(i, j), i, j)
                        for i, j in zip(range(k, -1, -1),
                                        range(len(x) - 1, -1, -1)))

    # returning the maximum of all the subproblems
    return max(digonal_computation(), key=itemgetter(0), default=(0, 0, 0))


def main(input_path):

    with open(input_path) as f:
        data = json.load(f)

    all_matches = list()
    all_lens = list()
    for sample in tqdm(data):
        curr_matches = list()
        has_match = list()
        for ctx in sample['ctxs']:
            count_match = False
            for i, pos in enumerate(sample['positive_ctxs']):
                if i in has_match:
                    continue
                if ctx['title'].lower() == pos['title'].lower():
                    length, _, _ = longest_common_substring(ctx['text'].lower(), pos['text'].lower())
                    if length / min(len(ctx['text']), len(pos['text'])) > 0.7:
                        count_match = True
                        has_match.append(i)
                        curr_matches.append(1)
                        break
                    elif length / min(len(ctx['text']), len(pos['text'])) < 0.5 and length / min(len(ctx['text']), len(pos['text'])) > 0.2:
                        print(length / min(len(ctx['text']), len(pos['text'])))
                        pass
            if not count_match:
                curr_matches.append(0)
        all_lens.append(len(sample['positive_ctxs']))
        all_matches.append(np.array(curr_matches))

    for i in [1, 5, 10, 20, 50, 100, 200]:
        print(f"{i}: {np.mean(np.sum(all_matches[:, :i]) / np.append(all_lens))}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Test retriever metrics')
    parser.add_argument('-i', '--input_path', type=str)
    args = parser.parse_args()
    main(**vars(args))