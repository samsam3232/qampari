import multiprocessing
import os
import argparse
from DataCreation.DataAlignment.WikiDataAlignment.align_query import align_queries

def main(input_dir, output_dir, indices_path, first_index, last_index):

    processes = list()
    for i in range(first_index, last_index):
        direc_inf = f'chunked/chunk_{i}/relevant_infos.json'
        direc_quests = f'chunked/chunk_{i}/question_couples.json'
        curr_dir = os.path.join(output_dir, f'chunk_{i+1}')
        if not os.path.exists(curr_dir):
            os.mkdir(curr_dir)
        new_p = multiprocessing.Process(target=align_queries, args=(os.path.join(input_dir, direc_quests), os.path.join(input_dir, direc_inf),
                                                                    curr_dir, indices_path, os.path.join(input_dir, 'all_mappings.json')))
        new_p.daemon = True
        new_p.start()
        processes.append(new_p)

    for process in processes:
        try:
            process.join(timeout=3600*48)
        except:
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Multiprocessing parser")
    parser.add_argument("-i", "--input_dir", type = str, help = "Path to the directory where the data is kept")
    parser.add_argument("-o", "--output_dir", type=str, help='Path where the results are to be kept')
    parser.add_argument("--indices_path", type=str, help="Path to where the indices file is kept.")
    parser.add_argument("-f", "--first_index", type=int)
    parser.add_argument('-l', '--last_index', type=int)
    args = parser.parse_args()
    main(**vars(args))