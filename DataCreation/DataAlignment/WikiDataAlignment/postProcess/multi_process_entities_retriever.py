import multiprocessing
import os
import argparse
from DataCreation.DataAlignment.WikiDataAlignment.postProcess.entities_retriever import filter as filter_main

def main(base_dir_questions: str, base_infos_wiki: str, output_txt: str, first_index: str, last_index: str,):

    processes = list()
    for i in range(first_index, last_index):
        direc = f'chunk_{i+1}/'
        curr_dir = os.path.join(base_dir_questions, f'chunk_{i+1}')
        relevant_infos_curr = os.path.join(base_infos_wiki, f'chunk_{i}/relevant_infos.json')
        if not os.path.exists(curr_dir):
            os.mkdir(curr_dir)
        new_p = multiprocessing.Process(target=filter_main, args=(os.path.join(base_dir_questions, direc), relevant_infos_curr,
                                                                  os.path.join(output_txt, f'chunk_{i}.txt'),))
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
    parser.add_argument('-q', '--base_dir_questions', type=str)
    parser.add_argument('-w', '--base_infos_wiki', type=str)
    parser.add_argument('-t', '--output_txt', type=str)
    parser.add_argument("-f", "--first_index", type=int)
    parser.add_argument('-l', '--last_index', type=int)
    args = parser.parse_args()
    main(**vars(args))