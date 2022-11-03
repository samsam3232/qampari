import multiprocessing
import os
import argparse
from DataCreation.complex_questions.WikiData.composition import main as comp_main

def main(input_dir, output_dir, list_indices_path, wikidata_mappings_path, relevant_infos_path, first_index, last_index):

    processes = list()
    for i in range(first_index, last_index):
        direc = f'chunk_{i+1}/'
        curr_dir = os.path.join(output_dir, f'chunk_{i+1}')
        if not os.path.exists(curr_dir):
            os.mkdir(curr_dir)
        new_p = multiprocessing.Process(target=comp_main, args=(os.path.join(relevant_infos_path, f'chunk_{i}/relevant_infos.json'),
                                                                os.path.join(input_dir, direc), curr_dir, wikidata_mappings_path,
                                                                list_indices_path,))
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
    parser.add_argument('-w', '--wikidata_mappings_path', type=str, help="Path to where the superclasses are kept")
    parser.add_argument('--list_indices_path', type=str)
    parser.add_argument("-f", "--first_index", type=int)
    parser.add_argument('-l', '--last_index', type=int)
    parser.add_argument('-r', '--relevant_infos_path', type=str, help='Path to where the mappings are kept')
    args = parser.parse_args()
    main(**vars(args))