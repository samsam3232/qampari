import multiprocessing
import os
import argparse
from DataCreation.DataAlignment.WikiDataAlignment.postProcess.typing_differentiation import main as typing_main

def main(input_dir, superclasses_path, relevant_infos_path, first_index, last_index):

    processes = list()
    for i in range(first_index, last_index):
        direc = f'chunk_{i+1}/'
        new_p = multiprocessing.Process(target=typing_main, args=(os.path.join(input_dir, direc), superclasses_path,
                                                                  os.path.join(relevant_infos_path, f'chunk_{i}/relevant_infos.json'),))
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
    parser.add_argument('-s', '--superclasses_path', type=str, help="Path to where the superclasses are kept;")
    parser.add_argument("-f", "--first_index", type=int)
    parser.add_argument('-l', '--last_index', type=int)
    parser.add_argument('-r', '--relevant_infos_path', type=str, help='Path to where the mappings are kept')
    args = parser.parse_args()
    main(**vars(args))