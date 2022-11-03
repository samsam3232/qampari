import gzip
import math
import os
import multiprocessing as mp
import json
import sys
from typing import Dict
import argparse
from DataCreation.DataAlignment.utils.alignment_utils import get_webpage_sentences, find_all_phrases, plural_to_singular


def read_tables(input_path):

    data = list()
    with gzip.open(input_path, 'r') as f:
        for line in f:
            curr_dat = json.loads(line)
            data.append(curr_dat)
    return data


def retrieve_sentences(table: Dict, colnum: int, rowindex: int, indices_path: str):

    """
    Retrieves the sentences in the hyperlink of a given cell.
    """

    cell = table['table']['table_rows'][rowindex][colnum]
    sentences = ""
    if 'links' in cell and (len(cell['links']) > 0):
        sentences = get_webpage_sentences([cell['links'][0]['url']], indices_path)
    return cell, sentences


def align_appartenance(table, col1, indices_path):

    """
    Receives a table and a column number and verifies that each element in the row of the table is aligned wth the table.
    """

    results = dict()
    col1_n = table['table']['header'][col1]['column_name']
    page_name = table['url'].split('/')[-1].split('_')[2:]
    page_name_sing = [plural_to_singular(i) for i in page_name]
    num_rows = 0
    for i in range(len(table['table']['table_rows'])):
        found = 0
        basis, basis_sentences = retrieve_sentences(table, col1, i, indices_path)
        if len(basis_sentences) == 0:
            continue
        num_rows += 1
        for word in page_name:
            if word in basis_sentences.lower():
                found += 1

        if found == len(page_name):
            results[f"appartenance##{col1_n}##{basis['text']}"] = [' '.join(page_name), basis['links'][0]['url']]
            continue

        found = 0
        for word in page_name_sing:
            if word in basis_sentences.lower():
                found += 1

        if found == len(page_name):
            results[f"appartenance##{col1_n}##{basis['text']}"] = [' '.join(page_name), basis['links'][0]['url']]
            continue

    return results, {f'{col1_n}' :num_rows}

def align_columns(table, col1, col2, indices_path):

    """
    Receives a table and two columns indices as inputs, and performs alignment on all the rows according to these two
    columns.
    """

    results = dict()
    col1_n = table['table']['header'][col1]['column_name']
    col2_n = table['table']['header'][col2]['column_name']
    num_cols = 0
    if "col2_n".lower() == "note":
        return results
    for i in range(len(table['table']['table_rows'])):
        basis, basis_sentences = retrieve_sentences(table, col1, i, indices_path)
        checking, checkin_sentences = retrieve_sentences(table, col2, i, indices_path)
        if len(basis_sentences) == 0:
            continue
        num_cols += 1
        phrases_curr = find_all_phrases(basis_sentences.lower(), checking['text'].lower(), True)
        if (len(phrases_curr) > 0):
            results[f"{col1_n}##{basis['text']}##{col2_n}##{checking['text']}"] = \
                [checking['text'].lower(), basis['links'][0]['url'], phrases_curr]
            continue

        phrases_curr = find_all_phrases(checkin_sentences.lower(), basis['text'].lower(), True)
        if (len(phrases_curr) > 0):
            results[f"{col1_n}##{basis['text']}##{col2_n}##{checking['text']}"] = \
                [basis['text'].lower(), checking['links'][0]['url'], phrases_curr]
            continue


    return results, {f'{col1_n}##{col2_n}': num_cols}


def align_table(table, indices_path):

    aligned = dict()
    num_rows = dict()
    for i in range(len(table['table']['header']) - 1):
        semantic = ('potential_semantic' in table['table']['header'][i]['metadata']) and (table['table']['header'][i]['metadata']['potential_semantic'])
        if not semantic:
            continue
        for j in range(len(table['table']['header'])):
            if i == j:
                continue
            curr_aligned, num_rows_curr = align_columns(table, i, j, indices_path)
            aligned.update(curr_aligned)
            num_rows.update(num_rows_curr)

        appartenance, num_rows_app = align_appartenance(table, i, indices_path)
        aligned.update(appartenance)
        num_rows.update(num_rows_app)

    return aligned, num_rows


def align_table_list(inputs):
    tables_list, indices_path, outpath = inputs

    aligned = dict()
    for table in tables_list:
        for i in range(len(table['context']['documents'])):
            if 'list_of' not in table['context']['documents'][i]['url'].lower():
                continue
            curr_aligned, num_rows = align_table(table['context']['documents'][i], indices_path)
            if num_rows == 0:
                continue
            aligned[f"{table['context']['documents'][i]['url']}, {table['context']['documents'][i]['table']['table_name']}_{i}"] = [num_rows, curr_aligned]

    with open(outpath, 'w') as f:
        json.dump(aligned, f)

    return 1


def main(input_path: str, indices_input_path: str, output_path: str, num_pools: int):
    tables = read_tables(input_path)
    with open(indices_input_path, 'r') as f:
        indices_path = json.load(f)

    # align_table_list([tables[150*45: 150*(45+1)], indices_path, os.path.join(output_path, f'table_chunk_40.json')])
    # sys.exit(1)
    # align_table_list((tables[:2], indices_path, os.path.join(output_path, f'table_chunk_0.json')))
    processes = mp.Pool(num_pools)
    processes.map(align_table_list, [(tables[int(i * 150):int(min((i + 1) * 150, len(tables)))], indices_path,
                                      os.path.join(output_path, f'table_chunk_{i}.json')) for i in
                                     range(math.ceil(len(tables) / 150.))])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Table multiprocessing")
    parser.add_argument("-i", "--input_path", type=str, help="Path to where the tables are kept", required=True)
    parser.add_argument("--indices_path", type=str, help="Path to the page's path", required=True)
    parser.add_argument("-o", "--output_path", type=str)
    parser.add_argument("-n", "--num_pools", type=int, help="Number of pool processes", default=50)
    args = parser.parse_args()
    main(**vars(args))