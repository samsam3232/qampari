import json
from DataCreation.DataAlignment.utils.properties_constants import FOR_SIMPLE, ALL_PROPERTIES
from typing import Dict, DefaultDict, List
from collections import defaultdict
import os
import logging
from tqdm import tqdm
import bz2
import argparse


def wikidata(filename: str):

    """
    Given a path to the wikidata dump, retrieves it entity per entity.
    """

    with bz2.open(filename, mode='rt') as f:
        f.read(2) # skip first two bytes: "{\n"
        for line in f:
            try:
                yield json.loads(line.rstrip(',\n'))
            except json.decoder.JSONDecodeError:
                continue


def retrieve_all(file_dir: str):

    """
    Goven a path to a directory of file, we load all the questions the couples of property and entity (potential simple
    question), then filter out all the questions that have less than 5 answers.
    We also filter out all the entities that are not the answer to some question.
    """

    subdirs = os.listdir(file_dir)
    relevant_infos, couples = defaultdict(lambda: {}), defaultdict(lambda: [])
    to_keep_answers = list()
    for dir in tqdm(subdirs):
        if 'coup' in dir:
            with open(os.path.join(file_dir, dir), 'r') as f:
                data = json.load(f)
            for key in data:
                couples[key] += data[key]

    to_del = list()
    for key in couples:
        if len(couples[key]) < 5:
            to_del.append(key)

    for key in to_del:
        couples.pop(key)

    for key in couples:
        for answer in couples[key]:
            if answer in relevant_infos and answer not in to_keep_answers:
                to_keep_answers.append(answer)

    for dir in tqdm(subdirs):
        if 'rele' in dir:
            with open(os.path.join(file_dir, dir), 'r') as f:
                data = json.load(f)
            for key in data:
                if key in to_keep_answers:
                    relevant_infos[key] = data[key]

    return relevant_infos, couples

def parse_answer(answer: str):

    """
    Parses a property related dict.
    """

    try:
        if answer['mainsnak']['datavalue']['type'] == 'wikibase-entityid':
            return answer['mainsnak']['datavalue']['value']['id']
        elif answer['mainsnak']['datavalue']['type'] == 'quantity':
            return float(answer['mainsnak']['datavalue']['value']['amount'])
        elif answer['mainsnak']['datavalue']['type'] == 'time':
            return int(answer['mainsnak']['datavalue']['value']['time'].split('-')[0])
        elif answer['mainsnak']['datavalue']['type'] == 'monolingualtext':
            if answer['mainsnak']['datavalue']['value']['language'] == 'en':
                return answer['mainsnak']['datavalue']['value']['text']
            else:
                return None
        elif answer['mainsnak']['datavalue']['type'] == 'string':
            return answer['mainsnak']['datavalue']['value']
        else:
            logging.info(f'Something is weird {answer}')
            return None
    except:
        logging.info('Error occured')
        logging.info(answer)
        return None


def save_cache(relevant_infos: dict, couples: dict, out_dir: str, cache_num: int):

    with open(os.path.join(out_dir, f'relevant_infos_{cache_num}.json'), 'w') as f:
        json.dump(relevant_infos, f)
    with open(os.path.join(out_dir, f'couples_{cache_num}.json'), 'w') as f:
        json.dump(couples, f)


def retrieve_website(single_prop: Dict) -> str:

    """
    Receives a dict and outputs the english wikipedia url of the object.
    """
    url = ""
    if 'sitelinks' in single_prop and 'enwiki' in single_prop['sitelinks']:
        url = f'https://en.wikipedia.org/wiki/{single_prop["sitelinks"]["enwiki"]["title"]}'
        url = url.replace(' ', '_')
    return url


def retrieve_labels(single_prop: Dict) -> List:

    """
    Given a property dictionary returns the label and all the alternative labels with a length larger than 3.
    """
    potential_labels = list()

    # check for the main label
    if 'labels' in single_prop:
        if 'en' in single_prop['labels']:
            if len(single_prop['labels']['en']['value']) > 3:
                potential_labels.append(single_prop['labels']['en']['value'])

    # check for alternative labels
    if 'aliases' in single_prop:
        for lang in single_prop['aliases']:
            if 'en' in lang:
                for alias in single_prop['aliases'][lang]:
                    if len(alias['value']) > 3:
                        potential_labels.append(alias['value'])

    return potential_labels


def retrieve_properties(single_prop: Dict) -> Dict:

    """
    Given the claims will return a dictionary of the properties and the objects they are about.
    """

    relations = defaultdict(lambda : [])
    if 'claims' in single_prop:
        for key in single_prop['claims']:
            if key in ALL_PROPERTIES:
                for answer in single_prop['claims'][key]:
                    if 'qualifiers' in answer and 'P582' in answer['qualifiers']:
                        continue
                    answer_parsed = parse_answer(answer)
                    if type(answer_parsed) == float or type(answer_parsed) == int:
                        relations[key].append(answer_parsed)
                        break
                    if answer_parsed is not None:
                        relations[key].append(answer_parsed)

    return relations


def retrieve_informations(wikidata_entity):

    """
    Given a wikidata entity dic retrieves all the informations needed to continue.
    """

    url = retrieve_website(wikidata_entity)
    potential_labels = retrieve_labels(wikidata_entity)
    relations = retrieve_properties(wikidata_entity)
    wiki_id = wikidata_entity['id']

    return url, potential_labels, relations, wiki_id



def parse_file(file_path: str, out_dir: str):

    """
    Given a file path to a wikidata json dump, parses it and returns a dictionary containing the relevant info for each
    wikidata entity and a dictionary of all the couples and their answers.
    Save every 3000000 entities to a cache.
    """

    relevant_infos, couples = defaultdict(lambda : {}), defaultdict(lambda : [])
    count, saved = 0, 0
    for single_prop in tqdm(wikidata(file_path)):
        if 'id' not in single_prop:
            continue

        url, potential_labels, relations, wiki_id = retrieve_informations(single_prop)

        if len(url) > 0:
            relevant_infos[wiki_id]['url'] = url
        if len(potential_labels) > 0:
            relevant_infos[wiki_id]['potential_labels'] = potential_labels

        added_comp = False
        for relation in relations:
            if relation in ALL_PROPERTIES and not added_comp:
                relevant_infos[wiki_id]['composition'] = defaultdict(lambda : [])
                added_comp = True
            for answer in relations[relation]:
                if relation in ALL_PROPERTIES:
                    relevant_infos[wiki_id]['composition'][relation].append(answer)
                if relation in FOR_SIMPLE:
                    couples[f'{relation}##{answer}'].append(wiki_id)

        count += 1

        if count % 3000000 == 2999999:
            save_cache(relevant_infos, couples, out_dir, saved)
            saved += 1

    save_cache(relevant_infos, couples, out_dir, saved)

    return relevant_infos, couples


def main(file_path: str, output_dir: str):

    """
    Given a file path to json parsed file, we parse it and output it to the dir that we have.
    """

    logging.basicConfig(filename=os.path.join(output_dir, "app.log"), filemode="w", level=logging.INFO)

    if not os.path.exists(os.path.join(output_dir, 'intermediate')):
        print('Parsing file')
        os.mkdir(os.path.join(output_dir, 'intermediate'))
        parse_file(file_path, os.path.join(output_dir, 'intermediate'))
    else:
        print('Skipping parsing')

    relevant_infos, couples = retrieve_all(os.path.join(output_dir, 'intermediate'))

    with open(os.path.join(output_dir, 'relevant_infos_fp.json'), 'w') as f:
        json.dump(relevant_infos, f)
    with open(os.path.join(output_dir, 'couple_answers_fp.json'), 'w') as f:
        json.dump(couples, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser('WikiData parsing.')
    parser.add_argument('-f', '--file_path', type=str, help="Path to the file/dir to parse.")
    parser.add_argument('-o', '--output_dir', type=str, help="Path to where to output the parsed data.")
    args = parser.parse_args()
    main(**vars(args))