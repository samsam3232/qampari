import multiprocessing as mp
import os
from tqdm import tqdm
import requests
from typing import Dict
from DataCreation.DataAlignment.utils.properties_constants import DESIRED_SUBCLASSES
from time import sleep
from random import randint
import argparse
import json


def get_superclasses(results: Dict):

    """
    Parses the list of all the results and organizes it in a list.
    """

    superclasses = list()
    for i in range(len(results['results']['bindings'])):
      curr_results = results['results']['bindings'][i]
      if curr_results['classLabel']['value'] not in superclasses:
        superclasses.append(curr_results['classLabel']['value'])
      if curr_results['superclassLabel']['value'] not in superclasses:
        superclasses.append(curr_results['superclassLabel']['value'])
    return superclasses


def get_entity_superclasses(entity: str):

    """
    Given an entity string, returns a list of all the superclasses of which this entity is an instance of
    """
    sleep(randint(10, 500) / 100.)
    url= 'https://query.wikidata.org/sparql'
    curr_query = """
      SELECT ?class ?classLabel ?superclass ?superclassLabel
      WHERE 
      {
      wd:%s wdt:P279* ?class.
      ?class wdt:P279 ?superclass.
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
      } 
    """ % (entity)
    header = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36'}
    res = requests.get(url, params={'format': 'json', 'query': curr_query}, headers=header)
    results = res.json()
    supers = get_superclasses(results)
    sleep(randint(10, 100) / 100.)
    return list(DESIRED_SUBCLASSES.intersection(set(supers)))


def retrieve_entities(inputs):

    entities, output_path, chunk_num = inputs

    superclasses = dict()
    for entity in tqdm(entities):
        curr_sup = get_entity_superclasses(entity)
        superclasses[entity] = curr_sup

    with open(os.path.join(output_path, f'chunk_{chunk_num}.json'), 'w') as f:
        json.dump(superclasses, f)


def main(entities_path: str, num_entities: int, first_chunk: int, last_chunk: int, output_path: str):

    with open(entities_path, 'r') as f:
        entities = f.read()
    entities = entities.split('\n')

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    processes = mp.Pool(50)
    processes.map(retrieve_entities, [(entities[int(i * num_entities):int(min((i + 1) * num_entities, len(entities)))],
                                       output_path, i) for i in range(first_chunk, last_chunk)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Superclass tree")
    parser.add_argument('-e', '--entities_path', type=str, help="Path to the entities list.")
    parser.add_argument('-n', '--num_entities', type=int, help="Number of entities per thread.")
    parser.add_argument('-f', '--first_chunk', type=int, help="First chunk")
    parser.add_argument('-l', '--last_chunk', type=int, help="Last chunk")
    parser.add_argument('-o', '--output_path', type=str, help="Path to where to keep the super classes")
    args = parser.parse_args()
    main(**vars(args))
