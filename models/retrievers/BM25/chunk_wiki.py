import argparse
from DataCreation.DataAlignment.utils import alignment_utils
import glob
from nltk.tokenize import word_tokenize,sent_tokenize
import numpy as np
import bisect
import os
from models.retrievers.BM25.pool import apply_pool
import more_itertools
from itertools import repeat
from sacremoses import MosesDetokenizer
import re
import json

detokenizer = MosesDetokenizer(lang='en')


def normalize(el):
    el = fix_qu(el.replace("\'","'"))
    tokens =el.split(" ")
    return detokenizer.detokenize(tokens).replace("\'","'")


def fix_qu(string):
    pat = re.compile('\"(.*?)\"')
    pat2 = re.compile('\" (.*?) \"')
    pat3 = re.compile("\'(.*?)\'")
    pat4 = re.compile("\' (.*?) \'")
    for x in pat.finditer(string):
        to_replace =x.group(0)
        res = pat2.match(to_replace)
        if res:
            replace_with = f'"{res.group(1)}"'
            string = string.replace(to_replace,replace_with)
    for x in pat3.finditer(string):
        to_replace =x.group(0)
        res = pat4.match(to_replace)
        if res:
            replace_with = f'"{res.group(1)}"'
            string = string.replace(to_replace,replace_with)
    return string

def read_jsonl(path):
    data_list= []
    with open(path) as f:
        for line in f:
            data_list.append( json.loads(line))
    return data_list


def _read_parsed_wiki(file_path):
    dump = alignment_utils.read_parsed_wikipedia(file_path)
    for el in dump:
        el['file_path'] = file_path
        yield el

def percent(i,p):
    return int(i*p)

def get_json(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data

def read_jsonl(path):
    data_list= []
    with open(path) as f:
        for line in f:
            data_list.append( json.loads(line))
    return data_list


def get_lengths(text):
    sents = sent_tokenize(text['text'])
    sents_lens =[len(word_tokenize(x)) for x in sents]
    text['sents'] = sents
    text['sents_lens'] = sents_lens
    return text


def get_idxs(len_array,total=100):
    idx_list = [0]
    orig_array = len_array
    while True:
        c_sum = np.cumsum(len_array)
        new_idx = bisect.bisect_right(c_sum,total)
        len_array = len_array[new_idx:]
        if new_idx==0:
            break
        idx_list.append(new_idx)
    idx_start_end=np.cumsum(idx_list).tolist()
    word_per_chunk=[sum(orig_array[s:e]) for s,e in list(more_itertools.windowed(idx_start_end,2))]
    return dict(idx_start_end=idx_start_end,word_per_chunk=word_per_chunk)


def get_split(args):
    len_array = args['sents_lens']
    el_list = [get_idxs(len_array,total=x) for x in range(80,140,5)]
    el_list = sorted(el_list,key = lambda el: abs(np.min(el['word_per_chunk'])-120))
    chunks = el_list[0]

    
    dict_list = []
    for chunk_id,(s,e) in enumerate(more_itertools.windowed(chunks['idx_start_end'],2)):
        text = " ".join(args['sents'][s:e])
        
        curr_dict = {key:args[key] for key in ['revid', 'url', 'title', 'file_path']}
        curr_dict['page_id'] = args['id']

        curr_dict['title'] = normalize(curr_dict['title'])
        curr_dict['content'] = normalize(text)
        curr_dict['chunk_id'] = f"{args['id']}__{chunk_id}"
        dict_list.append(curr_dict)
        
    return dict_list


def get_chunks(element):

    (chunk_id,path_name),output_path = element
    path = f"{output_path}/wikipedia_chunks_{chunk_id}.jsonl"
    with open(path,"w") as g:   
        for wikipedia_page in _read_parsed_wiki(path_name):
            wikipedia_page_wlens = get_lengths(wikipedia_page)
            if len(wikipedia_page_wlens['sents_lens'])==0:
                continue
            actions = get_split(wikipedia_page)
            for datum in actions:
                index_field = f"{datum['title']} {datum['content']}"
                datum = dict(id=datum["chunk_id"],contents=index_field,meta=datum)
                g.write(json.dumps(datum)+"\n")




def main(wikipedia_path: str, output_path: str):

    os.makedirs(output_path,exist_ok=True)
    glob_list = list(glob.glob(wikipedia_path))
    print(f"{len(glob_list)} files to process")
    path_list = list(zip(enumerate(glob_list),repeat(output_path)))
    apply_pool(get_chunks,path_list,len(glob_list),processes=os.cpu_count())  

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Wikipedia chunker')
    parser.add_argument('-w', '--wikipedia_path', type=str, help="Path where the Wikipedia pages are kept")
    parser.add_argument('-o', '--output_path', type=str, help="Path where we want to keep the results")
    args = parser.parse_args()
    main(**vars(args))