
import json
from timeit import repeat
import tqdm
import glob
from sacremoses import MosesDetokenizer

import os
# print(os.getcwd())
# os.chdir(os.path.expanduser("~/netapp/odqa_aggreg"))
import re
from itertools import repeat

import fire

import json
from timeit import repeat
import tqdm.auto as tqdm
import glob
from sacremoses import MosesDetokenizer
import re
from itertools import repeat
import multiprocessing


from fuzzysearch import find_near_matches

def apply_pool(func,iterable,iterable_size,processes=100):
    with multiprocessing.Pool(processes=processes) as pool:
        result_list = []
        for _, res in tqdm.tqdm(enumerate(pool.imap_unordered(func, iterable)),total=iterable_size):
            result_list.append(res)
    return result_list


import fire

def read_jsonl(path):
    data_list= []
    with open(path) as f:
        for line in f:
            data_list.append( json.loads(line))
    return data_list

def get_dict(file_path):
    proof_d = {}
    with open(file_path) as f:
        for x in  json.load(f):           
            proof_d[x['pid']] = x['query_res']
    return proof_d

def better_neg(aliases,query_res,gold_chunk_ids,pid):
    _ctxs = []
    for ctx in query_res:
        if (not contains_any_answer(aliases,ctx['contents'])) \
            and (ctx['id'] not in gold_chunk_ids):

            meta = ctx['meta']
            tmp = dict(chunk_id=ctx['id'],
                    title=meta['title'],
                    text=meta['content'],
                    score=ctx["score"],
                    pid=pid)
            _ctxs.append(tmp)
    return _ctxs


def neg_pos_datum(datum):
    pos_list = []
    neg_list = []
    aliases = []
    for answer in datum['answer_list']:
        aliases.extend(answer['aliases'])
    for answer in datum['answer_list']:
        for proof in answer['proof']:
            if proof['pid'] in aligned_proof_dict:
                proof_query_res = aligned_proof_dict[proof['pid']]            
                pos_list.append(parse_neg_pos(proof_query_res[0],pid=proof['pid']))
            
    gold_chunk_ids = [x['chunk_id'] for x in pos_list]
    if datum['qid'] in bm25_question_dict:
        question_query_res = bm25_question_dict[datum['qid']]
        neg_list = better_neg(aliases,question_query_res,gold_chunk_ids,pid=datum['qid'])[:len(pos_list)]
       
    return neg_list,pos_list

def parse_neg_pos(example,pid):
    result = {"text":example['meta']['content'],
             "title":example['meta']['title'],
             "score":example["score"],
             "chunk_id":example['meta']["chunk_id"],
             "pid":pid}
    return result

def contains_any_answer(aliases,text,max_l_dist=3):
    matches = []
    for alias in aliases:
        try:
            m = find_near_matches(alias.lower(),text.lower(), max_l_dist=max_l_dist)
            matches.extend(m)
        except:
            pass
    if len(matches)==0:
        return False
    else:
        return True


    
def parse_example(datum):
    neg_list,pos_list = neg_pos_datum(datum)
    datum['question'] = datum['question_text']
    datum['answers']=[ans['answer_text'] for ans in datum['answer_list']]
    datum['positive_ctxs']=pos_list
    datum['hard_negative_ctxs']=neg_list
    _ctxs = []
    if datum['qid'] in bm25_question_dict:
        question_query_res = bm25_question_dict[datum['qid']]
        
        for ctx in question_query_res:
            meta = ctx['meta']
            tmp = dict(id=ctx['id'],
                    title=meta['title'],
                    text=meta['content'],
                    score=ctx["score"],
                    )
            _ctxs.append(tmp)
    datum['ctxs']= _ctxs
    return datum

import more_itertools

bm25_question_dict = {}
aligned_proof_dict = {}

def run(bm25_path, aligned_path, example_path, output_path,shard_id=0,num_shards=1):
    print(shard_id,num_shards,bm25_path,aligned_path, example_path, output_path)
    # exit()
    os.makedirs(output_path,exist_ok=True)
    global bm25_question_dict
    global aligned_proof_dict

    bm25_files = glob.glob(f"{bm25_path}/chunk_*.json")
    aligned_files = glob.glob(f"{aligned_path}/chunk_*.json")
    
    

    aligned_proof_dict_list = apply_pool(get_dict,aligned_files,len(aligned_files),processes=os.cpu_count())  
    for el in aligned_proof_dict_list:
        aligned_proof_dict.update(el)
    
    # data_dict = {el['qid']:el for el in data}

    bm25_question_dict_list = apply_pool(get_dict,bm25_files,len(bm25_files),processes=os.cpu_count())  
    for el in bm25_question_dict_list:
        bm25_question_dict.update(el)
        # with open(f"{output_path}_tmp","w") as f:
        #     json.dump(proof_dict,f)
    # purint(proof_dict)
    data =  read_jsonl(example_path)
    print(f"{len(data)} total queries to process")
    data = list(more_itertools.distribute(num_shards, data)[shard_id])
    # if shard_id==0 and num_shards==5:
    #     data = data[:-200]
    print(f"{len(data)} queries to process by this shard")
    
    # itr = zip(train_data,repeat(proof_dict))
    # reformated_train_data = apply_pool(parse_example,train_data,len(train_data),processes=os.cpu_count())  
    mod_data = apply_pool(parse_example,data,len(data),processes=os.cpu_count())  

    with open(f"{output_path}_{shard_id}.json","w") as f:
        json.dump(mod_data,f)

if __name__ == '__main__':
  fire.Fire(run)

