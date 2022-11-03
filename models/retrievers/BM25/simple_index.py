from pool import apply_pool
import os
os.environ["JAVA_HOME"] = "/a/home/cc/students/cs/ohadr/netapp/jdk-11.0.12"
os.environ["JVM_PATH"] = "/a/home/cc/students/cs/ohadr/netapp/jdk-11.0.12/lib/server/libjvm.so"
import fire
from pyserini.search.lucene import LuceneSearcher
import tqdm.auto as tqdm
import json
from datasets import load_dataset

import more_itertools
print("after import")


def read_jsonl(path):
    data_list= []
    with open(path) as f:
        for line in f:
            data_list.append( json.loads(line))
    return data_list

def reformat_data(data):
    for datum in data:
        for answer in datum['answer_list']:
            for proof in answer['proof']:
                yield proof['proof_text'], proof['pid']

def idxformat_to_infformat(element):
    return dict(question=element['proof'],
                qid=element['proof'],
                answers=element['answers'],
                ctxs=[ctxqres_to_ctxdpr(ctx) for ctx in element['query_res']])

def ctxqres_to_ctxdpr(ctx):
    title,text = ctx['contents'].split("\n")
    title = title.strip('"')
    return  {'id': ctx['id'],
              'score':ctx['score'],
              'text': text,
              'title': title}


def run(query_path_or_split,input_type, output_path,index_path=None,k=200,shard_id=0,num_shards=1,use_cached=False):
    os.makedirs(output_path,exist_ok=True)
    assert input_type in ["proof","question","nq"]
    if input_type in ["proof","question"]:
        assert index_path is not None
        print("before reading data")
        query_path = query_path_or_split
        queries =  read_jsonl(query_path)
        print("before querie_list")
        if input_type=="proof":
            querie_list = list(reformat_data(queries))
        else:
            querie_list = [(query['question_text'],query['qid']) for query in queries]
        searcher = LuceneSearcher(index_path)
    else:
        assert input_type=="nq"
        assert index_path is None
        searcher = LuceneSearcher.from_prebuilt_index("wikipedia-dpr")
        nq_dataset = load_dataset("/home/joberant/home/ohadr/qampari/nq.py","inference_dprnq")
        split = query_path_or_split
        querie_list = [[element['question'],element['qid'],element['answers']] for element in nq_dataset[split]]
        # answer_list = [element['answers'] for element in nq_dataset[split]]

    print(f"{len(querie_list)} total queries to process")
    print(f"{len(querie_list)/num_shards} queries to process by this shard")


    querie_list_c =  more_itertools.chunked(querie_list, 800)
    querie_list_c = list(more_itertools.distribute(num_shards, querie_list_c)[shard_id])
    
    # answer_list

    for i,query_chunk in enumerate(tqdm.tqdm(querie_list_c)):
        if use_cached and os.path.exists(f"{output_path}/chunk_{shard_id}_{i}.json"):
            continue
        query_text,query_idxs,_ = list(zip(*query_chunk))        
        query_text_tmp = [" ".join(q_text.split(" ")[-950:]) for q_text in query_text]
        res = searcher.batch_search(queries=query_text_tmp,qids=query_idxs,threads=int(1.5*os.cpu_count()),k=k)
        final_list = []
        for (proof,pid,answer_list),query_res in zip(query_chunk,[res[y] for y in query_idxs]):
            q_res =[{"score":q.score,**json.loads(q.raw)} for q in query_res]
            r = dict(proof=proof,pid=pid,query_res=q_res,answers=answer_list)
            final_list.append(idxformat_to_infformat(r))

        with open(f"{output_path}/chunk_{shard_id}_{i}.json","w") as f:
            json.dump(final_list,f)


if __name__ == '__main__':
  fire.Fire(run)
