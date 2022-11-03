import streamlit as st
# To run use `streamlit run myapp_v0.py`

# streamlit run  src/demo.py args --server.fileWatcherType none 
import streamlit as st
import glob
import json
from st_aggrid import AgGrid

import pandas as pd
import json
import json
st.set_page_config(layout="wide")
import random
import os
import numpy as np

def sort_ctxs(exp):
    for el in exp:
        el['ctxs'] = sorted(el['ctxs'],key=lambda x:-x['score'])
    return exp
    

def read_glob(paths):
    paths = glob.glob(paths)
    data = []
    for path in paths:
        with open(path) as f:
            if path.endswith(".json"):
                data.extend(json.load(f))
            elif path.endswith(".jsonl"):
                for line in f:
                    data.append(json.loads(line))
    if "last.ckpt" in paths:
        return sort_ctxs(data)
    else:
        return data


@st.experimental_singleton
def get_rand(preds_path):
    preds = read_glob(preds_path)

    rand = random.Random(4)
    # r = list(zip(data,preds))
    rand.shuffle(preds)
    # data,preds = list(zip(*r))
    return preds


# data_path = "true_data/dev_data.jsonl"

# preds_path = "/a/home/cc/students/cs/ohadr/netapp/4TB_SSD/odqa_experiments/+batch_size-61_v0/dpr_biencoder.120.1303_preds_v3.json"
exp1 = "/home/joberant/home/ohadr/odqa_aggreg/odqa_experiments/+batch_size-120+gradient_accumulation_steps-4+learning_rate-6e-05_v2/dpr_biencoder.120.516_preds.json"
exp2 = "/specific/a/home/cc/students/cs/ohadr/netapp/odqa_aggreg/4TB_SSD/query_output_v5/bm25/dev_dumb_bm25_v5/preds.json"
exp3 = "/home/joberant/home/ohadr/testbed/logs/experiments/runs/v7.1/2022-04-19_15-14-24/checkpoints/last.ckpt_validation_predictions_final.json"
exp4 = "/a/home/cc/students/cs/ohadr/netapp/odqa_aggreg/odqa_experiments/+warmup_steps-4000+batch_size-120+num_train_epochs-300+gradient_accumulation_steps-4+learning_rate-6e-05_v0/dpr_biencoder.134.516_preds.json"
exp5= "/a/home/cc/students/cs/ohadr/netapp/odqa_aggreg/odqa_experiments/+batch_size-120+gradient_accumulation_steps-4+learning_rate-6e-05_v4/dpr_biencoder.120.516_preds.json"
exp_predictions = {"rag":exp3,
                    "dpr_v1":exp1,
                   "bm25":exp2,
                   "dpr_v2":exp4,
                   "dpr_simple":exp5,
                  }

preds_path = st.sidebar.selectbox('Model', list(exp_predictions.keys()),index=0)
preds_path = exp_predictions[preds_path] 

strict_eval = st.sidebar.radio(label="Strict",options=[True,False])

# unique_preds = st.sidebar.radio(label="Unique Predictions",options=[True,False],index=1)
# preds_path = st.text_area('Text to translate',preds_path)

def answer_is_in_ctx(answer,ctx):
    return (answer['answer_text'] in ctx['title']) or (answer['answer_text'] in ctx['text'])


def recall_at_k(pred,strict_eval,**kwargs):
    missing_answers = []
    correct_ranks = []
    if not strict_eval:
        res = []
        answer_set = set()
        ctx_list = []
        for ctx_idx,ctx in enumerate(pred['ctxs']):
            for answer in pred['answer_list']:
                if answer_is_in_ctx(answer,ctx):
                    if answer['aid'] not in answer_set :
                        answer_set.add(answer['aid'])
                        ctx_list.append([ctx_idx, answer['answer_text']])
            res.append(len(answer_set))
        # answer_idx = set([answer['aid'] ])
        for answer in pred['answer_list']:

            if answer['aid'] not in answer_set:
                missing_answers.append([answer['aid'],answer["answer_text"]])
        # missing_answers =  [[answer["answer_text"],answer["aid"]]
        #                     for answer in pred['answer_list'] if answer['aid'] in (answer_idx - answer_set)]
        cum_sum = np.array(res)
    else:
        res = []
        pos_set = set()
        ctx_list = []
        for ctx_idx,ctx in enumerate(pred['ctxs']):
            for proof in pred['positive_ctxs']:
                if ctx['id'] == proof['chunk_id']:
                    if proof['chunk_id'] not in pos_set :
                        pos_set.add(proof['chunk_id'])
                        ctx_list.append([ctx_idx, proof['ans_for']])
            res.append(len(pos_set))
        proof_idx = set([proof['chunk_id'] for proof in pred['positive_ctxs']])
        for proof in pred['positive_ctxs']:
            if proof['chunk_id'] not in pos_set:
                missing_answers.append([proof['ans_for'],proof['chunk_id']])
        cum_sum = np.array(res)
    # return cum_sum,ctx_list,missing_answers
    return cum_sum,missing_answers,ctx_list
        # missing_answers =  [[proof["ans_for"],proof['chunk_id']]
        #                     for proof in pred['positive_ctxs'] if proof['chunk_id'] in (proof_idx - pos_set)]
        # cum_sum = np.array(res)
    
    # 

preds = get_rand(preds_path)


# if 'count' not in st.session_state:
    # st.session_state.count = 0
# def increment_counter():
# 	st.session_state.count = min(len(data)-1, st.session_state.count + 1)
# def decrement_counter():
# 	st.session_state.count = max(0, st.session_state.count - 1)
# # but1, but2,_ = st.columns([1, 1,15])
# st.button('Increment', on_click=increment_counter)
import random

def set_num_in():

	st.session_state.num_in = random.randint(0, len(preds) - 1)
    # ,index=0,on_change=set_num_in
st.sidebar.button('Shuffle', on_click=set_num_in)


# datum,pred = data[st.session_state.count],preds[st.session_state.count]



num_in = st.sidebar.number_input('Dev instance number:', 0, len(preds),key="num_in")

# def get_len_proofs(datum):
#     # answer_id = 0
#     return len(datum['proofs'])
    # for answer in datum['answers']:
    #     if isinstance(answer=,list):
    #         for url,proof in zip(answer['found_in_url'],answer['proof']):
    #             answer_id +=1
    #     else:
    #         answer_id +=1
    # return answer_id

# def fix_preds(pred):
#     qid = pred['qid']
#     corr_preds = []
#     for ctx in pred['ctxs']:
#         # corr = ctx['id'].split("__")[0]==qid
#         # cor
#         corr_preds.append(ctx['has_answer'])
#         # if corr:
#         #     =True
#     return pred,corr_preds


# print(type(preds))
pred = preds[num_in]
cum_sum,missing_answers,ctx_list = recall_at_k(pred,strict_eval)

st.sidebar.write(f"Number of proofs: **{len(pred['positive_ctxs'])}**")
st.sidebar.write(f"Number of answers: **{len(pred['answers'])}**")
st.sidebar.write(f"Relevant passages (recall@200):  **{len(ctx_list)}**")
if "pred" in pred['ctxs'][0]:
    col_list = st.columns([1, 2,2,2])
    col_list[0].write("## Predictions")
    # ans_preds = [x['pred'][len("Answer: "):] for x in pred['ctxs']]
    ans_preds = []
    num_corr = 0
    if len(ctx_list)>0:
        answer_idx,answers =  list(zip(*ctx_list))
        # sanitized_answers = 
        ans_preds = [(i,x['pred'][len("Answer: "):]) for i,x in enumerate(pred['ctxs']) if x['pred'] not in  ["Not relevant"]]
        # col_list[0].write(ans_preds)
        ans_preds = [(f"{'✅ ' if i in answer_idx else '❌'}",x) for i,x in ans_preds]
        num_corr = len([x for x in ans_preds if x[0]=='✅ '])
        
        ans_preds = [f"{correct} {answer}" for correct,answer in ans_preds]
    col_list[0].write(ans_preds)
    st.sidebar.write(f"Correct answers for RAG: **{num_corr}**")
else:
    col_list = st.columns([1, 1,1])


# ranks = [ i for i,x in enumerate(corr_preds) if x]


st.sidebar.write("## Data")
q_index,subset,split_name =pred['qid'].split("__")
st.sidebar.write(f"Question index: **{q_index}**")
st.sidebar.write(f"Subset: **{subset}**")
st.sidebar.write(f"Split: **{split_name}**")
st.sidebar.write("### Question")
st.sidebar.write(pred['question'])
st.sidebar.write("### Answers")
st.sidebar.write(pred['answers'])
st.sidebar.write("Correct rank list:")
st.sidebar.write(ctx_list)

st.sidebar.write("Answers we missed:")
st.sidebar.write(missing_answers)

# col_list[0], col_list[1]  
col_list[-2].write("## Top 200 Passages")
col_list[-2].write([dict(rank=j,**x) for j,x in enumerate(pred['ctxs'][:100])])
col_list[-2].write([dict(rank=j+100,**x) for j,x in enumerate(pred['ctxs'][100:])])





col_list[-3].write("## Correct")
col_list[-3].write([dict(rank=j,answer=answer,**pred['ctxs'][j]) for j,answer in ctx_list])
col_list[-1].write("### Proofs")
col_list[-1].write(pred['positive_ctxs'])