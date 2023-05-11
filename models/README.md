# Models training

The trained model:
- [DPR](https://aggreg-qa.s3.amazonaws.com/models/dpr_model.tar.gz)
- [RAG-large multi](https://aggreg-qa.s3.amazonaws.com/models/RAG_model.tar.gz)
- [FiD-large multi](https://aggreg-qa.s3.amazonaws.com/models/FiD_model.tar.gz)

## Retrievers

Note that the retriever pipeline is long to run, and the results of both BM25 and DPR are available on our website.

We first chunk Wikipedia in chunks of 100 tokens max
`python models/retrievers/BM25/chunk_wiki.py -w path/to/wikipedia/ -o path/where/we/keep/chunks/`

### BM25

We need to run BM25 on the questions:

`python models/retrievers/BM25/simple_index.py --index_path path/to/chunked/wikipedia/ --query_path_or_split path/to/qampari/{train,dev,test}_data.jsonl 
    --shard_id shard_we_run_now --num_shards num_of_shard_in_indexing --input_type question --k 200 
    --output_path where/to/keep/bm25/results`

Since we also need to align our proofs with our smaller wikipedia chunks, so we can run:
`python models/retrievers/BM25/simple_index.py --index_path path/to/chunked/wikipedia/ --query_path_or_split path/to/qampari/{train,dev,test}_data.jsonl \
    --shard_id shard_we_run_now --num_shards num_of_shard_in_indexing --input_type proof --k 1 \
    --output_path where/to/keep/align/results`

We then only need to transfer to the same format as the dpr results:
`python models/retrievers/BM25/to_dpr.py --bm25_path path/to/bm25/resuts/  --aligned_path path/to/align/results/
    --example_path path/to/original/{train,dev,test}_data.jsonl --shard_id shard_we_run_now --num_shards num_of_shard_in_indexing
    --output_path path/where/to/keep/results`

### DPR

We first need to generate the embeddings for the contexts:

`python models/retrievers/GC-DPR/generate_dense_embeddings.py --model_file path_to_model --out_file where/to/keep/embeddings
    --ctx_file path/to/chunked/wikipedia --shard_id shard_in_case_we_split_runs --num_shards num_of_shards 
    --batch_size batch_size --fp16`

We then run DPR on the questions:

`python models/retrievers/dense_retriever.py --model_file path_to_model --encoded_ctx_file where/embeddings/are/kept
		--out_file where/to/keep/dpr/preds --ctx_file path/to/chunked/wikipedia
		--qa_file path/to/questions/data --save_or_load_index --fp16   --sequence_length 256`

## Readers

### FiD

Expects to receive the data as a .jsonl file (each line contains a json dict) with the following mandatory
keys: **_question_**, **_target_** (all the answers separated with the '#' sign) and **_ctxs_** (a list of
dictionaries with at least the _title_ and the _text_).

`python models/readers/FiD_model/train_reader.py --save_freq checkpoint_saving --train_data path/to/train/data/ 
--eval_data path/to/eval/data/ --model_size size_of_t5 --per_gpu_batch_size size_of_batch
--name name_of_experiment --lr learning_rate --optim optimizer_type --scheduler scheduler_type --weight_decay weight_decay_val
--total_step num_of_steps_to_run --main_port port_for_dataparallel --seed seed_value --eval_freq evaluation_frequence
--num_workers dataloader_workers --log_freq_train value_logging_freq --wandb_tags base_lr_0_00005 --n_context num_context_to_get
--use_previous_model use_already_saved_model --checkpoint_dir where/to/keep/checkpoints/`

To evaluate you can run:
`python models/readers/FiD_model/evaluate_fid.py  -i path/to/data/ -c checkpoint/to/run/ -n num_contexts -s model_size 
-p batch_size_per_gpu -w dataloader_workers -o where/to/keep/outputs.jsonl -l where/to/keep/log.log`

**Note that you can run zero-shot training and evaluation by defining num_contexts = 0**

### RAG

Expects to receive the data as a .json with the following mandatory keys: **_title_**, **_context_**, **_question_**, **_qid_**,
**_cid_**, **_answers_** (list of all the answers to the question), **_full\_answers_** (dictionary mapping from an answer
to a list of its aliases)

`python RAG/run_seq2seq_qa.py --per_device_eval_batch_size batch_size --model_name_or_path model_name --predict_with_generate True 
--evaluation_strategy steps --eval_steps steps_to_run_eval --cache_dir where/to/cache/model/
--train_file path/to/training/data --validation_file path/to/validation/data/  --test_file path/to/test/data/ 
--max_train_samples 6000000 --output_dir where/to/keep/checkpoint/ --do_train --do_eval --do_predict --learning_rate learning_rate 
--num_train_epochs num_epochs --per_device_train_batch_size batch_size_per_device --dataloader_num_workers 8 
--resume_from_checkpoint start/from/this/checpoint/`
