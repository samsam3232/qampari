# Data creation

## Wikidata 

### Simple questions

First step of the WikiData creation process is to parse Wikidata to get the potential question couples. To do so run the
following commands:  
`python DataCreation/wikidata_parsing/wikidata_parser.py -f path/to/wikidata/file -o path/to/intermediate/saved/info`  
`python DataCreation/wikidata_parsing/wikidata_chunker.py -i path/to/couples -o path/to/where/to/keep/chunked/wikidata
-n num_questions_in_chunk -r path/to/relevant_infos  (chunks wikidata for faster processing later)`

We can then begin to create simple questions by trying to align the potential answers to our questions.  
`python DataCreation/DataAlignment/WikiDataAlignment/multi_process_alignment.py -i path/to/chunked/wikidata -o path/to/where/to/keep/results/
--indices_path path/to/mapping/from/wikipedia_url/to/location/in/wikipedia -f first_chunk_to_treat -l last_chunk_to_treat`

We then want to collect all the entities appearing in our answers, what they are instances of, and then build a tree of what these
instances are a subclass of.  
`python DataCreation/DataAlignment/WikiDataAlignment/postProcess/multi_process_entities_retriever.py -i path/to/chunked/wikidata 
-q path/to/aligned/questions/ -t path/for/entity/outputs/  -f first_chunk_to_treat -l last_chunk_to_treat`  
`python DataCreation/DataAlignment/WikiDataAlignment/postProcess/superclass_tree.py -e path/to/entities/ -n entities_per_thread
-f first_chunk_of_entities -l last_chunk_of_entities -o path/for/tree/output/`  
`python DataCreation/DataAlignment/WikiDataAlignment/postProcess/multi_process_typing.py -i path/to/questions/ -s path/to/superclass/tree/
-f first/chunk/ -l last/chunk/ -r path/to/wikidata/infos/`

Finally, we want to use the NLI model to validate our proofs  
`python DataCreation/mnli_filtering/mnli_filtering.py -r path/to/questions/ -t threshold_for_data_filtering(wikidata_answers to found answers)`


### Intersection questions

To create the intersection we need to run the command  
`python DataCreation/complex_questions/WikiData/intersection.py -r path/to/simple_questions/ -o path/for/backward_intersection.jsonl
-c path/for/concatenated/intersec/answers.jsonl -n path/for/final/intersections.jsonl`


### Composition questions

We first create all the potential questions.   
`python DataCreation/complex_questions/WikiData/multi_process_comp.py -r path/for/relevant/infos/ -i path/to/questions 
-o path/for/output/ -w path/to/wikidata/mapping --list_indices_path path/to/wikipedia/indices/ -f first_chunk -l last_chunk`

We want to use the NLI model to validate our proofs  
`python DataCreation/mnli_filtering/mnli_filtering.py -r path/to/questions/ -t threshold_for_data_filtering(wikidata_answers to found answers)
-d composition`

## Wikipedia

We use code from HotPotQA to find the tables within the wikipedia pages and parse them. We then use:  
``python DataCreation/DataAlignment/WikipediaTables/align_tables.py -i path/to/tables/ --indices_path path/to/wikipedia/indces
-o path/where/results/are/saved/ -n number_of_processes``   

We now want to run:  
``python DataCreation/DataAlignment/WikipediaTables/align_tables.py -i path/to/aligned/tables/ -o path/where/questions/are/saved/``