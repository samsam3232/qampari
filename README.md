# QAMPARI
Official repository of the paper: "QAMPARI: An Open-domain Question Answering Benchmark for Questions with Many Answers from Multiple Paragraphs" (arxiv: https://arxiv.org/abs/2205.12665 website: https://samsam3232.github.io/qampari/)

QAMPARI is a dataset for Open Domain Question Answer (ODQA) composed of questions with many answers coming from multiple
paragraphes. Unlike most of ODQA tasks, each question in QAMPARI has multiple answers. For example:  
`What car models did Autozam produce? (simple question)`  
`Where are the papers owned at some point in time by Voice Media Group published? (composition question)`  
`Who studied at the Manhattan School of Music and also worked for Julliard School? (intersection question)`

QAMPARI data was created in a semi-automatic manners based on Wikidata's knowledge graph and Wikipedia tables. We provide,
along with 61911 train questions, 1000 dev questions and 1000 test questions. Answers for all dev and test questions were 
manually validated by crowd workers, and all dev-test questions were rephrased by workers as well. For 200 questions from
the test set, an expert annotator added as many answers missing from the gold set as possible under 12 minutes.

We trained SOTA retrieve and read models on QAMPARI (retrievers: BM25 and DPR, readers: FiD and Passage Independent Generator)
and found that their performance is not on par with their performance on other benchmarks. We used F1, recall, precision,
recall>=0.8 and F1>=0.5 as metrics.  


| Model    | Training data  | F1  | Recall | Precision | Recall >= 0.8 | F1>=0.5 |  
|:-----:   | :-------------:  | :---:  | :-----: | :---------: | :-------------: | :-------: |
| **FiD-BM25** | **QAMPARI** | 28.3 | 25.1 | 36.8 | 6.8 | 24.2 |
| **FiD-BM25** | **QAMPARI + NQ** | 29.7 | 26.9 | 37.7 | 7.4 | 25.6 |
| **FiD-DPR** | **QAMPARI**| 5.7 | 3.6 | 20.4 | 0 | 1.2 |
| **FiD-DPR** | **QAMPARI + NQ** | 7.7 | 5.3 | 21.3 | 0.1 | 2.4 |
| **PIG-BM25** | **QAMPARI** | 31.0 | 43.1 | 30.7 | 26.7 | 26 |
| **PIG-BM25** | **QAMPARI + NQ** | 30.5 | 47.9 | 28.2 | 31.2 | 22.3 |
| **PIG-DPR** | **QAMPARI** | 3.0 | 18.9 | 1.8 | 4.2 | 0 |
| **PIG-DPR** | **QAMPARI + NQ** | 3.1 | 18.0 | 1.9 | 4.2 | 0 |

The repository is composed of two parts: the [data creation](/DataCreation/README.md) part and the [model training](/models/README.md) part.
