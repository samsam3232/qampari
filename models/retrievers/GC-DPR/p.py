import glob
from dpr.indexer.faiss_indexers import DenseIndexer, DenseHNSWFlatIndexer, DenseFlatIndexer
index = DenseFlatIndexer(768, 20000)
ctx_files_pattern = "/media/disk2/odqa_experiments/+batch_size-61_v0/dpr_biencoder.120.1303_index_*.pkl"
input_paths = glob.glob(ctx_files_pattern)
index.index_data(input_paths)
