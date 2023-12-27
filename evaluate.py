import argparse

import pyterrier as pt
pt.init()

import pyt_splade
from pyterrier_doc2query import Doc2Query, QueryScorer, QueryFilter
from pyterrier_dr import ElectraScorer
from pyterrier.measures import *


def preprocess_documents(documents):
  for doc in documents:
    doc['text'] = (doc['title'] + " " + doc['body']).strip()
    yield doc


def eval(method, dataset_name):
  irds_name = "irds:" + dataset_name
  dataset = pt.get_dataset(irds_name)
  index_loc = "./" + method + "_" + dataset_name
  
  if method == "d2q":
    indexer = Doc2Query(append=True, batch_size=200, num_samples=10, fast_tokenizer=True) >> pt.IterDictIndexer(index_loc, overwrite=True)
    
    if "document" in dataset_name:
      preprocessed_documents = preprocess_documents(dataset.get_corpus_iter())
      indexref = indexer.index(preprocessed_documents, batch_size=512)
    else:
      indexref = indexer.index(dataset.get_corpus_iter(), batch_size=512)
      
    br = pt.BatchRetrieve(indexref, wmodel="BM25")
    retr_pipe = br
    
  elif method == "d2q--":
    doc2query = Doc2Query(append=False, batch_size=200, num_samples=20, fast_tokenizer=True)
    scorer = ElectraScorer(batch_size=200)
    indexer = pt.IterDictIndexer(index_loc, overwrite=True)
    pipeline = doc2query >> QueryScorer(scorer) >> QueryFilter(t=3.21484375) >> indexer # t=3.21484375 is the 70th percentile for generated queries on MS MARCO

    if "document" in dataset_name:
      preprocessed_documents = preprocess_documents(dataset.get_corpus_iter())
      indexref = pipeline.index(preprocessed_documents, batch_size=512)
    else:
      indexref = pipeline.index(dataset.get_corpus_iter(), batch_size=512)
    
    br = pt.BatchRetrieve(indexref, wmodel="BM25")
    retr_pipe = br
    
  elif method == "splade":
    factory = pyt_splade.SpladeFactory()
    doc_encoder = factory.indexing()
    indexer = pt.IterDictIndexer(index_loc, overwrite=True)
    indexer.setProperty("termpipelines", "")
    indexer.setProperty("tokeniser", "WhitespaceTokeniser")

    indexer_pipe = (doc_encoder >> pyt_splade.toks2doc() >> indexer)
    
    if "document" in dataset_name:
      preprocessed_documents = preprocess_documents(dataset.get_corpus_iter())
      indexref = indexer_pipe.index(preprocessed_documents, batch_size=640)
    else:
      indexref = indexer_pipe.index(dataset.get_corpus_iter(), batch_size=640)
    
    br = pt.BatchRetrieve(indexref, wmodel='BM25')
    query_splade = factory.query()
    retr_pipe = query_splade >> br
    
  results = pt.Experiment(
      [retr_pipe],
      pt.get_dataset(irds_name + "/trec-dl-2020").get_topics(),
      pt.get_dataset(irds_name + "/trec-dl-2020").get_qrels(),
      batch_size=400,
      filter_by_qrels=True,
      eval_metrics=[MAP, MRR, P@10],
      names=[method]
  )    
  
  print(results)
  
  res_file = f"./res_{method}_{dataset_name}.csv"
  results.to_csv(res_file)
  
  return results

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument(
    "--method",
    default=None,
    type=str,
    required=True,
    help="Specify d2q, d2q--, or splade"
  )
  
  parser.add_argument(
    "--dataset",
    default=None,
    type=str,
    required=True,
    help="Specify msmarco-document or msmarco-passage"
  )
  
  args = parser.parse_args()
  eval(args.method, args.dataset)