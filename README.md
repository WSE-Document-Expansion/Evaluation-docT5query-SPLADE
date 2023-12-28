# Evaluating Document Expansion Techniques: docT5query, doc2query--, and SPLADE++
This repository contains source code for the final project for NYU's CS-GY 6931 course. 

Evaluation metrics obtained are MAP, MRR, and P@10. Prior to evaluation, install the necessary Python packages in `requirements.txt`. There is the option to use docT5query, doc2query--, or SPLADE++ as the document expansion technique, and MSMARCO documents dataset or passages dataset. For example, in order to evaluate docT5query on MSMARCO documents dataset, use the following options:
```
python3 evaluate.py --method d2q --dataset msmarco-document
```
