# Final-Proj
Document expansion -- final project for Web Search Engines

## NYU HPC
To SSH into Greene:
```
ssh [accout-id]@greene.hpc.nyu.edu
```

You will want to save all your files in the `/scratch/[account-id]` folder

To copy files from local environment to Greene (make sure you are in your local environment):
```
scp [-options] /path/to/file [account-id]@greene.hpc.nyu.edu:/scratch/[account-id]
```

To copy files from Greene to your local environment (make sure you are in your local environment):
```
scp [-options] [account-id]@greene.hpc.nyu.edu:/scratch/[account-id]/path/to/file /path/in/local/
```

Set up your Anaconda environment in Greene according to https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/conda-environments-python-r?authuser=0

Load CUDA into your Greene environment.

## Doc2Query
We can find the source code for Doc2Query and Doc2Query-- here: https://github.com/terrierteam/pyterrier_doc2query/tree/master

To import MSMARCO documensa dataset for use wiht pyterrier: https://ir-datasets.com/msmarco-document.html
