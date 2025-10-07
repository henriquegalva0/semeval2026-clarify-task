import pandas as pd

# dataset CFGs

TRAININGDATASET=pd.read_parquet("hf://datasets/ailsntua/QEvasion/" + 'data/train-00000-of-00001.parquet')
TESTINGDATASET=pd.read_parquet("hf://datasets/ailsntua/QEvasion/" + 'data/test-00000-of-00001.parquet')

MULTIPLEQUESTIONS=False # True = enable multiple questions in the dataset

if MULTIPLEQUESTIONS==False:
    TRAININGDATASET=TRAININGDATASET[TRAININGDATASET["multiple_questions"] == "false"]

# transformers CFGs

NUMEPOCHS=5 # number of epochs
MAXSIZE=512 # transformers training data size
MAXLENGTH=512 # dataset data length
MODELNAME=str("albert/albert-base-v2")
EXPERIMENTNAME=str("evasion_based_clarity")
#EXPERIMENTNAME=str("direct_clarity")

# paths CFGs

OUTFILE=open(f"{MODELNAME.split('/')[-1]}-qaevasion-{EXPERIMENTNAME}")
OUTCSV=open(f"./results/{MODELNAME.split('/')[-1]}-{EXPERIMENTNAME}.csv")