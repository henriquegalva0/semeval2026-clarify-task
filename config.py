import pandas as pd

TRAININGDATASET = pd.read_parquet("hf://datasets/ailsntua/QEvasion/" + 'data/train-00000-of-00001.parquet')
TESTINGDATASET = pd.read_parquet("hf://datasets/ailsntua/QEvasion/" + 'data/test-00000-of-00001.parquet')

MODELNAME=str("albert/albert-base-v2")
EXPERIMENTNAME=str("direct")