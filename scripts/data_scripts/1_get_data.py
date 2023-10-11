#!/usr/bin/python3

import pandas as pd

# скачиваем данные и сохраняем в папке raw 

train = pd.read_csv('https://drive.google.com/uc?id=1gkDlwHivsO7mUBLpfOq5-SWwfaq7tnLV')

train.to_csv("data/raw/train.csv", index=False)
