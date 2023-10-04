#!/usr/bin/python3

from catboost.datasets import amazon

train, test = amazon()

train.to_csv("../../data/raw/train.csv", index=False)

test.to_csv("../../data/raw/test.csv", index=False)
