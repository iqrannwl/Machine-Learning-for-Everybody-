import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import preprocess, split_dataset, scale_data


def read_dataset():
    cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDost","class"]
    df = pd.read_csv("dataset/magic04.data", names=cols)
    return df

if __name__ == "__main__":
    #readin the dataset
    df = read_dataset()
    df = preprocess(df)
    train, validate, test = split_dataset(df)
    train, Xtrain, Ytrain = scale_data(train)