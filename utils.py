import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


def preprocess(df):
    (df["class"] == "g").astype(int)
    return df

def split_dataset(df):
    train, validate, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
    return train, validate, test

def scale_data(dataset, over_sample=False):
    X = dataset[dataset.columns[:-1]].values
    y = dataset[dataset.columns[-1]].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    if over_sample:
        ros = RandomOverSampler()
        X,y = ros.fit_resample(X, y)
    data = np.hstack((X, np.reshape(-1, 1)))
    return data, X, y
