import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def preprocess_data(data):
    """Common preprocessing for both
    training and testing data
    """
    df_sex = pd.get_dummies(data['Sex'])
    df = pd.concat((data, df_sex), axis=1)
    drop_col = [
        'Name',
        'Ticket',
        'Cabin',
        'Embarked',
        'Sex',
        'Age'
    ]
    leftover = df.drop(columns=drop_col)
    return leftover


def preprocess_train_data(data):
    # print(data.describe())
    print(data.shape)
    print(data.head())
    print(data.isnull().sum())
    leftover = preprocess_data(data)
    cols = list(leftover.columns)
    cols.remove('Survived')
    print(leftover[cols].head())
    x = np.array(leftover[cols], dtype=np.float32)
    y = np.array(leftover['Survived'], dtype=np.int32)
    print(x)
    print(len(y))
    return (x, y)


def preprocess_test_data(data):
    leftover = preprocess_data(data)
    # Fill the empty fare with a mean value
    col = "Fare"
    colval = leftover[col]
    leftover[col] = colval.fillna(colval.mean())
    # print(leftover.isnull().sum())
    cols = list(leftover.columns)
    # print(leftover[cols].head())
    x = np.array(leftover[cols], dtype=np.float32)
    return x


def train(data, label):
    clf = RandomForestClassifier()
    clf.fit(data, label)
    return clf


def test(model, data):
    return model.predict(data)


def calc_ratio(model, data, label):
    result = model.predict(data)
    assert len(label) == len(result)
    # ratio = 0.0
    sumval = sum(exp == act for exp, act in zip(label, result))
    return sumval / len(label)
    # for exp, act in zip(label, result):
    #     if exp == act
    # return ratio


if __name__ == "__main__":
    train_data = pd.read_csv("train.csv")
    pp_data, label = preprocess_train_data(train_data)
    model = train(pp_data, label)
    print(model)
    test_data = pd.read_csv("test.csv")
    ppt_data = preprocess_test_data(test_data)
    print(ppt_data)
    result = test(model, ppt_data)
    print(result)
    ratio = calc_ratio(model, pp_data, label)
    print(ratio)
