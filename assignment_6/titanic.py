import pandas as pd
import numpy as np
from sklearn.svm import SVC


def normalize_data(data):
    """Normalize the data for certain columns between min and max.
    This is needed for SVM classification"""
    target_col = [
        'Age',
        'Fare',
        'SibSp',
        'Parch'
    ]
    cols = data[target_col]
    # print(cols.head())
    mincols = cols.min()
    maxcols = cols.max()
    normalized = (cols - mincols) / (maxcols - mincols)
    # print(mincols)
    # print(maxcols)
    # print(normalized.head())
    data[target_col] = normalized
    return data

def convert_to_onehot(data):
    """Convert categorical data to one hot"""
    df_sex = pd.get_dummies(data['Sex'])
    df = pd.concat((data, df_sex), axis=1)
    df_pclass = pd.get_dummies(data['Pclass'], prefix='pclass')
    df = pd.concat((df, df_pclass), axis=1)
    df_embark = pd.get_dummies(data['Embarked'], prefix='embark')
    df = pd.concat((df, df_embark), axis=1)
    return df


def fill_missing(df):
    """Fill missing values to specific columns"""
    col = "Age"
    colval = df[col]
    df[col] = colval.fillna(colval.mean())
    col = "Embarked"
    colval = df[col]
    df[col] = colval.fillna(colval.mode())
    return df


def preprocess_data(data):
    """Common preprocessing for both
    training and testing data
    """
    df = fill_missing(data)
    df = convert_to_onehot(df)
    df = normalize_data(df)
    drop_col = [
        'PassengerId',
        'Name',
        'Ticket',
        'Cabin',
        'Sex',
        'Pclass',
        'Embarked',
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
    # clf = RandomForestClassifier()
    clf = SVC()
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

def create_output_data(index, value):
    return pd.concat((index, value), axis=1)


if __name__ == "__main__":
    train_data = pd.read_csv("train.csv")
    pp_data, label = preprocess_train_data(train_data)
    model = train(pp_data, label)
    # print(model)
    test_data = pd.read_csv("test.csv")
    ppt_data = preprocess_test_data(test_data)
    # print(ppt_data)
    result = test(model, ppt_data)
    print(result)
    ratio = calc_ratio(model, pp_data, label)
    print(ratio)
    val = pd.DataFrame({"Survived": result})
    output = create_output_data(test_data["PassengerId"], val)
    # print(output.head())
    output.to_csv("predict.csv", index=False)
