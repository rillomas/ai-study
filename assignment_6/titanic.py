import pandas as pd


def preprocess_data(data):
    # print(data.describe())
    print(data.shape)
    print(data.head())
    print(data.isnull().sum())
    drop_col = [
        'Name',
        'Ticket',
        'Cabin',
        'Embarked'
    ]
    output = data.drop(columns=drop_col)
    print(output.isnull().sum())
    print(output.head())
    return output


def train(data):
    model = "aardvark"
    return model


def test(model, data):
    print(model, data.head())
    result = "giraffe"
    return result


if __name__ == "__main__":
    train_data = pd.read_csv("train.csv")
    # test_data = pd.read_csv("test.csv")
    processed_train_data = preprocess_data(train_data)
    # processed_test_data = preprocess_data(test_data)
    # model = train(processed_train_data)
    # result = test(model, processed_test_data)
    # print(result)
