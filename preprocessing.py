import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path)
    # change target to 0-1
    data['target'] = data['increase_stock'].apply(lambda x: 1 if x == "high_bike_demand" else 0)
    data['isRaining'] = data['precip'].apply(lambda x: 1 if x > 0 else 0)

    data = data.drop(columns=['increase_stock', 'snow', 'holiday', 'precip'])
    
    return data

def split_data(data, test_size=0.2, random_state=23):
    X = data.drop(columns=['target'])
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    return X_train, X_test, y_train, y_test


def load_and_split(file_path, test_size=0.2, random_state=23):
    data = load_data(file_path)
    return split_data(data, test_size, random_state)

if __name__ == "__main__":
    file_path = "data/training_data_ht2025.csv"
    X_train, X_test, y_train, y_test = load_and_split(file_path)
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)


