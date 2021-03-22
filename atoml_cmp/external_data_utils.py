"""For the use of external data sets."""
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


def overwrite_dataset(external_dataset, placeholder_dataset,
                      external_path="external_data", test_folder="generated-tests"):
    framework_smoketest_paths = ["caret/smokedata", "sklearn/smokedata",
                                 "spark/src/test/resources/smokedata", "weka/src/test/resources/smokedata"]
    if Path(os.path.join(external_path, external_dataset)).is_file():
        for fw in framework_smoketest_paths:
            if Path(os.path.join(test_folder, fw, placeholder_dataset)).is_file():
                shutil.copy(os.path.join(external_path, external_dataset),
                            os.path.join(test_folder, fw, placeholder_dataset))
            else:
                print("placeholder path doesn't exist: %s" % os.path.join(test_folder, fw, placeholder_dataset))

    else:
        print("External dataset doesn't exist: %s" % os.path.join(external_path, external_dataset))


def rename_prediction_file(new, old):
    pred_path = "predictions"
    frameworks = ["sklearn", "caret", "spark", "weka"]
    for fw in frameworks:
        if Path(os.path.join(pred_path, fw)).is_dir():
            for f in os.listdir(os.path.join(pred_path, fw)):
                if old in f:
                    if Path(os.path.join(pred_path, fw, f)).is_file():
                        os.rename(os.path.join(pred_path, fw, f), os.path.join(pred_path, fw, f.replace(old, new)))


def transform_dataset(data, target, same=False, norm=None):
    if norm is not None:
        if norm == "minmax":
            data_min = data.min(axis=0)
            data_max = data.max(axis=0)
            data = (data - data_min) / (data_max-data_min)
        if norm == "z-score":
            data_mean = data.mean(axis=0)
            data_std = data.std(axis=0)
            data = (data - data_mean) / data_std
    if same:
        X_train, X_test = data, data
        y_train, y_test = target, target
    else:
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.5, random_state=42)
    return X_train, X_test, y_train, y_test


def create_arff(feature_names, feature_list, target_list, filename=None, path=None, relation="none"):
    arff_string = ""
    arff_string += "@RELATION " + relation.replace(" ", "_") + "\n\n"
    for feature in feature_names:
        feature = feature.replace(" ", "_")
        arff_string += "@ATTRIBUTE " + feature + " \tNUMERIC\n"
    arff_string += "@ATTRIBUTE classAtt {class_0,class_1}\n\n"
    arff_string += "@DATA\n"

    for x, y in zip(feature_list, target_list):
        for xi in x:
            arff_string += str(xi) + ","
        arff_string += str(y) + "\n"
    if path is not None and filename is not None:
        with open(os.path.join(path, filename), 'w') as file:
            file.write(arff_string)
    return arff_string


def create_wine_arff(path="external_data", use_uci_repo=True):
    if use_uci_repo:
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
                         sep=';')
    else:
        df = pd.read_csv(os.path.join(path, "raw_datasets", "wine.csv"))

    # form to binary classification (quality higher than 5)
    df['target'] = df['quality'].apply(lambda x: 'class_1' if x > 5 else 'class_0')

    wine_target = df['target'].to_numpy()
    wine_data = df.drop(['quality', 'target'], axis=1).to_numpy()
    wine_features = df.drop(['quality', 'target'], axis=1).columns

    X_train, X_test, y_train, y_test = transform_dataset(wine_data, wine_target)
    _ = create_arff(wine_features, X_train, y_train, filename="Wine_training.arff", path=path, relation="wine")
    _ = create_arff(wine_features, X_test, y_test, filename="Wine_test.arff", path=path, relation="wine")

    X_train, X_test, y_train, y_test = transform_dataset(wine_data, wine_target, norm="minmax")
    _ = create_arff(wine_features, X_train, y_train, filename="WineMinMaxNorm_training.arff", path=path,
                    relation="wine")
    _ = create_arff(wine_features, X_test, y_test, filename="WineMinMaxNorm_test.arff", path=path, relation="wine")

    X_train, X_test, y_train, y_test = transform_dataset(wine_data, wine_target, norm="z-score")
    _ = create_arff(wine_features, X_train, y_train, filename="WineZNorm_training.arff", path=path,
                    relation="wine")
    _ = create_arff(wine_features, X_test, y_test, filename="WineZNorm_test.arff", path=path, relation="wine")


def create_breast_cancer_arff(path="external_data", use_sklearn_repo=True):
    if use_sklearn_repo:
        dataset = load_breast_cancer()
        df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        df["target"] = np.where(dataset.target == 0, "class_0", "class_1")
    else:
        df = pd.read_csv(os.path.join(path, "raw_datasets", "breast_cancer.csv"))
        df["target"] = np.where(df.target == 0, "class_0", "class_1")

    bc_target = df['target'].to_numpy()
    bc_data = df.drop(['target'], axis=1).to_numpy()
    bc_features = df.drop(['target'], axis=1).columns

    X_train, X_test, y_train, y_test = transform_dataset(bc_data, bc_target)
    _ = create_arff(bc_features, X_train, y_train, filename="BreastCancer_training.arff", path=path,
                    relation="breast cancer")
    _ = create_arff(bc_features, X_test, y_test, filename="BreastCancer_test.arff", path=path,
                    relation="breast cancer")

    X_train, X_test, y_train, y_test = transform_dataset(bc_data, bc_target, norm="minmax")
    _ = create_arff(bc_features, X_train, y_train, filename="BreastCancerMinMaxNorm_training.arff", path=path,
                    relation="breast cancer")
    _ = create_arff(bc_features, X_test, y_test, filename="BreastCancerMinMaxNorm_test.arff", path=path,
                    relation="breast cancer")

    X_train, X_test, y_train, y_test = transform_dataset(bc_data, bc_target, norm="z-score")
    _ = create_arff(bc_features, X_train, y_train, filename="BreastCancerZNorm_training.arff", path=path,
                    relation="breast cancer")
    _ = create_arff(bc_features, X_test, y_test, filename="BreastCancerZNorm_test.arff", path=path,
                    relation="breast cancer")


if __name__ == "__main__":
    create_wine_arff()
    create_breast_cancer_arff()
