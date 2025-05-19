import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.compose import ColumnTransformer


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering function"""
    # Title extraction and grouping
    common_titles = ["Miss", "Mr", "Mrs", "Master"]
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].where(df["Title"].isin(common_titles), "Other")

    # Family features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"].replace(0, 1)
    df["IsChild"] = (df["Age"] <= 12).astype(int)
    df["IsFemale"] = (df["Sex"] == "female").astype(int)

    return df


def get_sets(filepath: str = "train.csv", is_test=False, preprocessor=None):
    """Main function to load and preprocess data"""
    df = pd.read_csv(filepath)

    # Extract target
    target_tensor = None
    if not is_test:
        target_tensor = torch.tensor(df["Survived"].values, dtype=torch.float32)

    # Feature engineering
    df = add_features(df)

    # Drop columns
    df.drop(["PassengerId", "Name", "Ticket", "Sex", "Cabin"], axis=1, inplace=True)
    if not is_test:
        df.drop(["Survived"], axis=1, inplace=True)
    expected_titles = ["Miss", "Mr", "Mrs", "Master", "Other"]
    expected_embarked = ["S", "C", "Q"]
    expected_pclass = [1, 2, 3]
    # Create new preprocessor only if none provided
    if preprocessor is None:
        preprocessor = ColumnTransformer(
            [
                (
                    "categorical",
                    OneHotEncoder(
                        categories=[
                            expected_embarked,
                            expected_pclass,
                            expected_titles,
                        ],
                        drop=None,
                        sparse_output=False,
                        handle_unknown="ignore",
                    ),
                    ["Embarked", "Pclass", "Title"],
                ),
                (
                    "numerical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            (
                                "binner",
                                KBinsDiscretizer(
                                    n_bins=4, encode="ordinal", strategy="quantile"
                                ),
                            ),
                        ]
                    ),
                    ["Age", "Fare", "FarePerPerson"],
                ),
                (
                    "binary",
                    "passthrough",
                    ["IsFemale", "IsChild", "SibSp", "Parch", "FamilySize"],
                ),
            ],
            remainder="drop",
        )
        # Fit only on training data
        features = preprocessor.fit_transform(df)
    else:
        # Reuse provided preprocessor
        features = preprocessor.transform(df)

    feature_tensor = torch.tensor(features, dtype=torch.float32)
    return feature_tensor, target_tensor, preprocessor  # Return the preprocessor!
