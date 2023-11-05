import glob
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import convert_to_int_else_object, detect_separator, remove_accents


def load_data(path, drop_cols=True):
    sep = detect_separator(path)
    df = pd.read_csv(path, index_col=None, header=0, sep=sep)
    df.columns = df.columns.str.lower().str.replace("_", "")
    if drop_cols:
        df = drop_columns(df)
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].str.lower()
            if pd.api.types.is_string_dtype(df[column]):
                df[column] = df[column].apply(remove_accents)
    df = df.applymap(convert_to_int_else_object)

    return df


def load_data_all_files(path):
    all_files = glob.glob(os.path.join(path, "*.csv"))
    all_files = [f for f in all_files]
    df_all_files = pd.concat([load_data(f) for f in all_files], ignore_index=True)
    df_all_files = handle_missing_values(df_all_files)
    return encode_categorical(df_all_files)


def drop_columns(df):
    # List of columns to be dropped due to salary information, household
    # income information or employment information
    cols_to_drop = [
        "mes",
        "secc",
        "nper",
        "subempleo",
        "pt2",
        "pt4",
        "ysvl",
        "ytotsinagui",
        "wsem",
        "e582",
        "egps8",
        "egps5",
        "e247",
        "d2114",
        "ysvlsag",
        "e45311cv",
        "e4521cv",
        "pobpcoac",
        "e5602",
        "ysvlcons",
        "d9",
        "e2141",
        "d212",
        "d2141",
        "d16",
    ]

    # Drop columns that start with specific letters
    cols_to_drop += [
        col for col in df.columns if col.startswith(("f", "g", "h", "H", "W", "i"))
    ]
    # Drop columns containing 'mto'
    cols_to_drop += [col for col in df.columns if "mto" in col]
    # Drop columns with 'PT' except for 'PT1'
    cols_to_drop += [col for col in df.columns if "PT" in col and col != "PT1"]
    # Drop columns with 'YT'
    cols_to_drop += [col for col in df.columns if "YT" in col]
    # Drop columns with '06'
    cols_to_drop += [col for col in df.columns if "06" in col]
    # Drop the columns from the dataframe
    df = df.drop(columns=cols_to_drop)
    return df


def encode_categorical(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def handle_missing_values(df):
    # Define a function to fill NaN with the mode, or a specified value if no mode
    def fill_mode(col):
        # Check if mode result is non-empty and the column is categorical
        if not col.mode().empty and col.dtype == "object":
            return col.fillna(col.mode().iloc[0])
        else:
            return col.fillna("Missing")  # Filling NaNs with 'Missing' for categorical

    # Apply the function to each column
    df = df.apply(fill_mode, axis=0)
    return df
