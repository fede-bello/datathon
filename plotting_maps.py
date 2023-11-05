"""
This is a script to plot maps of Uruguay using the data from the Encuesta Continua de Hogares (ECH)
from the Instituto Nacional de Estadística (INE) of Uruguay.
It is assumed that the data is in the folder 'data' and that the shapefile is in the folder 'data'. 

It is not automated, so you have to change the code manually to plot the map you want.
"""

# %%
import glob
import os
import unicodedata

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# %%
def detect_separator(filename):
    with open(filename, "r") as file:
        first_line = file.readline()
        if first_line.count(";") > first_line.count(","):
            return ";"
        else:
            return ","


# %%
def load_data_all_files(path) -> pd.DataFrame:
    all_files = glob.glob(os.path.join(path, "*.csv"))
    # all_files = all_files.sort()

    li = []

    for filename in all_files:
        if filename != "todo.csv":
            sep = detect_separator(filename)
            df = pd.read_csv(filename, index_col=None, header=0, sep=sep)
            df.columns = df.columns.str.lower()
            df = drop_columns(df)
            df.columns = df.columns.str.replace("_", "")
            for column in df.columns:
                if df[column].dtype == "object":
                    df[column] = df[column].str.lower()
                    if pd.api.types.is_string_dtype(df[column]):
                        df[column] = df[column].apply(remove_accents)

            li.append(df)

    return pd.concat(li, axis=0, ignore_index=True, join="outer")


# %%
def load_data(path) -> pd.DataFrame:
    sep = detect_separator(path)
    df = pd.read_csv(path, index_col=None, header=0, sep=sep)

    return df


# %%
def drop_columns(df):
    # List of columns to be dropped
    cols_to_drop = [
        "mes",
        "secc",
        "nper",
        "subempleo",
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


# %%
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


def convert_to_int_else_object(value):
    try:
        # Attempt to convert to an integer
        return int(value)
    except (ValueError, TypeError):
        # If conversion to an integer fails, return the value as a string
        return value


# Define a function to encode categorical variables
def encode_categorical(df):
    # Find categorical variables (assuming they are of type 'object')
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])

    return df


# Function to remove accents from strings
def remove_accents(input_str):
    if isinstance(input_str, str):
        nfkd_form = unicodedata.normalize("NFKD", input_str)
        only_ascii = nfkd_form.encode("ASCII", "ignore")
        return only_ascii.decode("ASCII")
    else:
        return input_str


# %%

df = load_data_all_files("data/sexo")

# %%
# Handle missing values
df = handle_missing_values(df)
# %%
# Handle missing values
# %%
# Selected columns and label
y = df["pt1"]
df = df.drop(
    columns=[
        "pt2",
        "pt4",
        "ysvl",
        "ytotsinagui",
        "wsem",
        "e582",
        "egps8",
        "egps5",
        "e5611",
        "e247",
        "d2114",
        "ysvlsag",
        "e45311cv",
        "e4521cv",
        "pobre",
        "pobpcoac",
        "e5602",
        "ysvlcons",
        "d9",
        "e2141",
        "lecheenpol",
        "d212",
        "d2141",
        "d16",
    ]
)


# %%


# Set the SHAPE_RESTORE_SHX option to YES
with fiona.Env(SHAPE_RESTORE_SHX="YES"):
    uruguay_map = gpd.read_file("data/departamentos.geojson", encoding="utf-8")

# %%
uruguay_map
# %%
# set column values to lowercase
uruguay_map.columns = map(str.lower, uruguay_map.columns)
# rename column admlnm to nomdpto
uruguay_map.rename(columns={"admlnm": "nomdpto"}, inplace=True)
# %%
for column in uruguay_map.columns:
    if uruguay_map[column].dtype == "object":
        uruguay_map[column] = uruguay_map[column].str.lower()
uruguay_map
# %%
data = df.groupby("nomdpto")["pt1"].mean().reset_index()
data
# %%
merged_data = uruguay_map.merge(data, on="nomdpto", how="left")
merged_data
# dicide the column pt1 by 39,71
merged_data["pt1"] = merged_data["pt1"] / 39.71
# %%


# For a simple choropleth map
merged_data.plot(
    column="pt1", scheme="naturalbreaks", figsize=(10, 10), legend=True, linewidth=4
)
# make the grid smaller
plt.grid(True, which="both", axis="both", linestyle="--")
plt.title("Average Salary by Department in Uruguay (U$S)")
plt.axis("off")  # This will remove the axis
plt.show()
# %%

# %%
df["e49"] = df["e49"].replace({1: 1, 2: 0, 3: 1})
percentage_df = df.groupby("nomdpto")["e49"].mean().reset_index()
percentage_df["e49"] = percentage_df["e49"] * 100  # Convert to percentage
# %%
merged_data = uruguay_map.merge(percentage_df, on="nomdpto", how="left")
merged_data
# %%


# For a simple choropleth map
merged_data.plot(
    column="e49", scheme="naturalbreaks", figsize=(10, 10), legend=True, linewidth=4
)
# make the grid smaller
plt.grid(True, which="both", axis="both", linestyle="--")
plt.title("Percentage of people that assited to school by Department in Uruguay")
plt.axis("off")  # This will remove the axis
plt.show()

# %%
"Percentage of people that assited to school by Department in Uruguay".lower().replace(
    " ", "_"
)
# %%
df["e511"] = df["e511"].replace("Missing", np.nan)

# Drop rows where 'e511' is NaN
df_ = df.dropna(subset=["e511"])
data = df_.groupby("nomdpto")["e511"].mean().reset_index()
data
# %%
merged_data = uruguay_map.merge(data, on="nomdpto", how="left")
merged_data
# %%

# For a simple choropleth map
merged_data.plot(
    column="e511", scheme="naturalbreaks", figsize=(10, 10), legend=True, linewidth=4
)
# make the grid smaller
plt.grid(True, which="both", axis="both", linestyle="--")
plt.title(
    "Average years of early childhood or preschool education by Department in Uruguay"
)
plt.axis("off")  # This will remove the axis
plt.show()

# %%
"Average years of early childhood or preschool education by Department in Uruguay".lower().replace(
    " ", "_"
)
# %%
df["e574"].unique()
# %%
# e574
df["e48"] = df["e48"].replace({1: 1, 2: 0})

# Calculate the percentage of people who can read and write per department
percentage_df = df.groupby("nomdpto")["e48"].mean().reset_index()
percentage_df["e48"] = percentage_df["e48"] * 100  # Convert to percentage

# Load your geographic data for the departments
# Replace 'path_to_geo_data' with the path to your shapefile or GeoJSON

# Merge the percentage data with the geographic data
merged_data = uruguay_map.merge(percentage_df, on="nomdpto", how="left")
# %%

# For a simple choropleth map
merged_data.plot(
    column="e48", scheme="naturalbreaks", figsize=(10, 10), legend=True, linewidth=4
)
# make the grid smaller
plt.grid(True, which="both", axis="both", linestyle="--")
plt.title("Percentage of people who can read and write by Department in Uruguay")
plt.axis("off")  # This will remove the axis

plt.show()

# %%
"Percentage of people who can read and write by Department in Uruguay".lower().replace(
    " ", "_"
)
# %%
df["e2181"] = df["e2181"].replace({1: 1, 2: 0})

# Calculate the percentage of people who can read and write per department
percentage_df = df.groupby("nomdpto")["e2181"].mean().reset_index()
percentage_df["e2181"] = percentage_df["e2181"] * 100  # Convert to percentage

# Load your geographic data for the departments
# Replace 'path_to_geo_data' with the path to your shapefile or GeoJSON

# Merge the percentage data with the geographic data
merged_data = uruguay_map.merge(percentage_df, on="nomdpto", how="left")
# %%

# For a simple choropleth map
merged_data.plot(
    column="e2181", scheme="naturalbreaks", figsize=(10, 10), legend=True, linewidth=4
)
# make the grid smaller
plt.grid(True, which="both", axis="both", linestyle="--")
plt.title(
    "Percentage of people who completed tertiary university studies or similar by Department in Uruguay"
)
plt.axis("off")  # This will remove the axis

plt.show()
# %%
"Percentage of people who completed tertiary university studies or similar by Department in Uruguay".lower().replace(
    " ", "_"
)
# %%
