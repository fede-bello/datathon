import pickle

import pandas as pd
import typer
from sklearn.model_selection import train_test_split

from data_preprocessing import load_data_all_files
from model_training import train_tabnet, train_trees
from utils import evaluate_model


def main(
    model: str = typer.Argument(
        ..., help="The model type to train ('tabnet' or 'trees')."
    ),
    data_path: str = typer.Option(
        "data", "--data-path", "-d", help="The path to the data directory."
    ),
    output_dir: str = typer.Option(
        "model_weights",
        "--output-dir",
        "-o",
        help="The output directory for the saved models.",
    ),
):
    """
    Load data, train the model, and evaluate it.
    """
    # Load and preprocess data
    df = load_data_all_files(data_path)
    y = df["pt1"]
    X = df.drop(columns=["pt1"])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # create a model name using the model and a timestamp
    model_name = f"{model}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize and train the selected model
    if model == "tabnet":
        regressor = train_tabnet(X_train, y_train, X_test, y_test)
        model_file_path = f"{output_dir}/{model_name}"
        regressor.save_model(model_file_path)
        y_test = y_test.values.reshape(-1, 1)
        X_test = X_test.values
    elif model == "trees":
        regressor = train_trees(X_train, y_train)
        model_file_path = f"{output_dir}/{model_name}.sav"
        with open(model_file_path, "wb") as f:
            pickle.dump(regressor, f)
    else:
        typer.echo(f"Model type '{model}' is not supported.")
        raise typer.Exit(code=1)

    # Evaluate the model
    MAE = evaluate_model(regressor, X_test, y_test)
    typer.echo(f"The Mean Absolute Error in test is: {MAE}")


if __name__ == "__main__":
    typer.run(main)
