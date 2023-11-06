from sklearn.model_selection import train_test_split
from data_preprocessing import load_data_all_files
from model_training import train_tabnet, train_trees
from utils import evaluate_model
import pickle

def main():
    # Load and preprocess data
    df = load_data_all_files("data")
    y = df["pt1"]
    X = df.drop(columns=["pt1"])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train TabNetRegressor
    # regressor = train_tabnet(X_train, y_train, X_test, y_test)
    # # Save the trained model
    # regressor.save_model("tabnet")

    # In case of wanting the AdaBooster Model, comment the previous lines and uncomment the next
    # Initialize and train AdaBoostRegressor
    regressor = train_trees(X_train, y_train)
    # Save the trained model
    pickle.dump(regressor, open("model_weights/tree_model.sav", "wb"))

    MAE = evaluate_model(regressor, X_test, y_test)

    print(f'The Mean Absolute Error in test is: {MAE}')


if __name__ == "__main__":
    main()
