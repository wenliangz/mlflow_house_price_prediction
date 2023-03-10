import os
import pickle
import shutil

import mlflow
import sys
import os
import pathlib
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.pyfunc import PythonModel


class WrappedLRModel(PythonModel):

    def __init__(self, sklearn_model_features, cat_features, model_artifact_name):
        """
        cat_features: Mapping from categorical feature names to all
        possible values, e.g.:
        {
            "Bldg Type": ["1Fam", "TwnhsE", ... ]
        }
        """
        self.feature_names = sklearn_model_features
        self.cat_features = cat_features

    def load_context(self, context):

        with open(context.artifacts["original_sklearn_model"], "rb") as m:
            self.lr_model = pickle.load(m)

    def _encode(self, row, colname):
        # 'colname' will be 'Bldg Type'
        value = row[colname]  # Value will be e.g. '1Fam'
        row[value] = 1
        return row

    def predict(self, context, model_input):
        # Expected model_input features: ["Lot Area", "Gr Liv Area", "Garage Area", "Bldg Type"]
        # Features required by the model: ["Lot Area", "Gr Liv Area", "Garage Area", "1Fam", "TwnhsE", ... ]
        model_features = model_input
        for col, unique_values in self.cat_features.items():
            for uv in unique_values:
                model_features[uv] = 0
            model_features = model_features.apply(lambda row: self._encode(row, col), axis=1)
        model_features = model_features.loc[:, self.feature_names]
        return self.lr_model.predict(model_features.to_numpy())

def prepare_data(dataframe, cat_features):
    df = dataframe
    cat_features_values = {}
    # Encode all the categorical features
    for col in list(dataframe.columns):
        if col in cat_features:
            cat_features_values[col] = list(dataframe[col].unique())
            # One-hot encoding
            dummies = pd.get_dummies(df[col])
            # Drop the original column
            df = pd.concat([df.drop([col], axis=1), dummies], axis=1)

    # Fill missing values with 0
    df = df.fillna(0)

    return df, cat_features_values


def train_and_evaluate(dataframe, cat_features_values):
    # Separate features from the target variable and convert to NumPy
    features = dataframe.drop(["SalePrice"], axis=1)
    target = dataframe.loc[:, "SalePrice"]
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features.to_numpy(), target.to_numpy(), test_size=0.2,
                                                        random_state=12)

    # Plot
    plot = dataframe.plot.scatter(x=0, y="SalePrice")
    fig = plot.get_figure()
    fig.savefig("tmp/plot.png")

    # Save the dataset
    dataframe.to_csv("tmp/dataset.csv", index=False)

    # Train the model
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    # Save the model
    serialized_model = pickle.dumps(model)
    with open("tmp/model.pkl", "wb") as f:
        f.write(serialized_model)

    model_artifact_name = "original_sklearn_model"
    model_artifacts = {
        model_artifact_name: "tmp/model.pkl"
    }

    mlflow.pyfunc.log_model(
        "custom_model",
        python_model=WrappedLRModel(sklearn_model_features=list(features.columns),
                                    cat_features=cat_features_values,
                                    model_artifact_name=model_artifact_name
                                    ),
        artifacts=model_artifacts,
        registered_model_name="House price predictions"
    )

    # Evaluate the model
    y_pred = model.predict(X_test)
    err = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("MSE", err)

    # Log the artifacts
    mlflow.log_artifacts("tmp")


if __name__ == '__main__':
    # Read and create datasets
    # data_path_args = sys.argv[1]
    # run_date = sys.argv[2]
    # MLFLOW_SERVER = sys.argv[3]
    # data_dir = pathlib.Path(f"{data_path_args}")

    # mlflow.set_tracking_uri(f"http://{MLFLOW_SERVER}")
    mlflow.set_tracking_uri(f"http://192.168.1.248:5000")
    # if mlflow.get_experiment_by_name(f"house_price_prediction") == None:
    #     mlflow.create_experiment(f"run_{run_date}")
    # mlflow.delete_experiment('house_price_prediction')
    mlflow.set_experiment("house_prediction")

    # mlflow.set_tracking_uri("http://localhost:5000")
    # mlflow.set_experiment("Models exercise")

    if not os.path.exists('tmp'):
        os.makedirs("tmp")

        # Load the complete dataset, select features + target variable
    data = pd.read_csv("housedata.csv")
    feature_columns = ["Lot Area", "Gr Liv Area", "Garage Area", "Bldg Type"]
    selected = data.loc[:, feature_columns + ["SalePrice"]]

    # Features that need encoding (categorical ones)
    cat_features = ["Bldg Type"]

    with mlflow.start_run():
        prepared, cat_features_values = prepare_data(selected, cat_features)
        train_and_evaluate(prepared, cat_features_values)

    shutil.rmtree("tmp")
