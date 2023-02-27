import pandas as pd

df = pd.read_csv("data/ames_housing.csv")

feature_columns = ["Lot Area", "Gr Liv Area", "Garage Area", "Bldg Type"]
selected = df.loc[10:30, feature_columns]
# selected.to_json("test.json", orient="split")
selected.to_csv("test.csv")