import pandas as pd

df = pd.read_csv("data/ames_housing.csv")
df_housedata = df.iloc[:-11,:]
df_housedata_new = df.iloc[-11:-1,:]
df_housedata.to_csv('data/housedata.csv')
df_housedata_new.to_csv('data/housedata_new.csv')
feature_columns = ["Lot Area", "Gr Liv Area", "Garage Area", "Bldg Type"]
selected = df.loc[10:30, feature_columns]
# selected.to_json("test.json", orient="split")
selected.to_csv("data/test.csv")