import requests
import json
import pandas as pd

# sample_input  = pd.DataFrame.from_records({"columns": ["Lot Area", "Gr Liv Area", "Garage Area", "SalePrice", "Bldg Type"], "data": [
#     [31770, 1656, 528.0, 215000, "Twnhs"]
# ] })

dataset = pd.DataFrame.from_records(
    columns=["Lot Area", "Gr Liv Area", "Garage Area", "SalePrice", "Bldg Type"],
    data=[[31770, 1656, 528.0, 215000, "Twnhs"]])

ds_dict = dataset.to_dict(orient='split')
del ds_dict['index']
ds_dict = {"dataframe_split": ds_dict}

model_call = requests.post(
    "http://localhost:5001/invocations",
    data=json.dumps(ds_dict),
    headers={"Content-type": "application/json"}
)
response_json = json.loads(model_call.text)
print(response_json)

