import pandas as pd
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
data = pd.read_csv('data/test.csv')

logged_model = 'models:/House price predictions/Production'
# logged_model = 'runs:/5e7fc662c3c540248834d6828549d050/custom_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
predict = loaded_model.predict(pd.DataFrame(data))

print(predict)