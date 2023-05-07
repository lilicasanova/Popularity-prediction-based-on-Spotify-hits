import xgboost as xgb
import pandas as pd
import datetime
import pickle 
import os

os.chdir(os.path.dirname(__file__))
dir_path = os.path.dirname(os.path.realpath(__file__))
csv_path = os.path.join(dir_path, 'train.csv')

#sacamos los datos de los .csv train

train_data = pd.read_csv(csv_path)

#dividimos features y target

X = train_data.drop('popularity_', axis=1)
Y = train_data['popularity_']

#creamos y entrenamos nuestro modelo

xgb_clas = xgb.XGBRFClassifier(random_state= 42)

xgb_clas.fit(X, Y)

#sacamos el output model fecha

filename = f"model_{datetime.datetime.now().strftime('%y%m%d%H%M%S')}.py"
with open(filename, 'wb') as f:
    pickle.dump(xgb_clas, f)