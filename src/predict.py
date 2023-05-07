import pandas as pd
import pickle
import sklearn
import os

os.chdir(os.path.dirname(__file__))
dir_path = os.path.dirname(os.path.realpath(__file__))
csv_path = os.path.join(dir_path, 'test.csv')


# Cargar el modelo entrenado
with open('model_230507192746.py', 'rb') as f:
    model = pickle.load(f)

# Cargar los datos de prueba
test_data = pd.read_csv(csv_path)
test_data.drop('popularity_', axis=1, inplace=True)

# Aplicar el modelo a los datos de prueba
predictions = model.predict(test_data)

# Guardar las predicciones en un archivo CSV
output = pd.DataFrame({'Prediction': predictions})
output.to_csv('predictions.csv', index=False)