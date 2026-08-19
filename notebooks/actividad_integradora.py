import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# PASO 2 · TAREA A
# Regresión: predecir el peso del pingüino
df = pd.read_csv('../data/penguins.csv')
print('Las 5 primeras filas:')
print(df.head())
print('\nForma del dataset:', df.shape)
print()
print(df.info())
print('\nValores nulos:')
print(df.isnull().sum())

# 1 Elimina las filas con nulos en las columnas que vas a usar, con .dropna(subset=[...])
df_reg = df.dropna(subset=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'species', 'island'])

# 2 Codifica las columnas de texto ( species , island , sex ) con pd.get_dummies()
df_enc = pd.get_dummies(df_reg, columns=['species', 'island', 'sex'], drop_first=False)

# 3 Separa X (todo menos body_mass_g ) e y ( body_mass_g ), y divide en train/test
X = df_enc.drop(columns=['body_mass_g'])
y = df_enc['body_mass_g']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4 Entrena, predice, y guarda cada métrica en su propia variable antes de imprimirla
modelo_reg = LinearRegression()
modelo_reg.fit(X_train, y_train)
y_pred = modelo_reg.predict(X_test)

# Guarda cada métrica en su propia variable, no solo la imprimas
mae_reg = mean_absolute_error(y_test, y_pred)
r2_reg = r2_score(y_test, y_pred)

print()
print('MAE:', round(mae_reg, 2))
print('R2:', round(r2_reg, 4))

# PASO 3 · TAREA B
# Clasificación: predecir el sexo del pingüino
df_clf = df.dropna(subset=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'species', 'island'])

df_clf_enc = pd.get_dummies(df_clf,)


































