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
print()

# PASO 3 · TAREA B
# Clasificación: predecir el sexo del pingüino
df_clf = df.dropna(subset=['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g','sex','species','island'])

df_clf_enc = pd.get_dummies(df_clf, columns=['species','island'], drop_first=False)
df_clf_enc['sex_binario'] = (df_clf_enc['sex']=='MALE').astype(int)

X = df_clf_enc.drop(columns=['sex','sex_binario'])
y = df_clf_enc['sex_binario']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar SOLO las columnas numéricas, ajustando solo con train
from sklearn.preprocessing import StandardScaler
cols_num = ['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']
scaler = StandardScaler()
X_train[cols_num] = scaler.fit_transform(X_train[cols_num])
X_test[cols_num] = scaler.transform(X_test[cols_num])

# PASO 3 · TAREA B (CONTINUACIÓN)
# Entrenar, predecir y evaluar
# Guardando cada métrica en su propia variable, no solo imprimiéndola

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

modelo_clf = LogisticRegression(max_iter=1000)
modelo_clf.fit(X_train, y_train)
y_pred = modelo_clf.predict(X_test)

# Guarda cada métrica en su propia variable, no solo la imprimas
acc_clf = accuracy_score(y_test, y_pred)
prec_clf = precision_score(y_test, y_pred)
rec_clf = recall_score(y_test, y_pred)
f1_clf = f1_score(y_test, y_pred)

print("Accuracy:", round(acc_clf, 4))
print("Precision:", round(prec_clf, 4))
print("Recall:", round(rec_clf, 4))
print("F1:", round(f1_clf, 4))
print(confusion_matrix(y_test, y_pred))
print()

# PASO 4
# Ambos resultados, uno al lado del otro
print("TAREA A: REGRESIÓN (peso del pingüino)")
print("MAE:", round(mae_reg, 2), "| R2:", round(r2_reg, 4))
print()
print("TAREA B: CLASIFICACIÓN (sexo del pingüino)")
print("Accuracy:", round(acc_clf, 4), "| Precision:", round(prec_clf, 4), "| Recall:", rec_clf)
