# -*- coding: utf-8 -*-
"""
Created on Mon May  7 10:56:33 2018

@author: Tomas Pernas Valcarce
"""

#Importo librerias que voy a utilizar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importamos el dataset
dataSet = pd.read_csv('../DATASET/weatherAUS.csv')

#Analizamos el dataset

#Creamos una funcion para contabilizar los NA:
def num_missing(x):
  return sum(x.isnull())

#Aplicamos la funcion por columna:
print ("Valores NA por columna:")
print (dataSet.apply(num_missing, axis=0)) #axis=0 define que la funcion va a ser aplicada por columna

#Aplicamos la funcion por fila:
print ("Valores NA por Fila:")
print (dataSet.apply(num_missing, axis=1).head()) #axis=1 define que la funcion va a ser aplicada por fila // lo restringimos a los 5 primeros registros

#Borramos los NA que aparecen en RaintToday y RainTomorrow
dataSet = dataSet.dropna(subset = ['RainToday'])
dataSet = dataSet.dropna(subset = ['RainTomorrow'])

#Todas las columnas que vamos a utilizar presentan en sus filas NA, por ello vamos a trabajarlos:
dataSet['MinTemp'].fillna(dataSet['MinTemp'].mean(), inplace=True)
dataSet['MaxTemp'].fillna(dataSet['MaxTemp'].mean(), inplace=True)
dataSet['Rainfall'].fillna(dataSet['Rainfall'].mean(), inplace=True)
dataSet['Evaporation'].fillna(dataSet['Evaporation'].mean(), inplace=True)
dataSet['Sunshine'].fillna(dataSet['Sunshine'].mean(), inplace=True)

#Separamos en variables. X para las independientes e Y para las dependientes.
X = dataSet.loc[:,['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','RainToday']].values
y = dataSet.loc[:, ['RainTomorrow']].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()

X[:, 5] = labelencoder.fit_transform(X[:, 5])
onehotencoder = OneHotEncoder(categorical_features = [5])
X = onehotencoder.fit_transform(X).toarray()


y[:, 0] = labelencoder.fit_transform(y[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()


#Borramos una de las columnas dummy para no tener el problema de la dummy trap
#Borro la columna de los YES, quedandome entonces la columna que obedece a los valores NO
X = np.delete(X,1,1)
y = np.delete(y,1,1)

#  Dividimos el ds en dos conjuntos. Training y Test.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Entrenamos nuestro modelo
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Ejecutamos nuestro modelo
y_pred = regressor.predict(X_test)

# Usando Backward Elimination observamos cuales son las variables de interes
import statsmodels.formula.api as sm

#Al momento de aplicar un modelo de optimizacion para hallar las variables
#mas importantes del modelo, hay que tener en cuenta la situacion cuando Y = 0
#Para ello lo que tenemos que hacer es agarrar nuestro dataset original y en la 
#Primer fila llenarlo de 1. De esta manera el modelo se dara cuenta que la condicion
#Y=0 se encuentra contenida dentro del dataset
X = np.append(arr = np.ones((140787, 1)).astype(int), values = X, axis = 1)

X_opt = X[:, [1, 2, 3, 4, 5, 6]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#Como podemos observar, todas las variables poseen un p <= 0.5 por lo que podemos decir que todas las variables independientes utilizadas son relevantes. 


#####################
#Evaluamos el modelo#
#####################

#Utilizamos la libreria statsmodels para hallar el valor de Tstudent y Pvalue
import statsmodels.api as sm

#Analizamos Tstudent y Pvalue
#Si el valor de Tstudent es cero, diremos que no hay relacion entre las variables X e Y.
#Si el valor de P-Value es muy bajo se suelen tomar valores menores a 0,05 para considerar que es bajo, decimos que hay relacion entre las variables X e Y

est = sm.OLS(y,X_opt).fit()
print(est.summary())

#Creamos una tabla para observar el valor real del numero y la prediccion que realizo el modelo.
pred_ds = pd.DataFrame(y_pred, columns=['Prediccion'])
pred_ds = pred_ds.assign(Valor_Real= y_test)
pred_ds


