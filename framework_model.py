"""
  Nombre:     Diego Armando Ulibarri Hernández
  Matricula:  A01636875
"""
# Nos sirve para dividir los datos entre prueba y entrenamiento
from sklearn.model_selection import train_test_split
# Se importa la librería de pandas para poder trabajar de una forma más fácil con el archivo csv
import pandas as pd
# Se importa la librería de matplotlib para visualizar el modelo
import matplotlib.pyplot as plt
# Modelo que utilizaremos para nuestras predicciones
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from mlxtend.evaluate import bias_variance_decomp


def regresionLineal(X_train, X_test, y_train, y_test):
    # creamos nuestro modelo con la librería de sklearn
    regr = LinearRegression()
    # entrenamos a nuestro modelo con nuestros datos de entrenamiento
    regr.fit(X_train, y_train)
    # Obtenemos nuestra prediccion con los datos prueba
    y_pred = regr.predict(X_test)
    # Imprimimos los resultados de nuestros scores
    print(f'Coeficiente de determinación: {r2_score(y_test, y_pred)}')
    print(f'Exactitud del modelo (train): {regr.score(X_train, y_train)}')
    # Imprimimos nuestros coeficientes
    print(f'Coeficientes: {regr.coef_}')
    # Mean squared error
    print(f'Mean squared error: {mean_squared_error(y_test, y_pred)}')
    # Impresion del modelo
    return y_pred


def main():
    """
      Función para la lectura, separación de datos y 
      llamar el modelo
    """
    df = pd.read_csv("./Data/Real-estate.csv")
    # Eliminamos el id y los precios de las casas para nuestras x
    X = df.drop(columns=['Y house price of unit area', 'No'], axis=1)
    y = df['Y house price of unit area']
    # Dividimos nuestro dataset en test y train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = regresionLineal(X_train, X_test, y_train, y_test)
    
    testing = pd.DataFrame({'Actual_Price':y_test, 'Price_Prediction':y_pred})
    print(testing)
    
    regr = LinearRegression()
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    mse, bias, var = bias_variance_decomp(regr, X_train, y_train, X_test, y_test, loss='mse', num_rounds=200, random_seed=123)
    print('MSE de bias_variance lib [pérdida promedio esperada]: %.3f' % mse)
    print('Sesgo promedio: %.3f' % bias)
    print('Varianza promedio: %.3f' % var)


if __name__ == "__main__":
    main()
