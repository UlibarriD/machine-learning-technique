"""
    Nombre:     Diego Armando Ulibarri Hernández
    Matricula:  A01636875
"""
# Libreria para escalar los datos
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
# Se importa la librería de pandas para poder trabajar de una forma más fácil con el archivo csv
import pandas as pd


def train_test(data_frame, train_percentage):
    """Funcion que divide nuestros datos en train y test

    Args:
        data_frame: nuestro dataFrame de real estate.
        train_percentage (float): Porcentaje para entrenar el modelo

    Returns:
        datos train y test
    """
    # Se seleccionan de forma aleatorea el 80% de nuestros datos para el train
    data_frame_train = data_frame.sample(frac=train_percentage, random_state=25)
    
    # Se seleccionan los datos restantes para el test
    data_frame_test = data_frame.drop(data_frame_train.index)
    
    # Ya que tenemos separados los datos los escalamos para poder trabajar mejor con ellos
    data_frame_train = pd.DataFrame(StandardScaler().fit_transform(data_frame_train))
    data_frame_test = pd.DataFrame(StandardScaler().fit_transform(data_frame_test))
    data_frame_train.columns = data_frame.columns
    data_frame_test.columns = data_frame.columns
    
    # Obtenemos nuestros datos x, y
    # Train
    X_train = data_frame_train.drop('Y house price of unit area', axis=1)
    y_train = data_frame_train['Y house price of unit area']
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    y_train = np.reshape(y_train, (y_train.shape[0],))
    # Test
    X_test = data_frame_test.drop('Y house price of unit area', axis=1)
    y_test = data_frame_test['Y house price of unit area']
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    
    return X_train, y_train, X_test, y_test


def linear_regression(x, m, b):
    return (x.dot(m)) + b


def get_loss(y, y_pred):
    return (1 / y.shape[0]) * np.sum(np.power(y - y_pred, 2))


def gradient_descent(x, y, m, b, epochs, learning_rate):
    # Se calcula la prediccion actual
    y_pred = linear_regression(x, m, b)
    curr_loss = 0

    for _ in range(epochs):
        y_pred = linear_regression(x, m, b)
        loss = get_loss(y, y_pred)

        # Si nuestra ultima perdida y la perdida actual son iguales, encontramos nuestra epoca minima
        if np.array_equal(curr_loss, loss):
            break
        else:
            # se calcula la ultima perdida
            curr_loss = loss
            
            dm = -(2 / x.shape[0]) * (x.T).dot(y - y_pred)
            db = -(2 / x.shape[0]) * np.sum(y - y_pred)
            # Ajuste de pesos
            m -= learning_rate * dm
            b -= learning_rate * db
    return m, b


def main():
    """
        Función principal que manda llamar a todas las 
        demas funciones y las integra
    """
    # Se leen del csv y se asignan los valores x, y
    data_frame = pd.read_csv("./Data/Real-estate.csv")
    # Se manda llamar la función creada para dividir nuestros datos en train y test
    X_train, y_train, X_test, y_test = train_test(data_frame, 0.8)
    
    # Prueba 1
    m, b = np.random.rand((X_train.shape[1])) * 10, np.random.random()
    epochs = 10000
    lr = 0.01
    train_m, train_b = gradient_descent(X_train, y_train, m, b, epochs, lr)
    y_pred = linear_regression(X_test, train_m, train_b)
    print(f'Coeficiente de determinación: {r2_score(y_test, y_pred)}')
    print(pd.DataFrame({'Valor deseado': y_test, 'Valor obtenido': y_pred}))
    
    # Prueba 2
    m, b = np.random.rand((X_train.shape[1])) * 10, np.random.random()
    epochs = 10000
    lr = 0.00001
    train_m, train_b = gradient_descent(X_train, y_train, m, b, epochs, lr)
    y_pred = linear_regression(X_test, train_m, train_b)
    print(f'Coeficiente de determinación: {r2_score(y_test, y_pred)}')
    print(pd.DataFrame({'Valor deseado': y_test, 'Valor obtenido': y_pred}))
    
    # Prueba 3
    m, b = np.random.rand((X_train.shape[1])) * 10, np.random.random()
    epochs = 100000
    lr = 0.1
    train_m, train_b = gradient_descent(X_train, y_train, m, b, epochs, lr)
    y_pred = linear_regression(X_test, train_m, train_b)
    print(f'Coeficiente de determinación: {r2_score(y_test, y_pred)}')
    print(pd.DataFrame({'Valor deseado': y_test, 'Valor obtenido': y_pred}))


if __name__ == "__main__":
    main()
