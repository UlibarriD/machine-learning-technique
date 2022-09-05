"""
  Nombre: Diego Armando Ulibarri Hernández 
  Matricula: A01636875
"""

# Se importa la librería de pandas para poder trabajar de una forma más fácil con el archivo csv
import pandas as pd
# Se importa la librería de matplotlib para visualizar el modelo
import matplotlib.pyplot as plt


def gradient_descent(epochs, m, c, L, x, y):
    """ función para obtener el gradiente descendiente 

    Args:
        epochs (int): epocas
        m (float): _description_
        y (float): _description_
        L (float): Learning rate
        x (): 
        y(): 
    """
    for _ in range(epochs):
        pred_act_y = m * x + c  # formula para la predicción actual de y
        D_m = (-2 / float(len(x))) * sum(x * (y - pred_act_y))
        D_c = (-2 / float(len(x))) * sum(y - pred_act_y) 
        m = m - L * D_m  # Predicción m
        c = c - L * D_c  # Predicción c
    
    return m, c


def main():
    """
      Esta función nos sirve para leer los datos del 
    """
    # Se leen del csv y se asignan los valores x, y, m, c
    dataFrame = pd.read_csv("./Data/winequality-red.csv")
    y = dataFrame['pH']
    x = dataFrame['fixed acidity']
    # Generar función para obtener un dataset para train y test...
    # Generar función para calcular los errores
    m, c = gradient_descent(10000, 0, 0, 0.001, x, y)
    print(m, c)
    # predicción y gráfica
    y_final = m * x + c
    plt.scatter(x, y)
    plt.plot([min(x), max(x)], [min(y_final), max(y_final)], color = 'red')
    plt.show()
    # Presentar el error


if __name__ == "__main__":
    main()
