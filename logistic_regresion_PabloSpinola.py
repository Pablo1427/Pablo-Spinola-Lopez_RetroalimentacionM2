# @author: Pablo Spínola López
#          A01753922
# Descripción: Acontinuación, se presenta un programa dedicado a calcular la regresión logística a partir de un dataset dado.
# Posterior al entrenamiento, se evalúa y se visualizan gráficas de desempeño.

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import exp
from math import log

def escalar_prom(var_indep):
    """
        Función para escalar una lista de valores utilizando media y valor máximo.

        Se recibe una lista de valores numéricos y los escala dividiento la diferencia
        de cada valor y la media de la lista entre el valor máximo de la lista. 
        Todo esto para normalizar los datos para centrar los datos.

        Args:
            var_indep (list): Lista de valores numéricos a escalar.

        Regresa:
            data_normal (list): Lista de valores escalados por media y valor máximo.
    """
    n = len(var_indep)
    max = 0
    suma = 0

    # Primer ciclo donde se realiza la suma de los valores para el promedio y se calcula el valr máximo
    for i in range(n):
        # Suma de los valores
        suma += var_indep[i]
        # Cálculo de valor máximo
        if var_indep[i] > max:
            max = var_indep[i]

    # Cálculo de media
    mean = suma / n
    
    data_normal = []

    # Segundo ciclo para llenar la lista con los valores normalizados
    for i in range(n):
        # Se utiliza la fórmula mencionada: (x - mean) / max
        normalizar = (var_indep[i] - mean) / max
        data_normal.append(normalizar)
    
    return data_normal

def split_data(X, Y, train_perc, val_perc):
    """
        Función que divide el conjunto de datos para entrenamiento, validación y prueba.

        Esta función toma las listas de variables independientes y dependientes,
        y las divide en tres subconjuntos: entrenamiento (idealmente mayor), validación y prueba,
        según las proporciones especificadas.

        Args:
            X (list): Lista de valores de la variable independiente (e.g., calificaciones).
            Y (list): Lista de valores de la variable dependiente (e.g., 1 para aprobado, 0 para no aprobado).
            train_ratio (float): Proporción del conjunto de datos que se asignará al conjunto de entrenamiento.
            val_ratio (float): Proporción del conjunto de datos que se asignará al conjunto de validación.
        
        Regresa:
            - X_train: Datos independientes para el entrenamiento.
            - Y_train: Datos dependientes para el entrenamiento.
            - X_val: Datos independientes para validación.
            - Y_val: Datos dependientes para validación.
            - X_test: Datos independientes para prueba.
            - Y_test: Datos dependientes para prueba.
    """
    n_total = len(X)

    # Se calcula la cantidad de datos que ha de tener cada lista según las proporciones dadas
    train_end = int(n_total * train_perc)
    val_end = train_end + int(n_total * val_perc)

    # Con los índices calculados, se distribuyen por partes del set completo a las listas correspondientes con slicing
    # Para el entrenamiento se selecciona desde el principio hasta la cantidad de entrenamiento.
    X_train = X[:train_end]
    Y_train = Y[:train_end]
    # Para el de validación, se inicia con el siguiente del entrenamiento hasta la cantidad de entrenamiento más validación
    # val_end = train_end + int(len(X) * val_perc)
    X_val = X[train_end:val_end]
    Y_val = Y[train_end:val_end]
    # Para el testeo, va desde después de la validación hasta el resto de la lista
    X_test = X[val_end:]
    Y_test = Y[val_end:]

    return list(X_train), list(Y_train), list(X_val), list(Y_val), list(X_test), list(Y_test)

class Reg_Evaluation:
    """
    Clase para evaluar el desempeño de un modelo de regresión logística.

    Esta clase proporciona métodos para evaluar las predicciones del modelo
    utilizando diversas métricas de evaluación, incluyendo la matriz de confusión,
    la curva precisión-recall, la puntuación F1 y la curva ROC.

    Atributos:
        y_real (list): Lista de valores verdaderos de la variable dependiente.
        y_predict (list): Lista de valores predichos por el modelo.
    """

    def __init__(self, y_real, y_predict):
        """
        Inicializa la clase con los valores reales y las predicciones.

        Args:
            y_real (list): Lista de valores verdaderos de la variable dependiente.
            y_predict (list): Lista de valores predichos por el modelo.
        """
        # Uso de numpy para los cálculos
        self.y_real = np.array(y_real)
        self.y_predict = np.array(y_predict)
        
        # Se calcula manualmente la matriz de confusión
        self.true_positive = np.sum((self.y_real == 1) & (self.y_predict == 1))
        self.true_negative = np.sum((self.y_real == 0) & (self.y_predict == 0))
        self.false_positive = np.sum((self.y_real == 0) & (self.y_predict == 1))
        self.false_negative = np.sum((self.y_real == 1) & (self.y_predict == 0))

    def calcula_precision(self, tp, fp):
        """
            Calcula la precisión dada la cantidad de verdaderos positivos (TP) y falsos positivos (FP).
            
            La precisión es la proporción de verdaderos positivos entre los casos que fueron
            clasificados como positivos por el modelo. Ayuda a entender la exactitud de las
            predicciones positivas del modelo.

            Args:
                - tp (int): Número de verdaderos positivos.
                - fp (int): Número de falsos positivos.

            Regresa:
                - precision (int): La precisión calculada.
            Regresa 0 si no hay casos positivos predichos.
        """
        if tp + fp > 0:
            precision = tp / (tp + fp)
            return precision
        return 0

    def calcula_recall(self, tp, fn):
        """
            Calcula el recall dado el número de verdaderos positivos (TP) y falsos negativos (FN).
            
            El recall mide la proporción entre verdaderos positivos y el total de casos realmente positivos.
            Ayuda a entender cuántos de los casos positivos fueron detectados por el modelo.

            Args:
                - tp (int): Número de verdaderos positivos.
                - fn (int): Número de falsos negativos.

            Regresa:
                - recall (int): La recall calculada.
            Regresa 0 si no hay casos positivos reales.
        """
        if tp + fn > 0:
            recall = tp / (tp + fn)
            return recall
        return 0

    def matrix_confusion(self):
        """
            Calcula y muestra la matriz de confusión utilizando un heatmap.
        """
        # Se crea una matriz de 2x2 con los valores de la matriz de confusión
        cm = np.array([[self.true_negative, self.false_positive],
                       [self.false_negative, self.true_positive]])

        # Se grafica como un heatmap, con un mayor sombreado donde existe más densidad de datos
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.show()

    def curve_precision_recall(self):
        """
            Calcula y muestra la curva precisión-recall.
        """
        # Se enlistan umbrales del 0 al 1 con un paso de 0.01
        umbrales = np.linspace(0, 1, 100)
        precisions = []
        recalls = []

        # Itera sobre cada umbral en la lista de umbrales para calcular la precisión y el recall
        for umbral in umbrales:
            # Convierte las probabilidades predichas en valores binarios usando el umbral actual
            y_pred_binary = (self.y_predict >= umbral).astype(int)
            # Calcula el número de verdaderos positivos
            true_positive = np.sum((self.y_real == 1) & (y_pred_binary == 1))
            # Calcula el número de falsos positivos
            false_positive = np.sum((self.y_real == 0) & (y_pred_binary == 1))
            # Calcular falsos negativos
            false_negative = np.sum((self.y_real == 1) & (y_pred_binary == 0))

            # Calcular precisión
            precision = self.calcula_precision(true_positive, false_positive)
            # Calcular recall
            recall = self.calcula_recall(true_positive, false_negative)
            
            # Añadir precisión y recall a las listas
            precisions.append(precision)
            recalls.append(recall)

        plt.plot(recalls, precisions, marker='.')
        plt.title('Curva Precision-Recall')
        plt.xlabel('Recall')
        plt.ylabel('Precisión')
        plt.show()

    def f1_score(self):
        """
            Calcula y muestra la puntuación F1.
        """
        # Calcular precisión
        precision = self.calcula_precision(self.true_positive, self.false_positive)
        # Calcular recall
        recall = self.calcula_recall(self.true_positive, self.false_negative)

        print(" Precisión:", precision)
        print(" Recall:", recall)

        f1 = 0
        # Asegurarse de que el denominador no sea cero
        if (precision + recall) > 0:
            # Calcular la puntuación F1 como la media armónica entre precisión y recall
            f1 = 2 * (precision * recall) / (precision + recall)

        print(' Puntuación F1:', f1)

    def curva_roc(self):
        """
        Calcula y muestra la curva ROC y el área bajo la curva (AUC).
        """
        # Se enlistan umbrales del 0 al 1 con un paso de 0.01
        umbrales = np.linspace(0, 1, 100)
        tp_rate = []
        fp_rate = []

        # Itera sobre cada umbral en la lista de umbrales
        for umbral in umbrales:
            # Convierte las probabilidades predichas en valores binarios usando el umbral actual
            y_pred_binary = (self.y_predict >= umbral).astype(int)
            # Calcula el número de verdaderos positivos
            true_positive = np.sum((self.y_real == 1) & (y_pred_binary == 1))
            # Calcula el número de falsos positivos
            false_positive = np.sum((self.y_real == 0) & (y_pred_binary == 1))
            # Calcula el número de verdaderos negativos
            true_negative = np.sum((self.y_real == 0) & (y_pred_binary == 0))
            # Calcula el número de falsos negativos
            false_negative = np.sum((self.y_real == 1) & (y_pred_binary == 0))

            true_positive_rate = 0
            false_positive_rate = 0

            # Calcula la tasa de verdaderos positivos y la tasa de falsos positivos, evitando 0s
            if (true_positive + false_negative) > 0:
                true_positive_rate = true_positive / (true_positive + false_negative)
            if (false_positive + true_negative) > 0:
                false_positive_rate = false_positive / (false_positive + true_negative)
            
            # Se guardan las tasas calculadas
            tp_rate.append(true_positive_rate)
            fp_rate.append(false_positive_rate)

        # Uso de numpy para facilitar las cosas
        fpr = np.array(fp_rate)
        tpr = np.array(tp_rate)
        
        # Calcula el área bajo la curva (AUC) con la integración trapezoidal (trapz)
        roc_auc = np.trapz(tpr, fpr)

        plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (área = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Curva de Característica Operativa del Receptor (ROC)')
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.legend(loc="lower right")
        plt.show()


class Logi_Reg():
    """ 
    Clase que implementa una regresión logística univariable.

    Esta clase implementa un modelo de regresión logística para una única variable independiente.
    La regresión logística nos ayuda a realizar un modelado estadístico, específicamente para predecir una
    variable binaria, lo cual quiere decir que se espera que los valores de la variable
    independinte sean 0 o 1.

    En esta implementación:

    - Se utiliza la función sigmoidal como función de activación. Esta función recibe una combinación lineal de
    los pesos con la variable independientes x, calculando una probabilidad de entre 0 y 1.
    
    - La pérdida se calcula utilizando cross-entropy (o log loss).
    Con esto podemos calcular que tanta diferencia existe entre las predicciones del modelo y los valores reales,
    para ajustar las predicciones de los pesos del modelo con el ritmo de aprendizaje.

    - Los pesos del modelo se actualizan usando Batch Gradient Descent. En cada época, los pesos se mejoran con
    el promedio del gradiente de la función de pérdida con respecto al ritmo de aprendizaje, siempre considerando
    todos los valores del conjunto de datos (batch) en cada actualización. Este tipo de gradient descent es
    eficiente para conjuntos pequeños de datos como este, ya que se actualizan los pesos de todo el dataset.

    Atributos:
        X: Lista de números (idealmente escalados), que representan la variable independiente.
        Y: Lista de valores binarios (0 o 1), que representan la variable dependiente.
        learning_rate: Ritmo de aprendizaje (alfa), que representa el tamaño de los pasos al momento de
                       actualizar  los pesos.
    """

    def __init__(self, Xtr, Ytr, Xv, Yv, Xte, Yte, learning_rate):
        """
            Se inicializan variables de instancia.
        """
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.Xval = Xv
        self.Yval = Yv
        self.Xte = Xte
        self.Yte = Yte
        self.lr = learning_rate
        self.weights = [0, 0]
        self.prev_weights = [0, 0]
        self.m = len(self.weights)
        self.n = len(self.Xtr)
        self.loss = 0
        self.test_loss = 0
        self.trained = False

    def unchanged(self):
        """
            Método para comprobar si el peso se mantuvo igual.
        """
        return self.weights == self.prev_weights

    def sigmoid(self, net):
        """
            Método para calcular la función sigmoidal.

            Con esta función de activación es posible convertir un valor dado en un valor que cae 
            dentro del rango entre 0 y 1; útil para calcular probabilidades de este tipo de regresión,
            por lo que retorna el valor de la probabilidad predicha. Se utiliza una transformación
            no-linear.

            La fórmula es:
                S(net) = 1 / (1 + exp(- net))
            
            Args:
                net (float): El valor a convertir usando la función sigmoidal.

            Regresa:
                hip (float): Un valor entre 0 y 1, que representa la probabilidad predicha.
        """
        S = 1 / (1 + exp(- net))
        return S
    
    def hipotesis(self, x):
        """
            Método para realizar una predicción (hipótesis) con la función sigmoidal.

            Este método hace una hipótesis para un valor de la variable independiente (X).
            Esta hipótesis se calcula de manera lineal al multiplicar los pesos actuales con el valor
            dependiente, y luego utilizar este valor a en la función sigmoidal para obtener una probabilidad.

            Dicha fórmula de combinación lineal es:
                net = w0 + x * w1
            
            Donde `w0` es el término independiente y el peso intercepto, y `w1` es el peso asociado a la variable
            independiente `x`.
            
            Args:
                x (float): El valor de la variable independiente a partir de donde se calcula la hipótesis.

            Regresa:
                hip (float): La predicción realizada, siendo un valor entre 0 y 1.
        """
        net = self.weights[0] + x * self.weights[1]
        hip = self.sigmoid(net)
        return hip
    
    def log_loss(self, X, Y):
        """
            Método para calcular la pérdida de logaritmo (log loss o cross-entropy loss).

            Se mide el error entre las predicciones (con hipótesis) y los valores de Y reales de validación.
            Se suman los logaritmos de las predicciones por el valor para los ejemplos positivos y los
            logaritmos de las probabilidades complementarias para los ejemplos negativos.
            
            La pérdida se calcula con:
                costo = - (1/n) * sum(Y * log(h(x)) + (1 - Y) * log(1 - h(x)))
        """
        error = 0
        perdida = 0
        n = len(X)
        
        for i in range(n):
            hi = self.hipotesis(X[i])
            
            # Si la predicción da 0 o 1, se ajusta ligeramente para no calcular el
            # logaritmo de 0 (indefinido), o el de 1 (0):
            #       Reemplazando con   1e-10 si es 0
            #                    con 1-1e-10 si es 1
            if hi == 0:
                hi = 1e-10
            elif hi == 1:
                hi = 1 - 1e-10
            
            # Se calcula la pérdida del valor antes de agregarla a la pérdida total:
            #      Y * log(h(x)) + (1 - Y) * log(1 - h(x))
            perdida = Y[i] * log(hi) + (1 - Y[i]) * log(1 - hi)
            
            # Se realiza la suma
            error += perdida
        
        # Se saca el promedio dividiéndolo entre el total de muestras, teniendo así la Cross Entropy y actualizándolo.
        costo = - error / n
        return costo
    
    def update_ws(self):
        """
            Método para actualizar los pesos usando Batch Gradient Descent.

            Este método calcula el gradiente de la función de pérdida (log_loss) con respecto a cada peso (w0 y w1).
            La actualización del peso w0 se calcula utilizando la diferencia entre la predicción con la hipótesis y
            el valor real de la variable dependiente Y. lo mismo para el peso w1, colo que también incluye el valor
            de la variable independiente X correspondiente.

            Los pesos se calculan de la siguiente forma:
                ws = ws - alfa * (1/n) * sum((h(x) - Y) * X)
        """
        new_weights = []

        # El ciclo exterior recorre los pesos involucrados (w0 y w1), actualizando uno por iteración, por lo tanto
        # son sólo 2
        for i in range(self.m):
            grad_des_sum = 0

            for j in range(self.n):
                # Se calcula la hipótesis y se saca la diferencia con el valor real correspondiente de Y
                hi = self.hipotesis(self.Xtr[j])
                dif = hi - self.Ytr[j]
                
                # Aquí, en caso de ser w0, sólo se usa la diferencia previa para el gradiente
                #       En caso de ser w1, se usa el producto de la diferencia por el valor de x correspondiente
                grad_des = dif if i == 0 else dif * self.Xtr[j]
                grad_des_sum += grad_des
            
            # Se obtiene el producto del ritmo de aprendizaje (alfa) con el promedio del gradiente entre el total
            # de muestras.
            lr_advance = self.lr * (grad_des_sum) / self.n
            # Se calculan finalmente los nuevos pesos al restarle a los actuales el producto previo con el ritmo
            # de aprendizaje.
            new_weights.append(self.weights[i] - lr_advance)
        
        # Finalmente, los pesos anteriores se guardan y se actualizan los pesos actuales, para comparación posterior.
        self.prev_weights = self.weights
        self.weights = new_weights

    def show_advance(self):
        """
            Método para mostrar los pesos finales calculados por el modelo.
        """
        print(" Pesos finales ajustados del modelo:")
        print(" \tw0 =", self.weights[0])
        print(" \tw1 =", self.weights[1])
        print(f"\n Pérdida con data de validación: {self.loss:.4f}")
        print(f" Pérdida con data de prueba: {self.test_loss:.4f}")

    def predict(self, X):
        """
            Método para hacer predicciones sobre un nuevo conjunto de datos utilizando el modelo entrenado.
    
            Se calcula la probabilidad de que cada punto de datos pertenezca a la clase positiva (que sea 1),
            utilizando el modelo entrenado. Estas probabilidades se transforman en predicciones binarias
            basado en un umbral de 0,5; probabilidad mayor a 0,5 da una predicción de 1, si no, 0.

            Args:
                X (list): Una lista con los valores independientes sobre los cuales se realiza la predicción.

            Regresa:
                y_pred (list): La lista con los valores dependientes que resultan de la predicción.
        """
        # Se revisa que el modelo esté entrenado, de lo contrario se alza una excepción.
        if self.trained:
            y_pred = []

            # Se recorre el dataset, calculando probabilidades y asignando predicciones a los valores como Y
            for i in X:
                
                # Se calcula la probabilidad con la hipótesis que utiliza la función sigmoidal
                probability = self.hipotesis(i)

                # En caso de que la probabilidad sea mayor o igual a 0.5 de ser positiva, predicción es 1
                if probability >= 0.5:
                    y_pred.append(1)
                
                # En caso de que la probabilidad sea menor a 0.5 de ser positiva, predicción es 0
                else:
                    y_pred.append(0)
            
            return y_pred
        
        else:
            # Error al no haber sido entrenado aun
            raise ValueError("El modelo no ha sido entrenado. Es necesario entrenarlo para realizar predicciones.")

    def do_training(self):
        """
            Método para entrenar el modelo de regresión logística utilizando Batch Gradient Descent.

            Se entrena al modelo ajustando los pesos de forma iterativa hasta cumplir alguna de las condiciones
            de aceptación.
        """
        while True:
            
            # Se actualizan los pesos con batch gradient descent.
            self.update_ws()
            # Se calcula la pérdida con los valores de validación como supervisión y asi conocer la
            # precisión tras actualizar los pesos previamente con los valores de entrenamiento.
            self.loss = self.log_loss(self.Xval, self.Yval)

            # El proceso continúa eternamente hasta que se cumpla alguna de estas condiciones:
            # - Los pesos no cambian entre iteraciones consecutivas, por lo que el modelo no muestra más mejoras.
            # - La pérdida calculada (log loss) es menor que un umbral de aceptación, por lo que el modelo
            # tiene nivel deseado de precisión.
            if self.unchanged() or self.loss < 0.01:
                self.trained = True
                break

    def execute_Logistic_Regression(self):
        """
            Método que inicia todo el modelo, realiza el entrenamiento y muestra los valores finales 
            para los pesos y las pérdidas en el dataset de validación adquiridas en entrenamiento y
            se calculan las pérdidas de prueba.
        """
        self.do_training()
        self.test_loss = self.log_loss(self.Xte, self.Yte)
        self.show_advance()


if __name__ == "__main__":

    # El dataset para los valores
    var_indep = [
        90, 89, 16, 87, 86,  42, 81, 83, 32, 40,
        94, 72, 88, 92, 100, 24, 30, 73, 66, 58,
        35, 22, 15, 69, 70,  74, 86, 62, 94, 77,
        10, 88, 82, 79, 61,  15, 96, 71, 13, 85,
        24, 49, 76, 84, 99,  2,  80, 67, 97, 99
    ]
    var_dep = [
        1, 1, 0, 1, 1, 0, 1, 1, 0, 0,
        1, 1, 1, 1, 1, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 1, 1,
        0, 1, 1, 1, 0, 0, 1, 1, 0, 1,
        0, 0, 1, 1, 1, 0, 1, 0, 1, 1
    ]

    # Valores para el aprendizaje
    alfa = 0.1
    train_perc = 0.6
    val_perc = 0.2
  # test_perc = Se asume que el porcentaje restante irá destinado al testeo

    # Se secciona el dataset para entrenamiento, validación y testeo
    var_indep_tr, var_dep_tr, var_indep_val, var_dep_val, var_indep_te, var_dep_te = split_data(var_indep, var_dep, train_perc, val_perc)
    
    print(" -------------------------------")
    print(" ----- Regresion logistica -----")
    print(" -------------------------------\n")
    
    print("  alpha:", alfa)
    print(f"  train %: 60%")
    print(f"  validation 20%:")
    print(f"  test %: 20%")
    print(" -------------------------------")
    print("  X entrenamiento:", var_indep_tr)
    print("  Y entrenamiento:", var_dep_tr)
    print(" -------------------------------")
    print("  X valicación:", var_indep_val)
    print("  Y valicación:", var_dep_val)
    print(" -------------------------------")
    print("  X test:", var_indep_te)
    print("  Y test:", var_dep_te)
    print(" -------------------------------")
    print()

    # Se normalizan las variables independientes de cada sección del dataset
    X_train = escalar_prom(var_indep_tr)
    X_val = escalar_prom(var_indep_val)
    X_test = escalar_prom(var_indep_te)
    
    print(" -------------------------------")
    print(" ----- Datasets X escalados ----")
    print(" -------------------------------")
    print("  X entrenamiento:", X_train)
    print("  X valicación:", X_val)
    print("  X test:", X_test)
    print(" -------------------------------")

    # Se hace una instancia de regresión logística con los valores previamente adquiridos y se entrena el modelo
    regresion1 = Logi_Reg(X_train, var_dep_tr, X_val, var_dep_val, X_test, var_dep_te, alfa)
    regresion1.execute_Logistic_Regression()

    # Una vez entrenado el modelo, podemos realizar predicciones para así evaluar el modelo.
    eval_data_indep = [
        47, 82, 14, 29, 56, 91, 67, 23, 34, 89,
        6,  52, 77, 16, 40, 3,  68, 95, 24, 70,
        86, 19, 44, 59, 84, 28, 11, 97, 81, 33
    ]
    real_Y = [
        0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 1,
        1, 0, 0, 0, 1, 0, 0, 1, 1, 0
    ]

    # Se normaliza el dataset de evaluación y se obtienen las predicciones del modelo.
    normalize_eval = escalar_prom(eval_data_indep)
    predicted_Y = regresion1.predict(normalize_eval)

    # Se evalua el modelo
    evaluacion1 = Reg_Evaluation(real_Y, predicted_Y)
    evaluacion1.matrix_confusion()
    evaluacion1.curve_precision_recall()
    evaluacion1.f1_score()
    evaluacion1.curva_roc()
