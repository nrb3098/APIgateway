from myapp.models import myApp
from myapp.serializers import myAppSerializer


from rest_framework import generics
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import JsonResponse


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Establecer el backend de Matplotlib en modo no interactivo

import matplotlib.pyplot as plt # Para graficar

import io
import base64


from sklearn import tree # Importa el módulo tree https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree
from sklearn.tree import DecisionTreeClassifier # Importa el Decision Tree Classifier
from sklearn.model_selection import train_test_split # Importa la función para dividir el conjunto de datos en train_test_split 
from sklearn import metrics #Importa metricas scikit-learn para el calculo del accuracy
from sklearn import preprocessing #Importa funciones para el procesamiento de los datos
from sklearn.metrics import confusion_matrix #Importa los métodos de cálculo de la matriz de confusión
from sklearn.preprocessing import StandardScaler #Escalamiento decimal
from sklearn.metrics import precision_recall_fscore_support # Importa módulos para calcular métricas a partir de la matriz de confusión
from sklearn.metrics import recall_score, f1_score, confusion_matrix # Importa módulos para calcular métricas a partir de la matriz de confusión
from sklearn.preprocessing import MinMaxScaler # Importa métodos para normalizar usando minMax
from sklearn.model_selection import cross_val_score # Importa módulos para hacer Cross Validation
from sklearn.model_selection import RepeatedStratifiedKFold #Para poder generar folds estratificados en CV
from sklearn.metrics import roc_curve, auc #Curva ROC y área bajo la curva
from sklearn.metrics import roc_auc_score #Para calcular el AUC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder #Para hacer One Hot Encoding
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import statsmodels.api as sm 
import seaborn as sns #Para graficar 



from django.shortcuts import render

class myAppCreateView(generics.ListCreateAPIView):
    queryset = myApp.objects.all()
    serializer_class = myAppSerializer

class myAppUpdateView(generics.RetrieveUpdateDestroyAPIView):
    queryset = myApp.objects.all()
    serializer_class = myAppSerializer

class FileUploadView(APIView):
    def post(self, request):

         # Crear un diccionario para almacenar las imágenes
        images = []
        file = request.FILES['file']

        df = pd.read_csv(file, sep=',', encoding= 'utf-8') # Lee el archivo csv y lo almacena en un dataframe
        df2 = df

        #Exploracion de los datos

        print(df.head()) #Imprime las primeras 5 filas del dataframe
        print(df.shape) #Imprime la cantidad de filas y columnas del dataframe
        print(df.describe()) #Imprime un resumen estadístico de las variables numéricas
        print(df.info()) #Imprime información del dataframe

        #Exploracion de los datos a través de gráficas

        #Matriz de correlacion
        #corr = df.set_index('Diagnosticos_Consulta').corr()
        #sm.graphics.plot_corr(corr, xnames=list(corr.columns))
        #plt.title("Matriz de correlación")
        #buffer = io.BytesIO()
        #plt.savefig(buffer, format='png')
        #buffer.seek(0)
        #image_png = buffer.getvalue()
        #buffer.close()
        #graphic = base64.b64encode(image_png)
        #graphic = graphic.decode('utf-8')
        #image_base64 = "data:image/png;base64," + graphic
#
        ## Agregar objeto de la imagen
        #images.append({
        #    "name": "corr_matrix",
        #    "data": image_base64
        #})

        
        #Histograma de la variable objetivo
        #plt.hist(df.outside)
        #plt.title("Histograma de la variable objetivo")
        #plt.xlabel("Outside")
        #plt.ylabel("Frecuencia")
        #buffer = io.BytesIO()
        #plt.savefig(buffer, format='png')
        #buffer.seek(0)
        #image_png = buffer.getvalue()
        #buffer.close()
        #graphic = base64.b64encode(image_png)
        #graphic = graphic.decode('utf-8')
        #image_base64 = "data:image/png;base64," + graphic
#
        ## Agregar objeto de la imagen
        #images.append({
        #    "name": "histogram",
        #    "data": image_base64
        #})


        plt.figure(figsize=(8, 6))
        df['Sexo'].value_counts().plot(kind='bar')
        plt.title('Recuento de valores en la columna Sexo')
        plt.xlabel('Sexo')
        plt.ylabel('Recuento')
        plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')
        image_base64 = "data:image/png;base64," + graphic
        
        # Agregar objeto de la imagen
        images.append({
            "name": "barplot",
            "data": image_base64
        })

        # Gráfico de líneas para la evolución de 'Presion_Sistolica' y 'Presion_Diastolica'
        plt.figure(figsize=(8, 6))
        df[['Presion_Sistolica', 'Presion_Diastolica']].plot(figsize=(10, 6))
        plt.title('Evolución de la Presión Sistólica y Diastólica')
        plt.xlabel('Índice')
        plt.ylabel('Presión')
        plt.legend()
        plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')
        image_base64 = "data:image/png;base64," + graphic

        # Agregar objeto de la imagen
        images.append({
            "name": "lineplot",
            "data": image_base64
        })


        #Boxplot de variables clinicas
        plt.figure(figsize=(8, 6))
        df.boxplot(column=['Frecuencia_Cardiaca', 'Frecuencia_Respiratoria'])
        plt.title('Boxplot de Variables Clínicas')
        plt.ylabel('Valor')
        plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')
        image_base64 = "data:image/png;base64," + graphic

        # Agregar objeto de la imagen
        images.append({
            "name": "boxplot",
            "data": image_base64
        })

        #Matriz de correlacion entre signos vitales principales QSofa.
        #plt.figure(figsize=(10, 8))
        #correlation_matrix = df[['Temperatura', 'Frecuencia_Cardiaca', 'Frecuencia_Respiratoria',
        #                        'Presion_Sistolica', 'Presion_Diastolica']].corr()
        #plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
        #plt.colorbar()
        #plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=45)
        #plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
        #plt.title('Matriz de Correlación')
        #plt.show()
        #buffer = io.BytesIO()
        #plt.savefig(buffer, format='png')
        #buffer.seek(0)
        #image_png = buffer.getvalue()
        #buffer.close()
        #graphic = base64.b64encode(image_png)
        #graphic = graphic.decode('utf-8')
        #image_base64 = "data:image/png;base64," + graphic
#
        ## Agregar objeto de la imagen
        #images.append({
        #    "name": "corr_matrix",
        #    "data": image_base64
        #})
        

        #Distribucion de la variable objetivo
        plt.figure(figsize=(8, 6))
        df['outside'].plot.hist(density=True, alpha=0.5)
        df['outside'].plot.kde()
        plt.title('Distribución de la Variable Objetivo')
        plt.xlabel('outside')
        plt.ylabel('Densidad')
        plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')
        image_base64 = "data:image/png;base64," + graphic
        
        # Agregar objeto de la imagen
        images.append({
            "name": "distribution",
            "data": image_base64
        })


        #Matriz de dispersion
        # Seleccionar las columnas numéricas relevantes
        #numeric_cols = ['IdCliente', 'Temperatura', 'Frecuencia_Cardiaca', 'Frecuencia_Respiratoria',
                        #'Presion_Sistolica', 'Presion_Diastolica', 'Temperatura_Consulta',
                        #'FC_Consulta', 'FR_Consulta', 'PS_Consulta', 'PD_Consulta', 'outside']

        # Crear una submuestra para una visualización más rápida (opcional)
        #subsample = df.sample(frac=0.2, random_state=42)

        # Crear la matriz de dispersión utilizando seaborn
        #sns.pairplot(subsample[numeric_cols], hue='outside', plot_kws={'alpha': 0.6})
        #plt.title('Matriz de Dispersión')
        #plt.show()
        #buffer = io.BytesIO()
        #plt.savefig(buffer, format='png')
        #buffer.seek(0)
        #image_png = buffer.getvalue()
        #buffer.close()
        #graphic = base64.b64encode(image_png)
        #graphic = graphic.decode('utf-8')
        #image_base64 = "data:image/png;base64," + graphic

        # Agregar objeto de la imagen
        #images.append({
            #"name": "scatter_matrix",
            #"data": image_base64
        #})

        #Distribucion de la Frecuencia Cardiaca por Categoria Variable Objetivo Outside
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, x='outside', y='Frecuencia_Cardiaca')
        plt.title('Distribución de Frecuencia Cardiaca por Categoría de outside')
        plt.xlabel('outside')
        plt.ylabel('Frecuencia Cardiaca')
        plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')
        image_base64 = "data:image/png;base64," + graphic

        # Agregar objeto de la imagen
        images.append({
            "name": "violinplot",
            "data": image_base64
        })

        #Distribucion de la Frecuencia Respiratoria por Categoria Variable Objetivo Outside
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, x='outside', y='Frecuencia_Respiratoria')
        plt.title('Distribución de Frecuencia Respiratoria por Categoría de outside')
        plt.xlabel('outside')
        plt.ylabel('Frecuencia Respiratoria')
        plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')
        image_base64 = "data:image/png;base64," + graphic

        # Agregar objeto de la imagen
        images.append({
            "name": "violinplot2",
            "data": image_base64
        })

        #Distribucion de la Presion Sistolica por Categoria Variable Objetivo Outside
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, x='outside', y='Presion_Sistolica')
        plt.title('Distribución de Presión Sistólica por Categoría de outside')
        plt.xlabel('outside')
        plt.ylabel('Presión Sistólica')
        plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')
        image_base64 = "data:image/png;base64," + graphic
        
        # Agregar objeto de la imagen
        images.append({
            "name": "violinplot3",
            "data": image_base64
        })

        df['Diagnosticos_Consulta']= df['Diagnosticos_Consulta'].astype(str).str[:4]

        #Se hace el proceso de One Hot Encoding para las variables categoricas

        one_hot_encoded_data = pd.get_dummies(df, columns=['Diagnosticos_Consulta'])

        # Eliminar las columnas que no se van a utilizar

        one_hot_encoded_data = one_hot_encoded_data.drop(['IdCliente', 'Sexo', 'IdAtencion', 'IdConsulta', 'Temperatura', 'Frecuencia_Cardiaca', 'Frecuencia_Respiratoria', 'Presion_Sistolica', 'Presion_Diastolica', 'Diagnosticos_Ingreso', 'Temperatura_Consulta', 'FC_Consulta', 'FR_Consulta', 'PS_Consulta', 'PD_Consulta', 'DesEnfermedadActual', 'DesPlanYConcepto'], axis=1)
        one_hot_encoded_data = one_hot_encoded_data.dropna()

        # Cambiar los valores de la columna "outside"
        df['outside'] = df['outside'].replace({0.0: 0, 1.1: 1})

        # Separar la columna 'outside' como variable objetivo y las demás columnas como variables predictoras
        X = one_hot_encoded_data.drop('outside', axis=1)
        y = one_hot_encoded_data['outside']

        # Dividir el conjunto de datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear y entrenar el modelo de regresión logística con un mayor número máximo de iteraciones
        model_logistic = LogisticRegression()
        model_logistic.fit(X_train, y_train)

        # Realizar predicciones con el modelo de regresión logística
        y_pred = model_logistic.predict(X_test)

        # Calcular las metricas del modelo de regresión logística
        accuracy_logistic = accuracy_score(y_test, y_pred)
        recall_logistic = recall_score(y_test, y_pred)
        f1_logistic = f1_score(y_test, y_pred)
        confusion_matrix_logistic = confusion_matrix(y_test, y_pred)
        print("Precision del modelo de regresión logística:", accuracy_logistic)
        print("Recall del modelo de regresión logística:", recall_logistic)
        print("Puntuación F1 del modelo de regresión logística:", f1_logistic)
        print("Matriz de confusión del modelo de regresión logística:")
        print(confusion_matrix_logistic)

        # Agregar imagen de resultados de regresión logística
        buffer = io.BytesIO()
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_logistic, annot=True, fmt='d', cmap='Blues')
        plt.title("Matriz de Confusión - Regresión Logística\nPrecisión: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(accuracy_logistic, recall_logistic, f1_logistic))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')
        image_base64 = "data:image/png;base64," + graphic

        images.append({
            "name": "confusion_matrix_logistic",
            "data": image_base64
        })


        # Crear y entrenar el modelo de árbol de decisión
        model_tree = DecisionTreeClassifier()
        model_tree.fit(X_train, y_train)

        # Realizar predicciones con el modelo de árbol de decisión
        y_pred = model_tree.predict(X_test)

        # Calcular las metricas del modelo de árbol de decisión
        accuracy_tree = accuracy_score(y_test, y_pred)
        recall_tree = recall_score(y_test, y_pred)
        f1_tree = f1_score(y_test, y_pred)
        confusion_matrix_tree = confusion_matrix(y_test, y_pred)
        print("Precisión del modelo de árbol de decisión:", accuracy_tree)
        print("Recall del modelo de árbol de decisión:", recall_tree)
        print("Puntuación F1 del modelo de árbol de decisión:", f1_tree)
        print("Matriz de confusión del modelo de árbol de decisión:")
        print(confusion_matrix_tree)

        # Agregar imagen de resultados de árbol de decisión
        buffer = io.BytesIO()
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_tree, annot=True, fmt='d', cmap='Blues')
        plt.title("Matriz de Confusión - Árbol de Decisión\nPrecisión: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(accuracy_tree, recall_tree, f1_tree))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')
        image_base64 = "data:image/png;base64," + graphic

        images.append({
            "name": "confusion_matrix_decision_tree",
            "data": image_base64
        })


        # Crear y entrenar el modelo SVM
        # model = SVC()
        # model.fit(X_train, y_train)

        # Realizar predicciones con el modelo SVM
        # y_pred = model.predict(X_test)


        # Calcular la precisión del modelo SVM
        # accuracy = accuracy_score(y_test, y_pred)
        # print("Precisión del modelo:", accuracy)

        # Agregar imagen curva ROC que compara los modelos de regresión logística y árbol de decisión
        buffer = io.BytesIO()
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--')
        # Calcular la curva ROC del modelo de regresión logística
        y_pred_proba = model_logistic.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label='Regresión Logística')
        # Calcular la curva ROC del modelo de árbol de decisión
        y_pred_proba = model_tree.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label='Árbol de Decisión')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curva ROC')
        plt.legend(loc='best')
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')
        image_base64 = "data:image/png;base64," + graphic

        images.append({
            "name": "roc_curve",
            "data": image_base64
        })


        # Devuelve el diccionario de imágenes en base64
        response_data = {
            'images': images
        }

        return Response(response_data)