import os
import numpy as np
from io import StringIO
from .models import Pruebas

import pandas as pd
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,  StandardScaler, FunctionTransformer, QuantileTransformer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.figure_factory as ff
import plotly.graph_objects as go



# Crear un modelo Keras como clasificador para usar con sklearn
class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  # Cambiar a softmax para múltiples clases
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X, y):
        self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        return self.model.predict(X)


def procesar(input_values):
    lista=[]
    procesados = set()
    print(f"Valores ingresados: {input_values}")
    for value in input_values:
        if value in procesados:
            print(f"El ID {value} ya ha sido procesado. Se omite este análisis.")
            continue
        pruebas = Pruebas.objects.filter(id=value)
        if pruebas.exists():
                print(f"Se encontraron {pruebas.count()} archivos para ID: {value}")
                procesados.add(value)
                for prueba in pruebas:
                    archivo_path = prueba.archivo.path
                    print(f"Intentando acceder al archivo: {archivo_path}")
                    if os.path.exists(archivo_path):
                        print(f"Archivo encontrado en la ruta: {archivo_path}")
                        try:
                            with open(archivo_path, 'r') as file:
                                contenido = file.read()
                                data = StringIO(contenido)
                                df = pd.read_table(data)
                                
                                if 'Unnamed: 0' in df.columns:
                                    df = df.drop('Unnamed: 0', axis=1)
                                    lista.append(df)
                                        
                                    
                                #print(df)
                                #print(f"{type(df)}")
                        except Exception as e:
                            print(f"Error al leer el archivo para ID {value}: {e}")
                    else:
                        print(f"El archivo no existe en la ruta: {archivo_path}")
        else:
                print(f"No se encontraron archivos para ID: {value}")
    
    print(len(lista))
    almacenando=np.concatenate(lista, axis=1)
    print (almacenando.shape)
    print (type(almacenando))
 
    almacenando = np.char.replace(almacenando.astype(str), ',', '.')
    if almacenando.dtype != np.float64:
        print("")
    try:
        almacenando = almacenando.astype(np.float64)
        print("")
    except Exception as e:
        print("")
    Gmax=almacenando.max(axis=0)
    Gmin=almacenando.min(axis=0)
    print("-----Gmax-----")
    print((Gmax.shape))
    print("-----Gmin-----")
    print((Gmin.shape))
    print("-----Df-----")
    DfGmaxGmin=Gmax-Gmin
    print(DfGmaxGmin.shape)
    print(DfGmaxGmin)
    num_sensores = 8
    datosDf=[]
    for i in range(0, len(DfGmaxGmin), num_sensores):
        datos_sensores = DfGmaxGmin[i:i+num_sensores]
        id_value = input_values[i // num_sensores] if (i // num_sensores) < len(input_values) else None
        pruebas_filtradas = Pruebas.objects.filter(id=id_value)
    
        for prueba_filtrada in pruebas_filtradas:
            diagnostico_valor = prueba_filtrada.diagnostico
            estado_canino = 0 if diagnostico_valor == 0 else 1
            #datos_sensores = np.append(datos_sensores, estado_canino)

            print(f"Sensores {i // num_sensores + 1} para ID {id_value}: {datos_sensores}")
            print(f"Procesando datos para el id: {prueba_filtrada.id} - Estado Canino: {estado_canino}")
        if estado_canino is not None:
            if len(datos_sensores) == num_sensores:
                datos_sensores = np.append(datos_sensores, estado_canino)  
                print(f"sensores {i // num_sensores + 1} para ID {id_value}: {datos_sensores}")       
                datosDf.append(datos_sensores)
            else:
                print(f" {num_sensores} datos de sensores {len(datos_sensores)}.")
        else:
            print(f"No se pudo determinar el estado del canino para ID {id_value}")
        #una consulta por filter pasar la lista {input_values} rrecorrer en el mismo for 
        #se va traer el campo diagnostico y lo guardo en una variable
        #crear un if para clasificar si tiene diagnostico 
        #guardar en una variable un numero ya sea que 0 sean enfermos y 1 sanos 
        #agregar un ultimo elemento a lista datos_sensores debe     quedar una lista con nueve datos
        
        #convertirlo a un numpy array
    datosDf = np.array(datosDf)    
    print(f" {type(datosDf)}")
    # print("-----Datos de Sensores------")
    # print(datosDf)

# Convertir numpy.ndarray a lista de listas
    datosDF_list = datosDf.tolist()
    print(datosDF_list)

    return datosDf

def entrenamientoCaracteristicas(input_values):
    # Convertir a DataFrame si es necesario
    if isinstance(input_values, np.ndarray):
        data = pd.DataFrame(input_values)

        # Asegurarse de que no hay valores nulos
        data = data.dropna()

        # Separar características (X) y etiquetas (y)
        X = data.iloc[:, :-1]  # Todas las columnas excepto la última
        y = data.iloc[:, -1]   # Última columna
    return X, y

def pca(X, y):
    """
    Realiza el análisis de componentes principales (PCA) y genera un gráfico interactivo con los índices de los puntos visibles.

    Argumentos:
        data (numpy.ndarray o pandas.DataFrame): Datos de entrada.

    Retorna:
        str: Gráfico en formato HTML.
    """

    # Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Aplicar PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Crear un DataFrame para los resultados del PCA
    pca_results = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_results["Label"] = y.values
    pca_results["Index"] = pca_results.index  # Agregar índice como columna

    # Convertir las etiquetas a tipo categórico para que Plotly use colores discretos
    pca_results["Label"] = pca_results["Label"].astype(str)

    # Mapear colores específicos para cada etiqueta
    color_map = {"0.0": "red", "1.0": "green"}  # Etiqueta 0 = rojo, Etiqueta 1 = verde

    # Crear el gráfico interactivo con Plotly
    fig = px.scatter(
        pca_results,
        x="PC1",
        y="PC2",
        color="Label",
        color_discrete_map=color_map,  # Mapear colores definidos
        title="Visualización de datos con PCA",
        labels={"PC1": "Componente Principal 1", "PC2": "Componente Principal 2", "Label": "Etiqueta"},
        hover_data={"PC1": True, "PC2": True, "Label": True, "Index": True}  # Mostrar índice en hover
    )

    # Agregar texto con los índices directamente en el gráfico
    fig.update_traces(
        text=pca_results["Index"],
        textposition="top center",  # Posición del texto sobre el punto
        marker=dict(size=10)  # Tamaño del marcador
    )

    # Ajustar la posición de la leyenda a la parte superior derecha
    fig.update_layout(
        legend=dict(
            title="Etiqueta",
            x=1,  # Posición en el eje horizontal (1 = extremo derecho)
            y=1,  # Posición en el eje vertical (1 = extremo superior)
            xanchor="right",
            yanchor="top"
        ),
        title=dict(
            text="Visualización de datos con PCA",
            x=0.5,  # Centrar el título
            xanchor="center",
            yanchor="top"
        )
    )

    # Convertir el gráfico a HTML
    graph_html = fig.to_html(full_html=False)
    return graph_html, X_pca

## Metodos de escalado

def escaladoMinMax(X):
    minMax=MinMaxScaler()
    minMaxScaler= minMax.fit_transform(X)
    return minMaxScaler

def escaladoStandard(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def escaladoFunctionTransformer(X):
    log_scaler = FunctionTransformer(np.log1p, validate=True)
    X_log_scaled = log_scaler.fit_transform(X)
    return X_log_scaled

def escaladoQuantileTransformer(X):
    quantile_transformer_normal = QuantileTransformer(output_distribution='normal', random_state=42)
    data_scaled_normal = quantile_transformer_normal.fit_transform(X)
    return data_scaled_normal

# ## Clasificador
# def clasificador(escalado, y):
#         # Crear el clasificador con Keras
#     keras_clf = KerasClassifier(input_dim=escalado.shape[1])

#     # Validación cruzada con StratifiedKFold
#     kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=25)
#     fold = 1
#     all_accuracies = []

#     for train_index, test_index in kf.split(escalado, y):
#         X_train, X_test = escalado[train_index], escalado[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#         # Entrenar el modelo
#         keras_clf.fit(X_train, y_train)

#         # Predecir en el conjunto de prueba
#         y_pred = keras_clf.predict(X_test)

#         # Calcular métricas
#         accuracy = accuracy_score(y_test, y_pred)
#         all_accuracies.append(accuracy)

#         # Matriz de confusión
#         conf_matrix = confusion_matrix(y_test, y_pred)

#         # Visualizar la matriz de confusión
#         plt.figure(figsize=(6, 4))
#         sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.title(f'Matriz de Confusión - Fold {fold}')
#         plt.show()

#         fold += 1

#     # Imprimir resultados generales
#     print("Accuracy por cada fold:", all_accuracies)
#     print("Accuracy promedio:", np.mean(all_accuracies))

def clasificador(escalado, y):
    # Crear el clasificador con Keras
    keras_clf = KerasClassifier(input_dim=escalado.shape[1])

    # Validación cruzada con StratifiedKFold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=25)
    fold = 1
    all_accuracies = []

    # Lista para almacenar las figuras de matrices de confusión
    conf_matrices_figures = []

    for train_index, test_index in kf.split(escalado, y):
        X_train, X_test = escalado[train_index], escalado[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Entrenar el modelo
        keras_clf.fit(X_train, y_train)

        # Predecir en el conjunto de prueba
        y_pred = keras_clf.predict(X_test)

        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        all_accuracies.append(accuracy)

        # Matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Crear figura de la matriz de confusión con Plotly
        fig = ff.create_annotated_heatmap(
            z=conf_matrix,
            x=["Predicted 0", "Predicted 1"],
            y=["True 0", "True 1"],  # Cambiar el orden para que True 0 esté arriba
            colorscale="Blues",
            showscale=False  # Ocultar la barra de color
        )

        # Calcular métricas generales
        accuracy_text = f"Accuracy del Fold {fold}: {accuracy:.2f}"
        all_accuracies_text = f"Accuracies por Fold: {all_accuracies}"

        # Agregar título y ajustar layout
        fig.update_layout(
            title=dict(
                text=f'Matriz de Confusión - Fold {fold}',
                x=0.5,  # Centrar el título
                xanchor='center',
                font=dict(size=12)  # Tamaño del título ajustado
            ),
            xaxis=dict(title='', tickfont=dict(size=10)),  # Reducir el tamaño de las etiquetas
            yaxis=dict(title='', tickfont=dict(size=10)),  # Reducir el tamaño de las etiquetas
            margin=dict(l=10, r=10, t=30, b=60),  # Márgenes ajustados
            autosize=False,
            width=250,  # Ajustar ancho del gráfico
            height=250,  # Ajustar altura del gráfico
        )
        
        # Agregar texto de accuracy debajo
        fig.add_annotation(
            text=accuracy_text,
            xref="paper", yref="paper",
            x=0.5, y=-0.3,  # Coordenadas (centrado, debajo de la gráfica)
            showarrow=False,
            font=dict(size=10, color="black")
        )

        # Agregar texto de all_accuracies debajo del gráfico
        fig.add_annotation(
            text=all_accuracies_text,
            xref="paper", yref="paper",
            x=0.5, y=-0.5,  # Coordenadas (centrado y más abajo)
            showarrow=False,
            font=dict(size=10, color="black")
        )

        # Almacenar la figura
        conf_matrices_figures.append(fig)

        fold += 1

    # Imprimir resultados generales
    print("Accuracy por cada fold:", all_accuracies)
    print("Accuracy promedio:", np.mean(all_accuracies))
    promedio = float(np.mean(all_accuracies))*100
    
    # Retornar las figuras para graficarlas o guardarlas
    return conf_matrices_figures, promedio
