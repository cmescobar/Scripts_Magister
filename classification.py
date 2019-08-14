from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pybalu.feature_selection import clean, sfs
from pybalu.feature_transformation import pca
from sklearn import preprocessing, svm
from sklearn.metrics import confusion_matrix
from descriptor_functions import normalize, get_mfcc
from prettytable import PrettyTable
import soundfile as sf
import numpy as np


# Selección del método de clasificación
op = input('Seleccione el método de clasificación\n'
            '[1] - KNN\n'
            '[2] - SVM\n')

# Cantidad de simulaciones por configuración
q_sim = 50
# Cantidad de características a clasificar
q_feats = [3, 5, 7, 9, 11, 13]
# Método de selección de características
sfs_method = 'fisher'
# Definición del test_size
test_size = 0.2

# Cargar datos
healthy_data = np.load('Features_Extracted/Healthy_all_features.npz')
pneumonia_data = np.load('Features_Extracted/Pneumonia_all_features.npz')

# Recuperar los datos
X_1 = healthy_data['features']
X_2 = pneumonia_data['features']

# Definición de las etiquetas
Y = np.array([1] * X_1.shape[0] +
             [2] * X_2.shape[0])

# Uniendo las submatrices
X = np.append(X_1, X_2, axis=0)

# Iterando sobre la cantidad de características
for nsel in q_feats:
    # Definición de lista de accuracies
    accuracy_list = []
    features_list = []
    
    # Definición de la tabla a guardar
    tabla = PrettyTable(['Classifier', 'sfs_method', 'Nsel', 'Features', 
                         'Accuracy'])

    # Iterando para el resumen
    for _ in range(q_sim):

        # Aplicando un stratify
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, 
                                                            test_size=test_size)

        # Aplicando selección de características
        feat_sel = sfs(X_train, Y_train, nsel, show=True, method=sfs_method)
        X_train_2 = X_train[:, feat_sel]
        X_test_2 = X_test[:, feat_sel]
        
        # Agregando a la lista de características
        features_list.append(feat_sel)

        if op == "1":
            # Definición del nombre de la clasificación
            clas_name = 'knn'
            
            # Diseño del clasificador
            knn_classifier = KNeighborsClassifier(n_neighbors=3, algorithm='auto')

            # Ajuste del clasificador
            knn_classifier.fit(X_train_2, Y_train)

            # Realizando la predicción de los datos
            Y_pred = knn_classifier.predict(X_test_2)

            # Calculando el accuracy
            accuracy = sum(Y_pred == Y_test) / len(Y_pred)
            # print(f"accuracy = {accuracy * 100} %")
            accuracy_list.append(accuracy)

            # Agregando los datos a la tabla
            tabla.add_row(['KNN', sfs_method, nsel, feat_sel, accuracy * 100])

        elif op == "2":
            # Definición del nombre de la clasificación
            clas_name = 'svm'
            
            # Diseñar clasificador SVM
            svm_classifier = svm.SVC(kernel='rbf', degree=50, gamma='auto')

            # Entrenando
            svm_classifier.fit(X_train_2, Y_train)

            # Realizando la predicción sobre los datos de testeo
            Y_pred = svm_classifier.predict(X_test_2)

            # Calculando el accuracy
            accuracy = sum(Y_pred == Y_test) / len(Y_pred)
            accuracy_list.append(accuracy)
            
            # Agregando los datos a la tabla
            tabla.add_row(['SVM', sfs_method, nsel, feat_sel, accuracy * 100])
    
    # Una vez terminado, se obtiene una tabla resumen
    resumen = PrettyTable(['Accuracy mean', np.mean(accuracy_list) * 100,
                           'Accuracy sdev', np.std(accuracy_list) * 100])

    # Guardando en el archivo txt
    with open(f'Results/Classification/classification_results.txt', 'a',
              encoding='utf8') as file:
        file.write(f'{tabla.get_string()}\n')
        file.write(resumen.get_string())
        file.write('\n\n\no-------------------------------------------o\n\n\n')
    
    # Guardando los datos, además, en un archivo npz
    filename = f'Results/Classification/{clas_name}_{sfs_method}_{nsel}.npz'
    np.savez(filename, accuracy_list=accuracy_list, features_list=features_list)
    