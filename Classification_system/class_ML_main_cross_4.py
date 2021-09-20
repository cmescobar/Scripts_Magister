# Este script incorpora la posibilidad de poder obtener los indices de train y test 
# de forma externa
__author__ = 'Christian Escobar Arce'

import os, gc, pickle
import numpy as np
import tensorflow as tf
import soundfile as sf
from ast import literal_eval
from datetime import datetime
from respiratory_sound_classification.paper_CNN_models import segnet_based_6_7, \
    definitive_segnet_based
from respiratory_sound_classification.respiratory_sound_management import \
    get_ML_datareg, get_model_data_idxs, get_training_weights, train_test_filebased, get_ML_data
from respiratory_sound_classification.classification_methods import crossval_results, events_eval_results


# Definición de la carpeta con la base de datos
db_folder = 'unpreprocessed_signals'
db_original = 'C:/Users/Chris/Desktop/Scripts_Magister/Respiratory_Sound_Database/audio_and_txt_files'



###############       Definición de parámetros       ###############

classifier_list = ['mlp', 'knn', 'svm']
objective_list = ['crackle', 'wheeze']
knn_param_list = [3, 5]
svm_param_list = ['rbf', 'poly']
mlp_param_list = [True]

# Carpeta a guardar
filepath_to_save = 'respiratory_sound_classification/Results v2'


# Parámetros de los espectrogramas generales
N = 1024
noverlap = int(0.9 * N)
spec_params = {'N': N, 'noverlap': noverlap, 'window': 'hann', 
               'padding': 0, 'repeat': 0}

# Parámetros MFCC
mfcc_params = {'n_mfcc': 50, 'n_filters': 50, 'spec_params': spec_params,
               'freq_lim': 2000, 'norm_filters': True, 'power': 2}
lfcc_params = {'n_mfcc': 50, 'n_filters': 50, 'spec_params': spec_params,
               'freq_lim': 2000, 'norm_filters': True, 'power': 2}
energy_params = {'spec_params': spec_params, 'fmin': 0, 'fmax': 1000, 
                 'fband': 20}

# Parámetros generales
clean_params = {'tol': 1e-5, 'show': True}
sel_params = {'n_features': 50, 'show': True}





###############       Rutina       ###############

# Nombre de los datos
filenames = [f'{db_folder}/{i[:-4]}' for i in os.listdir(db_folder) 
             if i.endswith('.wav')]

with open('respiratory_sound_classification/k-fold_groups.txt', 'r', encoding='utf8') as file:
    patient_groups = literal_eval(file.readline())


# Obteniendo los datos de entrenamiento
X_data, Y_wheeze, Y_crackl, patient_register = \
        get_ML_datareg(filenames, spec_params=spec_params, mfcc_params=mfcc_params, 
                       lfcc_params=lfcc_params, energy_params=energy_params)


for classify_method in classifier_list:    
    for objective_label in objective_list:
        
        # Definición de la característica a iterar
        if classify_method == 'mlp':
            feature_iter = mlp_param_list
            
        elif classify_method == 'knn':
            feature_iter = knn_param_list
        
        elif classify_method == 'svm':
            feature_iter = svm_param_list
            
        for param_i in feature_iter:
            # Definición del inicio del tiempo
            init_time = datetime.now()
            
            # Parámetros de clasificadores
            class_svm = {'classifier': 'svm', 'kernel': param_i}
            class_knn = {'classifier': 'knn', 'k_neigh': param_i}
            mlp_params = {'optimizer': 'Adam', 'loss': 'binary_crossentropy',
                        'batch_size': None, 'epochs': 30, 'verbose': 1, 
                        'metrics': ['accuracy', tf.keras.metrics.Recall(), 
                                    tf.keras.metrics.Precision()],
                        'out_layer': 'softmax', 'preprocessing': param_i}


            # Selección del clasificador
            if classify_method == 'svm':
                class_params = class_svm
                experiment_type = 'ML'

            elif classify_method == 'knn':
                class_params = class_knn
                experiment_type = 'ML'

            elif classify_method == 'mlp':
                class_params = None
                experiment_type = 'NN-MLP'

            else:
                raise Exception('Opción classifier no válida')


            # Selección de la etiqueta objetivo
            if objective_label == 'crackle':
                # Obtener
                confmat_list, accuracy_list = \
                        crossval_results(X_data, Y_wheeze, 
                                         patient_groups=patient_groups, 
                                         patient_register=patient_register, 
                                         experiment_type=experiment_type, 
                                         clean_params=clean_params, 
                                         sel_params=sel_params, 
                                         class_params=class_params,
                                         mlp_params=mlp_params,
                                         kfold=4)
            
            elif objective_label == 'wheeze':
                # Obtener
                confmat_list, accuracy_list = \
                        crossval_results(X_data, Y_wheeze, 
                                         patient_groups=patient_groups, 
                                         patient_register=patient_register, 
                                         experiment_type=experiment_type, 
                                         clean_params=clean_params, 
                                         sel_params=sel_params, 
                                         class_params=class_params,
                                         mlp_params=mlp_params,
                                         kfold=4)

            else:
                raise Exception('Opción objective_label no válida')



            ###############       Registro de los resultados       ###############

            # Definición del nombre del registro
            if classify_method == 'knn':
                filename_to_save = f"{class_params['classifier']}_{class_params['k_neigh']}"
            elif classify_method == 'svm':
                filename_to_save = f"{class_params['classifier']}_{class_params['kernel']}"
            elif classify_method == 'mlp':
                filename_to_save = f"mlp_{mlp_params['preprocessing']}"
                

            # Guardando el modelo
            if db_folder == 'preprocessed_signals':
                db_to = 'preprocessed'
            elif db_folder == 'unpreprocessed_signals':
                db_to = 'unpreprocessed'
            else:
                db_to = 'REVISAR'


            # Guardando los resultados
            txt_name = f'{filepath_to_save}/{filename_to_save}_{objective_label}_{db_to}_results.txt'
            with open(txt_name, 'w', encoding='utf8') as file:
                file.write(f'{confmat_list}')

            
            # Definición del tiempo final
            print(f'Duracion del entrenamiento: {datetime.now() - init_time} hrs.')
            
            
            # Eliminando las variables registradas que no se referencian en memoria
            print("Recolectando registros de memoria sin uso...")
            n = gc.collect()
            print("Número de objetos inalcanzables recolectados por el GC:", n)
            print("Basura incoleccionable:", gc.garbage)
