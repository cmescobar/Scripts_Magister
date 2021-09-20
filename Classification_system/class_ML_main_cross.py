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
    get_model_data_idxs, get_training_weights, train_test_filebased, get_ML_data
from respiratory_sound_classification.classification_methods import events_eval_results


# Definición de la carpeta con la base de datos
db_folder = 'unpreprocessed_signals'
db_original = 'C:/Users/Chris/Desktop/Scripts_Magister/Respiratory_Sound_Database/audio_and_txt_files'


# Definición del inicio del tiempo
init_time = datetime.now()


###############       Definición de parámetros       ###############

# Carpeta a guardar
filepath_to_save = 'respiratory_sound_classification'
objective_label = 'crackle'
classify_method = 'mlp'

# Parámetros de clasificadores
class_svm = {'classifier': 'svm', 'kernel': 'rbf'}
class_knn = {'classifier': 'knn', 'k_neigh': 5}
mlp_params = {'optimizer': 'Adam', 'loss': 'binary_crossentropy',
              'batch_size': None, 'epochs': 30, 'verbose': 1, 
              'metrics': ['accuracy', tf.keras.metrics.Recall(), 
                          tf.keras.metrics.Precision()],
              'out_layer': 'softmax', 'preprocessing': True}


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
sel_params = {'n_features': 60, 'show': True}



###############       Rutina       ###############



# Definición de la carpeta a buscar los archivos
train_test_folder = '/'.join(db_original.split('/')[:-1])
file_traintest = f'{train_test_folder}/ICBHI_challenge_train_test.txt'


# Definición de las listas de entrenamiento y testeo
train_list, test_list = \
        train_test_filebased(file_traintest, db_original, db_folder)


# Obteniendo los datos de entrenamiento
X_train, Y_wheeze_tr, Y_crackl_tr = \
        get_ML_data(train_list, spec_params=spec_params, mfcc_params=mfcc_params, 
                    lfcc_params=lfcc_params, energy_params=energy_params)

# Y de testeo
X_test, Y_wheeze_test, Y_crackl_test = \
        get_ML_data(test_list, spec_params=spec_params, mfcc_params=mfcc_params, 
                    lfcc_params=lfcc_params, energy_params=energy_params)


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
    confmat, accuracy, model = \
        events_eval_results(X_train, Y_crackl_tr, X_test, Y_crackl_test, 
                            experiment_type=experiment_type, 
                            clean_params=clean_params, 
                            sel_params=sel_params, 
                            class_params=class_params,
                            mlp_params=mlp_params)

elif objective_label == 'wheeze':
    confmat, accuracy, model = \
        events_eval_results(X_train, Y_wheeze_tr, X_test, Y_wheeze_test, 
                            experiment_type=experiment_type, 
                            clean_params=clean_params, 
                            sel_params=sel_params, 
                            class_params=class_params,
                            mlp_params=mlp_params)

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
    file.write(f'{confmat}\n{accuracy}')

# Guardando el clasificador
if classify_method == 'mlp':
    model_name = f'{filepath_to_save}/{filename_to_save}_{objective_label}_{db_to}.h5'
    tf.keras.models.save_model(model, filepath=model_name)
    
else:
    model_name = f'{filepath_to_save}/{filename_to_save}_{objective_label}_{db_to}.pickle'
    pickle.dump(model, open(model_name, 'wb'))


# Definición del tiempo final
print(f'Duracion del entrenamiento: {datetime.now() - init_time} hrs.')
