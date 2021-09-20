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
    get_ML_data_oncycles, get_model_data_idxs, get_training_weights, train_test_filebased, get_ML_data
from respiratory_sound_classification.classification_methods import events_eval_results, \
    conditioning_features_train_opt


# Definición de la carpeta con la base de datos
db_folder = 'unpreprocessed_signals'
db_original = 'C:/Users/Chris/Desktop/Scripts_Magister/Respiratory_Sound_Database/audio_and_txt_files'



###############       Definición de parámetros       ###############

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

# Método para los oncycle
oncycle = True





###############       Rutina       ###############

# Definición de la carpeta a buscar los archivos
train_test_folder = '/'.join(db_original.split('/')[:-1])
file_traintest = f'{train_test_folder}/ICBHI_challenge_train_test.txt'


# Definición de las listas de entrenamiento y testeo
train_list, test_list = \
        train_test_filebased(file_traintest, db_original, db_folder)


# Obteniendo los datos de entrenamiento
if oncycle:
    X_train, Y_wheeze_tr, Y_crackl_tr = \
            get_ML_data_oncycles(train_list, mfcc_params=mfcc_params, 
                        lfcc_params=lfcc_params, energy_params=energy_params)
else:
    X_train, Y_wheeze_tr, Y_crackl_tr = \
            get_ML_data(train_list, spec_params=spec_params, mfcc_params=mfcc_params, 
                        lfcc_params=lfcc_params, energy_params=energy_params)

        
# Obtención de los parámetros para el sistema crackle
params_crackl = \
    conditioning_features_train_opt(X_train, Y_crackl_tr, 
                                    clean_params=clean_params, 
                                    sel_params=sel_params)
    
# Obtención de los parámetros para el sistema wheeze
params_wheeze = \
    conditioning_features_train_opt(X_train, Y_wheeze_tr, 
                                    clean_params=clean_params, 
                                    sel_params=sel_params)
    
# Registrando
save_fold = 'respiratory_sound_classification'
db_to = db_folder.split('_')[0]

if oncycle:
    with open(f'{save_fold}/crackle_features_params_{db_to}_oncycle.txt', 'w', encoding='utf8') as file:
        file.write(f'{params_crackl}')
        
    with open(f'{save_fold}/wheeze_features_params_{db_to}_oncycle.txt', 'w', encoding='utf8') as file:
        file.write(f'{params_wheeze}')
else:
    with open(f'{save_fold}/crackle_features_params_{db_to}.txt', 'w', encoding='utf8') as file:
        file.write(f'{params_crackl}')
        
    with open(f'{save_fold}/wheeze_features_params_{db_to}.txt', 'w', encoding='utf8') as file:
        file.write(f'{params_wheeze}')

