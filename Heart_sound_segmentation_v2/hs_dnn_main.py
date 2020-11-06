__author__ = 'Christian Escobar Arce'

import os
import tensorflow as tf
from heart_sound_DNN_models import model_2, model_2_2, model_2_3, model_2_4, \
    model_2_5, model_2_6, model_3
from heart_sound_physionet_management import get_model_data


###############       Definición de parámetros       ###############

# Definición de la GPU con la que se trabajará
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Parámetros de get_model_data
test_size = 0.1
seed_split = 0
snr_list = [0, 1, 5, 10]
ind_beg = 0
ind_end = None
N = 128
step = 16
padding_value = 2

apply_bpfilter = True
bp_parameters = [40, 60, 230, 250]

append_envelopes = False
homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
hilbert_bool = False
simplicity_dict = {'N': 64, 'noverlap': 32, 'm': 10, 'tau': 1}
vfd_dict = {'N': 64, 'noverlap': 32, 'kmin': 2, 'kmax': 2, 'step_size_method': 'unit'}
wavelet_dict = {'wavelet': 'db4', 'levels': [2,3,4], 'start_level': 1, 'end_level': 5}
spec_track_dict = {'freq_obj': [100, 120], 'N': 128, 'noverlap': 100, 'padding': 0, 
                   'repeat': 0, 'window': 'hann'}


# Parámetros de Red neuronal
validation_split = 0.1
batch_size = 70
epochs = 30
model_name = 'Model_2_6'

###############       Definición de parámetros       ###############

### Checkeo de versiones ###
for filename in os.listdir('Models/'):
    if model_name in filename:
        check_pass = input(f'Ya existe una simulación de {model_name}. ¿Continuamos?\n' 
                           '[1] Si\n[2] No\nRespuesta: ')
        
        if check_pass == '1':
            break
        elif check_pass == '2':
            print('Simulación terminada.\n')
            exit()
        else:
            print('Opción no válida.')
            exit()


# Guardando los parámetros especificados para get_model_data
get_model_data_info = {'test_size': test_size, 'seed_split': seed_split,
                       'snr_list': snr_list, 'ind_beg': ind_beg, 
                       'ind_end': ind_end, 'N': N, 'noverlap': N - step,
                       'padding_value': padding_value, 
                       'append_envelopes': append_envelopes,
                       'apply_bpfilter': apply_bpfilter,
                       'bp_parameters': bp_parameters, 
                       'homomorphic_dict': homomorphic_dict, 
                       'hilbert_bool': hilbert_bool, 
                       'simplicity_dict': simplicity_dict, 
                       'vfd_dict': vfd_dict, 'wavelet_dict': wavelet_dict,
                       'spec_track_dict': spec_track_dict}

with open(f'Models/{model_name}-get_model_data_params.txt', 'w', encoding='utf8') as file:
    file.write(f'{get_model_data_info}')


# Definición de la carpeta con la base de datos
db_folder = 'PhysioNet 2016 CINC Heart Sound Database'

# Obtener los nombres de los archivos
filenames = [f'{db_folder}/{name[:-4]}' for name in os.listdir(db_folder) 
             if name.endswith('.wav')]


# Definición de los datos de entrenamiento y testeo
X_train, X_test, Y_train, Y_test = \
    get_model_data(db_folder, test_size=test_size, seed_split=seed_split, 
                   snr_list=snr_list, ind_beg=ind_beg, ind_end=ind_end, N=N, 
                   noverlap=N-step, padding_value=padding_value, 
                   activation_percentage=0.5, append_audio=True, 
                   append_envelopes=append_envelopes, 
                   apply_bpfilter=apply_bpfilter, bp_parameters=bp_parameters, 
                   homomorphic_dict=homomorphic_dict, hilbert_bool=hilbert_bool,
                   simplicity_dict=simplicity_dict, 
                   vfd_dict=vfd_dict, wavelet_dict=wavelet_dict, 
                   spec_track_dict=spec_track_dict)


# Imprimiendo la dimensión de los archivoss
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# Y transformando a tensor
X_train_tens  = tf.convert_to_tensor(X_train, dtype='float32')
s1_train_tens = tf.convert_to_tensor(Y_train[:,0], dtype='float32')
s2_train_tens = tf.convert_to_tensor(Y_train[:,1], dtype='float32')

X_test_tens  = tf.convert_to_tensor(X_test, dtype='float32')
s1_test_tens = tf.convert_to_tensor(Y_test[:,0], dtype='float32')
s2_test_tens = tf.convert_to_tensor(Y_test[:,1], dtype='float32')


# Print
print(X_train_tens.shape)
print(s1_train_tens.shape)
print(X_test_tens.shape)
print(s1_test_tens.shape)


# Creación del modelo
if model_name == 'Model_2':
    model = model_2(input_shape=(X_train.shape[1], X_train.shape[2]), 
                    padding_value=padding_value, name=model_name)
    
elif model_name == 'Model_2_2':
    model = model_2_2(input_shape=(X_train.shape[1], X_train.shape[2]), 
                      padding_value=padding_value, name=model_name)

elif model_name == 'Model_2_3':
    model = model_2_3(input_shape=(X_train.shape[1], X_train.shape[2]), 
                      padding_value=padding_value, name=model_name)
    
elif model_name == 'Model_2_4':
    model = model_2_4(input_shape=(X_train.shape[1], X_train.shape[2]), 
                      padding_value=padding_value, name=model_name)
    
elif model_name == 'Model_2_5':
    model = model_2_5(input_shape=(X_train.shape[1], X_train.shape[2]), 
                      padding_value=padding_value, name=model_name)

elif model_name == 'Model_2_6':
    model = model_2_6(input_shape=(X_train.shape[1], X_train.shape[2]), 
                      padding_value=padding_value, name=model_name)

elif model_name == 'Model_3':
    model = model_3(input_shape=(X_train.shape[1], X_train.shape[2]), 
                    padding_value=padding_value, name=model_name)



# Compilando las opciones del modelo
model.compile(optimizer='Adam', loss=['binary_crossentropy', 'binary_crossentropy'], 
              metrics=['accuracy'], loss_weights=[1., 1.])

# Mostrando el resumen
model.summary()

# Entrenando
if model_name in ['Model_2', 'Model_2_2', 'Model_2_3', 'Model_2_4', 'Model_2_5', 'Model_2_6']:
    print('\nTraining time\n------------\n')
    history = model.fit(x=X_train_tens, y=[s1_train_tens, s2_train_tens], epochs=epochs, 
                        batch_size=batch_size, verbose=1, validation_split=validation_split)

    print('\nTesting time\n------------\n')
    eval_info = model.evaluate(x=X_test_tens, y=[s1_test_tens, s2_test_tens], verbose=1,
                               return_dict=True)
    
elif model_name in ['Model_3']:
    print('\nTraining time\n------------\n')
    history = model.fit(x=X_train_tens, y=s1_train_tens + s2_train_tens, epochs=epochs, 
                        batch_size=batch_size, verbose=1, validation_split=validation_split)

    # Evaluando
    print('\nTesting time\n------------\n')
    eval_info = model.evaluate(x=X_test_tens, y=s1_test_tens + s2_test_tens, verbose=1,
                               return_dict=True)


# Guardando el modelo
model.save(f'Models/{model_name}.h5')

# Y guardando la información del entrenamiento con el testeo
with open(f'Models/{model_name}.txt', 'w', encoding='utf8') as file:
    file.write(f'{history.history}\n')
    file.write(f'{eval_info}')
