# Este script incorpora la posibilidad de poder obtener los indices de train y test 
# de forma externa
__author__ = 'Christian Escobar Arce'

import os
import numpy as np
import tensorflow as tf
from heart_sound_DNN_models import model_2_1, model_2_2, model_2_3, model_2_4, \
    model_2_5, model_2_6, model_2_7, model_2_8, model_3, model_4_1, model_4_2, \
    model_4_3, model_4_4, model_5_1, model_5_1_1, model_5_2_1, model_5_2_2, \
    model_5_2_3, model_5_2_4, model_5_2_4_1, model_5_2_5, model_5_2_6, model_5_2_7, \
    model_5_2_8, model_5_2_9, model_6_1, model_6_2, model_6_3, model_2_9, model_6_4, \
    model_5_2_4_2, model_5_2_4_3, model_7_1, model_7_2, model_7_1_2, model_5_2_4_3, \
    model_4_5, model_8_1
from heart_sound_physionet_management import get_model_data, get_model_data_idxs


# Definición de la carpeta con la base de datos
db_folder = 'PhysioNet 2016 CINC Heart Sound Database'

# Obtener los nombres de los archivos
filenames = [f'{db_folder}/{name[:-4]}' for name in os.listdir(db_folder) 
             if name.endswith('.wav')]

# Definición de la cantidad total de archivos de audio
N_data = len(filenames)

# Definición del orden de la lista de archivos a leer
np.random.seed(0)
order_list = np.random.choice(len(filenames), size=len(filenames), replace=False)


# Función que permitirá iterar sobre cada modelo, sin sobrepasar los límites de memoria
def model_train_iteration(model, model_name, index_list):
    # Definición de los datos de entrenamiento
    X_train, Y_train = \
        get_model_data_idxs(db_folder, snr_list=snr_list, index_list=index_list, N=N, 
                            noverlap=N-step, padding_value=padding_value, 
                            activation_percentage=activation_percentage, append_audio=True, 
                            append_envelopes=append_envelopes, 
                            apply_bpfilter=apply_bpfilter, bp_parameters=bp_parameters, 
                            homomorphic_dict=homomorphic_dict, hilbert_dict=hilbert_dict,
                            simplicity_dict=simplicity_dict, 
                            vfd_dict=vfd_dict, wavelet_dict=wavelet_dict, 
                            spec_track_dict=spec_track_dict, spec_energy_dict=spec_energy_dict,
                            append_fft=append_fft, print_indexes=False, return_indexes=True)
    
    # Entrenando
    if model_name in ['Model_2_1', 'Model_2_1_2', 'Model_2_1_no-noise', 'Model_2_1_hyper-noise',
                      'Model_2_2', 'Model_2_3', 'Model_2_4', 'Model_2_5', 'Model_2_6', 'Model_2_7',
                      'Model_2_7_2', 'Model_2_8', 'Model_4_1', 'Model_4_2', 'Model_4_3', 'Model_4_4',
                      'Model_5_1', 'Model_5_1_1', 'Model_5_2_1', 'Model_5_2_2']:
        print('\nTraining time\n------------\n')
        history = model.fit(x=X_train, y=[Y_train[:,0], Y_train[:,1]], epochs=epochs, 
                            batch_size=batch_size, verbose=1, validation_split=validation_split)

    
    elif model_name in ['Model_5_2_3', 'Model_5_2_4', 'Model_5_2_4_1', 'Model_5_2_5', 'Model_5_2_6',
                        'Model_5_2_7', 'Model_5_2_8', 'Model_5_2_9', 'Model_5_2_9_alt', 
                        'Model_5_2_9_alt_2', 'Model_5_2_4_again']:
        print('\nTraining time\n------------\n')
        history = model.fit(x=[X_train[:, :, i] for i in range(X_train.shape[2])], 
                            y=[Y_train[:,0], Y_train[:,1]], epochs=epochs, 
                            batch_size=batch_size, verbose=1, validation_split=validation_split)

    
    elif model_name in ['Model_3']:
        print('\nTraining time\n------------\n')
        history = model.fit(x=X_train, y=Y_train[:,0] + Y_train[:,1], epochs=epochs, 
                            batch_size=batch_size, verbose=1, validation_split=validation_split)

            
    elif model_name in ['Model_6_1', 'Model_6_1_noised', 'Model_6_1_onechannel', 
                        'Model_6_2', 'Model_6_3', 'Model_6_4_onechannel', 
                        'Model_6_4_typicalchannels', 'Model_6_1_pro_total',
                        'Model_8_1_pro_total']:
        print('\nTraining time\n------------\n')
        # Definición de las etiquetas de entrenamiento
        y1 = Y_train[:, :, 0]
        y2 = Y_train[:, :, 1]
        y0 = np.ones(Y_train.shape[:-1]) - y1 - y2

        # Acondicionando las etiquetas para entrenar el modelo
        y0 = np.expand_dims(y0, -1)     # Segmentos intermedios
        y1 = np.expand_dims(y1, -1)     # S1
        y2 = np.expand_dims(y2, -1)     # S2

        # Concatenando las etiquetas para el modelo
        y_to = np.concatenate((y0, y1, y2), axis=-1)
        
        # Entrenando
        history = model.fit(x=X_train, y=y_to, epochs=epochs, batch_size=batch_size, 
                            verbose=1, validation_split=validation_split)

    
    elif model_name in ['Model_7_1_pro_total', 'Model_7_2_pro_total', 'Model_7_1_pro_total']:
        print('\nTraining time\n------------\n')
        # Definición de las etiquetas de entrenamiento
        y1 = Y_train[:, :, 0]
        y2 = Y_train[:, :, 1]
        y0 = np.ones(Y_train.shape[:-1]) - y1 - y2

        # Acondicionando las etiquetas para entrenar el modelo
        y0 = np.expand_dims(y0, -1)     # Segmentos intermedios
        y1 = np.expand_dims(y1, -1)     # S1
        y2 = np.expand_dims(y2, -1)     # S2

        # Concatenando las etiquetas para el modelo
        y_to = np.concatenate((y0, y1, y2), axis=-1)
        
        # Entrenando
        history = model.fit(x=[X_train[:, :, i] for i in range(X_train.shape[2])], 
                            y=y_to, epochs=epochs, batch_size=batch_size, 
                            verbose=1, validation_split=validation_split)

    
    elif model_name in ['Model_7_1_2_pro_total']:
        print('\nTraining time\n------------\n')
        # Definición de las etiquetas de entrenamiento
        y1 = Y_train[:, :, 0] + Y_train[:, :, 1]
        y0 = np.ones(Y_train.shape[:-1]) - y1

        # Acondicionando las etiquetas para entrenar el modelo
        y0 = np.expand_dims(y0, -1)     # Segmentos intermedios
        y1 = np.expand_dims(y1, -1)     # S1 y S2
        
        # Concatenando las etiquetas para el modelo
        y_to = np.concatenate((y0, y1), axis=-1)
        
        # Entrenando
        history = model.fit(x=[X_train[:, :, i] for i in range(X_train.shape[2])], 
                            y=y_to, epochs=epochs, batch_size=batch_size, 
                            verbose=1, validation_split=validation_split)
    
    
    # No seguro de que esté bien
    elif model_name in ['Model_2_9', 'Model_4_5_pro_total', 'Model_4_5_pro']:
        print('\nTraining time\n------------\n')
        # Definición de las etiquetas de entrenamiento
        y1 = Y_train[:, 0]
        y2 = Y_train[:, 1]
        y0 = np.ones(Y_train.shape[:-1]) - y1 - y2

        # Acondicionando las etiquetas para entrenar el modelo
        y0 = np.expand_dims(y0, -1)     # Segmentos intermedios
        y1 = np.expand_dims(y1, -1)     # S1
        y2 = np.expand_dims(y2, -1)     # S2

        # Concatenando las etiquetas para el modelo
        y_to = np.concatenate((y0, y1, y2), axis=-1)
        
        # Entrenando
        history = model.fit(x=X_train, y=y_to, epochs=epochs, batch_size=batch_size, 
                            verbose=1, validation_split=validation_split)
        
    
    elif model_name in ['Model_5_2_4_2', 'Model_5_2_4_2_pro', 'Model_5_2_4_2_pro_total']:
        print('\nTraining time\n------------\n')
        # Definición de las etiquetas de testeo
        y1 = Y_train[:, 0]
        y2 = Y_train[:, 1]
        y0 = np.ones(Y_train.shape[0]) - y1 - y2

        # Acondicionando las etiquetas para testear el modelo
        y0 = np.expand_dims(y0, -1)     # Segmentos intermedios
        y1 = np.expand_dims(y1, -1)     # S1
        y2 = np.expand_dims(y2, -1)     # S2

        # Concatenando las etiquetas para el modelo
        y_to = np.concatenate((y0, y1, y2), axis=-1)

        # y_to = Y_train[:, 0] + 2 * Y_train[:, 1]
        # y_to = y_to.astype(int)
        
        # Entrenando
        history = model.fit(x=[X_train[:, :, i] for i in range(X_train.shape[2])], 
                            y=y_to, epochs=epochs, batch_size=batch_size, 
                            verbose=1, validation_split=validation_split)
    
    elif model_name in ['Model_5_2_4_3_pro_total']:
        print('\nTraining time\n------------\n')
        # Definición de las etiquetas de testeo
        y1 = Y_train[:, 0] + Y_train[:, 1]
        y0 = np.ones(Y_train.shape[0]) - y1

        # Acondicionando las etiquetas para testear el modelo
        y0 = np.expand_dims(y0, -1)     # Segmentos intermedios
        y1 = np.expand_dims(y1, -1)     # S1 y S2

        # Concatenando las etiquetas para el modelo
        y_to = np.concatenate((y0, y1), axis=-1)
        
        # Entrenando
        history = model.fit(x=[X_train[:, :, i] for i in range(X_train.shape[2])], 
                            y=y_to, epochs=epochs, batch_size=batch_size, 
                            verbose=1, validation_split=validation_split)
  
    
    # Y guardando la información del entrenamiento con el testeo
    with open(f'Models/{model_name}.txt', 'a', encoding='utf8') as file:
        file.write(f'{history.history}\n')
        
    return model


###############       Definición de parámetros       ###############

# Definición de la GPU con la que se trabajará
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Parámetros de get_model_data
test_size = 0.1
snr_list =  [] # [-1, 0, 1, 5] # [0, 1, 5, 10]
big_batch_size = None
N = 1024
step = 64
# N = 128
# step = 16
N_env = 128
step_env = 16
padding_value = 2
activation_percentage = None
# activation_percentage = 0.5

apply_bpfilter = True
bp_parameters = [20, 30, 190, 200]

append_envelopes = True
homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
hilbert_dict = {'analytic_env': True, 'inst_phase': False, 'inst_freq': False}
simplicity_dict =  None # {'N': 64, 'noverlap': 32, 'm': 10, 'tau': 1}
vfd_dict = {'N': N_env, 'noverlap': N_env - step_env, 'kmin': 2, 'kmax': 2, 
            'step_size_method': 'unit'}
wavelet_dict = {'wavelet': 'db4', 'levels': [3,4], 'start_level': 1, 'end_level': 4}
spec_track_dict =  {'freq_obj': [100, 150], 'N': N_env, 'noverlap': N_env - step_env, 
                    'padding': 0, 'repeat': 0, 'window': 'hann'}
spec_energy_dict = {'band_limits': [30, 180], 'alpha': 1, 'N': N_env, 
                    'noverlap': N_env - step_env, 'padding': 0, 'repeat': 0 , 
                    'window': 'hann'}
# spec_energy_dict = None
append_fft = False


# Parámetros de Red neuronal
validation_split = 0.1
batch_size = 70
epochs = 20
model_name = 'Model_8_1_pro_total'

# Parámetros de la función objetivo
optimizer = 'Adam'
# loss_func = 'binary_crossentropy'
loss_func = 'categorical_crossentropy'
metrics = ['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
loss_weights = None # [1., 1.]



###############       Definición de parámetros       ###############

# Definición de la cantidad de sonidos a usar en el training
q_train = int((1 - test_size) * len(filenames))

# Definición de la lista de entrenamiento
train_list = order_list[:q_train]
test_list  = order_list[q_train:]
    

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


# Definiendo los parámetros especificados para get_model_data
get_model_data_info = {'test_size': test_size, 'snr_list': snr_list, 
                       'big_batch_size': big_batch_size, 
                       'N': N, 'noverlap': N - step,
                       'padding_value': padding_value, 
                       'activation_percentage': activation_percentage,
                       'append_envelopes': append_envelopes,
                       'apply_bpfilter': apply_bpfilter,
                       'bp_parameters': bp_parameters, 
                       'homomorphic_dict': homomorphic_dict, 
                       'hilbert_dict': hilbert_dict, 
                       'simplicity_dict': simplicity_dict, 
                       'vfd_dict': vfd_dict, 'wavelet_dict': wavelet_dict,
                       'spec_track_dict': spec_track_dict, 
                       'spec_energy_dict': spec_energy_dict, 
                       'append_fft': append_fft}

# Definiendo los parámetros especificados para la función de costo
loss_func_info = {'optimizer': optimizer, 'loss': loss_func, 'metrics': metrics,
                  'loss_weights': loss_weights}

# Finalmente guardando los datos
with open(f'Models/{model_name}-get_model_data_params.txt', 'w', encoding='utf8') as file:
    file.write(f'{get_model_data_info}\n')
    file.write(f'{loss_func_info}')



###### Obtener los shapes #####

# Definición de los datos de entrenamiento y testeo
X_train, Y_train = \
    get_model_data_idxs(db_folder, snr_list=[], index_list=[0], N=N, 
                        noverlap=N-step, padding_value=padding_value, 
                        activation_percentage=activation_percentage, append_audio=True, 
                        append_envelopes=append_envelopes, 
                        apply_bpfilter=apply_bpfilter, bp_parameters=bp_parameters, 
                        homomorphic_dict=homomorphic_dict, hilbert_dict=hilbert_dict,
                        simplicity_dict=simplicity_dict, 
                        vfd_dict=vfd_dict, wavelet_dict=wavelet_dict, 
                        spec_track_dict=spec_track_dict, 
                        spec_energy_dict=spec_energy_dict,
                        append_fft=append_fft)


# Imprimiendo la dimensión de los archivos
print('Data shapes\n-----------')
print(X_train.shape)
print(Y_train.shape)

# Creación del modelo
if 'Model_2_1' in model_name:
    model = model_2_1(input_shape=(X_train.shape[1], X_train.shape[2]), 
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
    
elif 'Model_2_7' in model_name:
    model = model_2_7(input_shape=(X_train.shape[1], X_train.shape[2]), 
                      padding_value=padding_value, name=model_name)

elif model_name == 'Model_2_8':
    model = model_2_8(input_shape=(X_train.shape[1], X_train.shape[2]), 
                      padding_value=padding_value, name=model_name)

elif model_name == 'Model_2_9':
    model = model_2_9(input_shape=(X_train.shape[1], X_train.shape[2]), 
                      padding_value=padding_value, name=model_name)

elif model_name == 'Model_3':
    model = model_3(input_shape=(X_train.shape[1], X_train.shape[2]), 
                    padding_value=padding_value, name=model_name)

elif model_name == 'Model_4_1':
    model = model_4_1(input_shape=(X_train.shape[1], X_train.shape[2]), 
                      padding_value=padding_value, name=model_name)
    
elif model_name == 'Model_4_2':
    model = model_4_2(input_shape=(X_train.shape[1], X_train.shape[2]), 
                      padding_value=padding_value, name=model_name)
    
elif model_name == 'Model_4_3':
    model = model_4_3(input_shape=(X_train.shape[1], X_train.shape[2]), 
                      padding_value=padding_value, name=model_name)

elif model_name == 'Model_4_4':
    model = model_4_4(input_shape=(X_train.shape[1], X_train.shape[2]), 
                      padding_value=padding_value, name=model_name)

elif model_name in ['Model_4_5_pro', 'Model_4_5_pro_total']:
    model = model_4_5(input_shape=(X_train.shape[1], X_train.shape[2]), 
                      padding_value=padding_value, name=model_name)

elif model_name == 'Model_5_1':
    model = model_5_1(input_shape=(X_train.shape[1], X_train.shape[2]), 
                      padding_value=padding_value, name=model_name)
    
elif model_name == 'Model_5_1_1':
    model = model_5_1_1(input_shape=(X_train.shape[1], X_train.shape[2]), 
                        padding_value=padding_value, name=model_name)
    
elif model_name == 'Model_5_2_1':
    model = model_5_2_1(input_shape=(X_train.shape[1], X_train.shape[2]), 
                        padding_value=padding_value, name=model_name)

elif model_name == 'Model_5_2_2':
    model = model_5_2_2(input_shape=(X_train.shape[1], X_train.shape[2]), 
                        padding_value=padding_value, name=model_name)
    
elif model_name == 'Model_5_2_3':
    model = model_5_2_3(input_shape=(X_train.shape[1], X_train.shape[2]),
                        padding_value=padding_value, name=model_name)

# elif 'Model_5_2_4' in model_name:
#     model = model_5_2_4(input_shape=(X_train.shape[1], X_train.shape[2]),
#                         padding_value=padding_value, name=model_name)

elif model_name == 'Model_5_2_4_1':
    model = model_5_2_4_1(input_shape=(X_train.shape[1], X_train.shape[2]),
                          padding_value=padding_value, name=model_name)

elif model_name in ['Model_5_2_4_2', 'Model_5_2_4_2_pro', 'Model_5_2_4_2_pro_total'] :
    model = model_5_2_4_2(input_shape=(X_train.shape[1], X_train.shape[2]),
                          padding_value=padding_value, name=model_name)
    
elif model_name in ['Model_5_2_4_3_pro_total']:
    model = model_5_2_4_3(input_shape=(X_train.shape[1], X_train.shape[2]),
                          padding_value=padding_value, name=model_name)

elif model_name == 'Model_5_2_5':
    model = model_5_2_5(input_shape=(X_train.shape[1], X_train.shape[2]),
                        padding_value=padding_value, name=model_name)
    
elif model_name == 'Model_5_2_6':
    model = model_5_2_6(input_shape=(X_train.shape[1], X_train.shape[2]),
                        padding_value=padding_value, name=model_name)
    
elif model_name == 'Model_5_2_7':
    model = model_5_2_7(input_shape=(X_train.shape[1], X_train.shape[2]),
                        padding_value=padding_value, name=model_name)

elif model_name == 'Model_5_2_8':
    model = model_5_2_8(input_shape=(X_train.shape[1], X_train.shape[2]),
                        padding_value=padding_value, name=model_name)
    
elif 'Model_5_2_9' in model_name:
    model = model_5_2_9(input_shape=(X_train.shape[1], X_train.shape[2]),
                        padding_value=padding_value, name=model_name)

elif model_name in ['Model_6_1', 'Model_6_1_pro_total']:
    model = model_6_1(input_shape=(X_train.shape[1], X_train.shape[2]),
                      padding_value=padding_value, name=model_name)
    
elif 'Model_6_2' in model_name:
    model = model_6_2(input_shape=(X_train.shape[1], X_train.shape[2]),
                      padding_value=padding_value, name=model_name)

elif 'Model_6_3' in model_name:
    model = model_6_3(input_shape=(X_train.shape[1], X_train.shape[2]),
                      padding_value=padding_value, name=model_name)
    
elif 'Model_6_4' in model_name:
    model = model_6_4(input_shape=(X_train.shape[1], X_train.shape[2]),
                      padding_value=padding_value, name=model_name)

elif model_name in ['Model_7_1_pro_total', 'Model_7_1_pro_total']:
    model = model_7_1(input_shape=(X_train.shape[1], X_train.shape[2]),
                      padding_value=padding_value, name=model_name)
    
elif model_name in ['Model_7_1_2_pro_total']:
    model = model_7_1_2(input_shape=(X_train.shape[1], X_train.shape[2]),
                        padding_value=padding_value, name=model_name)
    
elif model_name in ['Model_7_2_pro_total']:
    model = model_7_2(input_shape=(X_train.shape[1], X_train.shape[2]),
                      padding_value=padding_value, name=model_name)
    
elif model_name in ['Model_8_1_pro_total']:
    model = model_8_1(input_shape=(X_train.shape[1], X_train.shape[2]),
                      padding_value=padding_value, name=model_name)


# Compilando las opciones del modelo
if model_name in ['Model_2_1', 'Model_2_1_2', 'Model_2_1_no-noise', 'Model_2_1_hyper-noise',
                  'Model_2_2', 'Model_2_3', 'Model_2_4', 'Model_2_5', 'Model_2_6', 'Model_2_7', 
                  'Model_2_7_2', 'Model_2_8', 'Model_4_1', 'Model_4_2', 'Model_4_3', 'Model_4_4', 
                  'Model_5_1', 'Model_5_1_1', 'Model_5_2_1', 'Model_5_2_2', 'Model_5_2_3', 
                  'Model_5_2_4', 'Model_5_2_4_again', 'Model_5_2_4_1', 'Model_5_2_5', 'Model_5_2_6', 
                  'Model_5_2_7', 'Model_5_2_8', 'Model_5_2_9', 'Model_5_2_9_alt', 'Model_5_2_9_alt_2']:
    loss_model = [loss_func, loss_func]

elif model_name in ['Model_2_9', 'Model_3', 'Model_6_1', 'Model_6_1_noised', 'Model_6_1_onechannel',
                    'Model_6_2', 'Model_6_3', 'Model_6_4_onechannel', 'Model_6_4_typicalchannels',
                    'Model_5_2_4_2', 'Model_5_2_4_2_pro', 'Model_5_2_4_2_pro_total','Model_5_2_4_3',
                    'Model_6_1_pro_total', 'Model_7_1_pro_total', 'Model_7_2_pro_total',
                    'Model_7_1_pro_total', 'Model_7_1_2_pro_total', 'Model_5_2_4_3_pro_total',
                    'Model_4_5_pro_total', 'Model_4_5_pro', 'Model_8_1_pro_total']:
    loss_model = loss_func

# Compilando las opciones
model.compile(optimizer=optimizer, loss=loss_model,
              metrics=metrics, loss_weights=loss_weights)


# Mostrando el resumen
model.summary()

# Y el gráfico del modelo
try:
    tf.keras.utils.plot_model(model, f'Models/{model_name}.png', 
                              show_shapes=True, expand_nested=True)
    tf.keras.utils.plot_model(model, f'Models/{model_name}_nested.png', 
                              show_shapes=True, expand_nested=False)
except:
    print('No se pudo graficar los modelos.')



############# Iteraciones por cada big batch #############

# Reseteando el archivo de historial
open(f'Models/{model_name}.txt', 'w', encoding='utf8').close()
open(f'Models/{model_name}_db.txt', 'w', encoding='utf8').close()



# Realizando las iteraciones
while train_list.size > 0:
    # Selección de archivos
    train_sel = train_list[:big_batch_size]
    
    # Cortando los archivos seleccionados
    if big_batch_size is None:
        train_list = train_list[:0]
    else:
        train_list = train_list[big_batch_size:]
    
    # Mensaje de progreso
    print(f'Faltan {train_list.size} sonidos por procesar...\n')
    
    # Aplicando la iteración
    model = model_train_iteration(model, model_name, train_sel)


# Guardando el modelo
model.save(f'Models/{model_name}.h5')


############# Testeando #############

# Definición de los datos de testeo
X_test, Y_test = \
    get_model_data_idxs(db_folder, snr_list=snr_list, index_list=test_list, N=N, 
                        noverlap=N-step, padding_value=padding_value, 
                        activation_percentage=activation_percentage, append_audio=True, 
                        append_envelopes=append_envelopes, 
                        apply_bpfilter=apply_bpfilter, bp_parameters=bp_parameters, 
                        homomorphic_dict=homomorphic_dict, hilbert_dict=hilbert_dict,
                        simplicity_dict=simplicity_dict, 
                        vfd_dict=vfd_dict, wavelet_dict=wavelet_dict, 
                        spec_track_dict=spec_track_dict, spec_energy_dict=spec_energy_dict,
                        append_fft=append_fft, print_indexes=False, return_indexes=True)


if model_name in ['Model_2_1', 'Model_2_1_2', 'Model_2_1_no-noise', 'Model_2_1_hyper-noise',
                  'Model_2_2', 'Model_2_3', 'Model_2_4', 'Model_2_5', 'Model_2_6', 'Model_2_7',
                  'Model_2_7_2', 'Model_2_8', 'Model_4_1', 'Model_4_2', 'Model_4_3', 'Model_4_4',
                  'Model_5_1', 'Model_5_1_1', 'Model_5_2_1', 'Model_5_2_2']:
    print('\nTesting time\n------------\n')
    eval_info = model.evaluate(x=X_test, y=[Y_test[:,0], Y_test[:,1]], verbose=1,
                               return_dict=True)


elif model_name in ['Model_5_2_3', 'Model_5_2_4', 'Model_5_2_4_1', 'Model_5_2_5', 'Model_5_2_6',
                    'Model_5_2_7', 'Model_5_2_8', 'Model_5_2_9', 'Model_5_2_9_alt', 
                    'Model_5_2_9_alt_2', 'Model_5_2_4_again']:
    print('\nTesting time\n------------\n')
    eval_info = model.evaluate(x=[X_test[:, :, i] for i in range(X_test.shape[2])], 
                               y=[Y_test[:,0], Y_test[:,1]], verbose=1,
                               return_dict=True)


elif model_name in ['Model_3']:
    print('\nTesting time\n------------\n')
    eval_info = model.evaluate(x=X_test, y=Y_test[:,0] + Y_test[:,1], verbose=1,
                                return_dict=True)


elif model_name in ['Model_6_1', 'Model_6_1_noised', 'Model_6_1_onechannel', 'Model_6_2', 
                    'Model_6_3', 'Model_6_4_onechannel', 'Model_6_4_typicalchannels', 
                    'Model_6_1_pro_total', 'Model_8_1_pro_total']:
    print('\nTesting time\n------------\n')
    # Definición de las etiquetas de testeo
    y1 = Y_test[:, :, 0]
    y2 = Y_test[:, :, 1]
    y0 = np.ones(Y_test.shape[:-1]) - y1 - y2

    # Acondicionando las etiquetas para testear el modelo
    y0 = np.expand_dims(y0, -1)     # Segmentos intermedios
    y1 = np.expand_dims(y1, -1)     # S1
    y2 = np.expand_dims(y2, -1)     # S2

    # Concatenando las etiquetas para el modelo
    y_to = np.concatenate((y0, y1, y2), axis=-1)
    
    # Evaluando
    eval_info = model.evaluate(x=X_test, y=y_to, verbose=1, return_dict=True)


elif model_name in ['Model_7_1_pro_total', 'Model_7_2_pro_total', 'Model_7_1_pro_total']:
    print('\nTesting time\n------------\n')
    # Definición de las etiquetas de testeo
    y1 = Y_test[:, :, 0]
    y2 = Y_test[:, :, 1]
    y0 = np.ones(Y_test.shape[:-1]) - y1 - y2

    # Acondicionando las etiquetas para testear el modelo
    y0 = np.expand_dims(y0, -1)     # Segmentos intermedios
    y1 = np.expand_dims(y1, -1)     # S1
    y2 = np.expand_dims(y2, -1)     # S2

    # Concatenando las etiquetas para el modelo
    y_to = np.concatenate((y0, y1, y2), axis=-1)
    
    # Evaluando
    eval_info = model.evaluate(x=[X_test[:, :, i] for i in range(X_test.shape[2])], 
                               y=y_to, verbose=1, return_dict=True)


elif model_name in ['Model_7_1_2_pro_total']:
    print('\nTesting time\n------------\n')
    # Definición de las etiquetas de testeo
    y1 = Y_test[:, :, 0] + Y_test[:, :, 1]
    y0 = np.ones(Y_test.shape[:-1]) - y1

    # Acondicionando las etiquetas para testear el modelo
    y0 = np.expand_dims(y0, -1)     # Segmentos intermedios
    y1 = np.expand_dims(y1, -1)     # S1 y S2
    
    # Concatenando las etiquetas para el modelo
    y_to = np.concatenate((y0, y1), axis=-1)
    
    # Evaluando
    eval_info = model.evaluate(x=[X_test[:, :, i] for i in range(X_test.shape[2])], 
                               y=y_to, verbose=1, return_dict=True)


elif model_name in ['Model_2_9', 'Model_4_5_pro_total', 'Model_4_5_pro']:
    print('\nTesting time\n------------\n')
    # Definición de las etiquetas de testeo
    y1 = Y_test[:, 0]
    y2 = Y_test[:, 1]
    y0 = np.ones(Y_test.shape[:-1]) - y1 - y2

    # Acondicionando las etiquetas para testear el modelo
    y0 = np.expand_dims(y0, -1)     # Segmentos intermedios
    y1 = np.expand_dims(y1, -1)     # S1
    y2 = np.expand_dims(y2, -1)     # S2

    # Concatenando las etiquetas para el modelo
    y_to = np.concatenate((y0, y1, y2), axis=-1)
    
    # Evaluando
    eval_info = model.evaluate(x=X_test, y=y_to, verbose=1, return_dict=True)


elif model_name in ['Model_5_2_4_2', 'Model_5_2_4_2_pro', 'Model_5_2_4_2_pro_total']:
    print('\nTesting time\n------------\n')
    # Definición de las etiquetas de testeo
    y1 = Y_test[:, 0]
    y2 = Y_test[:, 1]
    y0 = np.ones(Y_test.shape[0]) - y1 - y2

    # Acondicionando las etiquetas para testear el modelo
    y0 = np.expand_dims(y0, -1)     # Segmentos intermedios
    y1 = np.expand_dims(y1, -1)     # S1
    y2 = np.expand_dims(y2, -1)     # S2

    # Concatenando las etiquetas para el modelo
    y_to = np.concatenate((y0, y1, y2), axis=-1)
    
    # Evaluando
    eval_info = model.evaluate(x=[X_test[:, :, i] for i in range(X_test.shape[2])], 
                               y=y_to, verbose=1, return_dict=True)


elif model_name in ['Model_5_2_4_3_pro_total']:
    print('\nTesting time\n------------\n')
    # Definición de las etiquetas de testeo
    y1 = Y_test[:, 0] + Y_test[:, 1]
    y0 = np.ones(Y_test.shape[0]) - y1
    # Acondicionando las etiquetas para testear el modelo
    y0 = np.expand_dims(y0, -1)     # Segmentos intermedios
    y1 = np.expand_dims(y1, -1)     # S1 y S2

    # Concatenando las etiquetas para el modelo
    y_to = np.concatenate((y0, y1), axis=-1)
    
    # Evaluando
    eval_info = model.evaluate(x=[X_test[:, :, i] for i in range(X_test.shape[2])], 
                               y=y_to, verbose=1, return_dict=True)



# Y guardando la información del entrenamiento con el testeo
with open(f'Models/{model_name}.txt', 'a', encoding='utf8') as file:
    file.write(f'Testing_info: {eval_info}\n')
