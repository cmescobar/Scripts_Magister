__author__ = 'Christian Escobar Arce'

import os
import tensorflow as tf
from heart_sound_DNN_models import model_2_1, model_2_2, model_2_3, model_2_4, \
    model_2_5, model_2_6, model_2_7, model_2_8, model_3, model_4_1, model_4_2, \
    model_4_3, model_4_4, model_5_1, model_5_1_1, model_5_2_1, model_5_2_2, \
    model_5_2_3, model_5_2_4
from heart_sound_physionet_management import get_model_data


# Función que permitirá iterar sobre cada modelo, sin sobrepasar los límites de memoria
def model_iteration(model, model_name, ind_beg_iter, ind_end_iter):
    # Definición de los datos de entrenamiento y testeo
    X_train, X_test, Y_train, Y_test = \
        get_model_data(db_folder, test_size=test_size, seed_split=seed_split, 
                       snr_list=snr_list, ind_beg=ind_beg_iter, ind_end=ind_end_iter, N=N, 
                       noverlap=N-step, padding_value=padding_value, 
                       activation_percentage=0.5, append_audio=True, 
                       append_envelopes=append_envelopes, 
                       apply_bpfilter=apply_bpfilter, bp_parameters=bp_parameters, 
                       homomorphic_dict=homomorphic_dict, hilbert_bool=hilbert_bool,
                       simplicity_dict=simplicity_dict, 
                       vfd_dict=vfd_dict, wavelet_dict=wavelet_dict, 
                       spec_track_dict=spec_track_dict)

    # Entrenando
    if model_name in ['Model_2_1', 'Model_2_1_2', 'Model_2_1_no-noise', 'Model_2_1_hyper-noise',
                      'Model_2_2', 'Model_2_3', 'Model_2_4', 'Model_2_5', 'Model_2_6', 'Model_2_7',
                      'Model_2_7_2', 'Model_2_8', 'Model_4_1', 'Model_4_2', 'Model_4_3', 'Model_4_4',
                      'Model_5_1', 'Model_5_1_1', 'Model_5_2_1', 'Model_5_2_2']:
        print('\nTraining time\n------------\n')
        history = model.fit(x=X_train, y=[Y_train[:,0], Y_train[:,1]], epochs=epochs, 
                            batch_size=batch_size, verbose=1, validation_split=validation_split)

        print('\nTesting time\n------------\n')
        eval_info = model.evaluate(x=X_test, y=[Y_test[:,0], Y_test[:,1]], verbose=1,
                                   return_dict=True)
    
    elif model_name in ['Model_5_2_3', 'Model_5_2_4']:
        print('\nTraining time\n------------\n')
        history = model.fit(x=[X_train[:, :, i] for i in range(X_train.shape[2])], 
                            y=[Y_train[:,0], Y_train[:,1]], epochs=epochs, 
                            batch_size=batch_size, verbose=1, validation_split=validation_split)

        print('\nTesting time\n------------\n')
        eval_info = model.evaluate(x=[X_test[:, :, i] for i in range(X_test.shape[2])], 
                                   y=[Y_test[:,0], Y_test[:,1]], verbose=1,
                                   return_dict=True)
    
    elif model_name in ['Model_3']:
        print('\nTraining time\n------------\n')
        history = model.fit(x=X_train, y=Y_train[:,0] + Y_train[:,1], epochs=epochs, 
                            batch_size=batch_size, verbose=1, validation_split=validation_split)

        # Evaluando
        print('\nTesting time\n------------\n')
        eval_info = model.evaluate(x=X_test, y=Y_test[:,0] + Y_test[:,1], verbose=1,
                                   return_dict=True)

    # Y guardando la información del entrenamiento con el testeo
    with open(f'Models/{model_name}.txt', 'a', encoding='utf8') as file:
        file.write(f'{history.history},')
        file.write(f'{eval_info}\n')
        
    return model



# Definición de la carpeta con la base de datos
db_folder = 'PhysioNet 2016 CINC Heart Sound Database'

# Obtener los nombres de los archivos
filenames = [f'{db_folder}/{name[:-4]}' for name in os.listdir(db_folder) 
             if name.endswith('.wav')]

# Definición de la cantidad total de archivos de audio
N_data = len(filenames)



###############       Definición de parámetros       ###############

# Definición de la GPU con la que se trabajará
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



# Parámetros de get_model_data
test_size = 0.1
seed_split = 0
snr_list = [-1, 0, 1, 5]# [0, 1, 5, 10]
ind_beg = 0
ind_end = None
big_batch_size = 50
N = 128
step = 16
padding_value = 2

apply_bpfilter = True
bp_parameters = [40, 60, 230, 250]

append_envelopes = True
homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
hilbert_bool = True
simplicity_dict = {'N': 64, 'noverlap': 32, 'm': 10, 'tau': 1}
vfd_dict = {'N': 64, 'noverlap': 32, 'kmin': 2, 'kmax': 2, 'step_size_method': 'unit'}
wavelet_dict = {'wavelet': 'db4', 'levels': [2,3,4], 'start_level': 1, 'end_level': 4}
spec_track_dict = {'freq_obj': [100, 150], 'N': 128, 'noverlap': 100, 'padding': 0, 
                   'repeat': 0, 'window': 'hann'}


# Parámetros de Red neuronal
validation_split = 0.1
batch_size = 70 
epochs = 10
model_name = 'Model_5_2_4'

# Parámetros de la función objetivo
optimizer = 'Adam'
loss_func = 'binary_crossentropy'
metrics = ['accuracy']
loss_weights = [1., 1.]



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


# Definiendo los parámetros especificados para get_model_data
get_model_data_info = {'test_size': test_size, 'seed_split': seed_split,
                       'snr_list': snr_list, 'ind_beg': ind_beg, 
                       'ind_end': ind_end, 'big_batch_size': big_batch_size, 
                       'N': N, 'noverlap': N - step,
                       'padding_value': padding_value, 
                       'append_envelopes': append_envelopes,
                       'apply_bpfilter': apply_bpfilter,
                       'bp_parameters': bp_parameters, 
                       'homomorphic_dict': homomorphic_dict, 
                       'hilbert_bool': hilbert_bool, 
                       'simplicity_dict': simplicity_dict, 
                       'vfd_dict': vfd_dict, 'wavelet_dict': wavelet_dict,
                       'spec_track_dict': spec_track_dict}

# Definiendo los parámetros especificados para la función de costo
loss_func_info = {'optimizer': optimizer, 'loss': loss_func, 'metrics': metrics,
                  'loss_weights': loss_weights}

# Finalmente guardando los datos
with open(f'Models/{model_name}-get_model_data_params.txt', 'w', encoding='utf8') as file:
    file.write(f'{get_model_data_info}\n')
    file.write(f'{loss_func_info}')




###### Obtener los shapes #####

# Definición de los datos de entrenamiento y testeo
X_train, X_test, Y_train, Y_test = \
    get_model_data(db_folder, test_size=test_size, seed_split=seed_split, 
                   snr_list=[], ind_beg=0, ind_end=1, N=N, 
                   noverlap=N-step, padding_value=padding_value, 
                   activation_percentage=0.5, append_audio=True, 
                   append_envelopes=append_envelopes, 
                   apply_bpfilter=apply_bpfilter, bp_parameters=bp_parameters, 
                   homomorphic_dict=homomorphic_dict, hilbert_bool=hilbert_bool,
                   simplicity_dict=simplicity_dict, 
                   vfd_dict=vfd_dict, wavelet_dict=wavelet_dict, 
                   spec_track_dict=spec_track_dict)


# Imprimiendo la dimensión de los archivos
print('Data shapes\n-----------')
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

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

elif model_name == 'Model_5_2_4':
    model = model_5_2_4(input_shape=(X_train.shape[1], X_train.shape[2]),
                        padding_value=padding_value, name=model_name)

# Compilando las opciones del modelo
if model_name in ['Model_2_1', 'Model_2_1_2', 'Model_2_1_no-noise', 'Model_2_1_hyper-noise',
                  'Model_2_2', 'Model_2_3', 'Model_2_4', 'Model_2_5', 'Model_2_6', 'Model_2_7', 
                  'Model_2_7_2', 'Model_2_8', 'Model_4_1', 'Model_4_2', 'Model_4_3', 'Model_4_4', 
                  'Model_5_1', 'Model_5_1_1', 'Model_5_2_1', 'Model_5_2_2', 'Model_5_2_3', 
                  'Model_5_2_4']:
    loss_model = [loss_func, loss_func]

elif model_name in ['Model_3']:
    loss_model = loss_func

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
    print('No se pudo graficar los modelos uwu')



############# Iteraciones por cada big batch #############

# Definición de la cantidad de iteraciones
if N_data % big_batch_size == 0:
    n_iter = N_data // big_batch_size
else:
    n_iter = N_data // big_batch_size + 1

# Reseteando el archivo de historial
open(f'Models/{model_name}.txt', 'w', encoding='utf8').close()


# Realizando las iteraciones
for i in range(n_iter):
    # Definición de los intervalos a revisar
    if i < n_iter - 1:
        ind_beg_iter = big_batch_size * i
        ind_end_iter = ind_beg_iter + big_batch_size
    else:
        ind_beg_iter = big_batch_size * i
        ind_end_iter = None
    
    print(f'ind_beg: {ind_beg_iter}')
    print(f'ind_end: {ind_end_iter}')

    
    # Aplicando la iteración
    model = model_iteration(model, model_name, ind_beg_iter, ind_end_iter)
    
# Guardando el modelo
model.save(f'Models/{model_name}.h5')
