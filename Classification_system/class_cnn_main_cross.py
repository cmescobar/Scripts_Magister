# Este script incorpora la posibilidad de poder obtener los indices de train y test 
# de forma externa
__author__ = 'Christian Escobar Arce'

import os, gc
import numpy as np
import tensorflow as tf
import soundfile as sf
from ast import literal_eval
from scipy.io  import wavfile
from respiratory_sound_classification.paper_CNN_models import segnet_based_6_7, \
    definitive_segnet_based, definitive_segnet_based_2
from respiratory_sound_classification.respiratory_sound_management import \
    get_model_data_idxs, get_training_weights_resp


# Definición de la carpeta con la base de datos
db_folder = 'unpreprocessed_signals'
db_original = 'C:/Users/Chris/Desktop/Scripts_Magister/Respiratory_Sound_Database/audio_and_txt_files'


# Función que permitirá iterar sobre cada modelo, sin sobrepasar los límites de memoria
def model_train_iteration(model, model_name, index_list, epoch_train):
    # Definición de los datos de entrenamiento
    X_train, Y_train = \
            get_model_data_idxs(db_folder, snr_list=snr_list, index_list=index_list, N=N, 
                                noverlap=N-step, padding_value=padding_value, 
                                activation_percentage=activation_percentage)
    
    # Entrenando
    if model_name in ['segnet_based_6_7', 'definitive_segnet_based']:
        print('\nTraining time\n------------\n')
        if objective_label == 'wheeze':
            # Se obtiene la etiqueta de interés
            Y_to = np.expand_dims(Y_train[:, :, 0], axis=-1)
        
        elif objective_label == 'crackle':
            # Se obtiene la etiqueta de interés
            Y_to = np.expand_dims(Y_train[:, :, 1], axis=-1)
        
        else:
            raise Exception('Opción objective_label no válida.')
        
        # Y se transforma a one hot
        y_to = np.concatenate((Y_to, np.ones(Y_to.shape) - Y_to), axis=-1)
        
        # Entrenando
        history = model.fit(x=X_train, y=y_to, epochs=epoch_train, batch_size=batch_size, 
                            verbose=1)
        
    elif model_name in ['definitive_segnet_based_2']:
        print('\nTraining time\n------------\n')
        if objective_label == 'wheeze':
            # Se obtiene la etiqueta de interés
            y_to = np.expand_dims(Y_train[:, :, 0], axis=-1)
        
        elif objective_label == 'crackle':
            # Se obtiene la etiqueta de interés
            y_to = np.expand_dims(Y_train[:, :, 1], axis=-1)
        
        else:
            raise Exception('Opción objective_label no válida.')
                
        # Entrenando
        history = model.fit(x=X_train, y=y_to, epochs=epoch_train, batch_size=batch_size, 
                            verbose=1)

    # Y guardando la información del entrenamiento con el testeo
    with open(f'{filepath_to_save}/{model_name}_{objective_label}.txt', 
              'a', encoding='utf8') as file:
        file.write(f'{history.history}\n')
    
    # Eliminando las variables de entrenamiento de la memoria
    del Y_train
    del X_train
    
    # Eliminando las variables registradas que no se referencian en memoria
    print("Recolectando registros de memoria sin uso...")
    n = gc.collect()
    print("Número de objetos inalcanzables recolectados por el GC:", n)
    print("Basura incoleccionable:", gc.garbage)
    
    return model


def model_bigbatch_iteration(model, model_name, train_list_iter, big_batch_size,
                             epoch):
    # Definición de una lista de iteración auxiliar para cada loop
    # train_list_iter = train_list
    
    # Realizando las iteraciones
    while len(train_list_iter) > 0:
        # Selección de archivos
        train_sel = train_list_iter[:big_batch_size]
        
        # Cortando los archivos seleccionados
        if big_batch_size is None:
            train_list_iter = train_list_iter[:0]
        else:
            train_list_iter = train_list_iter[big_batch_size:]
        
        # Mensaje de progreso
        print(f'Epoch {epoch+1}: Faltan {len(train_list_iter)} sonidos por procesar...\n')
        
        # Aplicando la iteración
        model = model_train_iteration(model, model_name, train_sel, epoch_train=1)
               
    return model


def model_bigbatch_evaluation(model, model_name, index_list, epoch, type_op):
    if type_op == 'val':
        to_print = '\nValidation time\n---------------\n'
        print('Etapa de validación\n-------------------')
        snr_list_to = snr_list
    elif type_op == 'test':
        to_print = '\nTesting time\n------------\n'
        print('Etapa de testeo\n---------------')
        snr_list_to = list()
    
    # Definición de los datos de validación
    X_data, Y_data = \
        get_model_data_idxs(db_folder, snr_list=snr_list_to, index_list=index_list, N=N, 
                            noverlap=N-step, padding_value=padding_value, 
                            activation_percentage=activation_percentage)
    
    
    if model_name in ['segnet_based_6_7', 'definitive_segnet_based']:
        if objective_label == 'wheeze':
            # Se obtiene la etiqueta de interés
            Y_to = np.expand_dims(Y_data[:, :, 0], axis=-1)
        
        elif objective_label == 'crackle':
            # Se obtiene la etiqueta de interés
            Y_to = np.expand_dims(Y_data[:, :, 1], axis=-1)
        
        else:
            raise Exception('Opción objective_label no válida.')
        
        # Y se transforma a one hot
        y_to = np.concatenate((Y_to, np.ones(Y_to.shape) - Y_to), axis=-1)
        
        # Evaluando
        eval_info = model.evaluate(x=X_data, y=y_to, verbose=1, return_dict=True)
        
    elif model_name in ['definitive_segnet_based_2']:
        if objective_label == 'wheeze':
            # Se obtiene la etiqueta de interés
            y_to = np.expand_dims(Y_data[:, :, 0], axis=-1)
        
        elif objective_label == 'crackle':
            # Se obtiene la etiqueta de interés
            y_to = np.expand_dims(Y_data[:, :, 1], axis=-1)
        
        else:
            raise Exception('Opción objective_label no válida.')
                
        # Evaluando
        eval_info = model.evaluate(x=X_data, y=y_to, verbose=1, return_dict=True)
    
    # Y guardando la información del entrenamiento con el testeo
    with open(f'{filepath_to_save}/{model_name}_{objective_label}.txt', 
              'a', encoding='utf8') as file:
        if type_op == 'val':
            file.write(f'Validation_info_epoch_{epoch+1}: {eval_info}\n')
        elif type_op == 'test':
            file.write(f'Testing_info: {eval_info}\n')





###############       Definición de parámetros       ###############

# Definición de la GPU con la que se trabajará
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Carpeta a guardar
filepath_to_save = 'respiratory_sound_classification'

# Parámetros de get_model_data
snr_list = []
big_batch_size = 40 #// 4
padding_value = 2


# Parámetros de Red neuronal
validation_split = 0.1
batch_size = 70
epochs = 50
model_name = 'definitive_segnet_based_2'

# Parámetros de la función objetivo
optimizer = 'Adam'
loss_func = 'categorical_crossentropy'
metrics = ['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
loss_weights = 'median'
output_number = 2

# Definición de las ventanas a revisar
N = 2048
step = 1024
activation_percentage = None
objective_label = 'wheeze'





###############       Definición de parámetros       ###############

# Definición de la carpeta a buscar los archivos
train_test_folder = '/'.join(db_original.split('/')[:-1])
train_list = list()
test_list = list()

# Definición de la lista de nombres de entrenamiento y testeo
with open(f'{train_test_folder}/ICBHI_challenge_train_test.txt', 'r', encoding='utf8') as file:
    for line in file:
        # Obteniendo los datos de cada línea
        data = line.strip().split('\t')
        
        # Controlar que no se utilicen sonidos traqueales
        if 'Tc' in data[0]:
            continue
        
        # Definición del nombre del archivo de audio
        filename_audio = f'{db_original}/{data[0]}.wav'
        
        # Obteniendo el samplerate del archivo original
        try:
            sr, _ = wavfile.read(f'{filename_audio}')
        except:
            _, sr = sf.read(f'{filename_audio}')
        
        if data[1] == 'train':
            train_list.append((f'{db_folder}/{data[0]}_{sr}'))
        elif data[1] == 'test':
            test_list.append((f'{db_folder}/{data[0]}_{sr}'))



### Checkeo de versiones ###
for filename in os.listdir(f'{filepath_to_save}/'):
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




# Re Definición de los loss_weights
if loss_weights == 'median':
    print('\n\nGetting loss_weights\n---------------------')
    # Calculando los pesos
    class_weights_wheeze, class_weights_crackl = \
        get_training_weights_resp(train_list, freq_balancing='median')
    
    if objective_label == 'crackle':
        loss_weights = {'softmax_out': class_weights_crackl}
    elif objective_label == 'wheeze':
        loss_weights = {'softmax_out': class_weights_wheeze}
    
    # Eliminando las variables registradas que no se referencian en memoria
    print("Recolectando registros de memoria sin uso...")
    n = gc.collect()
    print("Número de objetos inalcanzables recolectados por el GC:", n)
    print("Basura incoleccionable:", gc.garbage)



# Creación del modelo
if model_name in ['segnet_based_6_7']:
    model = segnet_based_6_7(input_shape=(None, 1), padding_value=0, 
                             output_number=output_number, name=model_name)
    
elif model_name in ['definitive_segnet_based']:
    model = definitive_segnet_based(input_shape=(None, 1), padding_value=0, 
                                    output_number=output_number, name=model_name)
    
elif model_name in ['definitive_segnet_based_2']:
    model = definitive_segnet_based_2(input_shape=(None, 1),
                                      output_number=1, name=model_name)


# Compilando las opciones del modelo
if model_name in ['segnet_based_6_7', 'definitive_segnet_based']:
    loss_model = loss_func
    
elif model_name in ['definitive_segnet_based_2']:
    loss_model = 'binary_crossentropy'



# Compilando las opciones
model.compile(optimizer=optimizer, loss=loss_model,
              metrics=metrics, loss_weights=loss_weights)


# Mostrando el resumen
model.summary()


# Y el gráfico del modelo
try:
    tf.keras.utils.plot_model(model, 
                              f'{filepath_to_save}/{model_name}_{objective_label}.png', 
                              show_shapes=True, expand_nested=True)
    tf.keras.utils.plot_model(model, 
                              f'{filepath_to_save}/{model_name}_{objective_label}_nested.png', 
                              show_shapes=True, expand_nested=False)
except:
    print('No se pudo graficar los modelos.')





############# Iteraciones por cada big batch #############

# Reseteando el archivo de historial
open(f'{filepath_to_save}/{model_name}_{objective_label}.txt', 'w', encoding='utf8').close()

# Definiendo los parámetros especificados para get_model_data
get_model_data_info = {'test_size': 0.1, 'snr_list': snr_list, 
                       'big_batch_size': big_batch_size, 
                       'N': N, 'step': step}

# Definiendo los parámetros especificados para la función de costo
loss_func_info = {'optimizer': optimizer, 'loss': loss_func, 'metrics': metrics,
                  'loss_weights': loss_weights}

# Finalmente guardando los datos
with open(f'{filepath_to_save}/{model_name}_{objective_label}_get_model_data_params.txt', 
          'w', encoding='utf8') as file:
    file.write(f'{get_model_data_info}\n')
    file.write(f'{loss_func_info}')



# Retorna el modelo ya entrenado
for epoch in range(epochs):
    print(f'\nBig Epoch #{epoch+1}\n-------------\n')
    model = model_bigbatch_iteration(model, model_name, train_list_iter=train_list, 
                                     big_batch_size=big_batch_size, epoch=epoch)

    print('Guardando el modelo...\n\n')
    # Guardando el modelo en cada iteración
    model.save(f'{filepath_to_save}/{model_name}_{objective_label}.h5')
    

    print(f'\n---------- Fin epoch {epoch+1} ----------\n\n')
    
    # Y guardando la información del entrenamiento con el testeo
    with open(f'{filepath_to_save}/{model_name}_{objective_label}.txt', 
              'a', encoding='utf8') as file:
        file.write('---------------------------------------\n')


############# Testeando #############
model_bigbatch_evaluation(model, model_name, index_list=test_list, epoch=epoch, 
                          type_op='test')
