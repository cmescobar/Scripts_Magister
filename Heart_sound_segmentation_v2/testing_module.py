import os
import tensorflow as tf
import matplotlib.pyplot as plt
from ast import literal_eval
from heart_sound_physionet_management import get_windows_and_labels


def get_test_filenames(model_name, db_folder):
    # Definición de los índices de la lista de archivos de salida
    index_list = list()
    
    # Revisión de registro de base de datos usados para testear
    with open(f'Trained_models/{model_name}_db.txt', 'r', encoding='utf8') as file:
        for line in file:
            # Leyendo la línea
            dict_line = literal_eval(line.strip())
            
            # Y agregando a la lista de test de salida
            index_list.extend(dict_line['test_indexes'])
    
    # Una vez definidos los índices se obtienen las direcciones de los archivos de
    # audio
    filenames = [f'{db_folder}/{i[:-4]}' for i in os.listdir(db_folder) 
                 if i.endswith('.wav')]
    
    # Filtrando por los índices de interés
    filenames = [filename for num, filename in enumerate(filenames) if num in index_list]
    
    return filenames


def get_windowed_signal(model_name, filename):
    # Obtener los parámetros utilizados para obtener la señal ventaneada
    # y sus etiquetas
    with open(f'Trained_models/{model_name}-get_model_data_params.txt', 
              'r', encoding='utf8') as file:
        # Definición del diccionario de los parámetros de ventaneo
        data_dict = literal_eval(file.readline().strip())

    # Obteniendo el archivo de audio
    signal_wind, s1_wind, s2_wind = \
        get_windows_and_labels(filename, N=data_dict['N'], 
                               noverlap=data_dict['noverlap'], 
                               padding_value=data_dict['padding_value'], 
                               activation_percentage=0.5, append_audio=True, 
                               append_envelopes=data_dict['append_envelopes'], 
                               apply_bpfilter=data_dict['apply_bpfilter'],
                               bp_parameters=data_dict['bp_parameters'], 
                               apply_noise=False, snr_expected=0, seed_snr=None, 
                               homomorphic_dict=data_dict['homomorphic_dict'], 
                               hilbert_bool=data_dict['hilbert_bool'], 
                               simplicity_dict=data_dict['simplicity_dict'], 
                               vfd_dict=data_dict['vfd_dict'], 
                               wavelet_dict=data_dict['wavelet_dict'], 
                               spec_track_dict=data_dict['spec_track_dict'],
                               append_fft=data_dict['append_fft'])
        
    return signal_wind, s1_wind, s2_wind


def test_heart_sound(model_name, filename, db_folder):
    # Obtención del sonido cardiaco ventaneado y sus etiquetas
    signal_wind, s1_lab, s2_lab = get_windowed_signal(model_name, filename)
    
    # Cargar el modelo de interés
    model = tf.keras.models.load_model(f'Trained_models/{model_name}.h5')
    
    if model_name in ['Model_5_2_3', 'Model_5_2_4', 'Model_5_2_4_1', 'Model_5_2_5', 
                      'Model_5_2_6', 'Model_5_2_7', 'Model_5_2_8', 'Model_5_2_9']:
        # Evaluándolo
        s1_pred, s2_pred = model.predict([signal_wind[:, :, i] 
                                          for i in range(signal_wind.shape[2]) ])
    
    plt.subplot(2,1,1)
    plt.plot(s1_lab)
    plt.plot(s1_pred)
    
    plt.subplot(2,1,2)
    plt.plot(s2_lab)
    plt.plot(s2_pred)
    plt.show()
    
    # print(model.summary())


db_folder = 'PhysioNet 2016 CINC Heart Sound Database'
model_name = 'Model_5_2_9'

# test_heart_sound(model_name)
filenames = get_test_filenames(model_name, db_folder)
test_heart_sound(model_name, filenames[2], db_folder)
