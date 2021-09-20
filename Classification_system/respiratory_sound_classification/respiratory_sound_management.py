import os, sys, librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from ast import literal_eval
from collections import defaultdict
from scipy.io import wavfile, loadmat
from heart_sound_segmentation.filter_and_sampling import bandpass_filter
from heart_sound_segmentation.envelope_functions import get_envelope_pack, \
    get_spectral_info
from sklearn.model_selection import train_test_split
from heart_sound_segmentation.descriptor_functions import get_windowed_signal, \
    get_noised_signal, get_spectrogram
from respiratory_sound_classification.features import get_cepstral_coefficients, \
    get_energy_bands
from utils import get_resp_segments


def get_label_filename(filename, samplerate, length_desired):
    '''Función que permite retornar las etiquetas correspondientes a cada sonido
    respiratorio.
    
    Parameters
    ----------
    filename : str
        Nombre del archivo de interés (solo nombre).
    samplerate : float
        Tasa de muestreo original del sonido del archivo a revisar.
    length_desired : int
        Largo de la señal a obtener. Este largo es igual al largo del 
        sonido ya procesado (preprocessed_signals o unpreprocessed_signals),
        que en general ya se encuentra a una frecuencia de muestreo cercana
        a 4000 Hz.

    Returns
    -------
    Y_wheeze : ndarray
        Señal que contiene la ubicación de los wheezes.
    Y_crackl : ndarray
        Señal que contiene la ubicación de los crackles.
    '''
    # Definición de la carpeta de base de datos de los eventos
    db_events = 'C:/Users/Chris/Desktop/Scripts_Magister/Respiratory_Sound_Database/events'
        
    with open(f'{db_events}/{filename}_events.txt', 'r', encoding='utf8') as file:
        # Definición de las etiquetas para cada caso
        Y_wheeze = np.zeros(length_desired)
        Y_crackl = np.zeros(length_desired)
        
        # Obteniendo la información de cada segmento
        for line in file:
            # Obteniendo la información de la línea
            data = line.strip().split('\t')
            
            if data == ['']:
                continue
            
            # Definición de los límites inferior y superior
            lower = int(float(data[0]) * samplerate)
            upper = int(float(data[1]) * samplerate)
            
            # Revisando etiquetas
            if data[-1] == 'crackle':
                Y_crackl[lower:upper] = 1
            
            if data[-1] == 'wheeze':
                Y_wheeze[lower:upper] = 1
            
    return Y_wheeze.astype(int), Y_crackl.astype(int)


def get_windows_and_labels(filename, db_folder, N=512, noverlap=0, padding_value=2, 
                           activation_percentage=None, apply_noise=False, 
                           snr_expected=0, seed_snr=None):
    '''Función que, para un archivo especificado, permite obtener su 
    representación en matrices de delay y sus etiquetas.
    
    Parameters
    ----------
    filename : str
        Nombre del sonido a procesar y su largo.
    N : int, optional
        Cantidad de puntos a utilizar por ventana. Por defecto es 512.
    noverlap : int, optional
        Cantidad de puntos de traslape que se utiliza para calcular la 
        matriz. Por defecto es 0.
    padding_value : float, optional
        Valor que se utiliza para hacer padding de la señal cuando se 
        encuentra en la última ventana (que generalmente tiene menos) 
        puntos que las anteriores. Por defecto es 2.
    activation_percentage : float, optional
        Porcentaje de activación para el ventaneo de la señal etiqueta
        en la transformación a etiqueta por ventana. Si la presencia del
        sonido s1 o s2 (con valor 1) es mayor que este porcentaje en cada
        ventana, se considera válido. Por defecto es None. Si es None, no
        se aplica reducción de la ventana a una sola etiqueta.
    apply_noise : bool, optional
        Aplicar un ruido blanco gaussiano sobre el audio. Por defecto es False.
    snr_expected : float, optional
        Relación SNR deseada para la señal de salida. Por defecto es 0.
    seed_snr : int or None, optional
        Semilla a utilizar para la creación del ruido blanco gaussiano. Por
        defect es None.
        
    Returns
    -------
    audio_db : ndarray
        Matriz que contiene todas las ventanas de largo N de todos los archivos 
        de audio de la base de datos escogida.
    s1_labels : ndarray
        Matriz que contiene todas las etiquetas S1 de todos los archivos 
        de audio de la base de datos escogida.
    s2_labels : ndarray
        Matriz que contiene todas las etiquetas S2 de todos los archivos 
        de audio de la base de datos escogida.
    '''
    ### Archivo de audio ###
    # Obtención del archivo de audio .wav
    try:
        samplerate, audio = wavfile.read(f'{filename}.wav')
    except:
        audio, samplerate = sf.read(f'{filename}.wav')
    
    # Normalizando el audio
    audio = audio / max(abs(audio))
    
    
    # Aplicación de ruido blanco gaussiano si es que se espicifica
    if apply_noise:
        audio = get_noised_signal(audio, snr_expected, seed=seed_snr)
    
    
    # Definición de la variable en la que se almacenará la información
    audio_info = np.expand_dims(audio, -1)   
    
    
    ### Etiquetas de los estados ###
    
    # Dejando solo el nombre
    filename_clean = '_'.join(filename.split('/')[-1].split('_')[:-1])
    
    # Obteniendo las etiquetas en específico
    Y_wheeze, Y_crackl = \
            get_label_filename(filename_clean, samplerate=samplerate, 
                               length_desired=len(audio))
        
    # Agregando una dimensión a las etiquetas
    Y_wheeze = np.expand_dims(Y_wheeze, -1)
    Y_crackl = np.expand_dims(Y_crackl, -1)
        
    ### Transformación a señales ventaneadas ###
    ## Archivo de audio ##
    audio_info_matrix = get_windowed_signal(audio_info, samplerate, N=N, 
                                            noverlap=noverlap,
                                            padding_value=0)
    
    ## Etiquetas de los estados ##
    wheeze_matrix = get_windowed_signal(Y_wheeze, samplerate, N=N, 
                                    noverlap=noverlap, 
                                    padding_value=0)
    crackl_matrix = get_windowed_signal(Y_crackl, samplerate, N=N, 
                                    noverlap=noverlap, 
                                    padding_value=0)
    
    # Resumir a una sola etiqueta si es que se define esta variable
    if activation_percentage is not None:
        # Sin embargo, es necesario resumir en una etiqueta por ventana
        wheeze_info = wheeze_matrix.sum(axis=1) >= activation_percentage * N
        crackl_info = crackl_matrix.sum(axis=1) >= activation_percentage * N
    else:
        wheeze_info = wheeze_matrix
        crackl_info = crackl_matrix
    
    # Finalmente, pasando a números (0 o 1)
    wheeze_info = wheeze_info.astype(int)
    crackl_info = crackl_info.astype(int)
    
    return audio_info_matrix, wheeze_info, crackl_info


def get_heartsound_database(db_folder, seed_base, index_list, N=512, noverlap=0, 
                            padding_value=0, apply_noise=False, snr_expected=0, 
                            activation_percentage=None):
    '''Función que permite crear matrices de información y etiquetas en base a 
    los datos .wav y .mat de la carpeta db_folder para el problema de detección 
    de sonidos cardiacos.
    
    Parameters
    ----------
    db_folder : str
        Dirección de la carpeta a procesar.
    seed_base : int
        Número base para la semilla en la generación de ruido.
    index_list : list
        Lista de índices de archivos a revisar.
    (**kwargs) : De la función get_windows_and_labels.
        
    Returns
    -------
    audio_db : ndarray
        Matriz que contiene todas las ventanas de largo N de todos los archivos 
        de audio de la base de datos escogida.
    s1_labels : ndarray
        Matriz que contiene todas las etiquetas S1 de todos los archivos 
        de audio de la base de datos escogida.
    s2_labels : ndarray
        Matriz que contiene todas las etiquetas S2 de todos los archivos 
        de audio de la base de datos escogida.
    '''    
    # Definición de la matriz que concatenará la base de datos de audio
    audio_db = np.zeros((0, N, 1))
    
    # Definición de las matrices que concatenarán las etiquetas
    if activation_percentage is not None:
        wheeze_labels = np.zeros((0,1))
        crackl_labels = np.zeros((0,1))
    else:
        wheeze_labels = np.zeros((0, N, 1))
        crackl_labels = np.zeros((0, N, 1))
        
    for num, filename in enumerate(tqdm(index_list, desc='db', ncols=70)):
        # Obtención de los datos de interés para el archivo filename
        audio_mat, wh_lab, cr_lab = \
            get_windows_and_labels(filename, db_folder=db_folder, 
                                   N=N, noverlap=noverlap, 
                                   padding_value=padding_value, 
                                   activation_percentage=activation_percentage,
                                   apply_noise=apply_noise,
                                   snr_expected=snr_expected,
                                   seed_snr=num+seed_base)
        
        # Agregando la información a cada arreglo
        audio_db = np.concatenate((audio_db, audio_mat), axis=0)
        wheeze_labels = np.concatenate((wheeze_labels, wh_lab), axis=0)
        crackl_labels = np.concatenate((crackl_labels, cr_lab), axis=0)
    
    return audio_db, wheeze_labels, crackl_labels


def get_model_data_idxs(db_folder, snr_list=[], index_list=[], N=512, noverlap=0, 
                        padding_value=0, activation_percentage=0.5):
    '''Función que permite generar la base de datos final que se usará como entrada al 
    modelo. A diferencia de la función original, en esta se permite el ingreso de los
    índices de los archivos a considerar.
    
    Parameters
    ----------
    db_folder : str
        Dirección de la carpeta a procesar.
    test_size : float
        Porcentaje de los datos que se van a utilizar para el testing.
    snr_list : list, optional
        Lista de snr's a considerar para la generación de sonidos. Por defecto es
        una lista vacía.
    (**kwargs) : De la función get_heartsound_database.
        
    Returns
    -------
    audio_db : ndarray
        Matriz que contiene todas las ventanas de largo N de todos los archivos 
        de audio de la base de datos escogida.
    s1_labels : ndarray
        Matriz que contiene todas las etiquetas S1 de todos los archivos 
        de audio de la base de datos escogida.
    s2_labels : ndarray
        Matriz que contiene todas las etiquetas S2 de todos los archivos 
        de audio de la base de datos escogida.
    '''
    def _get_data(index_list, seed_base):
        '''Rutina auxiliar que obtiene los datos y sus respectivas etiquetas,
        incluso con una etapa en la que se añade ruido a la señal.
        '''
        # En primer lugar se obtiene la base de datos original
        audio_db, wheeze_labels, crackl_labels = \
            get_heartsound_database(db_folder, 0, index_list, N=N, 
                                    noverlap=noverlap, padding_value=padding_value, 
                                    activation_percentage=activation_percentage,
                                    apply_noise=False, snr_expected=0)

        # Para cada caso en las SNR definidas
        for snr_desired in snr_list:
            # Obteniendo la base de datos con ruido "snr_desired"
            audio_db_to, wheeze_labels_to, crackl_labels_to = \
                get_heartsound_database(db_folder, seed_base, index_list, N=N, 
                                        noverlap=noverlap, padding_value=padding_value, 
                                        activation_percentage=activation_percentage,
                                        apply_noise=True, snr_expected=snr_desired)

            # Aumentando la semilla base
            seed_base += 10

            # Y agregando a la base de datos
            audio_db  = np.concatenate((audio_db , audio_db_to),  axis=0)
            wheeze_labels = np.concatenate((wheeze_labels, wheeze_labels_to), axis=0)
            crackl_labels = np.concatenate((crackl_labels, crackl_labels_to), axis=0)

        # Se concatenan las etiquetas para tener una sola variable "Y"
        labels = np.concatenate((wheeze_labels, crackl_labels), axis=-1)
        
        return audio_db, labels
    
    
    # Obtener los datos de entrenamiento y testeo
    X, Y = _get_data(index_list, np.random.randint(0, 10000))
    
    # with open(f'Models/Last_model_reg.txt', 'a', encoding='utf8') as file:
    #     for name, size in sorted(((name, sys.getsizeof(value)) 
    #                             for name, value in locals().items() ), 
    #                             key= lambda x: -x[1])[:10]:
    #         text_to_disp = "{:>30}: {:>8}".format(name, sizeof_fmt(size))
    #         print(text_to_disp)
    #         file.write(f'{text_to_disp}\n')
    #     file.write('\n\n')
    
    return X, Y


def get_training_weights(db_folder, big_batch_size, index_array, N=512, noverlap=0, 
                         padding_value=2, activation_percentage=0.5, append_audio=True, 
                         freq_balancing='median'):
    '''Función que permite calcular los pesos de las etiquetas de entrenamiento. Para ello,
    se suma la cantidad de puntos de cada una de las etiquetas, considerando el proceso de
    ventaneo utilizado para obtener la base de datos de la entrada de la red.
    
    Parameters
    ----------
    db_folder : str
        Dirección de la carpeta a procesar.
    test_size : float
        Porcentaje de los datos que se van a utilizar para el testing.
    snr_list : list, optional
        Lista de snr's a considerar para la generación de sonidos. Por defecto es
        una lista vacía.
    (**kwargs) : De la función get_heartsound_database.
        
    Returns
    -------
    audio_db : ndarray
        Matriz que contiene todas las ventanas de largo N de todos los archivos 
        de audio de la base de datos escogida.
    s1_labels : ndarray
        Matriz que contiene todas las etiquetas S1 de todos los archivos 
        de audio de la base de datos escogida.
    s2_labels : ndarray
        Matriz que contiene todas las etiquetas S2 de todos los archivos 
        de audio de la base de datos escogida.
    '''
    def _get_data(index_list):
        '''Rutina auxiliar que obtiene los datos y sus respectivas etiquetas,
        incluso con una etapa en la que se añade ruido a la señal.
        '''
        # En primer lugar se obtiene la base de datos original
        _, s1_labels, s2_labels = \
            get_heartsound_database(db_folder, 0, index_list, N=N, 
                                    noverlap=noverlap, padding_value=padding_value, 
                                    activation_percentage=activation_percentage, 
                                    apply_noise=False, snr_expected=0)
        
        # Definición de S0
        s0_labels = np.ones(s1_labels.shape) - s1_labels - s2_labels
        
        # Contando la cantidad de etiquetas S1 y S2
        s0_count = np.sum(s0_labels)
        s1_count = np.sum(s1_labels)
        s2_count = np.sum(s2_labels)
        
        return s0_count, s1_count, s2_count
    
    # Definición de los contadores
    s0_count = 0
    s1_count = 0
    s2_count = 0
            
    # Realizando las iteraciones
    while index_array.size > 0:
        # Selección de archivos
        selected_index = index_array[:big_batch_size]
        
        # Cortando los archivos seleccionados
        if big_batch_size is None:
            index_array = index_array[:0]
        else:
            index_array = index_array[big_batch_size:]
        
        # Obtener los datos de entrenamiento y testeo
        s0_count_i, s1_count_i, s2_count_i = _get_data(selected_index)
        
        # Sumando a los contadores
        s0_count += s0_count_i
        s1_count += s1_count_i
        s2_count += s2_count_i
    
    # Calculando los pesos para cada caso
    if freq_balancing == 'median':
        # Cálculo de la mediana
        median_val = np.median([s0_count, s1_count, s2_count])
        
        # Definición de los pesos
        s0_val = median_val / s0_count
        s1_val = median_val / s1_count
        s2_val = median_val / s2_count

        # Definición del diccionario
        class_weights = {0: s0_val, 1: s1_val, 2: s2_val}
    
    return class_weights


def get_training_weights_resp(filenames, freq_balancing='median'):
    '''
    '''    
    # Contador de las etiquetas
    n_wheeze = 0
    n_crackl = 0
    n_total = 0
    
    # Nombre del archivo .wav a utilizar
    for filename in tqdm(filenames, ncols=100, desc='Weights'):
        # print(f'Iteración {num + 1}: {filename}')
        # print(f'--------------------------')
        
        # Cargando el archivo
        try:
            samplerate, resp_signal = wavfile.read(f'{filename}.wav')
        except:
            resp_signal, samplerate = sf.read(f'{filename}.wav')
        
        # print(f'Samplerate = {samplerate}, largo = {resp_signal.shape}')
        
        # Normalizando
        resp_signal = resp_signal / max(abs(resp_signal))
        
        # Obteniendo la información de los segmentos de este archivo de 
        # audio
        name_lab = '_'.join(filename.split('/')[-1].split('_')[:-1])
        Y_wheeze_i, Y_crackl_i = \
                get_label_filename(filename=name_lab, samplerate=samplerate, 
                                   length_desired=len(resp_signal))       
        
        # Sumando la cantidad de etiquetas de este sonido
        n_wheeze += np.sum(Y_wheeze_i)
        n_crackl += np.sum(Y_crackl_i)
        n_total  += len(resp_signal)
    
    # Definición de la cantidad de puntos que no son wheeze
    not_wheeze = n_total - n_wheeze
    not_crackl = n_total - n_crackl
    
    # Definiendo el diccionario de los pesos por la mediana
    # Calculando los pesos para cada caso
    if freq_balancing == 'median':
        # Cálculo de la mediana
        median_wheeze = np.median([not_wheeze, n_wheeze])
        median_crackl = np.median([not_crackl, n_crackl])
        
        # Definición de los pesos
        wheeze_0 = median_wheeze / not_wheeze
        wheeze_1 = median_wheeze / n_wheeze
        
        crackl_0 = median_crackl / not_crackl
        crackl_1 = median_crackl / n_crackl

        # Definición del diccionario
        class_weights_wheeze = {0: wheeze_0, 1: wheeze_1}
        class_weights_crackl = {0: crackl_0, 1: crackl_1}
    
    return class_weights_wheeze, class_weights_crackl


def train_test_indexes(ind_beg, ind_end, test_size, random_state=0):
    '''Función que permite obtener los índices de los audios que serán 
    utilizados para obtener los sonidos de entrenamiento y testeo.
    
    Parameters
    ----------
    ind_beg : int
        Indice del primer archivo de audio a considerar.
    ind_end : int
        Indice del último archivo de audio a considerar.
    test_size : float
        Porcentaje de datos utilizados para el testeo (valor entre 0 
        y 1).
    random_state : int, optional
        Semilla utilizada para generar los datos. Por defecto es 0.
    
    Returns
    -------
    train_indexes : list
        Lista que contiene los índices de la base de datos que serán 
        utilizadas para entrenamiento.
    test_indexes : list
        Lista que contiene los índices de la base de datos que serán 
        utilizadas para testeo.
    '''
    # Aplicación de la semilla para la separación de muestras
    np.random.seed(random_state)
    
    # Definición de la cantidad de datos
    N = abs(ind_end - ind_beg)
    
    # Definición de los índices de datos de entrenamiento
    train_indexes = np.random.choice(np.arange(ind_beg, ind_end), 
                                     size=int(round(N * (1 - test_size))),
                                     replace=False).tolist()
    train_indexes.sort()
    
    # Definición de los índices de datos de testeo
    test_indexes = list(set([i for i in range(ind_beg, ind_end)]) - 
                        set(train_indexes))
    test_indexes.sort()
    
    return train_indexes, test_indexes


def train_test_filebased(filename, db_root, db_folder):
    # Definición de las listas de entrenamiento y testeo
    train_list = list()
    test_list = list()

    # Definición de la lista de nombres de entrenamiento y testeo
    with open(filename, 'r', encoding='utf8') as file:
        for line in file:
            # Obteniendo los datos de cada línea
            data = line.strip().split('\t')

            # Controlar que no se utilicen sonidos traqueales
            if 'Tc' in data[0]:
                continue

            # Definición del nombre del archivo de audio
            filename_audio = f'{db_root}/{data[0]}.wav'

            # Obteniendo el samplerate del archivo original
            try:
                sr, _ = wavfile.read(f'{filename_audio}')
            except:
                _, sr = sf.read(f'{filename_audio}')

            if data[1] == 'train':
                train_list.append((f'{db_folder}/{data[0]}_{sr}'))
            elif data[1] == 'test':
                test_list.append((f'{db_folder}/{data[0]}_{sr}'))
                
    return train_list, test_list


def get_label_filename_ML(filename, time_array):
    '''Función que permite retornar las etiquetas correspondientes 
    a cada sonido respiratorio.
    
    Parameters
    ----------
    filename : str
        Nombre del archivo de interés (solo nombre).
    time_array : float
        Eje temporal del espectrograma para el cálculo de las 
        variables.

    Returns
    -------
    Y_wheeze : ndarray
        Señal que contiene la ubicación de los wheezes.
    Y_crackl : ndarray
        Señal que contiene la ubicación de los crackles.
    '''
    # Definición de la carpeta de base de datos de los eventos
    db_events = 'C:/Users/Chris/Desktop/Scripts_Magister/'\
                'Respiratory_Sound_Database/events'
    
    # Definición de las etiquetas para cada caso
    Y_wheeze = np.zeros(len(time_array), dtype=np.int8)
    Y_crackl = np.zeros(len(time_array), dtype=np.int8)
        
    with open(f'{db_events}/{filename}_events.txt', 'r', 
              encoding='utf8') as file:
        # Definición la información de los intervalos
        intervals_info = list()
        
        # Obteniendo la información de cada segmento
        for line in file:
            # Obteniendo la información de la línea
            data = line.strip().split('\t')
            
            if data == ['']:
                continue
            
            # Definición de los límites inferior y superior
            lower = float(data[0])
            upper = float(data[1])
            
            # Agregando los datos etiquetas
            intervals_info.append(((lower, upper), data[2]))
    
    # Etiquetando cada índice de tiempo
    for num, t in enumerate(time_array):
        # Revisando cada intervalo para ver si el tiempo "t"
        # está en alguno de ellos
        for interval in intervals_info:
            if interval[0][0] <= t <= interval[0][1]:
                if interval[1] == 'wheeze':
                    Y_wheeze[num] = 1
                if interval[1] == 'crackle':
                    Y_crackl[num] = 1
    
    return Y_wheeze, Y_crackl


def get_ML_data(filenames, spec_params, mfcc_params=None, 
                lfcc_params=None, energy_params=None):
    '''
    '''
    # Definición de la lista donde se acumulará la información
    X_data = list()
    
    # Definición de los arrays donde se acumularán las etiquetas
    Y_wheeze = list()
    Y_crackl = list()
    
    # Nombre del archivo .wav a utilizar
    for num, filename in enumerate(filenames):
        print(f'Iteración {num + 1}: {filename}')
        print(f'--------------------------')
        
        # Cargando el archivo
        try:
            samplerate, resp_signal = wavfile.read(f'{filename}.wav')
        except:
            resp_signal, samplerate = sf.read(f'{filename}.wav')
        
        print(f'Samplerate = {samplerate}, largo = {resp_signal.shape}')
        
        # Normalizando
        resp_signal = resp_signal / max(abs(resp_signal))
        
        
        # Obtener el tiempo y la dimensión de las características
        t, _, _ = get_spectrogram(resp_signal, samplerate, 
                                  N=spec_params['N'], 
                                  padding=spec_params['padding'], 
                                  repeat=spec_params['repeat'], 
                                  noverlap=spec_params['noverlap'], 
                                  window=spec_params['window'], 
                                  whole=False)
        
        # Obteniendo la información de los segmentos de este archivo de 
        # audio
        name_lab = '_'.join(filename.split('/')[-1].split('_')[:-1])
        Y_wheeze_i, Y_crackl_i = \
                get_label_filename_ML(filename=name_lab, time_array=t)       
        
        
        # Definición de la matriz de características
        feat_mat = np.zeros((0, len(t)))     
        
        
        ### Calculando las características ###

        # Cálculo del MFCC
        if mfcc_params is not None:
            # Calculando la característica
            mfcc_features = \
                get_cepstral_coefficients(resp_signal, samplerate, 
                                          spectrogram_params=mfcc_params['spec_params'],
                                          freq_lim=mfcc_params['freq_lim'], 
                                          n_filters=mfcc_params['n_filters'], 
                                          n_coefs=mfcc_params['n_mfcc'], 
                                          scale_type='mel', 
                                          filter_type='triangular', inverse_func='dct', 
                                          norm_filters=mfcc_params['norm_filters'], 
                                          plot_filterbank=False, 
                                          power=mfcc_params['power'])
            
            # Agregando
            feat_mat = np.concatenate((feat_mat, mfcc_features), axis=0)
            

        # Cálculo del LFCC
        if lfcc_params is not None:
            # Calculando la característica
            lfcc_features = \
                get_cepstral_coefficients(resp_signal, samplerate, 
                                          spectrogram_params=lfcc_params['spec_params'],
                                          freq_lim=lfcc_params['freq_lim'], 
                                          n_filters=lfcc_params['n_filters'], 
                                          n_coefs=lfcc_params['n_mfcc'], 
                                          scale_type='linear', 
                                          filter_type='triangular', inverse_func='dct', 
                                          norm_filters=lfcc_params['norm_filters'], 
                                          plot_filterbank=False, 
                                          power=lfcc_params['power'])
            
            # Agregando
            feat_mat = np.concatenate((feat_mat, lfcc_features), axis=0)

        
        # Cálculo de la energía por bandas
        if energy_params is not None:
            # Calculando la característica
            energy_S = \
                get_energy_bands(resp_signal, samplerate,
                                 spectrogram_params=energy_params['spec_params'],
                                 fmin=energy_params['fmin'], 
                                 fmax=energy_params['fmax'], 
                                 fband=energy_params['fband'])
            
            # Agregando
            feat_mat = np.concatenate((feat_mat, 10 * np.log10(energy_S + 1e-10)), axis=0)
            
        
        # Agregando la información a cada arreglo
        for i in range(feat_mat.shape[1]):
            X_data.append(feat_mat[:,i])
            Y_wheeze.append(Y_wheeze_i[i])
            Y_crackl.append(Y_crackl_i[i])

        print(f'Dimensión datos: {feat_mat.shape}')
        # print(Y_wheeze_i.shape)
        # print(Y_crackl_i.shape)

    
    return np.array(X_data), np.array(Y_wheeze), np.array(Y_crackl)


def get_ML_data_oncycles(filenames, mfcc_params=None, lfcc_params=None, 
                         energy_params=None):
    # Definición de los arrays donde se acumulará las características
    X_data = list()

    # Definición de los arrays donde se acumularán las etiquetas
    Y_wheeze = list()
    Y_crackl = list()

    # Dirección de la base de datos
    db_original = 'C:/Users/Chris/Desktop/Scripts_Magister/Respiratory_Sound_Database/audio_and_txt_files'


    # Nombre del archivo .wav a utilizar
    for num, filename in enumerate(filenames):
        print(f'Iteración {num + 1}: {filename}')
        print(f'--------------------------')
        
        # Cargando el archivo
        try:
            samplerate, resp_signal = wavfile.read(f'{filename}.wav')
        except:
            resp_signal, samplerate = sf.read(f'{filename}.wav')
        
        print(f'Samplerate = {samplerate}, largo = {resp_signal.shape}')
        
        # Normalizando
        resp_signal = resp_signal / max(abs(resp_signal))
        
        
        # Obteniendo la información de los segmentos de este archivo de audio
        name_txt = '_'.join(filename.split('/')[-1].split('_')[:-1])
        resp_list_info = get_resp_segments(resp_signal, samplerate, 
                                            filepath=f'{db_original}/{name_txt}.txt')
                
        # Para cada segmento, se obtiene la información de interés
        for resp_info in resp_list_info:            
            ### Calculando las características a partir del segmento ###
            
            # Cálculo del MFCC
            if mfcc_params is not None:
                mfcc_features = \
                    get_cepstral_coefficients(resp_info[0], samplerate, 
                                            spectrogram_params=mfcc_params['spec_params'],
                                            freq_lim=mfcc_params['freq_lim'], 
                                            n_filters=mfcc_params['n_filters'], 
                                            n_coefs=mfcc_params['n_mfcc'], 
                                            scale_type='mel', 
                                            filter_type='triangular', inverse_func='dct', 
                                            norm_filters=mfcc_params['norm_filters'], 
                                            plot_filterbank=False, 
                                            power=mfcc_params['power'])

            # Cálculo del LFCC
            if lfcc_params is not None:
                lfcc_features = \
                    get_cepstral_coefficients(resp_info[0], samplerate, 
                                            spectrogram_params=lfcc_params['spec_params'],
                                            freq_lim=lfcc_params['freq_lim'], 
                                            n_filters=lfcc_params['n_filters'], 
                                            n_coefs=lfcc_params['n_mfcc'], 
                                            scale_type='linear', 
                                            filter_type='triangular', inverse_func='dct', 
                                            norm_filters=lfcc_params['norm_filters'], 
                                            plot_filterbank=False, 
                                            power=lfcc_params['power'])
            
            if energy_params is not None:
                # Cálculo de la energía por bandas
                energy_S = \
                    get_energy_bands(resp_info[0], samplerate,
                                    spectrogram_params=energy_params['spec_params'],
                                    fmin=energy_params['fmin'], 
                                    fmax=energy_params['fmax'], 
                                    fband=energy_params['fband'])
            
            
            # Colapsando la información
            to_append = np.concatenate((mfcc_features.mean(axis=1),
                                        lfcc_features.mean(axis=1),
                                        20 * np.log10(energy_S.mean(axis=1))), axis=0)
            
            # Agregando la información a cada arreglo
            X_data.append(to_append)
            Y_crackl.append(resp_info[1])
            Y_wheeze.append(resp_info[2])

    # Transformando listas a arrays    
    return np.array(X_data), np.array(Y_wheeze), np.array(Y_crackl) 


def get_db_spectrograms(filenames, spec_params):
    '''
    '''
    def _get_intervals_filename(filename):
        # Definición de la carpeta de base de datos de los eventos
        db_events = 'C:/Users/Chris/Desktop/Scripts_Magister/'\
                    'Respiratory_Sound_Database/events'
            
        with open(f'{db_events}/{filename}_events.txt', 'r', 
                encoding='utf8') as file:
            # Definición la información de los intervalos
            intervals_info = list()
            
            # Obteniendo la información de cada segmento
            for line in file:
                # Obteniendo la información de la línea
                data = line.strip().split('\t')
                
                if data == ['']:
                    continue
                
                # Definición de los límites inferior y superior
                lower = float(data[0])
                upper = float(data[1])
                
                # Agregando los datos etiquetas
                intervals_info.append(((lower, upper), data[2]))
        
        return intervals_info

    
    # Definición de los arrays donde se acumularán las etiquetas
    S_silenc = list()
    S_wheeze = list()
    S_crackl = list()
    
    # Nombre del archivo .wav a utilizar
    for num, filename in enumerate(filenames):
        print(f'Iteración {num + 1}: {filename}')
        print(f'--------------------------')
        
        # Cargando el archivo
        try:
            samplerate, resp_signal = wavfile.read(f'{filename}.wav')
        except:
            resp_signal, samplerate = sf.read(f'{filename}.wav')
        
        print(f'Samplerate = {samplerate}, largo = {resp_signal.shape}')
        
        # Normalizando
        resp_signal = resp_signal / max(abs(resp_signal))
        
        
        # Obtener el tiempo y la dimensión de las características
        t, f, S = get_spectrogram(resp_signal, samplerate, 
                                  N=spec_params['N'], 
                                  padding=spec_params['padding'], 
                                  repeat=spec_params['repeat'], 
                                  noverlap=spec_params['noverlap'], 
                                  window=spec_params['window'], 
                                  whole=False)
        
        t = np.array(t)
        
        # Obteniendo la información de los segmentos de este archivo de 
        # audio
        name_lab = '_'.join(filename.split('/')[-1].split('_')[:-1])
        intervals_info = \
                 _get_intervals_filename(filename=name_lab)
        
        # índices de interés
        indexes_wheeze = np.array([])
        indexes_crackl = np.array([])
        
        for inter_i in intervals_info:
            if inter_i[1] == 'wheeze':
                indexes = np.where((t >= inter_i[0][0]) & (t <= inter_i[0][1]))[0]
                indexes_wheeze = np.append(indexes_wheeze, indexes)
                

            if inter_i[1] == 'crackle':
                indexes = np.where((t >= inter_i[0][0]) & (t <= inter_i[0][1]))[0]
                indexes_crackl = np.append(indexes_crackl, indexes)
                
        indexes_sil = np.array([i for i in range(len(t)) 
                                if i not in np.append(indexes_wheeze, indexes_crackl)])
        
        S_silenc.append(S[:, indexes_sil.astype(int)])
        S_wheeze.append(S[:, indexes_wheeze.astype(int)])
        S_crackl.append(S[:, indexes_crackl.astype(int)])
        
    return S_silenc, S_wheeze, S_crackl


def get_db_MFCC(filenames, spec_params):
    '''
    '''
    def _get_intervals_filename(filename):
        # Definición de la carpeta de base de datos de los eventos
        db_events = 'C:/Users/Chris/Desktop/Scripts_Magister/'\
                    'Respiratory_Sound_Database/events'
            
        with open(f'{db_events}/{filename}_events.txt', 'r', 
                encoding='utf8') as file:
            # Definición la información de los intervalos
            intervals_info = list()
            
            # Obteniendo la información de cada segmento
            for line in file:
                # Obteniendo la información de la línea
                data = line.strip().split('\t')
                
                if data == ['']:
                    continue
                
                # Definición de los límites inferior y superior
                lower = float(data[0])
                upper = float(data[1])
                
                # Agregando los datos etiquetas
                intervals_info.append(((lower, upper), data[2]))
        
        return intervals_info

    
    # Definición de los arrays donde se acumularán las etiquetas
    S_silenc = list()
    S_wheeze = list()
    S_crackl = list()
    
    # Nombre del archivo .wav a utilizar
    for num, filename in enumerate(filenames):
        print(f'Iteración {num + 1}: {filename}')
        print(f'--------------------------')
        
        # Cargando el archivo
        try:
            samplerate, resp_signal = wavfile.read(f'{filename}.wav')
        except:
            resp_signal, samplerate = sf.read(f'{filename}.wav')
        
        print(f'Samplerate = {samplerate}, largo = {resp_signal.shape}')
        
        # Normalizando
        resp_signal = resp_signal / max(abs(resp_signal))
        
        
        # Obtener el tiempo y la dimensión de las características
        S = librosa.feature.melspectrogram(y=resp_signal, sr=samplerate, 
                                           n_fft=spec_params['N'],
                                           hop_length=spec_params['noverlap'],
                                           n_mels=128, fmax=2000)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=50)
        
        # Obtener el tiempo y la dimensión de las características
        t, _, S = get_spectrogram(resp_signal, samplerate, 
                                  N=spec_params['N'], 
                                  padding=spec_params['padding'], 
                                  repeat=spec_params['repeat'], 
                                  noverlap=spec_params['noverlap'], 
                                  window=spec_params['window'], 
                                  whole=False)
        
        t = np.array(t)
                
        # Obteniendo la información de los segmentos de este archivo de 
        # audio
        name_lab = '_'.join(filename.split('/')[-1].split('_')[:-1])
        intervals_info = \
                 _get_intervals_filename(filename=name_lab)
        
        # índices de interés
        indexes_wheeze = np.array([])
        indexes_crackl = np.array([])
        
        for inter_i in intervals_info:
            if inter_i[1] == 'wheeze':
                indexes = np.where((t >= inter_i[0][0]) & (t <= inter_i[0][1]))[0]
                indexes_wheeze = np.append(indexes_wheeze, indexes)
                
            if inter_i[1] == 'crackle':
                indexes = np.where((t >= inter_i[0][0]) & (t <= inter_i[0][1]))[0]
                indexes_crackl = np.append(indexes_crackl, indexes)
                
        indexes_sil = np.array([i for i in range(len(t) - 1) 
                                if i not in np.append(indexes_wheeze, indexes_crackl)])
        
        S_silenc.append(mfcc[:, indexes_sil.astype(int)])
        S_wheeze.append(mfcc[:, indexes_wheeze.astype(int)])
        S_crackl.append(mfcc[:, indexes_crackl.astype(int)])
        
    return S_silenc, S_wheeze, S_crackl


def get_db_energy(filenames, spec_params, energy_params):
    '''
    '''
    def _get_intervals_filename(filename):
        # Definición de la carpeta de base de datos de los eventos
        db_events = 'C:/Users/Chris/Desktop/Scripts_Magister/'\
                    'Respiratory_Sound_Database/events'
            
        with open(f'{db_events}/{filename}_events.txt', 'r', 
                encoding='utf8') as file:
            # Definición la información de los intervalos
            intervals_info = list()
            
            # Obteniendo la información de cada segmento
            for line in file:
                # Obteniendo la información de la línea
                data = line.strip().split('\t')
                
                if data == ['']:
                    continue
                
                # Definición de los límites inferior y superior
                lower = float(data[0])
                upper = float(data[1])
                
                # Agregando los datos etiquetas
                intervals_info.append(((lower, upper), data[2]))
        
        return intervals_info

    
    # Definición de los arrays donde se acumularán las etiquetas
    S_silenc = list()
    S_wheeze = list()
    S_crackl = list()
    
    # Nombre del archivo .wav a utilizar
    for num, filename in enumerate(filenames):
        print(f'Iteración {num + 1}: {filename}')
        print(f'--------------------------')
        
        # Cargando el archivo
        try:
            samplerate, resp_signal = wavfile.read(f'{filename}.wav')
        except:
            resp_signal, samplerate = sf.read(f'{filename}.wav')
        
        print(f'Samplerate = {samplerate}, largo = {resp_signal.shape}')
        
        # Normalizando
        resp_signal = resp_signal / max(abs(resp_signal))
        
        
        # Obtener el tiempo y la dimensión de las características
        t, _, _ = get_spectrogram(resp_signal, samplerate, 
                                  N=spec_params['N'], 
                                  padding=spec_params['padding'], 
                                  repeat=spec_params['repeat'], 
                                  noverlap=spec_params['noverlap'], 
                                  window=spec_params['window'], 
                                  whole=False)
        
        t = np.array(t)
        
        # Calculando la característica
        energy_S = \
            get_energy_bands(resp_signal, samplerate,
                                spectrogram_params=energy_params['spec_params'],
                                fmin=energy_params['fmin'], 
                                fmax=energy_params['fmax'], 
                                fband=energy_params['fband'])
        
        # Obteniendo la información de los segmentos de este archivo de 
        # audio
        name_lab = '_'.join(filename.split('/')[-1].split('_')[:-1])
        intervals_info = \
                 _get_intervals_filename(filename=name_lab)
        
        # índices de interés
        indexes_wheeze = np.array([])
        indexes_crackl = np.array([])
        
        for inter_i in intervals_info:
            if inter_i[1] == 'wheeze':
                indexes = np.where((t >= inter_i[0][0]) & (t <= inter_i[0][1]))[0]
                indexes_wheeze = np.append(indexes_wheeze, indexes)
                

            if inter_i[1] == 'crackle':
                indexes = np.where((t >= inter_i[0][0]) & (t <= inter_i[0][1]))[0]
                indexes_crackl = np.append(indexes_crackl, indexes)
                
        indexes_sil = np.array([i for i in range(len(t)) 
                                if i not in np.append(indexes_wheeze, indexes_crackl)])
        
        S_silenc.append(energy_S[:, indexes_sil.astype(int)])
        S_wheeze.append(energy_S[:, indexes_wheeze.astype(int)])
        S_crackl.append(energy_S[:, indexes_crackl.astype(int)])
        
    return S_silenc, S_wheeze, S_crackl


def get_db_system(filenames, spec_params, mfcc_params=None, 
                  lfcc_params=None, energy_params=None):
    '''
    '''
    def _get_intervals_filename(filename):
        # Definición de la carpeta de base de datos de los eventos
        db_events = 'C:/Users/Chris/Desktop/Scripts_Magister/'\
                    'Respiratory_Sound_Database/events'
            
        with open(f'{db_events}/{filename}_events.txt', 'r', 
                encoding='utf8') as file:
            # Definición la información de los intervalos
            intervals_info = list()
            
            # Obteniendo la información de cada segmento
            for line in file:
                # Obteniendo la información de la línea
                data = line.strip().split('\t')
                
                if data == ['']:
                    continue
                
                # Definición de los límites inferior y superior
                lower = float(data[0])
                upper = float(data[1])
                
                # Agregando los datos etiquetas
                intervals_info.append(((lower, upper), data[2]))
        
        return intervals_info

    
    # Definición de los arrays donde se acumularán las etiquetas
    S_silenc = list()
    S_wheeze = list()
    S_crackl = list()

    
    # Nombre del archivo .wav a utilizar
    for num, filename in enumerate(filenames):
        print(f'Iteración {num + 1}: {filename}')
        print(f'--------------------------')
        
        # Cargando el archivo
        try:
            samplerate, resp_signal = wavfile.read(f'{filename}.wav')
        except:
            resp_signal, samplerate = sf.read(f'{filename}.wav')
        
        print(f'Samplerate = {samplerate}, largo = {resp_signal.shape}')
        
        # Normalizando
        resp_signal = resp_signal / max(abs(resp_signal))
        
        
        # Obtener el tiempo y la dimensión de las características
        t, _, _ = get_spectrogram(resp_signal, samplerate, 
                                  N=spec_params['N'], 
                                  padding=spec_params['padding'], 
                                  repeat=spec_params['repeat'], 
                                  noverlap=spec_params['noverlap'], 
                                  window=spec_params['window'], 
                                  whole=False)
        
        t = np.array(t)
        
        # Obteniendo la información de los segmentos de este archivo de 
        # audio
        name_lab = '_'.join(filename.split('/')[-1].split('_')[:-1])
        Y_wheeze_i, Y_crackl_i = \
                get_label_filename_ML(filename=name_lab, time_array=t)       
        
        
        # Definición de la matriz de características
        feat_mat = np.zeros((0, len(t)))     
        
        
        ### Calculando las características ###

        # Cálculo del MFCC
        if mfcc_params is not None:
            # Calculando la característica
            mfcc_features = \
                get_cepstral_coefficients(resp_signal, samplerate, 
                                          spectrogram_params=mfcc_params['spec_params'],
                                          freq_lim=mfcc_params['freq_lim'], 
                                          n_filters=mfcc_params['n_filters'], 
                                          n_coefs=mfcc_params['n_mfcc'], 
                                          scale_type='mel', 
                                          filter_type='triangular', inverse_func='dct', 
                                          norm_filters=mfcc_params['norm_filters'], 
                                          plot_filterbank=False, 
                                          power=mfcc_params['power'])
            
            # Agregando
            feat_mat = np.concatenate((feat_mat, mfcc_features), axis=0)
            

        # Cálculo del LFCC
        if lfcc_params is not None:
            # Calculando la característica
            lfcc_features = \
                get_cepstral_coefficients(resp_signal, samplerate, 
                                          spectrogram_params=lfcc_params['spec_params'],
                                          freq_lim=lfcc_params['freq_lim'], 
                                          n_filters=lfcc_params['n_filters'], 
                                          n_coefs=lfcc_params['n_mfcc'], 
                                          scale_type='linear', 
                                          filter_type='triangular', inverse_func='dct', 
                                          norm_filters=lfcc_params['norm_filters'], 
                                          plot_filterbank=False, 
                                          power=lfcc_params['power'])
            
            # Agregando
            feat_mat = np.concatenate((feat_mat, lfcc_features), axis=0)

        
        # Cálculo de la energía por bandas
        if energy_params is not None:
            # Calculando la característica
            energy_S = \
                get_energy_bands(resp_signal, samplerate,
                                 spectrogram_params=energy_params['spec_params'],
                                 fmin=energy_params['fmin'], 
                                 fmax=energy_params['fmax'], 
                                 fband=energy_params['fband'])
            
            # Agregando
            feat_mat = np.concatenate((feat_mat, 10 * np.log10(energy_S + 1e-10)), axis=0)
            
        # Obteniendo la información de los segmentos de este archivo de 
        # audio
        name_lab = '_'.join(filename.split('/')[-1].split('_')[:-1])
        intervals_info = \
                 _get_intervals_filename(filename=name_lab)
        
        # índices de interés
        indexes_wheeze = np.array([])
        indexes_crackl = np.array([])
        
        for inter_i in intervals_info:
            if inter_i[1] == 'wheeze':
                indexes = np.where((t >= inter_i[0][0]) & (t <= inter_i[0][1]))[0]
                indexes_wheeze = np.append(indexes_wheeze, indexes)
                

            if inter_i[1] == 'crackle':
                indexes = np.where((t >= inter_i[0][0]) & (t <= inter_i[0][1]))[0]
                indexes_crackl = np.append(indexes_crackl, indexes)
                
        indexes_sil = np.array([i for i in range(len(t)) 
                                if i not in np.append(indexes_wheeze, indexes_crackl)])
        
        S_silenc.append(feat_mat[:, indexes_sil.astype(int)])
        S_wheeze.append(feat_mat[:, indexes_wheeze.astype(int)])
        S_crackl.append(feat_mat[:, indexes_crackl.astype(int)])
    
    
    
    return S_silenc, S_wheeze, S_crackl


def get_ML_data_2(filenames, spec_params):
    '''
    '''
    def _get_intervals_filename(filename):
        # Definición de la carpeta de base de datos de los eventos
        db_events = 'C:/Users/Chris/Desktop/Scripts_Magister/'\
                    'Respiratory_Sound_Database/events'
            
        with open(f'{db_events}/{filename}_events.txt', 'r', 
                encoding='utf8') as file:
            # Definición la información de los intervalos
            intervals_info = list()
            
            # Obteniendo la información de cada segmento
            for line in file:
                # Obteniendo la información de la línea
                data = line.strip().split('\t')
                
                if data == ['']:
                    continue
                
                # Definición de los límites inferior y superior
                lower = float(data[0])
                upper = float(data[1])
                
                # Agregando los datos etiquetas
                intervals_info.append(((lower, upper), data[2]))
        
        return intervals_info

    
    # Definición de los arrays donde se acumularán las etiquetas
    S_silenc = list()
    S_wheeze = list()
    S_crackl = list()
    
    # Nombre del archivo .wav a utilizar
    for num, filename in enumerate(filenames):
        print(f'Iteración {num + 1}: {filename}')
        print(f'--------------------------')
        
        # Cargando el archivo
        try:
            samplerate, resp_signal = wavfile.read(f'{filename}.wav')
        except:
            resp_signal, samplerate = sf.read(f'{filename}.wav')
        
        print(f'Samplerate = {samplerate}, largo = {resp_signal.shape}')
        
        # Normalizando
        resp_signal = resp_signal / max(abs(resp_signal))
        
        
        # Obtener el tiempo y la dimensión de las características
        t, f, S = get_spectrogram(resp_signal, samplerate, 
                                  N=spec_params['N'], 
                                  padding=spec_params['padding'], 
                                  repeat=spec_params['repeat'], 
                                  noverlap=spec_params['noverlap'], 
                                  window=spec_params['window'], 
                                  whole=False)
        
        t = np.array(t)
        
        # Obteniendo la información de los segmentos de este archivo de 
        # audio
        name_lab = '_'.join(filename.split('/')[-1].split('_')[:-1])
        intervals_info = \
                 _get_intervals_filename(filename=name_lab)
        
        # índices de interés
        indexes_wheeze = np.array([])
        indexes_crackl = np.array([])
        
        for inter_i in intervals_info:
            if inter_i[1] == 'wheeze':
                indexes = np.where((t >= inter_i[0][0]) & (t <= inter_i[0][1]))[0]
                indexes_wheeze = np.append(indexes_wheeze, indexes)
                

            if inter_i[1] == 'crackle':
                indexes = np.where((t >= inter_i[0][0]) & (t <= inter_i[0][1]))[0]
                indexes_crackl = np.append(indexes_crackl, indexes)
                
        indexes_sil = np.array([i for i in range(len(t)) 
                                if i not in np.append(indexes_wheeze, indexes_crackl)])
        
        S_silenc.append(S[:, indexes_sil.astype(int)])
        S_wheeze.append(S[:, indexes_wheeze.astype(int)])
        S_crackl.append(S[:, indexes_crackl.astype(int)])
        
    return S_silenc, S_wheeze, S_crackl


def get_ML_datareg(filenames, spec_params, mfcc_params=None, 
                   lfcc_params=None, energy_params=None):
    '''
    '''
    # Definición de la lista donde se acumulará la información
    X_data = list()
    
    # Definición de los arrays donde se acumularán las etiquetas
    Y_wheeze = list()
    Y_crackl = list()
    
    # Diccionario que indica los segmentos que corresponden a cada paciente
    patient_register = defaultdict(list)
    
    # Contador de los segmentos
    seg_i = 0
        
    # Nombre del archivo .wav a utilizar
    for num, filename in enumerate(filenames):
        print(f'Iteración {num + 1}: {filename}')
        print(f'--------------------------')
                
        # Cargando el archivo
        try:
            samplerate, resp_signal = wavfile.read(f'{filename}.wav')
        except:
            resp_signal, samplerate = sf.read(f'{filename}.wav')
        
        print(f'Samplerate = {samplerate}, largo = {resp_signal.shape}')
        
        # Normalizando
        resp_signal = resp_signal / max(abs(resp_signal))
        
        
        # Obtener el tiempo y la dimensión de las características
        t, _, _ = get_spectrogram(resp_signal, samplerate, 
                                  N=spec_params['N'], 
                                  padding=spec_params['padding'], 
                                  repeat=spec_params['repeat'], 
                                  noverlap=spec_params['noverlap'], 
                                  window=spec_params['window'], 
                                  whole=False)
        
        # Obteniendo la información de los segmentos de este archivo de 
        # audio
        name_lab = '_'.join(filename.split('/')[-1].split('_')[:-1])
        Y_wheeze_i, Y_crackl_i = \
                get_label_filename_ML(filename=name_lab, time_array=t)       
        
        
        # Definición de la matriz de características
        feat_mat = np.zeros((0, len(t)))     
        
        
        ### Calculando las características ###

        # Cálculo del MFCC
        if mfcc_params is not None:
            # Calculando la característica
            mfcc_features = \
                get_cepstral_coefficients(resp_signal, samplerate, 
                                          spectrogram_params=mfcc_params['spec_params'],
                                          freq_lim=mfcc_params['freq_lim'], 
                                          n_filters=mfcc_params['n_filters'], 
                                          n_coefs=mfcc_params['n_mfcc'], 
                                          scale_type='mel', 
                                          filter_type='triangular', inverse_func='dct', 
                                          norm_filters=mfcc_params['norm_filters'], 
                                          plot_filterbank=False, 
                                          power=mfcc_params['power'])
            
            # Agregando
            feat_mat = np.concatenate((feat_mat, mfcc_features), axis=0)
            

        # Cálculo del LFCC
        if lfcc_params is not None:
            # Calculando la característica
            lfcc_features = \
                get_cepstral_coefficients(resp_signal, samplerate, 
                                          spectrogram_params=lfcc_params['spec_params'],
                                          freq_lim=lfcc_params['freq_lim'], 
                                          n_filters=lfcc_params['n_filters'], 
                                          n_coefs=lfcc_params['n_mfcc'], 
                                          scale_type='linear', 
                                          filter_type='triangular', inverse_func='dct', 
                                          norm_filters=lfcc_params['norm_filters'], 
                                          plot_filterbank=False, 
                                          power=lfcc_params['power'])
            
            # Agregando
            feat_mat = np.concatenate((feat_mat, lfcc_features), axis=0)

        
        # Cálculo de la energía por bandas
        if energy_params is not None:
            # Calculando la característica
            energy_S = \
                get_energy_bands(resp_signal, samplerate,
                                 spectrogram_params=energy_params['spec_params'],
                                 fmin=energy_params['fmin'], 
                                 fmax=energy_params['fmax'], 
                                 fband=energy_params['fband'])
            
            # Agregando
            feat_mat = np.concatenate((feat_mat, 10 * np.log10(energy_S + 1e-10)), axis=0)
            
        
        # Agregando la información a cada arreglo
        for i in range(feat_mat.shape[1]):
            X_data.append(feat_mat[:,i])
            Y_wheeze.append(Y_wheeze_i[i])
            Y_crackl.append(Y_crackl_i[i])
            
        # Definición del paciente de interés
        patient = name_lab.split('_')[0]    
        
        # Registrando
        patient_register[patient].extend([i for i in range(seg_i, seg_i + len(t))])
        seg_i += len(t)
        
        print(f'Dimensión datos: {feat_mat.shape}')
        # print(Y_wheeze_i.shape)
        # print(Y_crackl_i.shape)

    
    return np.array(X_data), np.array(Y_wheeze), np.array(Y_crackl), patient_register




# Módulo de testeo
if __name__ == '__main__':
    # Definición del testeo a realizar
    test_name = 'get_model_data_idxs_2'
    
    # Definición de la carpeta donde se almacena la base de datos
    db_folder = 'PhysioNet 2016 CINC Heart Sound Database'
    
    lista_random = np.random.choice(100, size=1, replace=False)
    print(lista_random)
    
    
    if test_name == 'get_model_data_idxs':
        # Parámetros
        N_env = 128
        step_env = 16
        file_to_index = 122
                
        data, labels = \
            get_model_data_idxs(db_folder, snr_list=[], index_list=[file_to_index],
                                N=3500, noverlap=0, padding_value=2, activation_percentage=None)
    