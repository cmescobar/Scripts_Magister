import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile, loadmat
from filter_and_sampling import bandpass_filter
from envelope_functions import get_envelope_pack
from sklearn.model_selection import train_test_split
from descriptor_functions import get_windowed_signal, get_noised_signal


def get_windows_and_labels(filename, N=512, noverlap=0, padding_value=2, 
                           activation_percentage=0.5, append_audio=True, 
                           append_envelopes=False, apply_bpfilter=False,
                           bp_parameters=None, apply_noise=False, 
                           snr_expected=0, seed_snr=None, 
                           homomorphic_dict=None, hilbert_bool=False, 
                           simplicity_dict=None, vfd_dict=None, 
                           wavelet_dict=None, spec_track_dict=None):
    '''Función que, para un archivo especificado, permite obtener su 
    representación en matrices de delay y sus etiquetas.
    
    Parameters
    ----------
    filename : str
        Nombre del sonido a procesar.
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
        ventana, se considera válido. Por defecto es 50%.
    append_audio : bool, optional
        Booleano que indica si se agrega el archivo de audio raw. Por defecto 
        es True.
    append_envelopes : bool, optional
        Booleano que indica si se agregan las envolventes de los archivos de
        audio. Por defecto es False.
    apply_bpfilter : bool, optional
        Aplicar un filtro pasa banda de manera previa sobre el audio.
        Por defecto es False.
    bp_parameters : list or ndarray, optional
        Arreglo de largo 4 indicando las frecuencias de corte en el orden:
        [freq_stop_1, freq_pass_1, freq_pass_2, freq_stop_2]. Por defecto 
        es None.
    apply_noise : bool, optional
        Aplicar un ruido blanco gaussiano sobre el audio. Por defecto es False.
    snr_expected : float, optional
        Relación SNR deseada para la señal de salida. Por defecto es 0.
    seed_snr : int or None, optional
        Semilla a utilizar para la creación del ruido blanco gaussiano. Por
        defect es None.
    homomorphic_dict : dict, optional
        Diccionario con los parámetros de la función 'homomorphic_filter'. 
        Por defecto es None.
    hilbert_bool : bool, optional
        Booleano que indica si es que se agrega las envolventes obtenidas a 
        partir de la transformada de Hilbert. Por defecto es False.
    simplicity_dict : dict, optional
        Diccionario con los parámetros de la función 
        'simplicity_based_envelope'. Por defecto es None.
    vfd_dict : dict, optional
        Diccionario con los parámetros de la función 
        'variance_fractal_dimension'. Por defecto es None.
    wavelet_dict : dict, optional
        Diccionario con los parámetros de la función 
        'stationary_multiscale_wavelets'. Por defecto es None.
    spec_track_dict : dict, optional
        Diccionario con los parámetros de la función 
        'modified_spectral_tracking'. Por defecto es None.
        
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
    samplerate, audio = wavfile.read(f'{filename}.wav')
    
    # Normalizando el audio
    audio = audio / max(abs(audio))
    
    # Aplicación de filtro pasa banda si es que se especifica
    if apply_bpfilter:
        audio = bandpass_filter(audio, samplerate, bp_method='scipy_fir',
                                freq_stop_1=bp_parameters[0], 
                                freq_pass_1=bp_parameters[1], 
                                freq_pass_2=bp_parameters[2], 
                                freq_stop_2=bp_parameters[3],
                                normalize=True)
    
    # Aplicación de ruido blanco gaussiano si es que se espicifica
    if apply_noise:
        audio = get_noised_signal(audio, snr_expected, seed=seed_snr)
    
    
    # Definición de la variable en la que se almacenará la información
    audio_info = np.zeros((len(audio), 0))
    
    # Preguntar si se agrega el archivo de audio
    if append_audio:
        # Y agregando una dimensión para dejarlo en formato matriz
        audio_mat = np.expand_dims(audio, -1)
        
        # Concatenando
        audio_info = np.concatenate((audio_info, audio_mat), axis=1)
    
    
    # Preguntar si se agrega el pack de envolventes
    if append_envelopes:
        # Calculando las envolventes
        envelopes = get_envelope_pack(audio, samplerate, 
                                      homomorphic_dict=homomorphic_dict, 
                                      hilbert_bool=hilbert_bool,
                                      simplicity_dict=simplicity_dict, 
                                      vfd_dict=vfd_dict, 
                                      wavelet_dict=wavelet_dict, 
                                      spec_track_dict=spec_track_dict)
        # Concatenando
        audio_info = np.concatenate((audio_info, envelopes), axis=1)
    
    
    ### Etiquetas de los estados ###
    # Obtención del archivo de las etiquetas .mat
    data_info = loadmat(f'{filename}.mat')
        
    # Etiquetas a 50 Hz de samplerate
    labels = data_info['PCG_states']
    
    # Pasando a 1000 Hz
    labels_adj = np.repeat(labels, 20)
    
    # Recuperación de las etiquetas de S1
    s1_labels = (labels_adj == 1)
    s2_labels = (labels_adj == 3)
    
    # Agregando una dimensión a las etiquetas
    s1_labels = np.expand_dims(s1_labels, -1)
    s2_labels = np.expand_dims(s2_labels, -1)
    
    ### Transformación a señales ventaneadas ###
    ## Archivo de audio ##
    audio_info_matrix = get_windowed_signal(audio_info, samplerate, N=N, 
                                            noverlap=noverlap,
                                            padding_value=padding_value)
    
    ## Etiquetas de los estados ##
    s1_matrix = get_windowed_signal(s1_labels, samplerate, N=N, 
                                    noverlap=noverlap, 
                                    padding_value=0)
    s2_matrix = get_windowed_signal(s2_labels, samplerate, N=N, 
                                    noverlap=noverlap, 
                                    padding_value=0)
    
    # Sin embargo, es necesario resumir en una etiqueta por ventana
    s1_wind = s1_matrix.sum(axis=1) >= activation_percentage * N
    s2_wind = s2_matrix.sum(axis=1) >= activation_percentage * N
    
    # Finalmente, pasando a números (0 o 1)
    s1_wind = s1_wind.astype(int)
    s2_wind = s2_wind.astype(int)
    
    return audio_info_matrix, s1_wind, s2_wind


def get_heartsound_database(db_folder, seed_base, ind_beg=0, ind_end=None, N=512, 
                            noverlap=0, padding_value=2, activation_percentage=0.5, 
                            append_audio=True, append_envelopes=False, 
                            apply_bpfilter=False, bp_parameters=None, 
                            apply_noise=False, snr_expected=0,
                            homomorphic_dict=None, hilbert_bool=False,
                            simplicity_dict=None, vfd_dict=None, 
                            wavelet_dict=None, spec_track_dict=None):
    '''Función que permite crear matrices de información y etiquetas en base a 
    los datos .wav y .mat de la carpeta db_folder para el problema de detección 
    de sonidos cardiacos.
    
    Parameters
    ----------
    db_folder : str
        Dirección de la carpeta a procesar.
    seed_base : int
        Número base para la semilla en la generación de ruido.
    ind_beg : int, optional
        Indice del primer archivo de audio a considerar. Por defecto es 0.
    ind_end : int, optional
        Indice del último archivo de audio a considerar. Por defecto es None.
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
    # Obtener los nombres de los archivos
    filenames = [f'{db_folder}/{name[:-4]}' for name in os.listdir(db_folder) 
                 if name.endswith('.wav')][ind_beg:ind_end]
    
    # Definción de la dimensión de los datos
    q_dim = 0
    
    if append_audio:
        q_dim += 1
    
    if append_envelopes:
        q_dim = q_dim if homomorphic_dict is None else q_dim + 1
        q_dim = q_dim if simplicity_dict is None else q_dim + 1
        q_dim = q_dim if vfd_dict is None else q_dim + 1
        q_dim = q_dim if wavelet_dict is None else q_dim + 1
        q_dim = q_dim if spec_track_dict is None \
                      else q_dim + len(spec_track_dict['freq_obj'])
        q_dim = q_dim + 2 if hilbert_bool else q_dim
    
    # Definición de la matriz que concatenará la base de datos de audio
    audio_db = np.zeros((0, N, q_dim))
    
    # Definición de las matrices que concatenarán las etiquetas
    s1_labels = np.zeros((0,1))
    s2_labels = np.zeros((0,1))
        
    for num, filename in enumerate(tqdm(filenames, desc='db', ncols=70)):
        # Obtención de los datos de interés para el archivo filename
        audio_mat, s1_lab, s2_lab = \
            get_windows_and_labels(filename, N=N, noverlap=noverlap, 
                                   padding_value=padding_value, 
                                   activation_percentage=activation_percentage, 
                                   apply_bpfilter=apply_bpfilter,
                                   bp_parameters=bp_parameters, 
                                   apply_noise=apply_noise, 
                                   snr_expected=snr_expected, 
                                   seed_snr=num+seed_base, 
                                   append_audio=append_audio, 
                                   append_envelopes=append_envelopes, 
                                   homomorphic_dict=homomorphic_dict, 
                                   hilbert_bool=hilbert_bool, 
                                   simplicity_dict=simplicity_dict, 
                                   vfd_dict=vfd_dict, wavelet_dict=wavelet_dict, 
                                   spec_track_dict=spec_track_dict)
        
        # Agregando la información a cada arreglo
        audio_db = np.concatenate((audio_db, audio_mat), axis=0)
        s1_labels = np.concatenate((s1_labels, s1_lab), axis=0)
        s2_labels = np.concatenate((s2_labels, s2_lab), axis=0)
        
    return audio_db, s1_labels, s2_labels


def get_model_data(db_folder, test_size, seed_split, snr_list=[], ind_beg=0, ind_end=None, 
                   N=512, noverlap=0, padding_value=2, activation_percentage=0.5, 
                   append_audio=True, append_envelopes=False, apply_bpfilter=False, 
                   bp_parameters=None, apply_noise=False, homomorphic_dict=None, 
                   hilbert_bool=False,simplicity_dict=None, vfd_dict=None, 
                   wavelet_dict=None, spec_track_dict=None):
    '''Función que permite generar la base de datos final que se usará
    como entrada al modelo.
    
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
    # En primer lugar se obtiene la base de datos original
    audio_db, s1_labels, s2_labels = \
        get_heartsound_database(db_folder, 0, ind_beg=ind_beg, 
                                ind_end=ind_end, N=N, noverlap=noverlap, 
                                padding_value=padding_value, 
                                activation_percentage=activation_percentage, 
                                append_audio=append_audio, 
                                append_envelopes=append_envelopes, 
                                apply_bpfilter=apply_bpfilter, 
                                bp_parameters=bp_parameters, 
                                apply_noise=False, snr_expected=0,
                                homomorphic_dict=homomorphic_dict, 
                                hilbert_bool=hilbert_bool,
                                simplicity_dict=simplicity_dict, 
                                vfd_dict=vfd_dict, 
                                wavelet_dict=wavelet_dict, 
                                spec_track_dict=spec_track_dict)
    
    
    # Definición de la semilla base
    seed_base = 0
        
    # Para cada caso en las SNR definidas
    for snr_desired in snr_list:
        # Obteniendo la base de datos con ruido "snr_desired"
        audio_db_to, s1_labels_to, s2_labels_to = \
            get_heartsound_database(db_folder, seed_base, ind_beg=ind_beg, 
                                    ind_end=ind_end, N=N, noverlap=noverlap, 
                                    padding_value=padding_value, 
                                    activation_percentage=activation_percentage, 
                                    append_audio=append_audio, 
                                    append_envelopes=append_envelopes, 
                                    apply_bpfilter=apply_bpfilter, 
                                    bp_parameters=bp_parameters, 
                                    apply_noise=True, 
                                    snr_expected=snr_desired,
                                    homomorphic_dict=homomorphic_dict, 
                                    hilbert_bool=hilbert_bool,
                                    simplicity_dict=simplicity_dict, 
                                    vfd_dict=vfd_dict, 
                                    wavelet_dict=wavelet_dict, 
                                    spec_track_dict=spec_track_dict)
        
        # Aumentando la semilla base
        seed_base += 10000
        
        # Y agregando a la base de datos
        audio_db  = np.concatenate((audio_db , audio_db_to),  axis=0)
        s1_labels = np.concatenate((s1_labels, s1_labels_to), axis=0)
        s2_labels = np.concatenate((s2_labels, s2_labels_to), axis=0)
    
    # Se concatenan las etiquetas para tener una sola variable "Y"
    labels = np.concatenate((s1_labels, s2_labels), axis=1)
    
    # Y finalmente es separan en train y test
    X_train, X_test, Y_train, Y_test = train_test_split(audio_db, labels, 
                                                        test_size=test_size,
                                                        random_state=seed_split)
    
    return X_train, X_test, Y_train, Y_test


def save_database(folder_to_save, db_folder, ind_beg=0, ind_end=None, N=512, noverlap=0, 
                  padding_value=2, activation_percentage=0.5, append_audio=True, 
                  append_envelopes=False, apply_bpfilter=False, bp_parameters=None, 
                  homomorphic_dict=None, hilbert_bool=False, simplicity_dict=None, 
                  vfd_dict=None, wavelet_dict=None, spec_track_dict=None):
    '''Rutina que permite guardar la base de datos de sonidos cardiacos ventaneados
    en un archivo .npz, en el cual se les puede especificar el uso de envolventes de los
    sonidos de interés.
    
    Parameters
    ----------
    folder_to_save : str
        Dirección donde se almacenerá la base de datos generada
    (**kwargs) : De la función get_heartsound_database.
    '''
    # Creación del nombre del archivo
    filename = 'db_'
    
    # Si se agrega el archivo de audio sin procesar
    if append_audio:
        filename += 'raw-'
    
    # Si se agregan envolventes, se ve para cada uno de los casos
    if append_envelopes:
        if homomorphic_dict is not None:
            filename += 'hom-'
        if hilbert_bool:
            filename += 'hil-'
        if simplicity_dict is not None:
            filename += 'sbe-'
        if vfd_dict is not None:
            filename += 'vfd-'
        if wavelet_dict is not None:
            filename += 'mwp-'
        if spec_track_dict is not None:
            filename += 'spt-'
    
    # Eliminar el último guión y agregar el formato
    filename = filename.strip('-') + '.npz'
    
    # Obtención de la base de datos de audio y etiquetas para S1-S2
    audio_db, s1_labels, s2_labels = \
        get_heartsound_database(db_folder, seed_base=0, ind_beg=ind_beg, ind_end=ind_end, N=N, 
                                noverlap=noverlap, padding_value=padding_value, 
                                activation_percentage=activation_percentage, 
                                append_audio=append_audio, 
                                append_envelopes=append_envelopes, 
                                apply_bpfilter=apply_bpfilter, 
                                bp_parameters=bp_parameters, 
                                homomorphic_dict=homomorphic_dict, 
                                hilbert_bool=hilbert_bool,
                                simplicity_dict=simplicity_dict, 
                                vfd_dict=vfd_dict, wavelet_dict=wavelet_dict, 
                                spec_track_dict=spec_track_dict)
    
    # Preguntar si es que la carpeta se ha creado. En caso de que no exista, 
    # se crea una carpeta
    if not os.path.isdir(folder_to_save):
        os.makedirs(folder_to_save)
    
    # Finalmente, guardando los datos en un archivo .npz
    np.savez(f'{folder_to_save}/{filename}', X=audio_db, Y_S1=s1_labels, 
             Y_S2=s2_labels)
