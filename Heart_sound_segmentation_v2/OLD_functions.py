import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from heart_sound_physionet_management import get_windows_and_labels


def get_heartsound_database_OLD(db_folder, seed_base, ind_beg=0, ind_end=None, N=512, 
                            noverlap=0, padding_value=2, activation_percentage=0.5, 
                            append_audio=True, append_envelopes=False, 
                            apply_bpfilter=False, bp_parameters=None, 
                            apply_noise=False, snr_expected=0,
                            homomorphic_dict=None, hilbert_dict=None,
                            simplicity_dict=None, vfd_dict=None, 
                            wavelet_dict=None, spec_track_dict=None,
                            spec_energy_dict=None, norm_type='minmax',
                            append_fft=False):
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
        q_dim = q_dim + 2 if hilbert_dict else q_dim
        q_dim = q_dim + 1 if append_fft else q_dim
    
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
                                   hilbert_dict=hilbert_dict, 
                                   simplicity_dict=simplicity_dict, 
                                   vfd_dict=vfd_dict, wavelet_dict=wavelet_dict, 
                                   spec_track_dict=spec_track_dict, 
                                   spec_energy_dict=spec_energy_dict, 
                                   norm_type=norm_type, append_fft=append_fft)
        
        # Agregando la información a cada arreglo
        audio_db = np.concatenate((audio_db, audio_mat), axis=0)
        s1_labels = np.concatenate((s1_labels, s1_lab), axis=0)
        s2_labels = np.concatenate((s2_labels, s2_lab), axis=0)
        
    return audio_db, s1_labels, s2_labels


def get_model_data_OLD(db_folder, test_size, seed_split, snr_list=[], ind_beg=0, ind_end=None, 
                   N=512, noverlap=0, padding_value=2, activation_percentage=0.5, 
                   append_audio=True, append_envelopes=False, apply_bpfilter=False, 
                   bp_parameters=None, apply_noise=False, homomorphic_dict=None, 
                   hilbert_dict=False,simplicity_dict=None, vfd_dict=None, 
                   wavelet_dict=None, spec_track_dict=None, append_fft=False):
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
        get_heartsound_database_OLD(db_folder, 0, ind_beg=ind_beg, 
                                ind_end=ind_end, N=N, noverlap=noverlap, 
                                padding_value=padding_value, 
                                activation_percentage=activation_percentage, 
                                append_audio=append_audio, 
                                append_envelopes=append_envelopes, 
                                apply_bpfilter=apply_bpfilter, 
                                bp_parameters=bp_parameters, 
                                apply_noise=False, snr_expected=0,
                                homomorphic_dict=homomorphic_dict, 
                                hilbert_dict=hilbert_dict,
                                simplicity_dict=simplicity_dict, 
                                vfd_dict=vfd_dict, 
                                wavelet_dict=wavelet_dict, 
                                spec_track_dict=spec_track_dict, 
                                append_fft=append_fft)
    
    
    # Definición de la semilla base
    seed_base = 0
        
    # Para cada caso en las SNR definidas
    for snr_desired in snr_list:
        # Obteniendo la base de datos con ruido "snr_desired"
        audio_db_to, s1_labels_to, s2_labels_to = \
            get_heartsound_database_OLD(db_folder, seed_base, ind_beg=ind_beg, 
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
                                    hilbert_dict=hilbert_dict,
                                    simplicity_dict=simplicity_dict, 
                                    vfd_dict=vfd_dict, 
                                    wavelet_dict=wavelet_dict, 
                                    spec_track_dict=spec_track_dict,
                                    append_fft=append_fft)
        
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
                  homomorphic_dict=None, hilbert_dict=None, simplicity_dict=None, 
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
        if hilbert_dict is not None:
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
        get_heartsound_database_OLD(db_folder, seed_base=0, ind_beg=ind_beg, ind_end=ind_end, N=N, 
                                noverlap=noverlap, padding_value=padding_value, 
                                activation_percentage=activation_percentage, 
                                append_audio=append_audio, 
                                append_envelopes=append_envelopes, 
                                apply_bpfilter=apply_bpfilter, 
                                bp_parameters=bp_parameters, 
                                homomorphic_dict=homomorphic_dict, 
                                hilbert_dict=hilbert_dict,
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