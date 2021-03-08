import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile, loadmat
from filter_and_sampling import bandpass_filter
from envelope_functions import get_envelope_pack


def get_signal_eval(filename, append_audio=True, append_envelopes=False, 
                    apply_bpfilter=False, bp_parameters=None, 
                    homomorphic_dict=None, hilbert_dict=None, 
                    simplicity_dict=None, vfd_dict=None, 
                    multiscale_wavelet_dict=None, spec_track_dict=None,
                    spec_energy_dict=None, wavelet_dict=None, 
                    norm_type='minmax', append_fft=False):
    '''Función que, para un archivo especificado, permite obtener su 
    representación en matrices de delay y sus etiquetas.
    
    Parameters
    ----------
    filename : str
        Nombre del sonido a procesar.
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
    homomorphic_dict : dict, optional
        Diccionario con los parámetros de la función 'homomorphic_filter'. 
        Por defecto es None.
    hilbert_dict : bool, optional
        hilbert_dict : dict or None, optional
        Diccionario con booleanos de inclusión de ciertas envolventes.
        'analytic_env' es el booleano para agregar la envolvente 
        analítica obtenida de la magntitud de la señal analítica.
        'inst_phase' es el booleano para agregar la fase instantánea
        obtenida como la fase de la señal analítica. 'inst_freq' es el
        booleano para agregar la frecuencia instantánea obtenida como 
        la derivada de la fase de la señal analítica. Por defecto es 
        None. Si es None, no se incluye como envolvente.
    simplicity_dict : dict, optional
        Diccionario con los parámetros de la función 
        'simplicity_based_envelope'. Por defecto es None.
    vfd_dict : dict, optional
        Diccionario con los parámetros de la función 
        'variance_fractal_dimension'. Por defecto es None.
    multiscale_wavelet_dict : dict, optional
        Diccionario con los parámetros de la función 
        'stationary_multiscale_wavelets'. Por defecto es None.
    spec_track_dict : dict, optional
        Diccionario con los parámetros de la función 
        'modified_spectral_tracking'. Por defecto es None.
    spec_energy_dict : dict or None, optional
        Diccionario con los parámetros de la función 
        "modified_spectral_tracking". Por defecto es None.
    wavelet_dict : dict, optional
        Diccionario con los parámetros de la función 
        'stationary_wavelets_decomposition'. Por defecto es None.
    append_fft : bool, optional
        Booleano que indica si se agregan la FFT unilateral de audio. Por 
        defecto es False.
        
    Returns
    -------
    audio_info_matrix : ndarray
        Matriz que contiene todas las ventanas de largo N de todos los archivos 
        de audio de la base de datos escogida.
    labels_adj : ndarray
        Matriz que contiene todas las etiquetas de todos los archivos 
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
                                      hilbert_dict=hilbert_dict,
                                      simplicity_dict=simplicity_dict, 
                                      vfd_dict=vfd_dict, 
                                      multiscale_wavelet_dict=multiscale_wavelet_dict,
                                      spec_track_dict=spec_track_dict,
                                      spec_energy_dict=spec_energy_dict, 
                                      wavelet_dict=wavelet_dict, 
                                      norm_type=norm_type)
        # Concatenando
        audio_info = np.concatenate((audio_info, envelopes), axis=1)
    
    
    ### Etiquetas de los estados ###
    # Obtención del archivo de las etiquetas .mat
    data_info = loadmat(f'{filename}.mat')
        
    # Etiquetas a 50 Hz de samplerate
    labels = data_info['PCG_states']
    
    # Pasando a 1000 Hz
    labels_adj = np.repeat(labels, 20)
    
    # Retornando
    return audio_info, labels_adj


def get_signal_eval_idx(db_folder, index, append_audio=True, 
                        append_envelopes=False, apply_bpfilter=False, 
                        bp_parameters=None, homomorphic_dict=None, 
                        hilbert_dict=None, simplicity_dict=None, vfd_dict=None, 
                        multiscale_wavelet_dict=None, spec_track_dict=None,
                        spec_energy_dict=None, wavelet_dict=None, 
                        norm_type='minmax', append_fft=False):
    '''Función que, para un archivo especificado, permite obtener su 
    representación en matrices de delay y sus etiquetas.
    
    Parameters
    ----------
    filename : str
        Nombre del sonido a procesar.
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
    homomorphic_dict : dict, optional
        Diccionario con los parámetros de la función 'homomorphic_filter'. 
        Por defecto es None.
    hilbert_dict : bool, optional
        hilbert_dict : dict or None, optional
        Diccionario con booleanos de inclusión de ciertas envolventes.
        'analytic_env' es el booleano para agregar la envolvente 
        analítica obtenida de la magntitud de la señal analítica.
        'inst_phase' es el booleano para agregar la fase instantánea
        obtenida como la fase de la señal analítica. 'inst_freq' es el
        booleano para agregar la frecuencia instantánea obtenida como 
        la derivada de la fase de la señal analítica. Por defecto es 
        None. Si es None, no se incluye como envolvente.
    simplicity_dict : dict, optional
        Diccionario con los parámetros de la función 
        'simplicity_based_envelope'. Por defecto es None.
    vfd_dict : dict, optional
        Diccionario con los parámetros de la función 
        'variance_fractal_dimension'. Por defecto es None.
    multiscale_wavelet_dict : dict, optional
        Diccionario con los parámetros de la función 
        'stationary_multiscale_wavelets'. Por defecto es None.
    spec_track_dict : dict, optional
        Diccionario con los parámetros de la función 
        'modified_spectral_tracking'. Por defecto es None.
    spec_energy_dict : dict or None, optional
        Diccionario con los parámetros de la función 
        "modified_spectral_tracking". Por defecto es None.
    wavelet_dict : dict, optional
        Diccionario con los parámetros de la función 
        'stationary_wavelets_decomposition'. Por defecto es None.
    append_fft : bool, optional
        Booleano que indica si se agregan la FFT unilateral de audio. Por 
        defecto es False.
        
    Returns
    -------
    audio_info_matrix : ndarray
        Matriz que contiene todas las ventanas de largo N de todos los archivos 
        de audio de la base de datos escogida.
    labels_adj : ndarray
        Matriz que contiene todas las etiquetas de todos los archivos 
        de audio de la base de datos escogida.
    '''
    # Obtener los nombres de los archivos
    filenames = [f'{db_folder}/{name[:-4]}' for name in os.listdir(db_folder) 
                 if name.endswith('.wav')]
    
    # Filtrando por los índices
    filename = filenames[index]
    print(filename)

    # Obtención de los datos de interés para el archivo filename
    audio_data, labels = \
        get_signal_eval(filename, append_audio=append_audio, 
                        append_envelopes=append_envelopes, 
                        apply_bpfilter=apply_bpfilter, 
                        bp_parameters=bp_parameters, 
                        homomorphic_dict=homomorphic_dict, 
                        hilbert_dict=hilbert_dict, 
                        simplicity_dict=simplicity_dict, 
                        vfd_dict=vfd_dict, 
                        multiscale_wavelet_dict=multiscale_wavelet_dict, 
                        spec_track_dict=spec_track_dict,
                        spec_energy_dict=spec_energy_dict, 
                        wavelet_dict=wavelet_dict, 
                        norm_type='minmax', append_fft=append_fft)
    
    return audio_data, labels


def eval_model(model_name, ):
    pass

