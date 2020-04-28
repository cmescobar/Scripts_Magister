import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from ast import literal_eval
from sklearn.decomposition import NMF
from matplotlib.widgets import Button
from fading_functions import fade_connect_signals
from filter_and_sampling import resampling_by_points
from math_functions import wiener_filter, raised_cosine_fading
from descriptor_functions import get_spectrogram, get_inverse_spectrogram


def nmf_decomposition(signal_in, samplerate, n_components=2, N=2048, noverlap=1024, 
                      padding=0, window='hamming', whole=False, alpha_wiener=1,  
                      wiener_filt=True, init='random', solver='cd', beta=2,
                      tol=1e-4, max_iter=200, alpha_nmf=0, l1_ratio=0,
                      random_state=None, W_0=None, H_0=None, same_outshape=True):
    '''Función que permite separar una señal utilizando la descomposición NMF,
    la cual usa como entrada al sistema el espectrograma de la señal de audio.
    Además utiliza todos los parámetros relevantes para este estudio del comando
    NMF programado en la librería sklearn [2].
        
    Parameters
    ----------
    signal_in : ndarray or list
        Señal a descomponer.
    samplerate : int
        Tasa de muestreo de la señal.
    n_components : int, optional
        Cantidad de componentes a separar la señal. Por defecto es 2.
    N : int, optional
        Cantidad de puntos utilizados en cada ventana de la STFT. Por defecto es 2048.
    noverlap : float, optional
        Cantidad de puntos de traslape que existe entre una ventana y la siguiente al 
        calcular la STFT. Por defecto es 1024
    padding : int, optional
        Cantidad de ceros añadidos al final para aplicar zero padding. Por defecto es 0.
    window : {None, 'hamming', 'hann', 'nutall', 'tukey'}, optional
        Opciones para las ventanas a utilizar en el cálculo de cada segmento del STFT.
        En caso de elegir None, se asume la ventana rectangular. Por defecto es 'hamming'.
    whole : bool, optional
        Indica si se retorna todo el espectro de frecuencia de la STFT o solo la mitad 
        (por redundancia). True lo entrega completo, False la mtiad. Por defecto es False.
    alpha_wiener : int, optional
        Exponente alpha del filtro de Wiener. Por defecto es 1.
    wiener_filt : bool, optional
        Indica si se aplica el filtro de wiener una vez separado ambas componentes.
        Por defecto es True.
    init : {'random', 'custom'}, optional
        Opción de puntos de inicio de la descomposición. 'random' inicia con puntos al
        azar, y 'custom' permite ingresar matrices en "W_0" y "H_0" como puntos iniciales.
        Por defecto es 'random'.
    solver : {'cd', 'mu'}, optional
        Solver numérico a usar. Por defecto es 'cd'.
    beta : {'frobenius', 'kullback-leibler', 'itakura-saito'}, float or string, optional
        Definición de la beta divergencia. Por defecto es 'frobenius' (o 2).
    tol: float, optional
        Tolerancia de la condición de parada. Por defecto es 1e-4.
    max_iter: int, optional
        Cantidad máxima de iteraciones. Por defecto es 200.
    alpha_nmf: float, optional
        Constante que multiplica los términos de regulación en la resolución del problema.
        Por defecto es 0.
    l1_ratio : float, optional
        Parámetro de regulación usado en 'cd'. Por defecto es 0.
    random_state : int, RandomState instance or None, optional
        En caso de ser un "int", actúa como semilla. Si es una instancia "RandomState",
        la variable es el generador de números aleatorios. Si es "None", el número aleatorio
        es un número aleatorio generado por np.random. Por defecto es None.
    W_0 : None or ndarray, optional
        Punto de inicio para W. Por defecto es None.
    H_0 : None or ndarray, optional
        Punto de inicio para H. Por defecto es None.
    same_outshape : bool, optional
        'True' para que la salida tenga el mismo largo que la entrada. 'False' entrega el
        largo de la señal obtenido después de la STFT. Por defecto es "True".
    
    Returns
    -------
    components : list
        Lista que contiene las componentes en el dominio del tiempo.
    Y_list : list
        Lista que contiene las componentes en espectrogramas.
    X : ndarray
        Magnitud del spectrograma de la señal de entrada (entrada NMF).
    W : ndarray
        Matriz W (plantillas espectrales) de la descomposición NMF.
    H : ndarray
        Matriz H (plantillas temporales) de la descomposición NMF.
    
    References
    ----------
    [1] Tutorial: https://ccrma.stanford.edu/~njb/teaching/sstutorial/part2.pdf
    [2] https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
    '''
    # Propiedad del overlap
    noverlap = 0 if noverlap <= 0 else noverlap
    noverlap = noverlap if noverlap < N else N - 1
    
    # Definición de una lista que almacene las componentes
    components = []
    # Listas de valores de interés
    Y_list = []
    
    # Obteniendo el espectrograma
    _, _, S = get_spectrogram(signal_in, samplerate, N=N, padding=padding, 
                              noverlap=noverlap, window=window, whole=whole)
    
    # Definiendo la magnitud del espectrograma (elemento a estimar)
    X = np.abs(S)
    
    # Definiendo el modelo de NMF
    model = NMF(n_components=n_components, init=init, solver=solver,
                beta_loss=beta, tol=tol, max_iter=max_iter, 
                random_state=random_state, alpha=alpha_nmf, l1_ratio=l1_ratio)
    
    # Ajustando W
    if init == 'random':
        W = model.fit_transform(X)
    elif init == 'custom':
        W = model.fit_transform(X, W=W_0, H=H_0)
    else:
        raise Exception('Opción de inicio no disponible. Por favor intente nuevamente.')
    
    # Ajustando H
    H = model.components_
    
    # Se define la función de transformación para Yi
    if wiener_filt:
        # Se aplica filtro de Wiener
        filt = lambda source_i: wiener_filter(X, source_i, W, H, alpha=alpha_wiener)
    else:
        # Solo se entrega la multiplicación W_i * H_i
        filt = lambda source_i: source_i
    
    # Obteniendo las fuentes
    for i in range(n_components):
        source_i = np.outer(W[:,i], H[i])
        
        # Aplicando el filtro
        Yi = filt(source_i) * np.exp(1j * np.angle(S))
        
        # Y posteriormente la transformada inversa
        yi = get_inverse_spectrogram(Yi, N=N, noverlap=noverlap, window=window, 
                                     whole=whole)
        
        if same_outshape:
            yi = yi[:len(signal_in)]
        
        # Agregando a la lista de componentes
        components.append(np.real(yi))
        Y_list.append(Yi)
        
    return components, Y_list, X, W, H


def nmf_decomposition_2(signal_in, samplerate, n_components=2, N=2048, noverlap=1024,
                      padding=0, window='hamming', whole=False, alpha_wiener=1,
                      wiener_filt=True):
    '''Función que permite separar una señal utilizando la descomposición NMF,
    la cual usa como entrada al sistema el espectrograma de la señal de audio.
    
    Parameters
    ----------
    signal_in : ndarray or list
        Señal a descomponer.
    samplerate : int
        Tasa de muestreo de la señal.
    n_components : int, optional
        Cantidad de componentes a separar la señal. Por defecto es 2.
    N : int, optional
        Cantidad de puntos utilizados en cada ventana de la STFT. Por defecto es 2048.
    noverlap : float, optional
        Cantidad de puntos de traslape que existe entre una ventana y la siguiente al 
        calcular la STFT. Por defecto es 1024.
    padding : int, optional
        Cantidad de ceros añadidos al final para aplicar zero padding. Por defecto es 0.
    window : {None, 'hamming', 'hann', 'nutall', 'tukey'}, optional
        Opciones para las ventanas a utilizar en el cálculo de cada segmento del STFT.
        En caso de elegir None, se asume la ventana rectangular. Por defecto es 'hamming'.
    whole : bool, optional
        Indica si se retorna todo el espectro de frecuencia de la STFT o solo la mitad 
        (por redundancia). True lo entrega completo, False la mtiad. Por defecto es False.
    alpha : int, optional
        Exponente alpha del filtro de Wiener. Por defecto es 1.
    wiener_filt : bool, optional
        Indica si se aplica el filtro de wiener una vez separado ambas componentes.
        Por defecto es True. 
    
    Returns
    -------
    components : list
        Lista que contiene las componentes en el dominio del tiempo.
    Y_list : list
        Lista que contiene las componentes en espectrogramas.
    X : ndarray
        Magnitud del spectrograma de la señal de entrada (entrada NMF).
    W : ndarray
        Matriz W (plantillas espectrales) de la descomposición NMF.
    H : ndarray
        Matriz H (plantillas temporales) de la descomposición NMF.
    
    References
    ----------
    - Tutorial: https://ccrma.stanford.edu/~njb/teaching/sstutorial/part2.pdf 
    '''
    # Definición de una lista que almacene las componentes
    components = []
    # Listas de valores de interés
    Y_list = []
    
    # Obteniendo el espectrograma
    _, _, S = get_spectrogram(signal_in, samplerate, N=N, padding=padding, 
                              noverlap=noverlap, window=window, whole=whole)
    
    # Definiendo la magnitud
    X = np.abs(S)
    
    # Definiendo el modelo de NMF
    model = NMF(n_components=n_components)#, beta_loss='itakura-saito', solver='mu')
    
    # Ajustando
    W = model.fit_transform(X)
    H = model.components_
    
    # Se define la función de transformación para Yi
    if wiener_filt:
        filt = lambda source_i: wiener_filter(X, source_i, W, H, alpha=alpha_wiener)
    else:
        filt = lambda source_i: source_i
    
    # Obteniendo las fuentes
    for i in range(n_components):
        source_i = np.outer(W[:,i], H[i])
        
        # Aplicando el filtro
        Yi = filt(source_i) * np.exp(1j * np.angle(S))
        
        # Y posteriormente la transformada inversa
        yi = get_inverse_spectrogram(Yi, N=N, noverlap=noverlap, window=window, 
                                     whole=whole)
        
        # Agregando a la lista de componentes
        components.append(np.real(yi))
        Y_list.append(Yi)
        
    return components, Y_list, X, W, H


def get_components_HR_sounds(filepath, sep_type='to all', assign_method='manual',  
                             n_components=2, N=2048, N_lax=1500, N_fade=500, 
                             noverlap=1024, padding=0, window='hamming', whole=False, 
                             alpha_wiener=1, wiener_filt=True, init='random', 
                             solver='cd', beta=2, tol=1e-4, max_iter=200, 
                             alpha_nmf=0, l1_ratio=0, random_state=0, 
                             W_0=None, H_0=None, plot_segments=False):
    '''Función que permite generar los archivos de componentes
    '''
    # Lista de los archivos de la carpeta
    filenames = [i for i in os.listdir(filepath) if i.endswith('.wav')]
    
    # Definición de la carpeta a almacenar
    if sep_type == 'to_all':
        filepath_to_save = f'{filepath}/Components/Separation to all'
    elif sep_type == 'on segments':
        filepath_to_save = f'{filepath}/Components/Separation on segments'
    else:
        raise Exception('Opción no válida para tipo de separación.')
    
    # Preguntar si es que la carpeta que almacenará las imágenes se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(filepath_to_save):
        os.makedirs(filepath_to_save)
    
    for audio_name in tqdm(filenames, desc='NMF decomp', ncols=70):
        # Dirección del archivo en la carpeta madre. Este archivo es el que se descompondrá
        dir_audio = f"{filepath}/{audio_name}"
                
        if sep_type == 'to_all':            
            # Lectura del archivo
            audio_file, samplerate = sf.read(dir_audio)
            
            # Aplicando la descomposición
            comps, _, _, _, _ = nmf_decomposition(audio_file, samplerate, 
                                                  n_components=n_components, 
                                                  N=N, noverlap=noverlap, padding=padding,
                                                  window=window, whole=whole, 
                                                  alpha_wiener=alpha_wiener,
                                                  wiener_filt=wiener_filt, init=init, 
                                                  solver=solver, beta=beta, tol=tol, 
                                                  max_iter=max_iter, alpha_nmf=alpha_nmf, 
                                                  l1_ratio=l1_ratio, random_state=random_state,
                                                  W_0=W_0, H_0=H_0)
            
            # Definiendo el nombre de los archivos
            name_to_save = f'{filepath_to_save}/{audio_name.strip(".wav")}'
            
            # Grabando cada componente
            sf.write(f'{name_to_save} Comp 1.wav', comps[0], samplerate)
            sf.write(f'{name_to_save} Comp 2.wav', comps[1], samplerate)
            
        elif sep_type == 'on segments':
            resp_signal, heart_signal = \
                nmf_applied_interest_segments(dir_audio, assign_method=assign_method, 
                                              n_components=n_components, N=N, 
                                              N_lax=N_lax, N_fade=N_fade, noverlap=noverlap, 
                                              padding=padding, window=window, whole=whole,
                                              alpha_wiener=alpha_wiener, 
                                              wiener_filt=wiener_filt, init=init, 
                                              solver=solver, beta=beta, tol=tol, 
                                              max_iter=max_iter, alpha_nmf=alpha_nmf, 
                                              l1_ratio=l1_ratio, random_state=random_state, W_0=W_0, H_0=H_0, plot_segments=plot_segments)


def nmf_applied_interest_segments(dir_file, assign_method='manual', n_components=2, 
                                  N=2048, N_lax=1500, N_fade=500, noverlap=1024, padding=0,
                                  window='hamming', whole=False, alpha_wiener=1, 
                                  wiener_filt=True, init='random', solver='cd', beta=2,
                                  tol=1e-4, max_iter=200, alpha_nmf=0, l1_ratio=0,
                                  random_state=0, W_0=None, H_0=None, 
                                  plot_segments=False):
    '''Función que permite obtener la descomposición NMF de una señal (ingresando su
    ubicación en el ordenador) 
    
    Parameters
    ----------
    dir_file : str
        Dirección del archivo de audio a segmentar.
    assign_method : {'auto', 'manual'}, optional
        Método de separación de sonidos. Para 'auto', se utiliza una lista de etiquetas
        creadas manualmente. Para 'manual' se etiqueta segmento a segmento cada componente, 
        las cuales son guardadas en un archivo .txt. Por defecto es 'manual'.
    n_components : int, optional
        Cantidad de componentes a separar la señal. Por defecto es 2.
    N : int, optional
        Cantidad de puntos utilizados en cada ventana de la STFT. Por defecto es 2048.
    N_lax : int, optional
        Cantidad de puntos adicionales que se consideran para cada lado más allá de los
        intervalos dados. Por defecto es 1500.
    N_fade : int, optional
        Cantidad de puntos utilizados para que la ventana se mezcle con fade. Por defecto
        es 500.
    noverlap : float, optional
        Cantidad de puntos de traslape que existe entre una ventana y la siguiente al 
        calcular la STFT. Por defecto es 1024.
    padding : int, optional
        Cantidad de ceros añadidos al final para aplicar zero padding. Por defecto es 0.
    window : {None, 'hamming', 'hann', 'nutall', 'tukey'}, optional
        Opciones para las ventanas a utilizar en el cálculo de cada segmento del STFT.
        En caso de elegir None, se asume la ventana rectangular. Por defecto es 'hamming'.
    whole : bool, optional
        Indica si se retorna todo el espectro de frecuencia de la STFT o solo la mitad 
        (por redundancia). True lo entrega completo, False la mtiad. Por defecto es False.
    alpha_wiener : int, optional
        Exponente alpha del filtro de Wiener. Por defecto es 1.
    wiener_filt : bool, optional
        Indica si se aplica el filtro de wiener una vez separado ambas componentes.
        Por defecto es True.
    **kwargs : Revisar nmf_decomposition para el resto.
    
    Returns
    -------
    resp_signal : ndarray
        Señal respiratoria aproximada mediante la descomposición.
    heart_signal : ndarray
        Señal cardíaca aproximada mediante la descomposición.
    '''
    # Abriendo el archivo de sonido
    signal_in, samplerate = sf.read(f'{dir_file}')
    
    # Definición de la carpeta donde se ubica
    filepath = '/'.join(dir_file.split('/')[:-1])
    
    # Definición del nombre del archivo
    filename = dir_file.strip('.wav').split('/')[-1]
        
    # Definición de la carpeta a guardar los segmentos
    filepath_to_save = f'{filepath}/Components/Separation on segments/{filename}'
    
    # Preguntar si es que la carpeta que almacenará las imágenes se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(filepath_to_save) and plot_segments:
        os.makedirs(filepath_to_save)
    
    # A partir del nombre del archivo es posible obtener también su lista de intervalos.
    ## Primero se obtiene el nombre del sonido cardíaco a revisar
    file_heart = filename.split(' ')[-1].strip('.wav')
    
    ## Luego se define la dirección del archivo de segmentos
    segment_folder = f'Database_manufacturing/db_heart/Manual Combinations'
    
    try:
        ## Y se retorna el archivo de segmentos correspondiente al nombre
        name_file_segment = [i for i in os.listdir(segment_folder) 
                             if i.endswith('.txt') and file_heart in i][0]
        
        ## Se abre el archivo y se obtiene la lista de intervalos
        with open(f'{segment_folder}/{name_file_segment}', 'r', encoding='utf8') as data:
            interval_list = literal_eval(data.readline())
        
    except:
        raise Exception(f'No se logra encontrar el archivo con intervalos cardíacos de '
                        f'{filename}')
    
    # Definición de la señal respiratoria de salida
    resp_signal = np.copy(signal_in)
    # Definición de la señal cardíaca de salida
    heart_signal = np.zeros(len(signal_in))
    
    # Definición del diccionario que contiene los parámetros de simulación
    dict_simulation = {'n_components': n_components, 'N': N, 'N_lax': N_lax,
                       'N_fade': N_fade, 'noverlap': noverlap, 'padding': padding,
                       'window': window, 'whole': whole, 'alpha_wiener': alpha_wiener,
                       'wiener_filt': wiener_filt, 'init': init, 
                       'solver': solver, 'beta': beta, 'tol': tol, 
                       'max_iter': max_iter, 'alpha_nmf': alpha_nmf, 
                       'l1_ratio': l1_ratio, 'random_state': random_state}
    
    # Control de parámetros simulación (consultar repetir) y asignación de id's
    dict_simulation, comps_choice, heart_comp_labels, plot_segments, assign_method =\
        _manage_registerdata_segments_nmf(dict_simulation, assign_method, 
                                      filepath_to_save, plot_segments)
    
    # Aplicando NMF en cada segmento de interés
    for num, interval in enumerate(interval_list, 1):
        # Definición del límite inferior y superior
        lower = interval[0] - N_lax
        upper = interval[1] + N_lax
        
        # Definición del segmento a transformar
        segment = signal_in[lower - N_fade:upper + N_fade]
        
        # Aplicando NMF 
        comps, _, _, W, H = nmf_decomposition(segment, samplerate, 
                                              n_components=n_components, 
                                              N=N, noverlap=noverlap, padding=padding,
                                              window=window, whole=whole, 
                                              alpha_wiener=alpha_wiener,
                                              wiener_filt=wiener_filt, init=init, 
                                              solver=solver, beta=beta, tol=tol, 
                                              max_iter=max_iter, alpha_nmf=alpha_nmf, 
                                              l1_ratio=l1_ratio, random_state=random_state,
                                              W_0=W_0, H_0=H_0)
    
        # Definición de guardado para cada id
        filepath_to_save_id = f'{filepath_to_save}/id {dict_simulation["id"]}'

        # Preguntar si es que la carpeta que almacenará las imágenes se ha
        # creado. En caso de que no exista, se crea una carpeta
        if not os.path.isdir(filepath_to_save_id):
            os.makedirs(filepath_to_save_id)
        
        # Graficos
        if plot_segments:
            decision_in_plot = _plot_segments_nmf(signal_in, comps, W, H, N_fade, 
                                                  lower, upper, num, filepath_to_save_id, 
                                                  assign_method)
            
        if assign_method == 'auto':
            # Se obtiene la decisión etiquetada como sonido cardíaco
            heart_decision = comps_choice[num - 1]
        
        elif assign_method == 'manual':
            # Se pregunta para la decisión del sonido cardíaco
            heart_decision = decision_in_plot
            
            # Se agrega la decisión a la lista de labels
            heart_comp_labels.append(heart_decision)
        
        # Y se complementa para el sonido respiratorio
        resp_decision = 0 if heart_decision == 1 else 1
        
        # Definición de la lista de señales a concatenar con fading para el corazón
        heart_connect = (heart_signal[:lower], comps[heart_decision][:len(segment)],
                         heart_signal[upper:])
        
        # Definición de la lista de señales a concatenar con fading para la respiración
        resp_connect = (resp_signal[:lower], comps[resp_decision][:len(segment)],
                        resp_signal[upper:])
        
        # Aplicando fading para cada uno
        heart_signal = fade_connect_signals(heart_connect, N=N_fade, beta=1)
        resp_signal = fade_connect_signals(resp_connect, N=N_fade, beta=1)
    
    if assign_method == 'manual' and plot_segments:
        # Finalmente, se escribe las etiquetas de las componentes
        with open(f'{filepath_to_save}/Heart comp labels.txt', 'a', encoding='utf8') as data:
            # Se agrega la información de las etiquetas generadas
            dict_simulation['heart_comp_labels'] = heart_comp_labels
            
            # Y se guarda
            data.write(f'{dict_simulation}\n')

    # Finalmente, grabando los archivos de audio
    sf.write(f'{filepath_to_save_id}/Respiratory signal.wav', resp_signal, 44100)
    sf.write(f'{filepath_to_save_id}/Heart signal.wav', heart_signal, 44100)
    
    return resp_signal, heart_signal


def comparison_components_nmf_ground_truth(filepath, sep_type='to all', plot_signals=False,
                                           plot_show=False):
    '''
    
    Parameters
    ----------
    sep_type : {'to all', 'on segments'}, optional
        Selección del tipo de la base generada por la separación mediante NMF. 'to all' 
        usa la base que separa toda la señal con NMF y 'on_segments' usa la base que
        solamente separa con NMF en segmentos de interés. Por defecto es 'to all'. 
    '''
    # Lista de los archivos de la carpeta
    filenames = [i for i in os.listdir(filepath) if i.endswith('.wav')]
    
    # Definición de la carpeta de componentes
    if sep_type == 'to all':
        filepath_comps = f'{filepath}/Components/Separation to all'
    elif sep_type == 'on segments':
        filepath_comps = f'{filepath}/Components/Separation on segments'
    else:
        raise Exception('Opción no válida para tipo de separación.')
    
    # Definición de carpeta donde se encuentran los archivos de sonido cardíaco
    heart_dir = 'Database_manufacturing/db_heart/Manual combinations'
    
    # Definición de carpeta donde se encuentran los archivos de sonido cardíaco
    resp_dir = 'Database_manufacturing/db_respiratory/Adapted'
    
    for audio_name in tqdm(filenames, desc='Comp. analysis', ncols=70):
        # Separando el nombre para lo que corresponda según la codificación utilizada
        _, resp_name, heart_name = audio_name.strip('.wav').split(' ')
        
        # Obtención del sonido respiratorio
        audio_resp, _ = sf.read(f'{resp_dir}/{resp_name}.wav')
        
        # Obtención del sonido cardíaco
        audio_heart, _ = sf.read(f'{heart_dir}/{heart_name}.wav')
        
        # Obtención de las componentes
        comp_1, _ = sf.read(f'{filepath_comps}/{audio_name.strip(".wav")} Comp 1.wav')
        comp_2, _ = sf.read(f'{filepath_comps}/{audio_name.strip(".wav")} Comp 2.wav')
        
        # Resampleando para igual número de muestras
        comp_1 = resampling_by_points(comp_1, 44100, len(audio_resp), trans_width=50,
                                      resample_method='interp1d', lp_method='fir', 
                                      fir_method='kaiser', gpass=1, gstop=80, 
                                      correct_by_gd=True, gd_padding='periodic',
                                      plot_filter=False, normalize=True)
        comp_2 = resampling_by_points(comp_2, 44100, len(audio_resp), trans_width=50,
                                      resample_method='interp1d', lp_method='fir', 
                                      fir_method='kaiser', gpass=1, gstop=80, 
                                      correct_by_gd=True, gd_padding='periodic',
                                      plot_filter=False, normalize=True)
        
        # Definición Heart comp
        heart_index = np.argmin((sum(abs(audio_heart - comp_1)),
                                 sum(abs(audio_heart - comp_2))))
        
        heart_comp = comp_1 if heart_index == 0 else comp_2
        
        # Definición de señal respiratoria como el complemento del resultado anterior
        resp_comp = comp_2 if heart_index == 0 else comp_1
        
        if plot_signals:
            # Ploteando las diferencias
            fig, ax = plt.subplots(2, 1, figsize=(17,7))
            
            ax[0].plot(audio_resp, label='Original')
            ax[0].plot(resp_comp, label='Componente resp')
            ax[0].set_ylabel('Signals')
            
            ax[1].plot(abs(audio_resp - resp_comp))
            ax[1].set_ylabel('Error')
            
            fig.suptitle(f'{audio_name} Respiratory')
            fig.savefig(f'{filepath_comps}/{audio_name.strip(".wav")} Resp.png')
            
            if plot_show:
                # Manager para modificar la figura actual y maximizarla
                manager = plt.get_current_fig_manager()
                manager.window.state('zoomed')
                
                plt.show()
            
            plt.close()
            
            fig, ax = plt.subplots(2, 1, figsize=(17,7))
            
            ax[0].plot(audio_heart, label='Original')
            ax[0].plot(heart_comp, label='Componente heart')
            ax[0].set_ylabel('Signals')
            
            ax[1].plot(abs(audio_heart - heart_comp))
            ax[1].set_ylabel('Error')
            
            fig.suptitle(f'{audio_name} Heart')
            fig.savefig(f'{filepath_comps}/{audio_name.strip(".wav")} Heart.png')
            
            if plot_show:
                # Manager para modificar la figura actual y maximizarla
                manager = plt.get_current_fig_manager()
                manager.window.state('zoomed')
                
                plt.show()
            
            plt.close()
            
            # Ploteando las diferencias
            fig, ax = plt.subplots(figsize=(17,7))
            
            ax.plot(audio_resp + audio_heart, label='Original')
            ax.plot(resp_comp + heart_comp, label='Componente resp')
            ax.set_ylabel('Signals')

            fig.suptitle(f'{audio_name} Sum comparation')
            fig.savefig(f'{filepath_comps}/{audio_name.strip(".wav")} Sum comps.png')
            
            if plot_show:
                # Manager para modificar la figura actual y maximizarla
                manager = plt.get_current_fig_manager()
                manager.window.state('zoomed')
                
                plt.show()
            
            plt.close()


def _decision_question_in_comps(n_components):
    '''Función que plantea la pregunta que permite decidir la componente que corresponde al
    sonido cardíaco.
    
    Parameters
    ----------
    n_components : int
        Cantidad de componentes a decidir para la señal.
    
    Returns
    -------
    decision : int
        Índice de la componente de la decisión tomada.
    '''
    while True:
        # Se pregunta
        decision = input('Seleccione el componente que corresponde al corazón: ')
        
        # Se asegura de que la respuesta sea correcta
        if decision in [str(i+1) for i in range(n_components)] and decision != '':
            decision = int(decision) - 1
            break
        else:
            print('La componente seleccionada no existe. Por favor, intente nuevamente.\n')
    
    return decision


def _plot_segments_nmf(signal_in, comps, W, H, N_fade, lower, upper, 
                       num, filepath_to_save, assign_method):
    '''Función auxiliar que permite realizar los gráficos en la función
    "nmf_applied_interest_segments".
    '''
    # Definición de la clase que usará el bóton para regular las funciones
    class Boton_seleccion:
        ''' Clase que permitirá controlar la variable de interés y los métodos
        para los botones implementados
        '''
        def __init__(self):
            self.value = None
        
        def componente_1(self, event):
            self.value = 0
            plt.close()
        
        def componente_2(self, event):
            self.value = 1
            plt.close()
    
    # Definición del backend para maximizar la ventana. Dependiendo del SO
    # puede variar. Para revisar usar comando matplotlib.get_backend()
    plt.switch_backend('TkAgg')
    
    # Creación del plot
    fig, ax = plt.subplots(1, 3, figsize=(17,7))
    
    # Plots
    ax[0].plot(comps[0], label='Comp 1')
    ax[0].plot(comps[1], label='Comp 2')
    ax[0].plot(signal_in[lower-N_fade:upper+N_fade], label='Original')
    ax[0].legend(loc='upper right')
    ax[0].set_title('Señales')
    
    ax[1].plot(W[:,0], label='Comp 1')
    ax[1].plot(W[:,1], label='Comp 2')
    ax[1].legend(loc='upper right')
    ax[1].set_xlim([0,100])
    ax[1].set_title('Matriz W')
    
    ax[2].plot(H[0], label='Comp 1')
    ax[2].plot(H[1], label='Comp 2')
    ax[2].legend(loc='upper right')
    ax[2].set_title('Matriz H')
    
    # Definición del título
    fig.suptitle(f'Segment #{num}')
    
    # Se guarda la figura
    fig.savefig(f'{filepath_to_save}/Segment {num}.png') 
    
    if assign_method == 'manual':
        # Manager para modificar la figura actual y maximizarla
        manager = plt.get_current_fig_manager()
        manager.window.state('zoomed')
        
        # Re ajuste del plot
        plt.subplots_adjust(bottom=0.15)
        
        # Definición de la clase que se utilizará para el callback
        callback_seleccion = Boton_seleccion()
        
        # Dimensión y ubicación de los botones
        ax_comp1 = plt.axes([0.7, 0.05, 0.1, 0.05])
        ax_comp2 = plt.axes([0.81, 0.05, 0.1, 0.05])
        
        # Definición de los botones
        bcomp1 = Button(ax_comp1, 'Componente 1', color='C0')
        bcomp2 = Button(ax_comp2, 'Componente 2', color='C1')
        
        # Y se conecta cada uno a una función en la clase definida arriba
        bcomp1.on_clicked(callback_seleccion.componente_1)
        bcomp2.on_clicked(callback_seleccion.componente_2)
        
        # Se muestra el gráfico
        plt.show()
        
        # Definición del valor a retornar
        to_return = callback_seleccion.value
        
        if to_return is None:
            print('Seleccione una opción...')
            return _plot_segments_nmf(signal_in, comps, W, H, N_fade, lower, upper, 
                                      num, filepath_to_save, assign_method)
        else:
            return to_return 
    
    # Se cierra la figura
    plt.close()


def _manage_registerdata_segments_nmf(dict_simulation, assign_method, 
                                      filepath_to_save, plot_segments):
    '''Rutina que permite realizar la gestión del registro de las simulaciones 
    realizadas. Se realiza el guardado los datos, corroboración de simulaciones
    ya existentes (preguntando si es que se quieren realizar nuevamente) y 
    creación de etiquetas para la señal simulada.
    
    Se asigna una etiqueta a cada señal nueva contando la cantidad de líneas 
    utilizadas anteriormente. Es decir, el id corresponde al número de la línea
    en el archivo
    '''
    # Definición previa de la selección de variables
    comps_choice = None
    heart_comp_labels = None
    
    # Definición del contador de líneas (que servirá como id de cada simulación)
    id_count = 1
    
    # Se intenta porque puede ser que el archivo no esté creado
    try:
        # Se obtienen las componetes a partir de las etiquetas generadas
        with open(f'{filepath_to_save}/Heart comp labels.txt', 'r', encoding='utf8') as data:
            for line in data:
                # Obtención del diccionario con el contexto
                dict_line = literal_eval(line.strip())
                
                # Se obtienen los componentes tentativos
                possible_comps = dict_line['heart_comp_labels']
                possible_id = dict_line['id']
                
                # Y se eliminan los labels para luego realizar una comparación
                del dict_line['heart_comp_labels']
                del dict_line['id']
                
                # Y se comparan los parámetros utilizados
                if dict_simulation == dict_line:
                    # Si son iguales, se asigna la variable
                    comps_choice = possible_comps
                    
                    # Y se agregan los valores al diccionario de la simulación
                    dict_id = {'id': possible_id}
                    
                    #Concatenando
                    dict_id.update(dict_simulation)
                    dict_simulation = dict_id
                    dict_simulation['heart_comp_labels'] = possible_comps
                    break
                
                # Se aumenta el contador por cada línea leída
                id_count += 1
    
    # En caso de que no exista
    except:
        # Definitivamente debe ingresarse manualmente
        print('Advertencia (!): Debido a que es la primera vez que se crea el archivo, '
              'el método debe ser manual (assign_method == "manual").\n')
        assign_method = 'manual'
    
    # Método de trabajo
    if assign_method == 'auto':
        if comps_choice is None:
            raise Exception('No existe una base de etiquetas para el archivo dado estos '
                            'parámetros, por lo que la asignación automática no puede '
                            'ser realizada (assign_method == "auto")\n')
    
    elif assign_method == 'manual':
        if comps_choice is None:
            # Si se usa el modo manual, se crea una lista que vaya guardando las etiquetas
            heart_comp_labels = list()
            # Y seleccionar modo plot por defecto
            plot_segments = True
            
            # Se agrega la id definida al diccionario de simulación
            dict_id = {'id': id_count}
            
            #Concatenando
            dict_id.update(dict_simulation)
            dict_simulation = dict_id
            
        else:
            while True:
                # Preguntar si es que se quiere hacer la simulación de nuevo en caso de que 
                # ya exista
                continue_decision = input('La simulación que quieres hacer ya fue realizada '
                                          f'(en id: {dict_simulation["id"]}), '
                                          '¿deseas hacerla nuevamente? [y/n]: ')
                
                if continue_decision == 'y':
                    # Si se usa el modo manual, se crea una lista que vaya guardando las etiquetas
                    heart_comp_labels = list()
                    # Y seleccionar modo plot por defecto
                    plot_segments = True
                    break
                
                elif continue_decision == 'n':
                    print('Función terminada.\n')
                    exit()
                    
                print('Opción no válida.\n')
    
    return dict_simulation, comps_choice, heart_comp_labels, plot_segments, assign_method


# Module testing
filepath = 'Database_manufacturing/db_HR/Source Separation/Seed-0 - 1_Heart 1_Resp 0_White noise'

get_components_HR_sounds(filepath, sep_type='on segments', assign_method='manual',  
                        n_components=2, N=2048, N_lax=1500, N_fade=500, 
                        noverlap=1024, padding=0, window='hamming', whole=False, 
                        alpha_wiener=1, wiener_filt=True, init='random', 
                        solver='cd', beta=2, tol=1e-4, max_iter=200, 
                        alpha_nmf=0, l1_ratio=0, random_state=0, 
                        W_0=None, H_0=None, plot_segments=False)

'''
get_components_HR_sounds(filepath, sep_type='on segments', assign_method='manual',  
                        n_components=2, N=2048, N_lax=1500, N_fade=500, 
                        noverlap=1024, padding=0, window='hamming', whole=False, 
                        alpha_wiener=1, wiener_filt=True, init='random', 
                        solver='cd', beta=2, tol=1e-4, max_iter=200, 
                        alpha_nmf=0, l1_ratio=0, random_state=0, 
                        W_0=None, H_0=None, plot_segments=False)


#comparison_components_nmf_ground_truth(filepath, plot_signals=True, plot_show=True)
'''