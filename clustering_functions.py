import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from filter_and_sampling import resampling_by_points
from math_functions import _correlation, _correlations, cosine_similarities


def spectral_correlation_criteria(W_i, W_dic, fcut_bin, measure='cosine',
                                  i_selection='max'):
    '''Función que retorna el valor de verdad de la pertenencia de una componente
    X_i al cluster de sonidos cardíacos, utilizando el criterio de la correlación 
    espectral entre la información de la matriz W de la componente y un diccionario
    preconfigurado de sonido cardíacos. Se escoge el máximo y luego ese valor se 
    compara con un umbral. Retorna True si es que corresponde a sonido cardíaco y 
    False si es que no.
    
    Parameters
    ----------
    W_i : ndarray
        Información espectral de la componente i a partir de la matriz W.
    W_dic : array_like
        Arreglo con las componentes preconfiguradas a partir de sonidos puramente 
        cardíacos externos a esta base de datos.
    fcut_bin : int
        Límite de frecuencia para considerar la medida de relación (en bins). Se
        corta debido a que en general los valores para frecuencias altas son cero,
        se parecen mucho, generando distorsión en las muestras.
    measure : {'cosine', 'correlation'}, optional
        Medida utilizada para calcular la similaridad. 'cosine' usa similaridad de 
        coseno, mientras que 'correlation' calcula el coeficiente de Pearson. Por
        defecto es 'cosine'.
    i_selection : {'max', 'mean'}, optional
        Método de selección del componente que identifica la similarida. Con 'max' 
        se utiliza el máximo de todas las medidas, mientras que con 'mean' se utiliza
        el promedio. Por defecto es 'max'.
    
    Referencias
    -----------
    [1] Canadas-Quesada, et. al (2017). A non-negative matrix factorization 
        approach  based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier. Chapter 3.2.1.
    '''    
    # Definición de la lista de valores a guardar
    if measure == 'cosine':
        SC_ij = cosine_similarities(W_dic[:,:fcut_bin], W_i[:fcut_bin])
    elif measure == 'correlation':
        SC_ij = _correlations(W_dic[:,:fcut_bin], W_i[:fcut_bin])
    else:
        raise Exception('Opción para "measure" no válida.')
    
    # Selección del índice
    if i_selection == 'max':
        S_i = max(SC_ij)
    elif i_selection == 'mean':
        S_i = np.mean(SC_ij)
    else:
        raise Exception('Opción para "i_selection" no válida.')
    
    return S_i


def roll_off_criteria(X, f1, f2, percentage=0.85):
    '''Función que retorna el valor de verdad de la pertenencia de una componente
    X_i al cluster de sonidos cardíacos, utilizando el criterio de la comparación 
    de energía bajo una frecuencia f0 con respecto a la energía total de sí misma. 
    Retorna True si es que corresponde a sonido cardíaco y False si es que no.
    
    Parameters
    ----------
    X : ndarray
        Espectrograma de la componente a clasificar.
    f1 : float
        Valor de la frecuencia de corte inferior en bins.
    f2 : float
        Valor de la frecuencia de corte superior en bins.
    percentage : float, optional
        Ponderación para la energía total del espectrograma en el criterio de
        selección. Por defecto es 0.85.
    
    Referencias
    -----------
    [1] Canadas-Quesada, et. al (2017). A non-negative matrix factorization 
        approach  based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier. Chapter 3.2.2.
    '''
    # Propiedad del porcentaje
    percentage = percentage if percentage <= 1 else 1
    
    # Definición del Roll-Off
    ro = np.sum(abs(X[f1:f2,:]) ** 1)
    
    # Definición de la energía del espectrograma
    er = np.sum(abs(X) ** 1)
    
    # Finalmente, se retorna la cualidad de verdad del criterio
    return ro >= percentage * er


def temporal_correlation_criteria(H_i, P, measure='correlation', H_binary=True, 
                                  show_plot=False):
    '''Función que retorna el valor de verdad de la pertenencia de una componente
    X_i al cluster de sonidos cardíacos, utilizando el criterio de la correlación
    temporal entre la información de la matriz H de la componente i, y el heart 
    rate del sonido cardiorespiratorio original. Retorna True si es que corresponde 
    a sonido cardíaco y False si es que no. 
    
    H_i : ndarray
        Información temporal de la componente i a partir de la matriz H.
    P : ndarray
        Heart rate de la señal cardiorespiratoria.
    measure : {'correlation', 'q_equal'}, optional
        
    H_binary : bool, optional
        Valor del umbral de decisión de la para clasificar una componente como
        sonido cardíaco. Por defecto es 0.
    
    Referencias
    -----------
    [1] Canadas-Quesada, et. al (2017). A non-negative matrix factorization 
        approach  based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier. Chapter 3.2.3.
    '''
    # Selección del tipo de H a procesar
    if H_binary:
        # Obtener el promedio de la señal
        H_i_mean =  np.mean(H_i)
        # Preprocesando la matriz H_i
        H_in = np.where(H_i >= H_i_mean, 1, 0)
    else:
        H_in = H_i
    
    # Gráfico de P y H_in
    if show_plot:
        plt.plot(P)
        plt.plot(H_in)
        plt.show()
    
    # Selección medida de desempeño
    if measure == 'correlation':
        # Calculando la correlación
        TC_i = _correlation(P, H_in)
    elif measure == 'q_equal':
        TC_i = sum(np.equal(P, H_in)) / len(H_in)
    
    return TC_i


def spectral_correlation_test(W, samplerate, fcut, N, noverlap, n_comps_dict, 
                              beta, filepath_data, padding=0, repeat=0, 
                              measure='cosine', i_selection='max', threshold=0.5):
    '''Función que permite realizar un testeo de cada componente de la matriz
    W sobre el diccionario construido a partir de sonidos puramente cardíacos.
    Retorna un arreglo de booleanos. Si la entrada i es True, se trata de un
    sonido cardíaco. Si es False, se trata de un sonido respiratorio.
    
    Parameters
    ----------
    W : ndarray
        Matriz W de la descomposición NMF.
    samplerate : float
        Tasa de muestreo de la base de datos utilizada en el diccionario. Esta debe coincidir
        con la tasa de muestreo de la señal a descomponer.
    N : float
        Largo de la ventana de análisis utilizada para construir el espectrograma.
    noverlap : float
        Cantidad de puntos de traslape para la base de datos utilizada en el diccionario.
    n_comps_dict : float
        Revisar la base de datos que se haya descompuesto en "n_comps_dict" 
        componentes a partir de la base puramente cardíaca.
    beta : float
        Beta utilizado para aplicar NMF (en la definición de la divergencia).
    filepath_data : str
        Directorio donde se encuentran las carpetas de los diccionarios.
    threshold : float, optional
        Valor del umbral de decisión para clasificar una componente de la matriz
        W como sonido cardíaco. Por defecto es 0.5.
        
    Referencias
    -----------
    [1] Canadas-Quesada, et. al (2017). A non-negative matrix factorization 
        approach  based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier. Chapter 3.2.1.
    '''
    # Definición de la cantidad de componentes
    q_points_nyq, n_components = W.shape
    
    # Definición del fcut_bin en base a la frecuencia de corte a considerar fcut
    fcut_bin = int(q_points_nyq / (samplerate / 2) * fcut)
    
    # Abrir el diccionario
    filepath = f'{filepath_data}/SR {samplerate}/'\
               f'W_dict_comps{n_comps_dict}_N{N}_noverlap{noverlap}_'\
               f'padding{padding}_repeat{repeat}_beta{beta}.npz'
    data = np.load(filepath)
    
    # Definición del diccionario
    W_dict = data['W_list']
    
    # Definición de la lista de booleanos de salida
    S_i_list = list()
    
    for i in range(n_components):
        # Se obtiene el bool correspondiente a la componente i. True si
        # es sonido cardíaco y False si es sonido respiratorio
        S_i = spectral_correlation_criteria(W[:,i], W_dict, fcut_bin=fcut_bin, 
                                            i_selection=i_selection,
                                            measure=measure)
        # Agregando...
        S_i_list.append(S_i)
    
    # Transformando a array
    S_i_array = np.array(S_i_list)
    
    # Definición de umbral
    if threshold == 'mean':
        threshold = np.mean(S_i_array)
    
    # Aplicando umbral
    return S_i_array >= threshold, S_i_array


def roll_off_test(X_list, f1, f2, samplerate, whole=False, percentage=0.85):
    '''Función que permite realizar un testeo del espectrograma de cada componente. 
    Si la entrada i es True, se trata de un sonido cardíaco. Si es False, se trata 
    de un sonido respiratorio.
    
    Parameters
    ----------
    X_list : list or array
        Lista de espectrogramas de las componentes obtenidas mediante NMF.
    f1 : float
        Valor de la frecuencia de corte inferior. Se recomienda usar en 20 Hz.
    f2 : float
        Valor de la frecuencia de corte superior. Se recomienda usar en 150 Hz.
    samplerate : float
        Tasa de muestreo de la señal a testear.
    whole : bool, optional
        Indica si los espectrogramas de X_list están hasta samplerate (True) o 
        hasta samplerate // 2 (False).
    percentage : float, optional
        Ponderación para la energía total del espectrograma en el criterio de
        selección. Por defecto es 0.85.
    
    Referencias
    -----------
    [1] Canadas-Quesada, et. al (2017). A non-negative matrix factorization 
        approach  based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier. Chapter 3.2.2.
    '''
    # Definición de la lista de booleanos de salida
    bool_list = list()
    
    # Definición frecuencia de corte
    if whole:
        f1_bin = int(f1 / samplerate * X_list[0].shape[0])
        f2_bin = int(f2 / samplerate * X_list[0].shape[0])
    else:
        f1_bin = int(f1 / (samplerate // 2) * X_list[0].shape[0])
        f2_bin = int(f2 / (samplerate // 2) * X_list[0].shape[0])
        
    for X_i in X_list:
        bool_list.append(roll_off_criteria(X_i, f1=f1_bin, f2=f2_bin, percentage=percentage))
        
    return np.array(bool_list)


def temporal_correlation_test(H, filename, samplerate_original, samplerate_signal, N, 
                              N_audio, noverlap, threshold=0, measure='correlation', 
                              H_binary=True, reduce_to_H=False, version=2, 
                              ausc_zone='Anterior'):
    '''Función que permite realizar un testeo de cada componente de la matriz
    H en comparación con el heart rate obtenido a partir de la señal original.
    Retorna un arreglo de booleanos. Si la entrada i es True, se trata de un
    sonido cardíaco. Si es False, se trata de un sonido respiratorio.
    
    Parameters
    ----------
    H : ndarray
        Matriz H de la descomposición NMF.
    filename : str
        Nombre del archivo de audio.
    samplerate_original : int
        Samplerate de la señal original. Se utiliza este parámetro para obtener
        el heart rate, ya que los segmentos están expresados en muestras con la tasa
        de muestreo de la señal original.
    samplerate_signal : int
        Samplerate de la señal descompuesta en NMF. Puede ser distinta a 
        samplerate_original (por downsampling, por ejemplo).
    N : int
        Largo de la ventana de análisis utilizada para construir el espectrograma.
    N_audio : int
        Largo del archivo de audio a descomponer.
    noverlap : int
        Cantidad de puntos de traslape utilizados para construir el espectrograma.
    threshold : float, optional
        Valor del umbral de decisión para clasificar una componente de la matriz
        H como sonido cardíaco. Por defecto es 0.
        
    Referencias
    -----------
    [1] Canadas-Quesada, et. al (2017). A non-negative matrix factorization 
        approach  based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier. Chapter 3.2.3.
    '''
    # Definición del nombre del archivo con segmentos de sonido cardiaco.
    for i in filename.split(" "):
        if "Seed" in i:
            name = i.split('/')[-1]
            continue
    
    # Archivo con el nombre de los segmentos 
    filename_heart_segments = f'{name} - segments.txt'
    
    # Agregando la carpeta donde se ubica
    filename_heart_segments = f'Database_manufacturing/db_heart/Manual combinations v{version}/'\
                              f'{ausc_zone}/{filename_heart_segments}'
    
    # Definición del heart rate
    P = get_heart_rate(filename_heart_segments, 
                       samplerate_original=samplerate_original, 
                       samplerate_signal=samplerate_signal, 
                       N=N, N_audio=N_audio, noverlap=noverlap,
                       reduce_to_H=reduce_to_H)
        
    # Definición de la lista de booleanos de salida
    TC_i_list = list()
    
    for i in range(H.shape[0]):
        # Si es que se quiere hacer en la dimension de P, se debe hacer un resample
        if not reduce_to_H:
            h_interest = resampling_by_points(H[i], samplerate_signal, N_desired=len(P),
                                              normalize=True)
        else:
            h_interest = H[i]
        
        # Aplicando el criterio
        TC_i = temporal_correlation_criteria(h_interest, P, measure=measure, 
                                             H_binary=H_binary)
        # Agregando
        TC_i_list.append(TC_i)
    
    # Pasando a array
    TC_i_array = np.array(TC_i_list)
    
    if threshold == 'mean':
        threshold = np.mean(TC_i_array)
    
    return TC_i_array >= threshold, TC_i_array


def get_heart_rate(filename, samplerate_original, samplerate_signal, N, N_audio,
                   noverlap, reduce_to_H=False):
    '''Obtención del heart rate a partir de los segmentos de sonido cardíaco
    presente en la señal.
    
    Parameters
    ----------
    filename : str
        Dirección del archivo de segmentos a revisar.
    samplerate_original : int
        Samplerate de la señal original. Se utiliza este parámetro para obtener
        el heart rate, ya que los segmentos están expresados en muestras con la tasa
        de muestreo de la señal original.
    samplerate_signal : int
        Samplerate de la señal descompuesta en NMF. Puede ser distinta a 
        samplerate_original (por downsampling, por ejemplo).
    N : int
        Largo de la ventana de análisis utilizada para construir el espectrograma.
    N_audio : int
        Largo del archivo de audio a descomponer.
    noverlap : int
        Cantidad de puntos de traslape utilizados para construir el espectrograma.
    reduce_to_H : bool, optional
        Indica si es que se quiere trabajar en el largo de la componente H (True) o
        en el largo de la señal original (False). Por defecto es False.
    
    Referencias
    -----------
    [1] Canadas-Quesada, et. al (2017). A non-negative matrix factorization 
        approach  based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier. Chapter 3.2.3.
    '''
    with open(filename, 'r', encoding='utf8') as file:
        segments = literal_eval(file.readline())
    
    # Definición del arreglo que contendrá la posición del sonido cardíaco
    heart_indexes = np.zeros(N_audio)
    
    for i in segments:
        # Definición del origen y el fin del segmento dependiendo del samplerate
        beg = int(i[0] * samplerate_signal / samplerate_original)
        end = int(i[1] * samplerate_signal / samplerate_original)
        # Se definen como 1 los segmentos del sonido cardíaco
        heart_indexes[beg:end] = 1
    
    # Realizando un proceso de ventaneo similar al STFT
    if reduce_to_H:
        heart_rate = _from_segments_to_Hdim(heart_indexes, N, noverlap)
    else:
        heart_rate = heart_indexes
    
    return heart_rate


def _from_segments_to_Hdim(signal_in, N, noverlap):
    '''Obtención del heart reate mediante el prototipo de heart rate dado por los
    segmentos de presencia de sonido cardíaco. Es necesario realizar este proceso
    ya que al utilizar la información de la matriz H producto de la descomposición
    NMF del espectrograma, queda en la dimensión columna (horizontal o de tiempo).
    Por ende, es necesario hacer un procedimiento similar a la obtención del
    espectrograma para poder transformar la señal en el tiempo (muestras) al tiempo
    overlapped del espectrograma.
    
    Parameters
    ----------
    signal_in : ndarray
        Señal heart rate a transformar en dimensión de las componentes matriz H.
    N : int
        Largo de la ventana de análisis utilizada para construir el espectrograma.
    noverlap : int
        Cantidad de puntos de traslape utilizados para construir el espectrograma.
    '''
    # Corroboración de criterios: noverlap <= N - 1
    if N <= noverlap:
        raise Exception('noverlap debe ser menor que N.')
    elif noverlap < 0:
        raise Exception('noverlap no puede ser negativo')
    else:
        noverlap = int(noverlap)
    
    # Definición del paso de avance
    step = N - noverlap
        
    # Definición de bordes de signal_in (se hace porque en el espectrograma
    # también se define así, y es necesario ser coherente con eso)
    signal_in = np.concatenate((np.zeros(N//2), signal_in, np.zeros(N//2)))
    
    # Definición de la salida
    signal_out = list()
    
    # Iteración sobre el audio
    while signal_in.size != 0:
        # Se corta la cantidad de muestras que se necesite, o bien, las que se
        # puedan cortar
        if len(signal_in) >= N:
            # Se obtienen las N muestras de interés
            signal_frame = signal_in[:N]
            
            # Y se corta la señal para la siguiente iteración
            signal_in = signal_in[step:]
            
        # En la última iteración se añaden ceros para lograr el largo N
        else:
            # Definición del último frame
            last_frame = signal_in[:]
            
            # Se rellena con ceros hasta lograr el largo            
            signal_frame = np.append(last_frame, [0] * (N - len(last_frame)))
            
            # Y se corta la señal para la siguiente iteración
            signal_in = signal_in[:0]
    
        if np.sum(signal_frame) >= N * (0.4):
            signal_out.append(1)
        else:
            signal_out.append(0)
            
    return signal_out


# Olvido
"""
def clustering_test(W, H, samplerate_original, samplerate_signal, N, N_audio, noverlap, 
                    n_comps_dict, beta, filepath_dict, Xcomps_list, filename_heart_segments, 
                    uf=0.5, f1=20, f2=150, percentage=0.85, ut=0, decision_kind='or'):
    '''Función que retorna el valor de verdad de las componentes que pertenecen
    al cluster de sonido cardíacos. Cada entrada True indica que la componente 
    corresponde a sonido cardíaco, y cada False a sonido respiratorio.
    
    Parameters
    ----------
    W : ndarray
        Matriz W de la descomposición NMF.
    H : ndarray
        Matriz H de la descomposición NMF.
    samplerate_original : float
        Samplerate de la señal original. Se utiliza este parámetro para obtener
        el heart rate, ya que los segmentos están expresados en muestras con la tasa
        de muestreo de la señal original.
    samplerate_signal : float
        Samplerate de la señal descompuesta en NMF. Puede ser distinta a 
        samplerate_original (por downsampling, por ejemplo). Esta es la que se usa
        para obtener la base de datos y transformar el heart rate.
    N : int
        Largo de la ventana de análisis utilizada para construir el espectrograma.
    N_audio : int
        Largo del archivo de audio a descomponer.
    noverlap : int
        Cantidad de puntos de traslape utilizados para construir el espectrograma.
    n_comps_dict : float
        Revisar la base de datos que se haya descompuesto en "n_comps_dict" 
        componentes a partir de la base puramente cardíaca.
    beta : float
        Beta utilizado para aplicar NMF (en la definición de la divergencia).
    filepath_dict : str
        Directorio donde se encuentran las carpetas de los diccionarios.
    Xcomps_list : list or array
        Lista de espectrogramas de las componentes obtenidas mediante NMF.
    filename_heart_segments : str
        Dirección del archivo a revisar para los segmentos de sonido cardíaco.
    uf : float, optional
        Valor del umbral de decisión para clasificar una componente de la matriz
        W como sonido cardíaco. Por defecto es 0.5.
    f0 : float, optional
        Valor de la frecuencia de corte en bins. Se recomienda usar en 260 Hz.
    percentage : float, optional
        Ponderación para la energía total del espectrograma en el criterio de
        selección. Por defecto es 0.85.
    ut : float, optional
        Valor del umbral de decisión para clasificar una componente de la matriz
        H como sonido cardíaco. Por defecto es 0.
    decision_kind : {'or', 'vote'}, optional
        Método de decisión utilizada para clasificar como sonido cardíaco o respiratorio.
        Si se usa 'or' basta con que alguno de los criterios sea True para que se declare
        sonido cardíaco. Si se usa 'vote' se elige por mayoría de votos. Por defecto es 'or'.
        
    Referencias
    -----------
    [1] Canadas-Quesada, et. al (2017). A non-negative matrix factorization 
        approach  based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier. Chapter 3.2.
    '''
    # Bools para criterio espectral
    bool_spectral = spectral_correlation_test(W=W, samplerate_dict=samplerate_signal, N=N, 
                                              noverlap=noverlap, n_comps_dict=n_comps_dict, 
                                              beta=beta, filepath_data=filepath_dict, 
                                              threshold=uf)
    
    # Bools para criterio roll-off
    bool_roll_off = roll_off_test(X_list=Xcomps_list, f1=f1, f2=f2, percentage=percentage)
    
    # Bools para criterio temporal
    bool_temporal = temporal_correlation_test(H=H, filename_heart_segments=filename_heart_segments,
                                              samplerate_original=samplerate_original, 
                                              samplerate_signal=samplerate_signal, 
                                              N=N, N_audio=N_audio, noverlap=noverlap, 
                                              threshold=ut)
    
    if decision_kind == 'or':
        # Se retorna el "or" entre todas las listas (basta con que al menos
        # una se declare corazón (True) para que sea corazón)
        return bool_spectral | bool_roll_off | bool_temporal
    
    elif decision_kind == 'vote':
        # Si es que hay mayoría de votación, se declara la variable (2 True y 1 False,
        # es True; y viceversa).
        return (bool_spectral.astype(int) + bool_roll_off.astype(int) + 
                bool_temporal.astype(int)) >= 2
    
    else:
        raise Exception('Opción no válida para "decision_kind".')
"""