import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.decomposition import PCA
from evaluation_metrics import get_PSD
from sklearn import preprocessing, svm
from sklearn.neighbors import KNeighborsClassifier
from librosa.feature import mfcc, spectral_centroid, spectral_rolloff, \
    spectral_bandwidth, spectral_contrast, spectral_flatness, \
    zero_crossing_rate
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
        Tipo de métrica para realizar la clasificación. 'correlation' calcula la
        correlación entre el heart rate y el H de cada componente. 'q_equal'
        calcula el porcentaje de puntos iguales en ambas representaciones.
        Por defect es 'correlation'. 
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
    padding : int, optional
        Cantidad de puntos adicionales para zero_padding. Por defecto es 0.
    repeat : int, optional
        Cantidad de repeticiones de la señal. Por defecto es 0.
    measure : {'cosine', 'correlation'}, optional
        Medida utilizada para calcular la similaridad. 'cosine' usa similaridad de 
        coseno, mientras que 'correlation' calcula el coeficiente de Pearson. Por
        defecto es 'cosine'.
    i_selection : {'max', 'mean'}, optional
        Método de selección del componente que identifica la similarida. Con 'max' 
        se utiliza el máximo de todas las medidas, mientras que con 'mean' se utiliza
        el promedio. Por defecto es 'max'.
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


def spectral_correlation_test_2_v1(W, signal_in, samplerate, N_lax, filepath_data, 
                                prom_spectra=False, measure='correlation', 
                                i_selection='max', threshold='mean'):
    '''Función que permite realizar un testeo de cada componente de la matriz
    W sobre el diccionario construido a partir de sonidos puramente cardíacos.
    Retorna un arreglo de booleanos. Si la entrada i es True, se trata de un
    sonido cardíaco. Si es False, se trata de un sonido respiratorio.
    
    Parameters
    ----------
    W : ndarray
        Matriz W de la descomposición NMF.
    signal_in : ndarray
        Señal de entrada a comparar con las plantillas espectrales obtenidas.
    samplerate : float
        Tasa de muestreo de la base de datos utilizada en el diccionario. Esta debe coincidir
        con la tasa de muestreo de la señal a descomponer.
    N_lax : int, optional
        Cantidad de puntos adicionales que se consideran para cada lado más allá de los
        intervalos dados.
    filepath_data : str
        Directorio donde se encuentran las carpetas de los diccionarios.
    prom_spectra : bool, optional
        Opción de resumir todos los espectros al promedio (estilo Welch PSD). Por defecto es
        False.
    measure : {'cosine', 'correlation'}, optional
        Medida utilizada para calcular la similaridad. 'cosine' usa similaridad de 
        coseno, mientras que 'correlation' calcula el coeficiente de Pearson. Por
        defecto es 'cosine'.
    i_selection : {'max', 'mean'}, optional
        Método de selección del componente que identifica la similarida. Con 'max' 
        se utiliza el máximo de todas las medidas, mientras que con 'mean' se utiliza
        el promedio. Por defecto es 'max'.
    threshold : float, optional
        Valor del umbral de decisión para clasificar una componente de la matriz
        W como sonido cardíaco. Por defecto es 0.5.
        
    Referencias
    -----------
    [1] Canadas-Quesada, et. al (2017). A non-negative matrix factorization 
        approach  based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier. Chapter 3.2.1.
    [2] Elaboración propia.
    '''
    # Se obtienen los segmentos de sonido respiratorio
    with open(f'{filepath_data}', 'r', encoding='utf8') as file:
        intervals = literal_eval(file.readline())
        
    # Definición de la razón entre el samplerate original y el samplerate utilizado
    sr_ratio = 44100 // samplerate
    
    # Definición de la lista de segmentos de sonido respiratorio
    resp_list = list()
    
    # Variable auxiliar que indica el inicio de una señal
    beg = 0
    
    for i in intervals:
        # Definición de los límites a revisar
        lower = beg // sr_ratio - N_lax if beg != 0 else 0
        upper = i[0] // sr_ratio + N_lax
        
        # Segmento de señal respiratoria
        resp_signal = signal_in[lower:upper]
        
        # Agregando a las listas
        resp_list.append(resp_signal)
        
        # Se redefine la variable auxiliar
        beg = i[1]
    
    # Finalmente se agrega el segmento final
    resp_signal = signal_in[i[1] // sr_ratio - N_lax:]
    resp_list.append(resp_signal)
    
    # Definición de la cantidad de puntos a utilizar
    N = (W.shape[0]-1) * 2
    
    # Definición de la matriz a rellenar
    resp_array = np.empty((len(resp_list), W.shape[0]))
    
    # Re acondicionando las señales
    for i in range(len(resp_list)):
        # Resampleando a 4 veces la cantidad de puntos que se necesita
        resp = resampling_by_points(resp_list[i], samplerate, N)
        resp_array[i] = 20 * np.log10(1 / N * abs(np.fft.fft(resp)) 
                                      + 1e-12)[:W.shape[0]]
    
    # Si es que se promedian los espectros
    if prom_spectra:
        resp_array = np.array([resp_array.mean(axis=0)])
    
    # Definición de la lista de booleanos de salida
    S_i_list = list()
    
    for i in range(W.shape[1]):
        # Se obtiene el bool correspondiente a la componente i. True si
        # es sonido cardíaco y False si es sonido respiratorio
        S_i = spectral_correlation_criteria(20 * np.log10(W[:,i] + 1e-12), 
                                            resp_array, fcut_bin=-1, 
                                            i_selection=i_selection,
                                            measure=measure)
        # Agregando...
        S_i_list.append(S_i)
    
    # Transformando a array
    S_i_array = np.array(S_i_list)
        
    # Definición de umbral
    if threshold == 'mean':
        threshold = np.mean(S_i_array)
    elif threshold == 'median':
        threshold = np.median(S_i_array)
    
    # Aplicando umbral
    return S_i_array < threshold, S_i_array


def spectral_correlation_test_2_v2(W, signal_in, samplerate, N_lax, filepath_data, 
                                prom_spectra=False, measure='correlation', 
                                i_selection='max', threshold='mean'):
    '''Función que permite realizar un testeo de cada componente de la matriz
    W sobre el diccionario construido a partir de sonidos puramente cardíacos.
    Retorna un arreglo de booleanos. Si la entrada i es True, se trata de un
    sonido cardíaco. Si es False, se trata de un sonido respiratorio.
    
    OJO: Se diferencia de la versión 1 en que este aplica zero padding en vez de
    resamplear.
    
    Parameters
    ----------
    W : ndarray
        Matriz W de la descomposición NMF.
    signal_in : ndarray
        Señal de entrada a comparar con las plantillas espectrales obtenidas.
    samplerate : float
        Tasa de muestreo de la base de datos utilizada en el diccionario. Esta debe coincidir
        con la tasa de muestreo de la señal a descomponer.
    N_lax : int, optional
        Cantidad de puntos adicionales que se consideran para cada lado más allá de los
        intervalos dados.
    filepath_data : str
        Directorio donde se encuentran las carpetas de los diccionarios.
    prom_spectra : bool, optional
        Opción de resumir todos los espectros al promedio (estilo Welch PSD). Por defecto es
        False.
    measure : {'cosine', 'correlation'}, optional
        Medida utilizada para calcular la similaridad. 'cosine' usa similaridad de 
        coseno, mientras que 'correlation' calcula el coeficiente de Pearson. Por
        defecto es 'cosine'.
    i_selection : {'max', 'mean'}, optional
        Método de selección del componente que identifica la similarida. Con 'max' 
        se utiliza el máximo de todas las medidas, mientras que con 'mean' se utiliza
        el promedio. Por defecto es 'max'.
    threshold : float, optional
        Valor del umbral de decisión para clasificar una componente de la matriz
        W como sonido cardíaco. Por defecto es 0.5.
        
    Referencias
    -----------
    [1] Canadas-Quesada, et. al (2017). A non-negative matrix factorization 
        approach  based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier. Chapter 3.2.1.
    [2] Elaboración propia.
    '''
    # Se obtienen los segmentos de sonido respiratorio
    with open(f'{filepath_data}', 'r', encoding='utf8') as file:
        intervals = literal_eval(file.readline())
        
    # Definición de la razón entre el samplerate original y el samplerate utilizado
    sr_ratio = 44100 // samplerate
    
    # Definición de la lista de segmentos de sonido respiratorio
    resp_list = list()
    
    # Variable auxiliar que indica el inicio de una señal
    beg = 0
    
    for i in intervals:
        # Definición de los límites a revisar
        lower = beg // sr_ratio - N_lax if beg != 0 else 0
        upper = i[0] // sr_ratio + N_lax
        
        # Segmento de señal respiratoria
        resp_signal = signal_in[lower:upper]
        
        # Agregando a las listas
        resp_list.append(resp_signal)
        
        # Se redefine la variable auxiliar
        beg = i[1]
    
    # Finalmente se agrega el segmento final
    resp_signal = signal_in[i[1] // sr_ratio - N_lax:]
    resp_list.append(resp_signal)
    
    # Definición de la cantidad de puntos a utilizar
    N = (W.shape[0]-1) * 2
    
    # Definición de la matriz a rellenar
    resp_array = np.empty((len(resp_list), W.shape[0]))
    
    # Re acondicionando las señales
    for i in range(len(resp_list)):
        # Resampleando a 4 veces la cantidad de puntos que se necesita
        resp = np.concatenate((resp_list[i], [0] * (N - len(resp_list[i])) ))
        resp_array[i] = 20 * np.log10(1 / N * abs(np.fft.fft(resp)) 
                                      + 1e-12)[:W.shape[0]]
    
    # Si es que se promedian los espectros
    if prom_spectra:
        resp_array = np.array([resp_array.mean(axis=0)])
    
    # Definición de la lista de booleanos de salida
    S_i_list = list()
    
    for i in range(W.shape[1]):
        # Se obtiene el bool correspondiente a la componente i. True si
        # es sonido cardíaco y False si es sonido respiratorio
        S_i = spectral_correlation_criteria(20 * np.log10(W[:,i] + 1e-12), 
                                            resp_array, fcut_bin=-1, 
                                            i_selection=i_selection,
                                            measure=measure)
        # Agregando...
        S_i_list.append(S_i)
    
    # Transformando a array
    S_i_array = np.array(S_i_list)
        
    # Definición de umbral
    if threshold == 'mean':
        threshold = np.mean(S_i_array)
    elif threshold == 'median':
        threshold = np.median(S_i_array)
    
    # Aplicando umbral
    return S_i_array < threshold, S_i_array


def machine_learning_clustering(comps, signal_in, samplerate, N_lax, filepath_data, 
                                N=4096, classifier='svm', n_neighbors=1, pca_comps=30,
                                db_basys=1e-12):
    # Se obtienen los segmentos de sonido respiratorio
    with open(f'{filepath_data}', 'r', encoding='utf8') as file:
        intervals = literal_eval(file.readline())
        
    # Definición de la razón entre el samplerate original y el samplerate utilizado
    sr_ratio = 44100 // samplerate
    
    # Definición de la lista de segmentos de sonido respiratorio y cardiaco
    resp_list = list()
    heart_list = list()
    
    # Variable auxiliar que indica el inicio de una señal
    beg = 0
    
    for i in intervals:
        # Definición de los límites a revisar para el sonido respiratorio
        lower_resp = beg // sr_ratio - N_lax if beg != 0 else 0
        upper_resp = i[0] // sr_ratio + N_lax
        
        # Definición de los límites a revisar para el sonido cardiaco
        lower_heart = i[0] // sr_ratio - N_lax
        upper_heart = i[1] // sr_ratio + N_lax
        
        # Segmento de señal respiratoria
        resp_signal = signal_in[lower_resp:upper_resp]
        # Y cardiaca
        heart_signal = signal_in[lower_heart:upper_heart]
                
        # Agregando a las listas
        resp_list.append(resp_signal)
        heart_list.append(heart_signal)
        
        # Se redefine la variable auxiliar
        beg = i[1]
    
    # Finalmente se agrega el segmento final
    resp_signal = signal_in[i[1] // sr_ratio - N_lax:]
    resp_list.append(resp_signal)
    
    # Definición de la matriz a rellenar
    resp_array = np.empty((0, N//2+1))
    heart_array = np.empty((0, N//2+1))
    
    # Re acondicionando las señales
    for i in range(len(resp_list)):
        # Resampleando a 4 veces la cantidad de puntos que se necesita
        resp = resampling_by_points(resp_list[i], samplerate, N)
        resp_array = np.vstack((resp_array, 
                                20 * np.log10(1 / N * abs(np.fft.fft(resp)) 
                                              + db_basys)[:N//2+1]))
    
    for i in range(len(heart_list)):
        # Resampleando a 4 veces la cantidad de puntos que se necesita
        heart = resampling_by_points(heart_list[i], samplerate, N)
        heart_array = np.vstack((heart_array, 
                                 20 * np.log10(1 / N * abs(np.fft.fft(heart)) 
                                               + db_basys)[:N//2+1]))
    
    # Definición de la matriz de entrenamiento
    X_train = np.vstack((resp_array, heart_array))
    
    '''
    print(X_train)
    print(np.argwhere(np.isnan(X_train)))
    
    plt.figure()
    plt.pcolormesh(X_train, cmap='jet')
    plt.colorbar()
    plt.show()
    return
    '''
    
    # Definición de la matriz de etiquetas de entrenamiento. Se define como
    # 0 para respiración y 1 para corazón
    Y_train =  np.array([0] * resp_array.shape[0] +
                        [1] * heart_array.shape[0])
    
    # Reducción de dimensiones
    pca = PCA(n_components=pca_comps)
    X_pca = pca.fit_transform(X_train)
    
    
    for num, i in enumerate(X_pca):
        if Y_train[num] == 0:
            color = 'r'
        elif Y_train[num] == 1:
            color = 'b'
        
        plt.scatter(i[0], i[1], color=color)
    
    
    if classifier == 'knn':
        clas = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='auto')
    elif classifier == 'svm':
        clas = svm.SVC(kernel='linear', degree=50, gamma='auto')
    else:
        raise Exception('Opción "classifier" no válida.')
    
    # Ajuste del clasificador
    clas.fit(X_pca, Y_train)
    
    # Definición de la variable de decisión
    decision = list()
    
    # Para cada componente
    for comp_sound in comps:
        # Para cada iteración, se define nuevamente esta lista auxiliar que sirve para
        # tomar la decisión de la componente i-ésima en base a la mayoría de votos de 
        # las características del sonido cardiorrespiratorio
        decision_comp = list()
        
        for i in intervals:
            # Definición de los límites a revisar para el sonido cardiaco
            lower = i[0] // sr_ratio - N_lax
            upper = i[1] // sr_ratio + N_lax
            
            # Definición del sonido cardiorrespiratorio i en el sonido completo
            hr_sound = resampling_by_points(comp_sound[lower:upper], samplerate, N)
            
            # Se calcula su magnitud
            feat = 20 * np.log10(1 / N * abs(np.fft.fft(hr_sound)) + db_basys)[:N//2+1]
            
            # Y se transforma
            feat = pca.transform(feat.reshape(1,-1))  
            # plt.scatter(feat[:,0], feat[:,1], color='cyan', marker='.')
            
            # Resultado clasificación
            Y_pred = clas.predict(feat)
            
            # Agregando a la lista
            decision_comp.append(Y_pred)
        
        # Una vez completada la lista de decisiones, se procede a contar sus propiedades
        q_hearts = np.sum(decision_comp)
        
        # Si la cantidad de componentes de corazón es mayor al 50%, entonces es corazón,
        # en caso contrario es respiración
        if q_hearts / len(decision_comp) >= 0.5:
            decision.append(True)
        else:
            decision.append(False)
    print(decision)
    # plt.show()
    return decision


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
    measure : {'correlation', 'q_equal'}, optional
        Tipo de métrica para realizar la clasificación. 'correlation' calcula la
        correlación entre el heart rate y el H de cada componente. 'q_equal'
        calcula el porcentaje de puntos iguales en ambas representaciones.
        Por defect es 'correlation'. 
    H_binary : bool, optional
        Valor del umbral de decisión de la para clasificar una componente como
        sonido cardíaco. Por defecto es 0.
    
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
            h_interest = resampling_by_points(H[i], samplerate_signal, 
                                              N_desired=len(P),
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


def temporal_correlation_test_segment(H, lower, upper, N_fade, N_lax, 
                                      samplerate_signal, threshold=0, 
                                      measure='correlation', H_binary=True, 
                                      reduce_to_H=False):
    '''Función que permite realizar un testeo de cada componente de la matriz
    H en comparación con el heart rate obtenido a partir de la señal original.
    Retorna un arreglo de booleanos. Si la entrada i es True, se trata de un
    sonido cardíaco. Si es False, se trata de un sonido respiratorio.
    
    Parameters
    ----------
    H : ndarray
        Matriz H de la descomposición NMF.
    lower : int
        Índice inferior del segmento.
    upper : int
        Índice superior del segmento.
    N_fade : int
        Cantidad de puntos de fade.
    threshold : float, optional
        Valor del umbral de decisión para clasificar una componente de la matriz
        H como sonido cardíaco. Por defecto es 0.
    measure : {'correlation', 'q_equal'}, optional
        Tipo de métrica para realizar la clasificación. 'correlation' calcula la
        correlación entre el heart rate y el H de cada componente. 'q_equal'
        calcula el porcentaje de puntos iguales en ambas representaciones.
        Por defect es 'correlation'. 
    H_binary : bool, optional
        Valor del umbral de decisión de la para clasificar una componente como
        sonido cardíaco. Por defecto es 0.
    
    Referencias
    -----------
    [1] Canadas-Quesada, et. al (2017). A non-negative matrix factorization 
        approach  based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier. Chapter 3.2.3.
    '''
    # Definición del heart rate en ese segmento
    P = np.array([0] * (N_fade + N_lax) + 
                 [1] * abs(upper - lower - 2 * N_lax) + 
                 [0] * (N_fade + N_lax))
        
    # Definición de la lista de booleanos de salida
    TC_i_list = list()
    
    for i in range(H.shape[0]):
        # Si es que se quiere hacer en la dimension de P, se debe hacer un resample
        if not reduce_to_H:
            h_interest = resampling_by_points(H[i], samplerate_signal, 
                                              N_desired=len(P),
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


def energy_percentage_test(W, percentage=0.85):
    '''Criterio de "centroide" propuesto.
    
    Parameters
    ----------
    W : ndarray
        Matriz de plantillas espectrales W del NMF.
    percentage : float, optional
        Porcentaje de energía límite a evaluar. Por defecto es 85%.
    '''
    # Definición de la lista de puntos límite
    limit_point_list = list()
    
    # Agregando los valores para cada componente
    for i in range(W.shape[1]):
        limit_point_list.append(p_percentage_energy(W[:,i], percentage=percentage))
    
    # Pasando a array
    limit_points = np.array(limit_point_list)
    
    # Cálculo de la media de los puntos
    mu_c = limit_points.mean()
    
    return limit_points >= mu_c, limit_points


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


def p_percentage_energy(signal_in, percentage=0.85):
    '''Función que retorna el mínimo índice de un arreglo que cumple que 
    la energía a este ese índica sea mayor que el "x"% de su energía.
    
    Parameters
    ----------
    signal_in : ndarray
        Señal de entrada.
    percentage : float, optional
        Porcentaje de energía límite a evaluar. Por defecto es 85%.
    
    Returns
    -------
    index : int
        Primer índice que cumple el criterio.
    '''
    # Cálculo de la energía total
    total_energy = np.sum(abs(signal_in ** 2))
    
    # Si es que la suma hasta el punto i es mayor que el "x"% de la
    # energía total, se retorna ese primer punto que ya cumple el
    # criterio
    for i in range(len(signal_in)):
        if np.sum(abs(signal_in[:i] ** 2)) >= 0.85 * total_energy:
            return i


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


def machine_learning_clustering_OLD(comps, signal_in, samplerate, N_lax, filepath_data, 
                                n_mfcc=20, classifier='svm'):
    # Se obtienen los segmentos de sonido respiratorio
    with open(f'{filepath_data}', 'r', encoding='utf8') as file:
        intervals = literal_eval(file.readline())
        
    # Definición de la razón entre el samplerate original y el samplerate utilizado
    sr_ratio = 44100 // samplerate
    
    # Definición de la lista de segmentos de sonido respiratorio y cardiaco
    resp_list = list()
    resp_mfcc = np.empty((0, n_mfcc))
    heart_list = list()
    heart_mfcc = np.empty((0, n_mfcc))
    
    # Variable auxiliar que indica el inicio de una señal
    beg = 0
    
    for i in intervals:
        # Definición de los límites a revisar para el sonido respiratorio
        lower_resp = beg // sr_ratio - N_lax if beg != 0 else 0
        upper_resp = i[0] // sr_ratio + N_lax
        
        # Definición de los límites a revisar para el sonido cardiaco
        lower_heart = i[0] // sr_ratio - N_lax
        upper_heart = i[1] // sr_ratio + N_lax
        
        # Segmento de señal respiratoria
        resp_signal = signal_in[lower_resp:upper_resp]
        # Y cardiaca
        heart_signal = signal_in[lower_heart:upper_heart]
                
        # Agregando a las listas
        resp_list.append(resp_signal)
        heart_list.append(heart_signal)
        resp_mfcc = np.vstack((resp_mfcc, mfcc(resp_signal, sr=samplerate, 
                                               n_mfcc=n_mfcc).T))
        heart_mfcc = np.vstack((heart_mfcc, mfcc(heart_signal, sr=samplerate, 
                                                 n_mfcc=n_mfcc).T))
        
        # Se redefine la variable auxiliar
        beg = i[1]
    
    # Finalmente se agrega el segmento final
    resp_signal = signal_in[i[1] // sr_ratio - N_lax:]
    resp_list.append(resp_signal)
    resp_mfcc = np.vstack((resp_mfcc, mfcc(resp_signal, sr=samplerate, 
                                           n_mfcc=n_mfcc).T))
    
    # Definición de la matriz de entrenamiento
    X_train = np.vstack((resp_mfcc, heart_mfcc))
    # Definición de la matriz de etiquetas de entrenamiento. Se define como
    # 0 para respiración y 1 para corazón
    Y_train =  np.array([0] * resp_mfcc.shape[0] +
                        [1] * heart_mfcc.shape[0])
    
    # Reducción de dimensiones
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_train)
    
    '''
    %matplotlib notebook
    for num, i in enumerate(X_pca):
        if Y_train[num] == 0:
            color = 'r'
        elif Y_train[num] == 1:
            color = 'b'
        
        plt.scatter(i[0], i[1], color=color)
    '''
    
    if classifier == 'knn':
        clas = KNeighborsClassifier(n_neighbors=1, algorithm='auto')
    elif classifier == 'svm':
        clas = svm.SVC(kernel='linear', degree=50, gamma='auto')
    
    # Ajuste del clasificador
    clas.fit(X_pca, Y_train)
    
    # Para cada componente
    for comp_sound in comps:
        for i in intervals:
            # Definición de los límites a revisar para el sonido cardiaco
            lower = i[0] // sr_ratio - N_lax
            upper = i[1] // sr_ratio + N_lax
            
            hr_sound = comp_sound[lower:upper]
            
            # Se calcula su MFCC
            mfcc_i = mfcc(np.asfortranarray(hr_sound), sr=samplerate, n_mfcc=n_mfcc).T
            
            # Y se transforma
            mfcc_i = pca.transform(mfcc_i).mean(axis=0)
            
            plt.scatter(mfcc_i[0], mfcc_i[1], color='C2')
                        
            Y_pred = clas.predict(mfcc_i.reshape(1,-1))
            print(Y_pred)
     
    plt.show()
    return


def machine_learning_clustering_OLD_2(comps, signal_in, samplerate, N_lax, filepath_data, 
                                n_bands=5, N=4096, classifier='svm', n_neighbors=1, 
                                pca_comps=30, db_basys=1e-12):
    # Se obtienen los segmentos de sonido respiratorio
    with open(f'{filepath_data}', 'r', encoding='utf8') as file:
        intervals = literal_eval(file.readline())
        
    # Definición de la razón entre el samplerate original y el samplerate utilizado
    sr_ratio = 44100 // samplerate
    
    # Definición de la lista de segmentos de sonido respiratorio y cardiaco
    resp_list = list()
    heart_list = list()
    
    # Definición de las matrices de características a rellenar
    resp_feats = np.empty((0, n_bands + 1 + 5))
    heart_feats = np.empty((0, n_bands + 1 + 5))
    
    # Variable auxiliar que indica el inicio de una señal
    beg = 0
    
    for i in intervals:
        # Definición de los límites a revisar para el sonido respiratorio
        lower_resp = beg // sr_ratio - N_lax if beg != 0 else 0
        upper_resp = i[0] // sr_ratio + N_lax
        
        # Definición de los límites a revisar para el sonido cardiaco
        lower_heart = i[0] // sr_ratio - N_lax
        upper_heart = i[1] // sr_ratio + N_lax
        
        # Segmento de señal respiratoria
        resp_signal = signal_in[lower_resp:upper_resp]
        # Y cardiaca
        heart_signal = signal_in[lower_heart:upper_heart]
        
        # Calculando las características de interés
        resp_centroid = spectral_centroid(resp_signal, samplerate, 
                                          n_fft=N, hop_length=N//2)
        resp_flatness = spectral_flatness(resp_signal, n_fft=N, 
                                          hop_length=N//2)
        resp_bandwidth = spectral_bandwidth(resp_signal, samplerate,
                                            n_fft=N, hop_length=N//2)
        resp_rolloff = spectral_rolloff(resp_signal, samplerate,
                                        n_fft=N, hop_length=N//2, 
                                        roll_percent=0.85)
        resp_contrast = spectral_contrast(resp_signal, samplerate,
                                          n_fft=N, hop_length=N//2,
                                          n_bands=n_bands, fmin=100)
        resp_zerocross = zero_crossing_rate(resp_signal, frame_length=N,
                                            hop_length=N//2)
        
        heart_centroid = spectral_centroid(heart_signal, samplerate, 
                                           n_fft=N, hop_length=N//2)
        heart_flatness = spectral_flatness(heart_signal, n_fft=N, 
                                           hop_length=N//2)
        heart_bandwidth = spectral_bandwidth(heart_signal, samplerate,
                                             n_fft=N, hop_length=N//2)
        heart_rolloff = spectral_rolloff(heart_signal, samplerate,
                                         n_fft=N, hop_length=N//2, 
                                         roll_percent=0.85)
        heart_contrast = spectral_contrast(heart_signal, samplerate,
                                           n_fft=N, hop_length=N//2,
                                           n_bands=n_bands, fmin=100)
        heart_zerocross = zero_crossing_rate(heart_signal, frame_length=N,
                                             hop_length=N//2)
        
        # Agregando a las listas
        resp_list.append(resp_signal)
        heart_list.append(heart_signal)
        
        # Agregando las características
        resp_feats_i = np.concatenate(([resp_centroid.mean()], 
                                       [resp_flatness.mean()],
                                       [resp_bandwidth.mean()],
                                       [resp_rolloff.mean()],
                                       resp_contrast.mean(axis=1),
                                       [resp_zerocross.mean()]))
        heart_feats_i = np.concatenate(([heart_centroid.mean()], 
                                        [heart_flatness.mean()],
                                        [heart_bandwidth.mean()],
                                        [heart_rolloff.mean()],
                                        heart_contrast.mean(axis=1),
                                        [heart_zerocross.mean()]))
        
        # Para incorporarlas al vector de características
        resp_feats = np.vstack((resp_feats, resp_feats_i))
        heart_feats = np.vstack((heart_feats, heart_feats_i))
        
        # Se redefine la variable auxiliar
        beg = i[1]
    
    # Finalmente se agrega el segmento final
    resp_signal = signal_in[i[1] // sr_ratio - N_lax:]
    resp_list.append(resp_signal)
    
    # Calculando las características de interés
    resp_centroid = spectral_centroid(resp_signal, samplerate, 
                                      n_fft=N, hop_length=N//2)
    resp_flatness = spectral_flatness(resp_signal, n_fft=N, 
                                      hop_length=N//2)
    resp_bandwidth = spectral_bandwidth(resp_signal, samplerate,
                                        n_fft=N, hop_length=N//2)
    resp_rolloff = spectral_rolloff(resp_signal, samplerate,
                                    n_fft=N, hop_length=N//2, 
                                    roll_percent=0.85)
    resp_contrast = spectral_contrast(resp_signal, samplerate,
                                      n_fft=N, hop_length=N//2,
                                      n_bands=n_bands, fmin=100)
    resp_zerocross = zero_crossing_rate(resp_signal, frame_length=N,
                                        hop_length=N//2)
    
    # Agregando las características
    resp_feats_i = np.concatenate(([resp_centroid.mean()], 
                                   [resp_flatness.mean()],
                                   [resp_bandwidth.mean()],
                                   [resp_rolloff.mean()],
                                   resp_contrast.mean(axis=1),
                                   [resp_zerocross.mean()]))
    
    # Para incorporarlas al vector de características
    resp_feats = np.vstack((resp_feats, resp_feats_i))
        
    # Definición de la matriz de entrenamiento
    X_train = np.vstack((resp_feats, heart_feats))
    
    '''
    print(X_train)
    print(np.argwhere(np.isnan(X_train)))
    
    plt.figure()
    plt.pcolormesh(X_train, cmap='jet')
    plt.colorbar()
    plt.show()
    return
    '''
    
    # Definición de la matriz de etiquetas de entrenamiento. Se define como
    # 0 para respiración y 1 para corazón
    Y_train =  np.array([0] * resp_feats.shape[0] +
                        [1] * heart_feats.shape[0])
    
    # Reducción de dimensiones
    pca = PCA(n_components=pca_comps)
    X_pca = pca.fit_transform(X_train)
    
    '''
    %matplotlib notebook
    for num, i in enumerate(X_pca):
        if Y_train[num] == 0:
            color = 'r'
        elif Y_train[num] == 1:
            color = 'b'
        
        plt.scatter(i[1], i[2], color=color)
    
    plt.show()
    return
    '''
    
    if classifier == 'knn':
        clas = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='auto')
    elif classifier == 'svm':
        clas = svm.SVC(kernel='linear', degree=50, gamma='auto')
    else:
        raise Exception('Opción "classifier" no válida.')
    
    # Ajuste del clasificador
    clas.fit(X_pca, Y_train)
    
    # Definición de la variable de decisión
    decision = list()
    
    # Para cada componente
    for comp_sound in comps:
        # Para cada iteración, se define nuevamente esta lista auxiliar que sirve para
        # tomar la decisión de la componente i-ésima en base a la mayoría de votos de 
        # las características del sonido cardiorrespiratorio
        decision_comp = list()
        
        for i in intervals:
            # Definición de los límites a revisar para el sonido cardiaco
            lower = i[0] // sr_ratio - N_lax
            upper = i[1] // sr_ratio + N_lax
            
            # Definición del sonido cardiorrespiratorio i en el sonido completo
            hr_sound = resampling_by_points(comp_sound[lower:upper], samplerate, N)
            
            # Se calculan los features
            comp_centroid = spectral_centroid(hr_sound, samplerate, 
                                              n_fft=N, hop_length=N//2)
            comp_flatness = spectral_flatness(hr_sound, n_fft=N, 
                                              hop_length=N//2)
            comp_bandwidth = spectral_bandwidth(hr_sound, samplerate,
                                                n_fft=N, hop_length=N//2)
            comp_rolloff = spectral_rolloff(hr_sound, samplerate,
                                            n_fft=N, hop_length=N//2, 
                                            roll_percent=0.85)
            comp_contrast = spectral_contrast(hr_sound, samplerate,
                                              n_fft=N, hop_length=N//2,
                                              n_bands=n_bands, fmin=100)
            comp_zerocross = zero_crossing_rate(hr_sound, frame_length=N,
                                                hop_length=N//2)
            
            feat = np.concatenate(([comp_centroid.mean()], 
                                   [comp_flatness.mean()],
                                   [comp_bandwidth.mean()],
                                   [comp_rolloff.mean()],
                                   comp_contrast.mean(axis=1),
                                   [comp_zerocross.mean()]))
            
            # Y se transforma
            feat = pca.transform(feat.reshape(1,-1))           
            # plt.scatter(feat[0], feat[1], color='cyan', marker='.')
            
            # Resultado clasificación
            Y_pred = clas.predict(feat)
            
            # Agregando a la lista
            decision_comp.append(Y_pred[0])
        print(decision_comp)
        # Una vez completada la lista de decisiones, se procede a contar sus propiedades
        q_hearts = np.sum(decision_comp)
        
        # Si la cantidad de componentes de corazón es mayor al 50%, entonces es corazón,
        # en caso contrario es respiración
        if q_hearts / len(decision_comp) >= 0.7:
            decision.append(True)
        else:
            decision.append(False)
     
    return decision


def centroid_test(W, limit='upper', gamma=1):
    '''Criterio de centroide propuesto.
    
    Parameters
    ----------
    W : ndarray
        Matriz de plantillas espectrales W del NMF.
    limit : {'upper', 'mean_range'}, float, optional
        Tipo de límite utilizado para clasificar las componetntes. 'upper' 
        clasifica como corazón a las componentes cuyas centroides estén 
        sobre el promedio. 'mean_range' clasifica como corazón a las 
        componentes que se encuentren en el rango mu +- gamma*sigma. Se puede 
        utilizar un float como límite específico también. Por defecto es 
        'upper'.
    gamma : float, optional
        Factor de ponderación de sigma en el criterio "mean_range" del 
        parámetro limit. Por defecto es 1.
    '''
    # Cálculo de los centroides de la matriz W
    centroid_W = np.matmul(W.T, np.arange(0, W.shape[0])) / W.shape[0]
    print(centroid_W)
    # Cálculo de la media del centroide
    mu_c = centroid_W.mean()
    sd_c = centroid_W.std()
    
    if limit == 'upper':
        # Para valores mayores -> corazón. Caso contrario -> pulmón
        bool_out = centroid_W >= mu_c
    elif limit == 'mean_range':
        # Para valores dentro del rango, es corazón. Fuera es pulmón
        bool_out = (mu_c - gamma * sd_c <= centroid_W) & \
                   (centroid_W <= mu_c + gamma * sd_c)
    else:
        bool_out = centroid_W <= limit
        
    return bool_out, centroid_W

"""