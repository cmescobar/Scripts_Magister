def nmf_decomposition_1(signal_in, samplerate, n_components=2, N=2048, noverlap=1024,
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


def nmf_decomposition_2(signal_in, samplerate, n_components=2, N=2048, noverlap=1024, 
                      padding=0, window='hamming', whole=False, alpha_wiener=1,  
                      wiener_filt=True, init='random', solver='cd', beta=2,
                      tol=1e-4, max_iter=200, alpha_nmf=0, l1_ratio=0,
                      random_state=None, W_0=None, H_0=None, same_outshape=True,
                      plot_spectrogram=False):
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
    t, f, S = get_spectrogram(signal_in, samplerate, N=N, padding=padding, 
                              noverlap=noverlap, window=window, whole=whole)
        
    # Definiendo la magnitud del espectrograma (elemento a estimar)
    X = np.abs(S)
    
    if plot_spectrogram:
        plt.pcolormesh(t, f, 20*np.log10(X + 1e-12), cmap='inferno')
        plt.colorbar()
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
    
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


def nmf_decomposition_3(signal_in, samplerate, n_components=2, N=2048, noverlap=1024, 
                      iter_prom=1, padding=0, window='hann', whole=False, alpha_wiener=1,  
                      filter_out='wiener', init='random', solver='cd', beta=2,
                      tol=1e-4, max_iter=200, alpha_nmf=0, l1_ratio=0,
                      random_state=0, W_0=None, H_0=None, same_outshape=True,
                      plot_spectrogram=False, scale='abs', db_basys=1e-15):
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
        calcular la STFT. Por defecto es 1024.
    iter_prom : int, optional
        Cantidad N de soluciones obtenidas a partir de la descomposición a promediar para
        obtener las matrices W y H. Por defecto es 1.
    padding : int, optional
        Cantidad de ceros añadidos al final para aplicar zero padding. Por defecto es 0.
    window : {None, 'hamming', 'hann', 'nutall', 'tukey'}, optional
        Opciones para las ventanas a utilizar en el cálculo de cada segmento del STFT.
        En caso de elegir None, se asume la ventana rectangular. Por defecto es 'hann'.
    whole : bool, optional
        Indica si se retorna todo el espectro de frecuencia de la STFT o solo la mitad 
        (por redundancia). True lo entrega completo, False la mitad. Por defecto es False.
    alpha_wiener : int, optional
        Exponente alpha del filtro de Wiener. Por defecto es 1.
    filter_out : {None, 'wiener', 'binary'}, optional
        Tipo de filtro utilizado para la reconstrucción de la señal. Si es None, se reconstruye
        directamente utilizando lo obtenido. Si es 'wiener', se aplica un filtro de Wiener. 
        Si es 'binary' se aplica un filtro binario. Por defecto es 'wiener'.
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
    scale : {'abs', 'dB'}, optional
        Opción de escala utilizada para la entrada. 'abs' utiliza el espectrograma de la
        señal, mientras que 'dB' utiliza el espectrograma en decibeles. Por defecto es 'abs'.
    db_basys : float, optional
        Valor base del espectrograma en decibeles (para evitar divisiones por cero). 
        Por defecto es 1e-15.
    
    Returns
    -------
    components : list
        Lista que contiene las componentes en el dominio del tiempo.
    Y_list : list
        Lista que contiene las componentes en espectrogramas.
    S : ndarray
        Espectrograma de la señal de entrada.
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
        
    # Obteniendo el espectrograma
    t, f, S = get_spectrogram(signal_in, samplerate, N=N, padding=padding, 
                              noverlap=noverlap, window=window, whole=whole)
    
    # Graficando
    if plot_spectrogram:
        plt.pcolormesh(t, f, 20 * np.log10(abs(S) + db_basys), cmap='inferno')
        plt.colorbar()
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
    
    # Definiendo la magnitud del espectrograma (elemento a estimar)
    if scale == 'abs':
        X = np.abs(S)
        to_up = 0       #Se define para evitar errores
    elif scale == 'dB':
        X = 20 * np.log10(np.abs(S) + db_basys)
        
        # Sumando el mínimo a todas las entradas (para que todo sea no negativo)
        to_up = abs(np.min(X))
        X += to_up
    else:
        raise Exception('Opción ingresada en "scale" no soportada.')
    
    # Dimensión del espectrograma
    n_row, n_col = X.shape
    
    # Definición de la matriz W y H
    W = np.zeros((n_row, n_components))
    H = np.zeros((n_components, n_col))
    
    for i in range(iter_prom):
        # Definiendo el modelo de NMF
        model = NMF(n_components=n_components, init=init, solver=solver,
                    beta_loss=beta, tol=tol, max_iter=max_iter, 
                    random_state=random_state + i, alpha=alpha_nmf, l1_ratio=l1_ratio)

        # Ajustando W
        if init == 'random':
            W_iter = model.fit_transform(X)
        elif init == 'custom':
            W_iter = model.fit_transform(X, W=W_0, H=H_0)
        else:
            raise Exception('Opción ingresada en "init" no soportada.')

        # Ajustando H
        H_iter = model.components_
        
        # Agregando a W y H
        W += W_iter
        H += H_iter
    
    # Promediando
    W /= iter_prom
    H /= iter_prom
    
    # Filtro de salida
    if filter_out == 'binary':
        components, Y_list = _binary_masking(signal_in, W, H, S, n_components, N=N, 
                                             noverlap=noverlap, window=window, 
                                             whole=whole, same_outshape=same_outshape)
    elif filter_out == 'wiener':
        components, Y_list = _wiener_masking(signal_in, W, H, S, n_components, N=N, 
                                             noverlap=noverlap, window=window, 
                                             whole=whole, alpha_wiener=alpha_wiener,
                                             same_outshape=same_outshape,)
    elif filter_out is None:
        components, Y_list = _no_masking(signal_in, W, H, S, n_components, N=N, 
                                         noverlap=noverlap, window=window, 
                                         whole=whole, same_outshape=same_outshape,
                                         scale=scale)
    else:
        raise Exception('Opción ingresada en "filter_out" no soportada.')
    
    return components, Y_list, S, W, H


def _binary_masking(signal_in, W, H, S, k, N, noverlap, window, whole, same_outshape=True):
    '''Función que permite aplicar enmascaramiento binario a las componentes obtenidas
    mediante descomposición NMF. Esto se obtiene de la forma:
    M_i = {1 si X_i > X_j donde j in {componentes} y j != i
           0 en otro caso}
    
    Esta máscara, en síntesis, indica dónde se encuentra el máximo de todas las 
    componentes en una entrada (p,q) de la matriz (donde p indica alguna fila y q
    alguna columna). La componente que tenga la entrada (p,q) máxima en comparación
    con las demás componentes, tendrá el valor 1. Así, cuando se recomponga la señal,
    la componente i tendrá el valor de la señal original "S" en (p,q) y ninguna otra
    componente lo tendrá.
    
    Parameters
    ----------
    signal_in : ndarray or list
        Señal a descomponer.
    W : ndarray
        Matriz que contiene la información espectral de las componentes
    H : ndarray
        Matriz que contiene la información temporal de las componentes
    S : ndarray
        Espectrograma de la señal a descomponer
    k : int
        Cantidad de componentes a descomponer (rango interno)
    noverlap : float
        Cantidad de puntos de traslape que existe entre una ventana y la siguiente al 
        calcular la STFT
    window : {None, 'hamming', 'hann', 'nutall', 'tukey'}
        Opciones para las ventanas a utilizar en el cálculo de cada segmento del STFT.
    whole : bool
        Indica si se retorna todo el espectro de frecuencia de la STFT o solo la mitad 
        (por redundancia). True lo entrega completo, False la mitad.
    same_outshape : bool, optional
        'True' para que la salida tenga el mismo largo que la entrada. 'False' entrega el
        largo de la señal obtenido después de la STFT. Por defecto es "True".
        
    Returns
    -------
    components : list
        Lista que contiene las componentes en el dominio del tiempo.
    Y_list : list
        Lista que contiene las componentes en espectrogramas.
        
    Referencias
    -----------
    [1] ChingShun Lin and Erwin Hasting. Blind Source Separation of Heart and Lung Sounds 
        Based on Nonnegative Matrix Factorization. Department of Electronic and Computer 
        Engineering. 2013.
    [2] Ghafoor Shah, Peter Koch, and Constantinos B. Papadias. On the Blind Recovery of
        Cardiac and Respiratory Sounds. IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, 
        VOL. 19, NO. 1, JANUARY 2015
    '''
    # Definición de la lista para almacenar las fuentes
    sources = np.zeros((W.shape[0], H.shape[1], k))

    # Obteniendo las fuentes y aplicando los filtros
    for i in range(k):
        source_i = np.outer(W[:,i], H[i])
        sources[:,:,i] = source_i

    # Obteniendo el arreglo donde se encuentra cada máximo
    masks_indexes = np.argmax(sources, axis=2)
    
    # Definición de la lista de componentes
    Y_list = list()
    components = list()
        
    for i in range(k):
        Yi = np.where(masks_indexes == i, abs(S), 0)
        
        # Y posteriormente la transformada inversa
        yi = get_inverse_spectrogram(Yi * np.exp(1j * np.angle(S)), 
                                     N=N, noverlap=noverlap, window=window, 
                                     whole=whole)
        
        if same_outshape:
            yi = yi[:len(signal_in)]
        
        # Agregando a la lista de componentes
        Y_list.append(Yi)
        components.append(np.real(yi))
        
    return components, Y_list


def _wiener_masking(signal_in, W, H, S, k, N, noverlap, window, whole, alpha_wiener, 
                    same_outshape=True):
    '''Función que permite aplicar enmascaramiento por filtro de Wiener a 
    las componentes obtenidas mediante descomposición NMF.
    
    Parameters
    ----------
    signal_in : ndarray or list
        Señal a descomponer.
    W : ndarray
        Matriz que contiene la información espectral de las componentes
    H : ndarray
        Matriz que contiene la información temporal de las componentes
    S : ndarray
        Espectrograma de la señal a descomponer
    k : int
        Cantidad de componentes a descomponer (rango interno)
    noverlap : float
        Cantidad de puntos de traslape que existe entre una ventana y la siguiente al 
        calcular la STFT
    window : {None, 'hamming', 'hann', 'nutall', 'tukey'}
        Opciones para las ventanas a utilizar en el cálculo de cada segmento del STFT.
    whole : bool
        Indica si se retorna todo el espectro de frecuencia de la STFT o solo la mitad 
        (por redundancia). True lo entrega completo, False la mitad.
    alpha_wiener : int
        Exponente alpha del filtro de Wiener.
    same_outshape : bool, optional
        'True' para que la salida tenga el mismo largo que la entrada. 'False' entrega el
        largo de la señal obtenido después de la STFT. Por defecto es "True".
        
    Returns
    -------
    components : list
        Lista que contiene las componentes en el dominio del tiempo.
    Y_list : list
        Lista que contiene las componentes en espectrogramas.
        
    Referencias
    -----------
    [1] Canadas-Quesada, F. J., Ruiz-Reyes, N., Carabias-Orti, J., Vera-Candeas, P., &
        Fuertes-Garcia, J. (2017). A non-negative matrix factorization approach based on 
        spectro-temporal clustering to extract heart sounds. Applied Acoustics.
    '''
    # Definición de la lista de componentes
    Y_list = list()
    components = list()
    
    # Obteniendo las fuentes y aplicando los filtros
    for i in range(k):
        source_i = np.outer(W[:,i], H[i])
        
        # Aplicando el filtro de wiener
        filt_source_i = wiener_filter(abs(S), source_i, W, H, alpha=alpha_wiener)
        
        # Aplicando el filtro
        Yi = filt_source_i * np.exp(1j * np.angle(S))
        
        # Y posteriormente la transformada inversa
        yi = get_inverse_spectrogram(Yi, N=N, noverlap=noverlap, window=window, 
                                     whole=whole)

        if same_outshape:
            yi = yi[:len(signal_in)]
        
        # Agregando a la lista de componentes
        components.append(np.real(yi))
        Y_list.append(Yi)
    
    return components, Y_list


def _no_masking(signal_in, W, H, S, k, N, noverlap, window, whole, scale, same_outshape=True):
    '''Función que recompone las componentes obtenidas mediante descomposición NMF
    sin aplicar ninguna máscara.
    
    Parameters
    ----------
    signal_in : ndarray or list
        Señal a descomponer.
    W : ndarray
        Matriz que contiene la información espectral de las componentes
    H : ndarray
        Matriz que contiene la información temporal de las componentes
    S : ndarray
        Espectrograma de la señal a descomponer
    k : int
        Cantidad de componentes a descomponer (rango interno)
    noverlap : float
        Cantidad de puntos de traslape que existe entre una ventana y la siguiente al 
        calcular la STFT
    window : {None, 'hamming', 'hann', 'nutall', 'tukey'}
        Opciones para las ventanas a utilizar en el cálculo de cada segmento del STFT.
    whole : bool
        Indica si se retorna todo el espectro de frecuencia de la STFT o solo la mitad 
        (por redundancia). True lo entrega completo, False la mitad.
    same_outshape : bool, optional
        'True' para que la salida tenga el mismo largo que la entrada. 'False' entrega el
        largo de la señal obtenido después de la STFT. Por defecto es "True".
    
    Returns
    -------
    components : list
        Lista que contiene las componentes en el dominio del tiempo.
    Y_list : list
        Lista que contiene las componentes en espectrogramas.
    '''
    # Criterio de salida
    if scale == 'dB':
        raise Exception('No es posible reconstruir sin máscara en escala dB.')
    
    # Definición de la lista de componentes
    Y_list = list()
    components = list()
    
    # Obteniendo las fuentes y aplicando los filtros
    for i in range(k):
        source_i = np.outer(W[:,i], H[i])
        
        # Aplicando el filtro
        Yi = source_i * np.exp(1j * np.angle(S))
        
        # Y posteriormente la transformada inversa
        yi = get_inverse_spectrogram(Yi, N=N, noverlap=noverlap, window=window, 
                                     whole=whole)
        
        if same_outshape:
            yi = yi[:len(signal_in)]
        
        # Agregando a la lista de componentes
        components.append(np.real(yi))
        Y_list.append(Yi)
    
    return components, Y_list
