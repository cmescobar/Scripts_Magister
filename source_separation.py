import os
import ctypes
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from ast import literal_eval
from sklearn.decomposition import NMF
from matplotlib.widgets import Button
from wavelet_functions import wavelet_denoising
from fading_functions import fade_connect_signals, fading_signal
from math_functions import wiener_filter, raised_cosine_fading
from descriptor_functions import get_spectrogram, get_inverse_spectrogram, centroide
from filter_and_sampling import resampling_by_points, lowpass_filter, downsampling_signal
from clustering_functions import spectral_correlation_test, centroid_test,\
    temporal_correlation_test, temporal_correlation_test_segment, machine_learning_clustering


def nmf_decomposition(signal_in, samplerate, n_components=2, N=2048, noverlap=1024, 
                      iter_prom=1, padding=0, repeat=0, window='hann', whole=False, 
                      alpha_wiener=1, filter_out='wiener', init='random', solver='cd', 
                      beta=2, tol=1e-4, max_iter=200, alpha_nmf=0, l1_ratio=0,
                      random_state=None, W_0=None, H_0=None, same_outshape=True,
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
    repeat : int, optional
        Cantidad de veces que se repite la señal en el cálculo de la STFT. Por defecto es 0.
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
    t, f, S = get_spectrogram(signal_in, samplerate, N=N, padding=padding, repeat=repeat, 
                              noverlap=noverlap, window=window, whole=whole)
    
    # Graficando
    if plot_spectrogram:
        plt.pcolormesh(t, f, 20 * np.log10(abs(S) + db_basys), cmap='jet')
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
                                             padding=padding, repeat=repeat,
                                             whole=whole, same_outshape=same_outshape)
    elif filter_out == 'wiener':
        components, Y_list = _wiener_masking(signal_in, W, H, S, n_components, N=N, 
                                             noverlap=noverlap, window=window, 
                                             padding=padding, repeat=repeat,
                                             whole=whole, alpha_wiener=alpha_wiener,
                                             same_outshape=same_outshape,)
    elif filter_out is None:
        components, Y_list = _no_masking(signal_in, W, H, S, n_components, N=N, 
                                         noverlap=noverlap, window=window, 
                                         padding=padding, repeat=repeat,
                                         whole=whole, same_outshape=same_outshape,
                                         scale=scale)
    else:
        raise Exception('Opción ingresada en "filter_out" no soportada.')
    
    return components, Y_list, S, W, H


def nmf_decomposition_denoising(signal_in, samplerate, n_components=2, N=2048, noverlap=1024, 
                                iter_prom=1, padding=0, repeat=0, window='hann', whole=False, 
                                alpha_wiener=1, filter_out='wiener', init='random', solver='cd', 
                                beta=2, tol=1e-4, max_iter=200, alpha_nmf=0, l1_ratio=0,
                                random_state=0, W_0=None, H_0=None, same_outshape=True,
                                plot_spectrogram=False, scale='abs', db_basys=1e-15, 
                                wav_denoising=False, wavelet='db4', level=10, 
                                threshold_criteria='soft', threshold_delta='universal', 
                                log_base='e'):
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
    # Opción de denoising previo
    if wav_denoising:
        signal_to_spect = wavelet_denoising(signal_in, wavelet=wavelet, mode='periodization', 
                                            level=level, threshold_criteria=threshold_criteria, 
                                            threshold_delta=threshold_delta, min_percentage=None, 
                                            print_delta=False, log_base=log_base,
                                            plot_levels=False, delta=None)
    else:
        signal_to_spect = signal_in

    # Propiedad del overlap
    noverlap = 0 if noverlap <= 0 else noverlap
    noverlap = noverlap if noverlap < N else N - 1
        
    # Obteniendo el espectrograma
    t, f, S = get_spectrogram(signal_to_spect, samplerate, N=N, padding=padding, 
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
                                             padding=padding, repeat=repeat,
                                             whole=whole, same_outshape=same_outshape)
    elif filter_out == 'wiener':
        components, Y_list = _wiener_masking(signal_in, W, H, S, n_components, N=N, 
                                             noverlap=noverlap, window=window, 
                                             padding=padding, repeat=repeat,
                                             whole=whole, alpha_wiener=alpha_wiener,
                                             same_outshape=same_outshape,)
    elif filter_out is None:
        components, Y_list = _no_masking(signal_in, W, H, S, n_components, N=N, 
                                         noverlap=noverlap, window=window, 
                                         padding=padding, repeat=repeat,
                                         whole=whole, same_outshape=same_outshape,
                                         scale=scale)
    else:
        raise Exception('Opción ingresada en "filter_out" no soportada.')
    
    return components, Y_list, S, W, H


def nmf_decomposition_bandlimited(signal_in, samplerate, n_components=2, N=2048, noverlap=1024, 
                                  iter_prom=1, padding=0, repeat=0, window='hann', whole=False, 
                                  alpha_wiener=1, filter_out='wiener', init='random', solver='cd', 
                                  beta=2, tol=1e-4, max_iter=200, alpha_nmf=0, l1_ratio=0,
                                  random_state=None, W_0=None, H_0=None, same_outshape=True,
                                  plot_spectrogram=False, scale='abs', db_basys=1e-15, 
                                  interest_band=None):
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
    
    try:
        if len(interest_band) == 1:
            X_to_proc = X[:interest_band[0], :]
        elif len(interest_band) == 2:
            X_to_proc = X[interest_band[0]:interest_band[1], :]
        elif len(interest_band) > 2:
            raise Exception('"interest_band" puede ser de largo máximo 2.')
    except:
        X_to_proc = X
    
    # Dimensión del espectrograma
    n_row, n_col = X_to_proc.shape
    
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
            W_iter = model.fit_transform(X_to_proc)
        elif init == 'custom':
            W_iter = model.fit_transform(X_to_proc, W=W_0, H=H_0)
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
    
    sources = []
    for i in range(n_components):
        source_i = np.outer(W[:,i], H[i])        
        sources.append(source_i)
        
        plt.figure()
        try:
            if len(interest_band) == 1:
                plt.pcolormesh(source_i[:interest_band[0],:], cmap='inferno')
            elif len(interest_band) == 2:
                plt.pcolormesh(source_i[interest_band[0]:interest_band[1],:], cmap='inferno')
        except:
            plt.pcolormesh(source_i, cmap='inferno')
        
        plt.colorbar()
        plt.ylabel(f'Opción {i}')
    
    plt.show()
    dec = int(input('Decision heart: '))
    undec = 1 if dec == 0 else 0
    
    if filter_out == 'binary':
        _, Y_list = _binary_masking(signal_in, W, H, X_to_proc, k=n_components, N=N, 
                                    noverlap=noverlap, window=window, whole=whole, 
                                    padding=padding, repeat=repeat, get_inverse=False)
    
    elif filter_out == 'wiener':
        _, Y_list = _wiener_masking(signal_in, W, H, X_to_proc, k=n_components, N=N, 
                                    noverlap=noverlap, window=window, whole=whole, 
                                    alpha_wiener=alpha_wiener, padding=padding, repeat=repeat, 
                                    get_inverse=False)
    
    elif filter_out is None:
        _, Y_list = _no_masking(signal_in, W, H, X_to_proc, k=n_components, N=N,
                                noverlap=noverlap, window=window, whole=whole,
                                padding=padding, repeat=repeat, scale=scale, get_inverse=False)
    
    # Concatenar lo necesario para cada señal
    try:
        if len(interest_band) == 1:
            # Con la decisión, se concatenan ceros encima del corazón
            source_heart = np.concatenate((Y_list[dec], 
                                           np.zeros((S.shape[0] - interest_band[0], S.shape[1]))), 
                                          axis=0)
            # Y se concatena el resto de la señal encima de la respiración
            source_resp = np.concatenate((Y_list[undec], abs(S[interest_band[0]:,:])), 
                                          axis=0)
        
        elif len(interest_band) == 2:
            # Con la decisión, se concatenan ceros encima del corazón
            source_heart = np.concatenate((np.zeros((interest_band[0], S.shape[1])),
                                           Y_list[dec], 
                                           np.zeros((S.shape[0] - interest_band[1], S.shape[1] )) ), 
                                          axis=0)
            # Y se concatena el resto de la señal encima de la respiración
            source_resp = np.concatenate((abs(S[:interest_band[0],:]), 
                                          Y_list[undec], 
                                          abs(S[interest_band[1]:,:])), 
                                          axis=0)
    except:
        source_heart = Y_list[dec]
        source_resp = Y_list[undec]
    
    plt.figure()
    plt.pcolormesh(20*np.log10(abs(source_resp) + 1e-8), cmap='inferno')
    plt.colorbar()
    plt.show()
    
    plt.figure()
    plt.pcolormesh(20*np.log10(abs(source_heart) + 1e-8), cmap='inferno')
    plt.colorbar()
    plt.show()
    
    # Incorporando la fase
    Yheart = source_heart * np.exp(1j * np.angle(S))
    Yresp = source_resp * np.exp(1j * np.angle(S))

    # Y posteriormente la transformada inversa
    yheart = get_inverse_spectrogram(Yheart, N=N, noverlap=noverlap, window=window, 
                                     whole=whole)
    yresp = get_inverse_spectrogram(Yresp, N=N, noverlap=noverlap, window=window, 
                                    whole=whole)

    if same_outshape:
        yheart = yheart[:len(signal_in)]
        yresp = yresp[:len(signal_in)]
    
    # Definición de la lista de las componentes
    components = list()
    Y_list = list()
    
    # Agregando a la lista de componentes
    components.append(np.real(yheart))
    components.append(np.real(yresp))
    Y_list.append(Yheart)
    Y_list.append(Yresp)
    
    return components, Y_list, S, W, H


def nmf_decomposition_k_more(signal_in, samplerate, n_components=2, N=2048, noverlap=1024, 
                             iter_prom=1, padding=0, repeat=0, window='hann', whole=False, 
                             alpha_wiener=1, filter_out='wiener', init='random', solver='cd', 
                             beta=2, tol=1e-4, max_iter=200, alpha_nmf=0, l1_ratio=0,
                             random_state=None, W_0=None, H_0=None, same_outshape=True,
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
                                             padding=padding, repeat=repeat, 
                                             whole=whole, same_outshape=same_outshape)
    elif filter_out == 'wiener':
        components, Y_list = _wiener_masking(signal_in, W, H, S, n_components, N=N, 
                                             noverlap=noverlap, window=window, 
                                             padding=padding, repeat=repeat, 
                                             whole=whole, alpha_wiener=alpha_wiener,
                                             same_outshape=same_outshape,)
    elif filter_out is None:
        components, Y_list = _no_masking(signal_in, W, H, S, n_components, N=N, 
                                         noverlap=noverlap, window=window, 
                                         padding=padding, repeat=repeat, 
                                         whole=whole, same_outshape=same_outshape,
                                         scale=scale)
    else:
        raise Exception('Opción ingresada en "filter_out" no soportada.')
    
    return components, Y_list, S, W, H


def get_components_HR_sounds(filepath, samplerate_des, sep_type='to all', assign_method='manual', 
                             clustering=False, n_components=2, N=2048, N_lax=1500, N_fade=500, 
                             noverlap=1024, padding=0, repeat=0, window='hamming', 
                             whole=False, alpha_wiener=1, filter_out='wiener', 
                             init='random', solver='cd', beta=2, tol=1e-4, max_iter=200, 
                             alpha_nmf=0, l1_ratio=0, random_state=0, 
                             W_0=None, H_0=None, plot_segments=False, scale='abs', 
                             ausc_zone='Anterior', fcut_spect_crit=200, 
                             measure_spect_crit='correlation', i_selection='max', f1_roll=20, 
                             f2_roll=150, measure_temp_crit='q_equal', H_binary=True, 
                             reduce_to_H=False, dec_criteria='or', version=2, only_centroid=False):
    '''Función que permite generar los archivos de audio de las componentes separadas
    mediante el proceso de NMF.
    
    Parameters
    ----------
    filepath : str
        Dirección dónde se encuentran los archivos de audio a procesar.
    sep_type : {'to all', 'on segments', 'masked segments'}, optional
        Selección del tipo de la base generada por la separación mediante NMF. 'to all' 
        usa la base que separa toda la señal con NMF, 'on_segments' usa la base que 
        solamente separa con NMF en segmentos de interés y 'masked segments' usa la base 
        que es separada luego de aplicar una máscara que hace que solo quede lo que debiera 
        ser el corazón. Por defecto es 'to all'.
    assign_method : {'auto', 'manual'}, optional
        Método de separación de sonidos. Para 'auto', se utiliza una lista de etiquetas 
        creadas manualmente. Para 'manual' se etiqueta segmento a segmento cada componente, 
        las cuales son guardadas en un archivo .txt. Por defecto es 'manual'. (Solo útil 
        para sep_type == "on segments" o "masked segments").
    kwargs : Parámetros a revisar en NMF segments
    '''
    # Lista de los archivos de la carpeta
    filenames = [i for i in os.listdir(filepath) if i.endswith('.wav')]
    
    # Definición de 
    if sep_type == 'to all':
        # Definición del diccionario que contiene los parámetros de simulación
        dict_simulation = {'n_components': n_components, 'N': N, 'clustering': clustering, 
                           'noverlap': noverlap, 'repeat': repeat, 'padding': padding,
                           'window': window, 'filter_out': filter_out, 'init': init, 
                           'solver': solver, 'beta': beta, 'tol': tol, 
                           'max_iter': max_iter, 'alpha_nmf': alpha_nmf, 
                           'l1_ratio': l1_ratio, 'random_state': random_state,
                           'scale': scale, 'ausc_zone': ausc_zone, 
                           'fcut_spect_crit': fcut_spect_crit, 
                           'measure_spect_crit': measure_spect_crit, 
                           'i_selection': i_selection, 'f1_roll': f1_roll, 
                           'f2_roll': f2_roll, 'measure_temp_crit': measure_temp_crit, 
                           'H_binary': H_binary, 'reduce_to_H': reduce_to_H, 
                           'dec_criteria': dec_criteria}
        
        # Definición del filepath a guardar
        filepath_to_save = f'{filepath}/Components/Separation to all'
    
    elif sep_type == 'on segments':
        # Definición del diccionario que contiene los parámetros de simulación
        dict_simulation = {'n_components': n_components, 'N': N, 'clustering': clustering,
                           'N_lax': N_lax, 'N_fade': N_fade, 'noverlap': noverlap, 
                           'repeat': repeat, 'padding': padding, 'window': window, 
                           'filter_out': filter_out, 'init': init, 'solver': solver, 
                           'beta': beta, 'tol': tol, 
                           'max_iter': max_iter, 'alpha_nmf': alpha_nmf, 
                           'l1_ratio': l1_ratio, 'random_state': random_state,
                           'scale': scale, 'ausc_zone': ausc_zone, 
                           'fcut_spect_crit': fcut_spect_crit, 
                           'measure_spect_crit': measure_spect_crit, 
                           'i_selection': i_selection, 'f1_roll': f1_roll, 
                           'f2_roll': f2_roll, 'measure_temp_crit': measure_temp_crit, 
                           'H_binary': H_binary, 'reduce_to_H': reduce_to_H, 
                           'dec_criteria': dec_criteria, 'only_centroid': only_centroid}
        
        # Definición del filepath a guardar
        filepath_to_save = f'{filepath}/Components/Separation on segments'
    
    elif sep_type == 'masked segments':
        # Definición del diccionario que contiene los parámetros de simulación
        dict_simulation = {'n_components': n_components, 'N': N, 'clustering': clustering,
                           'N_lax': N_lax, 'N_fade': N_fade, 'noverlap': noverlap, 
                           'repeat': repeat, 'padding': padding, 'window': window, 
                           'filter_out': filter_out, 'init': init, 'solver': solver, 
                           'beta': beta, 'tol': tol, 
                           'max_iter': max_iter, 'alpha_nmf': alpha_nmf, 
                           'l1_ratio': l1_ratio, 'random_state': random_state,
                           'scale': scale, 'ausc_zone': ausc_zone, 
                           'fcut_spect_crit': fcut_spect_crit, 
                           'measure_spect_crit': measure_spect_crit, 
                           'i_selection': i_selection, 'f1_roll': f1_roll, 
                           'f2_roll': f2_roll, 'measure_temp_crit': measure_temp_crit, 
                           'H_binary': H_binary, 'reduce_to_H': reduce_to_H, 
                           'dec_criteria': dec_criteria}
        
        # Definición del filepath a guardar
        filepath_to_save = f'{filepath}/Components/Masking on segments'
    
    # Preguntar si es que la carpeta que almacenará la simulación se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(filepath_to_save):
        os.makedirs(filepath_to_save)
    
    # Refresco del diccionario de simulación
    dict_simulation, continue_dec = _manage_registerdata_nmf(dict_simulation, 
                                                             filepath_to_save)
    # Seguir con la rutina
    if not continue_dec:
        return None
    
    # Creación de carpeta para opción to all (para las otras se crea en la rutina)
    if sep_type == 'to all':
        # Definición del filepath a guardar para la simulación
        filepath_to_save_id = f'{filepath_to_save}/id {dict_simulation["id"]}'
        
        # Preguntar si es que la carpeta que almacenará las imágenes se ha
        # creado. En caso de que no exista, se crea una carpeta
        if not os.path.isdir(filepath_to_save_id):
            os.makedirs(filepath_to_save_id)
    
    for audio_name in tqdm(filenames, desc='NMF decomp', ncols=70):
        # Dirección del archivo en la carpeta madre. Este archivo es el que se descompondrá
        dir_audio = f"{filepath}/{audio_name}"
        
        if sep_type == 'to all':
            nmf_applied_all(dir_audio, filepath_to_save_id, 
                            assign_method=assign_method,  
                            samplerate_des=samplerate_des,
                            clustering=clustering, n_components=n_components, N=N, 
                            noverlap=noverlap, padding=padding, repeat=repeat, 
                            window=window, whole=whole,
                            alpha_wiener=alpha_wiener, 
                            filter_out=filter_out, init=init, 
                            solver=solver, beta=beta, tol=tol, 
                            max_iter=max_iter, alpha_nmf=alpha_nmf, 
                            l1_ratio=l1_ratio, random_state=random_state, 
                            W_0=W_0, H_0=H_0, scale=scale, version=version,
                            ausc_zone=ausc_zone, fcut_spect_crit=fcut_spect_crit, 
                            measure_spect_crit=measure_spect_crit, i_selection=i_selection, 
                            f1_roll=f1_roll, f2_roll=f2_roll, 
                            measure_temp_crit=measure_temp_crit, H_binary=H_binary, 
                            reduce_to_H=reduce_to_H, dec_criteria=dec_criteria)
        
        elif sep_type == 'on segments':
            nmf_applied_interest_segments(dir_audio, samplerate_des=samplerate_des,
                                          assign_method=assign_method, 
                                          n_components=n_components, N=N, 
                                          N_lax=N_lax, N_fade=N_fade, noverlap=noverlap, 
                                          padding=padding, repeat=repeat, window=window, 
                                          whole=whole, alpha_wiener=alpha_wiener, 
                                          filter_out=filter_out, init=init, 
                                          solver=solver, beta=beta, tol=tol, 
                                          max_iter=max_iter, alpha_nmf=alpha_nmf, 
                                          l1_ratio=l1_ratio, random_state=random_state, 
                                          W_0=W_0, H_0=H_0, plot_segments=plot_segments, 
                                          scale=scale, ausc_zone=ausc_zone, 
                                          fcut_spect_crit=fcut_spect_crit, 
                                          measure_spect_crit=measure_spect_crit, 
                                          i_selection=i_selection, 
                                          f1_roll=f1_roll, f2_roll=f2_roll, 
                                          measure_temp_crit=measure_temp_crit,
                                          H_binary=H_binary, reduce_to_H=reduce_to_H, 
                                          dec_criteria=dec_criteria, version=version, 
                                          clustering=clustering, only_centroid=only_centroid)
        
        elif sep_type == 'masked segments':
            nmf_applied_masked_segments(dir_audio, samplerate_des=samplerate_des,
                                        assign_method=assign_method, 
                                        n_components=n_components, N=N, 
                                        N_lax=N_lax, N_fade=N_fade, noverlap=noverlap, 
                                        padding=padding, repeat=repeat, window=window, 
                                        whole=whole, alpha_wiener=alpha_wiener, 
                                        filter_out=filter_out, init=init, 
                                        solver=solver, beta=beta, tol=tol, 
                                        max_iter=max_iter, alpha_nmf=alpha_nmf, 
                                        l1_ratio=l1_ratio, random_state=random_state, 
                                        W_0=W_0, H_0=H_0, plot_segments=plot_segments, 
                                        scale=scale, ausc_zone=ausc_zone, 
                                        fcut_spect_crit=fcut_spect_crit, 
                                        measure_spect_crit=measure_spect_crit, 
                                        i_selection=i_selection, 
                                        f1_roll=f1_roll, f2_roll=f2_roll, 
                                        measure_temp_crit=measure_temp_crit,
                                        H_binary=H_binary, reduce_to_H=reduce_to_H, 
                                        dec_criteria=dec_criteria, version=version, 
                                        clustering=clustering)
            
        else:
            raise Exception('Opción para "sep_type" no soportada.')


def nmf_applied_all(dir_file, filepath_to_save_id, samplerate_des, assign_method, 
                    clustering=True, n_components=2, N=2048, noverlap=1024, repeat=0, 
                    padding=0, window='hamming', whole=False, alpha_wiener=1, 
                    filter_out='wiener', init='random', solver='cd', beta=2, tol=1e-4, 
                    max_iter=200, alpha_nmf=0, l1_ratio=0, random_state=0, W_0=None, H_0=None, 
                    scale='abs', version=2, ausc_zone='Anterior', fcut_spect_crit=200, 
                    measure_spect_crit='correlation', i_selection='max', f1_roll=20, 
                    f2_roll=150, measure_temp_crit='q_equal', H_binary=True, 
                    reduce_to_H=False, dec_criteria='or'):
    '''Función que permite obtener la descomposición NMF de una señal (ingresando su
    ubicación en el ordenador), la cual descompone toda la señal.
    
    Parameters
    ----------
    dir_file : str
        Dirección del archivo de audio a segmentar.
    filepath_to_save_id : str
        Dirección donde se almacenerá las componentes segmentadas.
    samplerate_des : int
        Tasa de muestreo deseada.
    clustering : Bool, optional
        Tipo de decisión del clustering. Por defecto es automático (True).
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
    alpha_wiener : int, optional
        Exponente alpha del filtro de Wiener. Por defecto es 1.
    filter_out : {None, 'wiener', 'binary'}, optional
        Tipo de filtro utilizado para la reconstrucción de la señal. Si es None, se reconstruye
        directamente utilizando lo obtenido. Si es 'wiener', se aplica un filtro de Wiener. 
        Si es 'binary' se aplica un filtro binario. Por defecto es 'wiener'.
    **kwargs : Revisar nmf_decomposition para el resto.
    
    Returns
    -------
    comps : list of ndarray
        Lista que contiene ambas señales descompuestas mediante NMF.
    '''
    # Recuperando el nombre del archivo
    audio_name = dir_file.split('/')[-1]
    
    # Lectura del archivo
    signal_in, samplerate = sf.read(dir_file)
    
    # Solo si es que hay que bajar puntos se baja, en caso contrario se mantiene
    if samplerate_des < samplerate:
        _, signal_to = downsampling_signal(signal_in, samplerate, 
                                           samplerate_des//2-100, 
                                           samplerate_des//2)
    else:
        signal_to = signal_in
        samplerate_des = samplerate
    
    # Aplicando la descomposición
    comps, _, _, W, H = nmf_decomposition(signal_to, samplerate_des, 
                                          n_components=n_components, 
                                          N=N, noverlap=noverlap, padding=padding,
                                          repeat=repeat, window=window, whole=whole, 
                                          alpha_wiener=alpha_wiener,
                                          filter_out=filter_out, init=init, 
                                          solver=solver, beta=beta, tol=tol, 
                                          max_iter=max_iter, alpha_nmf=alpha_nmf, 
                                          l1_ratio=l1_ratio, random_state=random_state,
                                          W_0=W_0, H_0=H_0, scale=scale)
    
    # Definiendo el nombre de los archivos
    dir_to_save = f'{filepath_to_save_id}/{audio_name.strip(".wav")}/'
    
    # Preguntar si es que la carpeta que almacenará las imágenes se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(dir_to_save):
        os.makedirs(dir_to_save)
    
    # Criterios de decisión
    if n_components == 2 and not clustering:
        if assign_method == 'auto':
            # Se calculan los centroides
            comp1_centroid = centroide(W[:,0])
            comp2_centroid = centroide(W[:,1])
            
            # Definición de la decisión
            heart_dec = 0 if comp1_centroid >= comp2_centroid else 1
            
            # Generar plots
            _ = _plot_nmf(signal_to, samplerate_des, comps, W, H, 
                          filepath_to_save=dir_to_save,
                          assign_method=assign_method)
            
            # Definición del texto a guardar
            save_txt = ' centroid k2'
        
        elif assign_method == 'manual':
            # Decisión a tomar
            heart_dec = _plot_nmf(signal_to, samplerate_des, comps, W, H, 
                                  filepath_to_save=dir_to_save,
                                  assign_method=assign_method)
            # Definición del texto a guardar
            save_txt = ''
        
        elif assign_method == 'labeled':
            # Se obtiene la decisión etiquetada como sonido cardíaco
            with open(f'{dir_to_save}/Heart decision.txt', 'r', encoding='utf8') as data:
                heart_dec = literal_eval(data.readline().strip())
            
            # Definición del texto a guardar
            save_txt = ''
        
        # Definición de la decisión de respiración
        resp_dec = 1 if heart_dec == 0 else 0
        
        # Definiendo las señales
        heart_signal = comps[heart_dec]
        resp_signal = comps[resp_dec]
    
    elif n_components > 2 or clustering:
        # Definición de la carpeta a revisar
        folder = f'{dir_file.split("/")[0]}'
        
        # Definición del nombre del archivo de sonidos cardiacos
        heart_segments = f'{dir_file.split("/")[-1].strip(".wav").split(" ")[-1]}'
        
        # Definición de la ubicación de los segmentos de sonido cardiaco de interés
        if version == 1:
            filename_heart_segments = f'{folder}/db_heart/Manual combinations v{version}/'\
                                    f'{heart_segments} - segments.txt'
        else:
            filename_heart_segments = f'{folder}/db_heart/Manual combinations v{version}/'\
                                    f'{ausc_zone}/{heart_segments} - segments.txt'
        
        
        if assign_method == 'auto':
            # CRITERIO 1: Definición de la ubicación del diccionario
            filepath_dict = f'Heart component dictionaries/{scale} scale decomposition'
            
            # Aplicación de los criterios
            ## CRITERIO 1: Correlación espectral
            a1_bool, a1 = spectral_correlation_test(W, samplerate_des, fcut_spect_crit, 
                                                    N=N, noverlap=noverlap, 
                                                    n_comps_dict=4, beta=beta, 
                                                    filepath_data=filepath_dict, 
                                                    padding=padding, repeat=repeat, 
                                                    measure=measure_spect_crit, 
                                                    i_selection=i_selection, threshold='mean')

            ## CRITERIO 2: Centroide espectral
            a2_bool, a2 = centroid_test(W, limit='upper', gamma=1)
            
            ## CRITERIO 3: Correlación temporal
            a3_bool, a3 = temporal_correlation_test(H, filename_heart_segments, samplerate, 
                                                    samplerate_des, N=N, N_audio=len(signal_in), 
                                                    noverlap=noverlap, threshold='mean', 
                                                    measure=measure_temp_crit, 
                                                    H_binary=H_binary, reduce_to_H=reduce_to_H,
                                                    version=version, ausc_zone=ausc_zone)
            
            # Decisión final
            if dec_criteria == 'or':
                heart_dec = a1_bool | a2_bool | a3_bool
            elif dec_criteria == 'vote':
                heart_dec = (a1_bool.astype(int) + a2_bool.astype(int) + 
                             a3_bool.astype(int)) >= 2
            elif dec_criteria == 'and':
                heart_dec = a1_bool & a2_bool & a3_bool
            
            # Definición de las señales a grabar
            heart_signal = np.zeros(len(comps[0]))
            resp_signal = np.zeros(len(comps[0]))
            
            # Finalmente, grabando los archivos
            for num, dec in enumerate(heart_dec):
                # Grabando cada componente
                if dec:
                    heart_signal += comps[num]
                else:
                    resp_signal += comps[num]
            
            # Definición del texto a guardar
            save_txt = ' clustering'
            
            # Graficar
            _plot_clustering_points(dir_to_save, a1, a2, a3, 
                                    measure_temp_crit)
            
        elif assign_method == 'machine':
            # CRITERIO: Machine Learning
            heart_dec = \
                machine_learning_clustering(comps, signal_to, samplerate_des, N_lax=20, 
                                            filepath_data=filename_heart_segments, 
                                            N=4096, classifier='knn', n_neighbors=1, 
                                            pca_comps=30, db_basys=1e-12)
                                        
            
            # Definición de las señales a grabar
            heart_signal = np.zeros(len(comps[0]))
            resp_signal = np.zeros(len(comps[0]))
            
            # Finalmente, grabando los archivos
            for num, dec in enumerate(heart_dec):
                # Grabando cada componente
                if dec:
                    heart_signal += comps[num]
                else:
                    resp_signal += comps[num]
            
            # Definición del texto a guardar
            save_txt = ' machine'
    
    # Registrando
    if assign_method in ['auto_multi', 'auto_machine', 'manual']:
        with open(f'{dir_to_save}/Heart decision{save_txt}.txt', 'w', encoding='utf8') as data:
            data.write(str(heart_dec))
    
    # Finalmente, grabando los archivos
    sf.write(f'{dir_to_save} Heart Sound{save_txt}.wav', heart_signal, samplerate_des)
    sf.write(f'{dir_to_save} Respiratory Sound{save_txt}.wav', resp_signal, samplerate_des)
    
    return comps


def nmf_applied_interest_segments(dir_file, samplerate_des, assign_method='manual', n_components=2, 
                                  N=2048, N_lax=0, N_fade=500, noverlap=1024, padding=0,
                                  repeat=0, window='hamming', whole=False, alpha_wiener=1, 
                                  filter_out='wiener', init='random', solver='cd', beta=2,
                                  tol=1e-4, max_iter=200, alpha_nmf=0, l1_ratio=0,
                                  random_state=0, W_0=None, H_0=None, 
                                  plot_segments=False, scale='abs',
                                  ausc_zone='Anterior', fcut_spect_crit=200, 
                                  measure_spect_crit='correlation', i_selection='max', 
                                  f1_roll=20, f2_roll=150, measure_temp_crit='q_equal', 
                                  H_binary=True, reduce_to_H=False, dec_criteria='or', 
                                  version=2, clustering=False, only_centroid=False):
    '''Función que permite obtener la descomposición NMF de una señal (ingresando su
    ubicación en el ordenador), la cual solamente descompone los segmentos de interés
    previamente etiquetados, uno a uno.
    
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
    
    # Solo si es que hay que bajar puntos se baja, en caso contrario se mantiene
    if samplerate_des < samplerate:
        _, signal_to = downsampling_signal(signal_in, samplerate, 
                                           samplerate_des//2-100, 
                                           samplerate_des//2)
        sr_ratio = samplerate // samplerate_des
    else:
        signal_to = signal_in
        samplerate_des = samplerate
        sr_ratio = 1
    
    # Definición de la carpeta donde se ubica
    filepath = '/'.join(dir_file.split('/')[:-1])
    
    # Definición del nombre del archivo
    filename = dir_file.strip('.wav').split('/')[-1]
        
    # Definición de la carpeta a guardar los segmentos
    filepath_to_save = f'{filepath}/Components/Separation on segments'
    
    # Preguntar si es que la carpeta que almacenará las simulaciones se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(filepath_to_save) and plot_segments:
        os.makedirs(filepath_to_save)
    
    # Definición del diccionario que contiene los parámetros de simulación
    dict_simulation = {'n_components': n_components, 'N': N, 'clustering': clustering,
                        'N_lax': N_lax, 'N_fade': N_fade, 'noverlap': noverlap, 
                        'repeat': repeat, 'padding': padding, 'window': window, 
                        'filter_out': filter_out, 'init': init, 'solver': solver, 
                        'beta': beta, 'tol': tol, 
                        'max_iter': max_iter, 'alpha_nmf': alpha_nmf, 
                        'l1_ratio': l1_ratio, 'random_state': random_state,
                        'scale': scale, 'ausc_zone': ausc_zone, 
                        'fcut_spect_crit': fcut_spect_crit, 
                        'measure_spect_crit': measure_spect_crit, 
                        'i_selection': i_selection, 'f1_roll': f1_roll, 
                        'f2_roll': f2_roll, 'measure_temp_crit': measure_temp_crit, 
                        'H_binary': H_binary, 'reduce_to_H': reduce_to_H, 
                        'dec_criteria': dec_criteria, 'only_centroid': only_centroid}
    
    # Control de parámetros simulación (consultar repetir) y asignación de id's
    dict_simulation, _ = _manage_registerdata_nmf(dict_simulation, 
                                                  filepath_to_save,
                                                  in_func=True)
    
    # Definición del filepath a guardar para la simulación
    filepath_to_save_id = f'{filepath_to_save}/id {dict_simulation["id"]}'
    filepath_to_save_name = f'{filepath_to_save_id}/{filename}'
    
    # Preguntar si es que la carpeta que almacenará las imágenes se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(filepath_to_save_name):
        os.makedirs(filepath_to_save_name)
    
    # A partir del nombre del archivo es posible obtener también su lista de intervalos.
    ## Primero se obtiene el nombre del sonido cardíaco a revisar
    file_heart = filename.split(' ')[-1].strip('.wav')
    
    ## Luego se define la dirección del archivo de segmentos
    if version == 1:
        segment_folder = f'Database_manufacturing/db_heart/Manual combinations v{version}'
    else:
        segment_folder = f'Database_manufacturing/db_heart/Manual combinations v{version}/{ausc_zone}'
    
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
    resp_signal = np.copy(signal_to)
    # Definición de la señal cardíaca de salida
    heart_signal = np.zeros(len(signal_to))
    
    # Definición de la lista de puntos para ambos casos
    if assign_method == 'labeled':
        if n_components == 2 and not clustering:
            # Se obtiene la decisión etiquetada como sonido cardíaco
            with open(f'{filepath_to_save_name}/Heart comp labels.txt',
                        'r', encoding='utf8') as data:
                comps_choice = literal_eval(data.readline().strip())
        
        elif n_components > 2 or clustering:
            pass
    
    elif assign_method in ['auto', 'manual']:
        heart_comp_labels = list()
    
    # Aplicando NMF en cada segmento de interés
    for num, interval in enumerate(interval_list, 1):
        # Definición del límite inferior y superior
        lower = (interval[0] - N_lax) // sr_ratio
        upper = (interval[1] + N_lax) // sr_ratio
        
        # Definición del segmento a transformar
        segment = signal_to[lower - N_fade:upper + N_fade]
        
        # Aplicando NMF 
        comps, _, _, W, H = nmf_decomposition(segment, samplerate_des, 
                                              n_components=n_components, 
                                              N=N, noverlap=noverlap, padding=padding, 
                                              repeat=repeat, window=window, whole=whole, 
                                              alpha_wiener=alpha_wiener, 
                                              filter_out=filter_out, init=init, 
                                              solver=solver, beta=beta, tol=tol, 
                                              max_iter=max_iter, alpha_nmf=alpha_nmf, 
                                              l1_ratio=l1_ratio, random_state=random_state,
                                              W_0=W_0, H_0=H_0, scale=scale)
        
        # Graficos
        if plot_segments or assign_method == 'manual':
            if n_components == 2:
                decision_in_plot = _plot_segments_nmf(signal_to, samplerate_des, comps, 
                                                    W, H, N_fade, lower, upper, num, 
                                                    filepath_to_save_name, assign_method)
        
        # Método a implementar
        if n_components == 2 and not clustering:
            if assign_method == 'auto':
                # Se calculan los centroides
                comp1_centroid = centroide(W[:,0])
                comp2_centroid = centroide(W[:,1])
                
                # Definición de la decisión
                heart_decision = 0 if comp1_centroid >= comp2_centroid else 1
                
                # Definición del texto a guardar
                save_txt = ' centroid k2'
            
            elif assign_method == 'manual':
                # Se pregunta para la decisión del sonido cardíaco
                heart_decision = decision_in_plot
                
                # Definición del texto a guardar
                save_txt = ''
            
            elif assign_method == 'labeled':
                # Leyendo los registros
                heart_decision = comps_choice[num - 1]
                
                # Definición del texto a guardar
                save_txt = ''
            
            # Se agrega la decisión a la lista de labels
            heart_comp_labels.append(heart_decision)
            
            # Y se complementa para el sonido respiratorio
            resp_decision = 0 if heart_decision == 1 else 1
            
            # Definición de la componente de señal respiratoria y cardiaca
            heart_comps = comps[heart_decision][:len(segment)]
            resp_comps = comps[resp_decision][:len(segment)]

        elif n_components > 2 or clustering:
            if assign_method == 'auto':
                # CRITERIO 1: Definición de la ubicación del diccionario
                filepath_dict = f'Heart component dictionaries/{scale} scale decomposition'

                # Aplicación de los criterios
                ## CRITERIO 1: Correlación espectral
                a1_bool, a1 = spectral_correlation_test(W, samplerate_des, fcut_spect_crit, 
                                                        N=N, noverlap=noverlap, 
                                                        n_comps_dict=4, beta=beta, 
                                                        filepath_data=filepath_dict, 
                                                        padding=padding, repeat=repeat, 
                                                        measure=measure_spect_crit, 
                                                        i_selection=i_selection, threshold='mean')
                
                ## CRITERIO 2: Centroide espectral
                a2_bool, a2 = centroid_test(W, limit='upper', gamma=1)
                
                ## CRITERIO 3: Correlación temporal
                a3_bool, a3 = \
                     temporal_correlation_test_segment(H, lower, upper, N_fade, 
                                                       N_lax // sr_ratio, 
                                                       samplerate_des,
                                                       threshold='mean', 
                                                       measure=measure_temp_crit, 
                                                       H_binary=H_binary)
                
                # Decisión final
                if dec_criteria == 'or':
                    heart_dec = a1_bool | a2_bool | a3_bool
                elif dec_criteria == 'vote':
                    heart_dec = (a1_bool.astype(int) + a2_bool.astype(int) + 
                                 a3_bool.astype(int)) >= 2
                elif dec_criteria == 'and':
                    heart_dec = a1_bool & a2_bool & a3_bool
                
                if only_centroid:
                    heart_dec = a2_bool
                
                # Definición de las señales a grabar
                heart_comps = np.zeros(len(comps[0]))
                resp_comps = np.zeros(len(comps[0]))
                
                # Finalmente, grabando los archivos
                for num_comp, dec in enumerate(heart_dec):
                    # Grabando cada componente
                    if dec:
                        heart_comps += comps[num_comp]
                    else:
                        resp_comps += comps[num_comp]
                
                # Definición del texto a guardar
                save_txt = ' clustering'
                
                # Graficar
                _plot_clustering_points(filepath_to_save_name, a1, a2, a3, 
                                        measure_temp_crit)
                _plot_segments_nmf_kmore(signal_to, heart_comps, resp_comps, N_fade, 
                                         lower, upper, num, filepath_to_save_name)
            
            elif assign_method == 'labeled':
                pass
        
        # Definición de la lista de señales a concatenar con fading para el corazón
        heart_connect = (heart_signal[:lower], heart_comps, heart_signal[upper:])
        
        # Definición de la lista de señales a concatenar con fading para la respiración
        resp_connect = (resp_signal[:lower], resp_comps, resp_signal[upper:])
        
        # Aplicando fading para cada uno
        heart_signal = fade_connect_signals(heart_connect, N=N_fade, beta=1)
        resp_signal = fade_connect_signals(resp_connect, N=N_fade, beta=1)
    
    if assign_method in ['auto', 'manual']:
        # Guardando el archivo de registro
        with open(f'{filepath_to_save_name}/Heart comp labels{save_txt}.txt', 'w', 
                encoding='utf8') as data:
            data.write(f'{heart_comp_labels}')
        
    # Finalmente, grabando los archivos de audio
    sf.write(f'{filepath_to_save_name}/Respiratory signal{save_txt}.wav', 
                resp_signal, samplerate_des)
    sf.write(f'{filepath_to_save_name}/Heart signal{save_txt}.wav', 
                heart_signal, samplerate_des)
        
    return resp_signal, heart_signal


def nmf_applied_masked_segments(dir_file, samplerate_des, assign_method='manual', n_components=2, 
                                N=2048, N_lax=0, N_fade=100, noverlap=1024, padding=0,
                                repeat=0, window='hamming', whole=False, alpha_wiener=1, 
                                filter_out='wiener', init='random', solver='cd', beta=2,
                                tol=1e-4, max_iter=200, alpha_nmf=0, l1_ratio=0,
                                random_state=0, W_0=None, H_0=None, 
                                plot_segments=False, scale='abs', 
                                ausc_zone='Anterior', fcut_spect_crit=200, 
                                measure_spect_crit='correlation', i_selection='max', 
                                f1_roll=20, f2_roll=150, measure_temp_crit='q_equal', 
                                H_binary=True, reduce_to_H=False, dec_criteria='or', 
                                version=2, clustering=False):
    '''Función que permite obtener la descomposición NMF de una señal (ingresando su
    ubicación en el ordenador), la cual solamente descompone en segmentos de interés
    previamente etiquetados, aplicando una máscara y descomponiendo la señal completa.
    
    Parameters
    ----------
    dir_file : str
        Dirección del archivo de audio a segmentar.
    assign_method : {'auto_labeled', 'auto_centroid', 'manual'}, optional
        Método de separación de sonidos. Para 'auto_labeled'se utiliza una lista de etiquetas
        creadas manualmente. Para 'auto_centroid' se utiliza el criterio del centroide para 
        seleccionar las componentes. Para 'manual' se etiqueta segmento a segmento cada 
        componente, las cuales son guardadas en un archivo .txt. Por defecto es 'manual'.
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
    
    # Solo si es que hay que bajar puntos se baja, en caso contrario se mantiene
    if samplerate_des < samplerate:
        _, signal_to = downsampling_signal(signal_in, samplerate, 
                                           samplerate_des//2-100, 
                                           samplerate_des//2)
        sr_ratio = samplerate // samplerate_des
    else:
        signal_to = signal_in
        samplerate_des = samplerate
        sr_ratio = 1
    
    # Definición de la carpeta donde se ubica
    filepath = '/'.join(dir_file.split('/')[:-1])
    
    # Definición del nombre del archivo
    filename = dir_file.strip('.wav').split('/')[-1]
    
    # Definición de la carpeta a guardar los segmentos
    filepath_to_save = f'{filepath}/Components/Masking on segments'
    
    # Preguntar si es que la carpeta que almacenará las imágenes se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(filepath_to_save):
        os.makedirs(filepath_to_save)
    
    # Definición del diccionario que contiene los parámetros de simulación
    dict_simulation = {'n_components': n_components, 'N': N, 'clustering': clustering,
                       'N_lax': N_lax, 'N_fade': N_fade, 'noverlap': noverlap, 
                       'repeat': repeat, 'padding': padding, 'window': window, 
                       'filter_out': filter_out, 'init': init, 'solver': solver, 
                       'beta': beta, 'tol': tol, 
                       'max_iter': max_iter, 'alpha_nmf': alpha_nmf, 
                       'l1_ratio': l1_ratio, 'random_state': random_state,
                       'scale': scale, 'ausc_zone': ausc_zone, 
                       'fcut_spect_crit': fcut_spect_crit, 
                       'measure_spect_crit': measure_spect_crit, 
                       'i_selection': i_selection, 'f1_roll': f1_roll, 
                       'f2_roll': f2_roll, 'measure_temp_crit': measure_temp_crit, 
                       'H_binary': H_binary, 'reduce_to_H': reduce_to_H, 
                       'dec_criteria': dec_criteria}
    
    # Control de parámetros simulación (consultar repetir) y asignación de id's
    dict_simulation, _ = _manage_registerdata_nmf(dict_simulation, 
                                                  filepath_to_save,
                                                  in_func=True)
    
    # Definición del filepath a guardar para la simulación
    filepath_to_save_id = f'{filepath_to_save}/id {dict_simulation["id"]}'
    filepath_to_save_name = f'{filepath_to_save_id}/{filename}'
    
    # Preguntar si es que la carpeta que almacenará las imágenes se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(filepath_to_save_name):
        os.makedirs(filepath_to_save_name)
    
    # A partir del nombre del archivo es posible obtener también su lista de intervalos.
    ## Primero se obtiene el nombre del sonido cardíaco a revisar
    file_heart = filename.split(' ')[-1].strip('.wav')
    
    ## Luego se define la dirección del archivo de segmentos
    if version == 1:
        segment_folder = f'Database_manufacturing/db_heart/Manual combinations v{version}'
    else:
        segment_folder = f'Database_manufacturing/db_heart/Manual combinations v{version}/{ausc_zone}'
    
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
    
    # Definición de la señal a descomponer mediante NMF
    to_decompose = np.zeros(len(signal_to))
    # Definición de la señal respiratoria de salida
    resp_signal = np.copy(signal_to)
    # Definición de la señal cardíaca de salida
    heart_signal = np.zeros(len(signal_to))
    
    # Transformando la señal 
    for interval in interval_list:
        # Definición del límite inferior y superior
        lower = (interval[0] - N_lax) // sr_ratio
        upper = (interval[1] + N_lax) // sr_ratio
        
        # Agregando los valores de la señal a descomponer
        to_decompose_faded = fading_signal(signal_to[lower - N_fade:upper + N_fade],
                                           N_fade, beta=1, side='both')
        
        to_decompose[lower - N_fade:upper + N_fade] += to_decompose_faded
        resp_signal[lower:upper] -= signal_to[lower:upper]

    # Aplicando NMF 
    comps, _, _, W, H = nmf_decomposition(to_decompose, samplerate_des, 
                                          n_components=n_components, 
                                          N=N, noverlap=noverlap, padding=padding,
                                          window=window, whole=whole, 
                                          alpha_wiener=alpha_wiener,
                                          filter_out=filter_out, init=init, 
                                          solver=solver, beta=beta, tol=tol, 
                                          max_iter=max_iter, alpha_nmf=alpha_nmf, 
                                          l1_ratio=l1_ratio, random_state=random_state,
                                          W_0=W_0, H_0=H_0, scale=scale) #, plot_spectrogram=True)

    # Graficos
    if plot_segments or assign_method == 'manual':
        if n_components == 2:
            decision_in_plot = _plot_masked_nmf(signal_to, samplerate_des, comps, W, H, 
                                                filepath_to_save_name, assign_method)
    
    if n_components == 2 and not clustering:
        if assign_method == 'auto':
            # Se calculan los centroides
            comp1_centroid = centroide(W[:,0])
            comp2_centroid = centroide(W[:,1])
            
            # Definición de la decisión
            heart_decision = 0 if comp1_centroid >= comp2_centroid else 1
            
            # Definición del texto a guardar
            save_txt = ' centroid k2'

        elif assign_method == 'manual':
            # Se pregunta para la decisión del sonido cardíaco
            heart_decision = decision_in_plot
            
            # Definición del texto a guardar
            save_txt = ''
            
        elif assign_method == 'labeled':
            # Se obtiene la decisión etiquetada como sonido cardíaco
            with open(f'{filepath_to_save_name}/Heart decision.txt', 'r', 
                      encoding='utf8') as data:
                heart_decision = literal_eval(data.readline().strip())
        
        # Y se complementa para el sonido respiratorio
        resp_decision = 0 if heart_decision == 1 else 1
        
        # Para conectarlas adecuadamente a la señal de interés
        for num, interval in enumerate(interval_list, 1):
            # Definición del límite inferior y superior
            lower = (interval[0] - N_lax) // sr_ratio
            upper = (interval[1] + N_lax) // sr_ratio

            # Graficando los segmentos
            _plot_masked_segments_nmf(signal_to, comps, heart_decision, resp_decision, 
                                      lower, upper, N_fade, num, filepath_to_save_name)
            
            # Definición de la lista de señales a concatenar con fading para el corazón
            heart_connect = (heart_signal[:lower], 
                             comps[heart_decision][lower-N_fade:upper+N_fade], 
                             heart_signal[upper:])
            # Definición de la lista de señales a concatenar con fading para la respiración
            resp_connect = (resp_signal[:lower], 
                            comps[resp_decision][lower - N_fade:upper + N_fade], 
                            resp_signal[upper:])

            #plt.plot(range(0,lower), resp_signal[:lower], color='C0', zorder=2)
            #plt.plot(range(lower - N_fade,upper + N_fade), 
            #comps[resp_decision][lower - N_fade:upper + N_fade], color='C1', zorder=2)
            #plt.plot(range(upper, len(resp_signal)), resp_signal[upper:], color='C2', zorder=2)
            #plt.plot(resp_signal, linewidth=5, color='C3', zorder=1)
            
            # Aplicando fading para cada uno
            heart_signal = fade_connect_signals(heart_connect, N=N_fade, beta=1)
            resp_signal = fade_connect_signals(resp_connect, N=N_fade, beta=1)
        
        # Guardando
        if assign_method in ['auto', 'manual']:
            # Registro del archivo de la simulación
            with open(f'{filepath_to_save_name}/Heart decision{save_txt}.txt', 
                    'w', encoding='utf8') as data:
                data.write(str(heart_decision))
    
    elif n_components > 2 or clustering:
        if assign_method == 'auto':
            # CRITERIO 1: Definición de la ubicación del diccionario
            filepath_dict = f'Heart component dictionaries/{scale} scale decomposition'
            
            # Definición de la carpeta a revisar
            folder = f'{dir_file.split("/")[0]}'
            
            # Definición del nombre del archivo de sonidos cardiacos
            heart_segments = f'{dir_file.split("/")[-1].strip(".wav").split(" ")[-1]}'
            
            # Definición de la ubicación de los segmentos de sonido cardiaco de interés
            if version == 1:
                filename_heart_segments = f'{folder}/db_heart/Manual combinations v{version}/'\
                                          f'{heart_segments} - segments.txt'
            else:
                filename_heart_segments = f'{folder}/db_heart/Manual combinations v{version}/'\
                                          f'{ausc_zone}/{heart_segments} - segments.txt'
            
            # Aplicación de los criterios
            ## CRITERIO 1: Correlación espectral
            a1_bool, a1 = spectral_correlation_test(W, samplerate_des, fcut_spect_crit, 
                                                    N=N, noverlap=noverlap, 
                                                    n_comps_dict=4, beta=beta, 
                                                    filepath_data=filepath_dict, 
                                                    padding=padding, repeat=repeat, 
                                                    measure=measure_spect_crit, 
                                                    i_selection=i_selection, threshold='mean')
            
            ## CRITERIO 2: Centroide espectral
            a2_bool, a2 = centroid_test(W, limit='upper', gamma=1)
            
            ## CRITERIO 3: Correlación temporal
            a3_bool, a3 = temporal_correlation_test(H, filename_heart_segments, samplerate, 
                                                    samplerate_des, N=N, N_audio=len(signal_in), 
                                                    noverlap=noverlap, threshold='mean', 
                                                    measure=measure_temp_crit, 
                                                    H_binary=H_binary, reduce_to_H=reduce_to_H,
                                                    version=version, ausc_zone=ausc_zone)
            
            # Decisión final
            if dec_criteria == 'or':
                heart_dec = a1_bool | a2_bool | a3_bool
            elif dec_criteria == 'vote':
                heart_dec = (a1_bool.astype(int) + a2_bool.astype(int) + 
                             a3_bool.astype(int)) >= 2
            elif dec_criteria == 'and':
                heart_dec = a1_bool & a2_bool & a3_bool
            
            # Definición de las señales a grabar
            heart_comps = np.zeros(len(comps[0]))
            resp_comps = np.zeros(len(comps[0]))
            
            # Finalmente, grabando los archivos
            for num_comp, dec in enumerate(heart_dec):
                # Grabando cada componente
                if dec:
                    heart_comps += comps[num_comp]
                else:
                    resp_comps += comps[num_comp]
            
            # Definición del texto a guardar
            save_txt = ' clustering'
            
            # Para conectarlas adecuadamente a la señal de interés
            for num_int, interval in enumerate(interval_list, 1):
                # Definición del límite inferior y superior
                lower = (interval[0] - N_lax) // sr_ratio
                upper = (interval[1] + N_lax) // sr_ratio

                # Graficando los segmentos
                _plot_masked_segments_nmf_kmore(signal_to, heart_comps, resp_comps, lower, 
                                                upper, N_fade, num_int, filepath_to_save_name)
                
                # Definición de la lista de señales a concatenar con fading para el corazón
                heart_connect = (heart_signal[:lower], 
                                 heart_comps[lower - N_fade:upper + N_fade], 
                                 heart_signal[upper:])
                # Definición de la lista de señales a concatenar con fading para la respiración
                resp_connect = (resp_signal[:lower], 
                                resp_comps[lower - N_fade:upper + N_fade], 
                                resp_signal[upper:])

                # plt.plot(range(0,lower), resp_signal[:lower], color='C0', zorder=2)
                # plt.plot(range(lower - N_fade,upper + N_fade), 
                # comps[resp_decision][lower - N_fade:upper + N_fade], color='C1', zorder=2)
                # plt.plot(range(upper, len(resp_signal)), resp_signal[upper:], color='C2', zorder=2)
                # plt.plot(resp_signal, linewidth=5, color='C3', zorder=1)
                
                # Aplicando fading para cada uno
                heart_signal = fade_connect_signals(heart_connect, N=N_fade, beta=1)
                resp_signal = fade_connect_signals(resp_connect, N=N_fade, beta=1)
            
            # Graficar
            _plot_clustering_points(filepath_to_save_name, a1, a2, a3, 
                                    measure_temp_crit)
        
        elif assign_method == 'labeled':
            pass
    
    # Finalmente, grabando los archivos de audio
    sf.write(f'{filepath_to_save_name}/Respiratory signal{save_txt}.wav', 
                resp_signal, samplerate_des)
    sf.write(f'{filepath_to_save_name}/Heart signal{save_txt}.wav', 
                heart_signal, samplerate_des)
    
    return resp_signal, heart_signal


def comparison_components_nmf_ground_truth(filepath, sep_type='to all', plot_signals=True,
                                           plot_show=False, id_rev=1):
    '''Rutina que permite realizar la comparación entre las señales obtenidas mediante
    distintas variantes de descomposición mediante NMF con sus respectivas señales 
    originales.
    
    Parameters
    ----------
    filepath : str
        Dirección dónde se encuentran los archivos de audio a procesar.
    id_rev : int
        En caso de seleccionar sep_type == "on segments", corresponde al identificador 
        de la base a usar.
    sep_type : {'to all', 'on segments', 'masked segments'}, optional
        Selección del tipo de la base generada por la separación mediante NMF. 'to all' 
        usa la base que separa toda la señal con NMF, 'on_segments' usa la base que 
        solamente separa con NMF en segmentos de interés y 'masked segments' usa la base 
        que es separada luego de aplicar una máscara que hace que solo quede lo que debiera 
        ser el corazón. Por defecto es 'to all'.
    plot_signals : bool, optional
        Decidir si se realizan gráficos.
    plot_show : bool, optional
        Decidir si se muestran los gráficos a medida que salen (debe cumplirse que 
        plot_signals == True). Por defecto es False.
    '''
    # Lista de los archivos de la carpeta
    filenames = [i for i in os.listdir(filepath) if i.endswith('.wav')]
    
    # Definición de la carpeta de componentes
    if sep_type == 'to all':
        sep = 'Separation to all'
    elif sep_type == 'on segments':
        sep = 'Separation on segments'
    elif sep_type == 'masked segments':
        sep = 'Masking on segments'
    else:
        raise Exception('Opción no válida para tipo de separación.')
    
    # Definición de la dirección a revisar para obtener las componentes
    filepath_comps = f'{filepath}/Components/{sep}'
    filepath_comps_id = f'{filepath_comps}/id {id_rev}'
    
    # Definición de carpeta donde se encuentran los archivos de sonido cardíaco
    heart_dir = 'Database_manufacturing/db_heart/Manual combinations'
    
    # Definición de carpeta donde se encuentran los archivos de sonido cardíaco
    resp_dir = 'Database_manufacturing/db_respiratory/Adapted'
    
    # Obteniendo la información de la simulación a partir del identificador id
    with open(f'{filepath_comps}/Simulation register.txt', 'r', encoding='utf8') as data:
        for line in data:
            data_dict = literal_eval(line.strip())
            
            if data_dict['id'] == id_rev:
                data_table = data_dict
    
    # Comprobando si es que el registro de la simulación existe
    try:
        keys = list(data_table.keys())
        values = list(data_table.values())
        
        table_info = [[keys[i], values[i]]for i in range(len(keys))]
    except:
        raise Exception('Registro de la simulación no existe.')
    
    # Definición de los arreglos de error
    error_heart_list = list()
    error_resp_list = list()
    error_signal_list = list()
    
    for audio_name in tqdm(filenames, desc='Comp. analysis', ncols=70):
        # Separando el nombre para lo que corresponda según la codificación utilizada
        _, resp_name, heart_name = audio_name.strip('.wav').split(' ')
        
        # Obtención del sonido respiratorio
        audio_resp, _ = sf.read(f'{resp_dir}/{resp_name}.wav')
        
        # Obtención del sonido cardíaco
        audio_heart, _ = sf.read(f'{heart_dir}/{heart_name}.wav')
        
        # Obtención de las componentes
        if sep_type == 'to all':
            comp_1, _ = sf.read(f'{filepath_comps_id}/{audio_name.strip(".wav")} Comp 1.wav')
            comp_2, _ = sf.read(f'{filepath_comps_id}/{audio_name.strip(".wav")} Comp 2.wav')

            # Definición Heart comp
            heart_index = np.argmin((sum(abs(audio_heart - comp_1)),
                                    sum(abs(audio_heart - comp_2))))

            heart_comp = comp_1 if heart_index == 0 else comp_2
        
            # Definición de señal respiratoria como el complemento del resultado anterior
            resp_comp = comp_2 if heart_index == 0 else comp_1
            
        elif sep_type in ['on segments', 'masked segments']:
            heart_comp, _ = sf.read(f'{filepath_comps_id}/{audio_name.strip(".wav")}/'
                                    f'Heart signal.wav')
            resp_comp, _ = sf.read(f'{filepath_comps_id}/{audio_name.strip(".wav")}/'
                                   f'Respiratory signal.wav')
        
        # Normalización de todas las señales
        audio_resp_norm = audio_resp / max(abs(audio_resp))
        audio_heart_norm = audio_heart / max(abs(audio_heart))
        resp_comp_norm = resp_comp / max(abs(resp_comp))
        heart_comp_norm = heart_comp / max(abs(heart_comp))
        
        # Definición de los errores
        error_resp = abs(audio_resp_norm - resp_comp_norm)
        error_heart = abs(audio_heart_norm - heart_comp_norm)
        error_signal = abs(original_signal_norm - sum_comp_signal_norm)
        
        if plot_signals:
            # Definición del directorio a guardar las imágenes
            savefig = f'{filepath_comps_id}/{audio_name.strip(".wav")}'
            
            # Ploteando las diferencias
            fig, ax = plt.subplots(2, 1, figsize=(15,7))
            
            ax[0].plot(audio_resp_norm, label='Original', linewidth=4)
            ax[0].plot(resp_comp_norm, label='Componente resp', linewidth=1)
            ax[0].set_ylabel('Signals')
            ax[0].legend(loc='upper right')
            
            ax[1].plot(error_resp)
            ax[1].set_ylabel('Error')
            
            # Se reajusta los plots
            fig.subplots_adjust(right=0.82)
            # Finalmente se añaden tablas resumen
            table_pos = plt.table(cellText=table_info,
                                  rowLoc='center', colLoc='center',
                                  colColours=['yellowgreen', 'yellowgreen'],
                                  colLabels=['Parameter', 'Value'],
                                  bbox=[1.01, 0.3, 0.17, 1.9],)
            
            table_error = plt.table(cellText=[[round(np.sum(error_resp), 2), 
                                               round(max(error_resp), 5)]],
                                   rowLoc='center', colLoc='center',
                                   colColours=['cyan', 'magenta'],
                                   colLabels=['Sum error', 'Max error'],
                                   bbox=[1.01, 0, 0.17, 0.28],)
            
            # Se setean sus fuentes
            table_pos.set_fontsize(7)
            table_error.set_fontsize(7)
                        
            fig.suptitle(f'{audio_name} Respiratory Normalized')
            fig.savefig(f'{savefig}/Resp.png')
            
            if plot_show:
                # Manager para modificar la figura actual y maximizarla
                manager = plt.get_current_fig_manager()
                manager.window.state('zoomed')
                
                plt.show()
            
            plt.close()
            
            fig, ax = plt.subplots(2, 1, figsize=(15,7))
            
            ax[0].plot(audio_heart_norm, label='Original', linewidth=4)
            ax[0].plot(heart_comp_norm, label='Componente heart', linewidth=1)
            ax[0].set_ylabel('Signals')
            ax[0].legend(loc='upper right')
            
            ax[1].plot(error_heart)
            ax[1].set_ylabel('Error')
            
            # Se reajusta los plots
            fig.subplots_adjust(right=0.82)
            # Finalmente se añaden tablas resumen
            table_pos = plt.table(cellText=table_info,
                                  rowLoc='center', colLoc='center',
                                  colColours=['yellowgreen', 'yellowgreen'],
                                  colLabels=['Parameter', 'Value'],
                                  bbox=[1.01, 0.3, 0.17, 1.9],)
            
            table_error = plt.table(cellText=[[round(np.sum(error_heart), 2), 
                                               round(max(error_heart), 5)]],
                                   rowLoc='center', colLoc='center',
                                   colColours=['cyan', 'magenta'],
                                   colLabels=['Sum error', 'Max error'],
                                   bbox=[1.01, 0, 0.17, 0.28],)
            
            # Se setean sus fuentes
            table_pos.set_fontsize(7)
            table_error.set_fontsize(7)
            
            fig.suptitle(f'{audio_name} Heart Normalized')
            fig.savefig(f'{savefig}/Heart.png')
            
            if plot_show:
                # Manager para modificar la figura actual y maximizarla
                manager = plt.get_current_fig_manager()
                manager.window.state('zoomed')
                
                plt.show()
            
            plt.close()
            
            # Ploteando las diferencias
            fig, ax = plt.subplots(2, 1, figsize=(15,7))
            
            # Definición de los coeficientes a ponderar para esta base
            name_info = filepath.split('/')[-1].split(' ')
            coef_heart = int(name_info[4].split('_')[0])
            coef_resp = int(name_info[5].split('_')[0])
            
            # Señales a comparar
            original_signal = coef_resp * audio_resp + coef_heart * audio_heart
            sum_comp_signal = coef_resp * resp_comp + coef_heart * heart_comp
            
            # Y re normalizando
            original_signal_norm = original_signal/max(abs(original_signal))
            sum_comp_signal_norm = sum_comp_signal/max(abs(sum_comp_signal))
            
            ax[0].plot(original_signal_norm, label='Original', linewidth=3)
            ax[0].plot(sum_comp_signal_norm, label='Suma comps', linewidth=1)
            ax[0].set_ylabel('Signals')
            ax[0].legend(loc='upper right')
            
            ax[1].plot(error_signal)
            ax[1].set_ylabel('Error')
            
            # Se reajusta los plots
            fig.subplots_adjust(right=0.82)
            # Finalmente se añaden tablas resumen
            table_pos = plt.table(cellText=table_info,
                                  rowLoc='center', colLoc='center',
                                  colColours=['yellowgreen', 'yellowgreen'],
                                  colLabels=['Parameter', 'Value'],
                                  bbox=[1.01, 0.3, 0.17, 1.9],)
            
            table_error = plt.table(cellText=[[round(np.sum(error_signal), 2), 
                                               round(max(error_signal), 5)]],
                                   rowLoc='center', colLoc='center',
                                   colColours=['cyan', 'magenta'],
                                   colLabels=['Sum error', 'Max error'],
                                   bbox=[1.01, 0, 0.17, 0.28],)
            
            # Se setean sus fuentes
            table_pos.set_fontsize(7)
            table_error.set_fontsize(7)
            
            fig.suptitle(f'{audio_name} Sum comparation')
            fig.savefig(f'{savefig}/Sum comps.png')
            
            if plot_show:
                # Manager para modificar la figura actual y maximizarla
                manager = plt.get_current_fig_manager()
                manager.window.state('zoomed')
                
                plt.show()
            
            plt.close()
            
        # Finalmente, se agregan todos los errores a las listas correspondientes
        error_heart_list.append((np.sum(error_heart), max(error_heart)))
        error_resp_list.append((np.sum(error_resp), max(error_resp)))
        error_signal_list.append((np.sum(error_signal), max(error_signal)))
    
    return error_heart_list, error_resp_list, error_signal_list


def check_auto_criteria(filepath, sep_type, id_rev, criteria='centroid'):
    '''Función que permite registrar el accuracy del criterio de selección entre las 2 
    variables obtenidas a partir de NMF (en donde la selección que se compara es la
    selección del sonido cardíaco)
    
    Parameters
    ----------
    filepath : str
        Dirección dónde se encuentran los archivos de audio a procesar.
    sep_type : {'to all', 'on segments', 'masked segments'}, optional
        Selección del tipo de la base generada por la separación mediante NMF. 'to all' 
        usa la base que separa toda la señal con NMF, 'on_segments' usa la base que 
        solamente separa con NMF en segmentos de interés y 'masked segments' usa la base 
        que es separada luego de aplicar una máscara que hace que solo quede lo que debiera 
        ser el corazón. Por defecto es 'to all'.
    id_rev : int
        En caso de seleccionar sep_type == "on segments", corresponde al identificador 
        de la base a usar.
    criteria : {'centroid'}, optional
        Criterio de selección entre 
    '''
    # Definición de la carpeta de separación a revisar
    if sep_type == 'to all':
        sep = 'Separation to all'
    elif sep_type == 'on segments':
        sep = 'Separation on segments'
    elif sep_type == 'masked segments':
        sep = 'Masking on segments'
    else:
        raise Exception('Opción de separación inválida.')
    
    # Definición del directorio a revisar
    filepath_id = f'{filepath}/Components/{sep}/id {id_rev}'
    
    # Definición de la variable que cuenta los aciertos
    ok_detections = 0
    
    for folder in os.listdir(filepath_id):
        # Definición del directorio donde se encuentran los archivos
        filepath_txt = f'{filepath_id}/{folder}'
        
        # Lectura de los registros. En primer lugar se obtiene el etiquetado manualmente
        with open(f'{filepath_txt}/Heart decision.txt', 'r', encoding='utf8') as data:
            manual_label = data.readline()
        
        # Y luego el etiquetado mediante el algoritmo
        with open(f'{filepath_txt}/Heart decision {criteria}.txt', 'r', encoding='utf8') as data:
            auto_label = data.readline()
        
        if auto_label == manual_label:
            ok_detections += 1
    
    # Obtención del accuracy
    accuracy = ok_detections / len(os.listdir(filepath_id))
    
    try:
        # Rutina para asegurarse de no re escribir una simulación ya realizada
        with open(f'{filepath}/Components/{sep}/{criteria} test.txt', 'r+', 
                  encoding='utf8') as data:
            # Se lee el archivo
            info = data.readlines()
            # Y se ordena
            info.sort()
            
            # Se guarda la información del checkeo
            info.append('id {:d} = {:.2f} %'.format(id_rev, accuracy * 100))
            
            # Guardando
            data.writelines(info)
    except:
        with open(f'{filepath}/Components/{sep}/{criteria} test.txt', 'w', 
                  encoding='utf8') as data:
            data.write('id {:d} = {:.2f} %'.format(id_rev, accuracy * 100))


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


def _plot_nmf(signal_in, samplerate, comps, W, H, filepath_to_save, assign_method):
    # Definición de la clase que usará el bóton para regular las funciones
    class Boton_seleccion:
        ''' Clase que permitirá controlar la variable de interés y los métodos
        para los botones implementados.
        
        Aclaración: La correspondencia es la siguiente.
        [None, 1, 2, 3, 'up', 'down']
        {None, MouseButton.LEFT, MouseButton.MIDDLE, MouseButton.RIGHT, 'up', 'down'}
        '''
        def __init__(self):
            self.value = None
        
        def componente_1(self, event):
            if event.button == 1:
                self.value = 0
                plt.close()
        
        def componente_2(self, event):
            if event.button == 1:
                self.value = 1
                plt.close()
    
    # Definición del backend para maximizar la ventana. Dependiendo del SO
    # puede variar. Para revisar usar comando matplotlib.get_backend()
    plt.switch_backend('TkAgg')
    
    # Creación del plot
    fig, ax = plt.subplots(2, 2, figsize=(17,10))
    
    # Plots
    ax[0][0].plot(signal_in, label='Original')
    ax[0][0].legend(loc='upper right')
    ax[0][0].set_title('Original signal')
    ax[0][0].set_xlabel('Samples')
    
    ax[1][0].plot(comps[0], label='Comp 1', color='C0')
    ax[1][0].plot(comps[1], label='Comp 2', color='C1')
    ax[1][0].legend(loc='upper right')
    ax[1][0].set_title('NMF components')
    ax[1][0].set_xlabel('Samples')
    
    f = np.linspace(0, samplerate // 2, W.shape[0])
    ax[0][1].plot(f, W[:,0], label='Comp 1', color='C0')
    ax[0][1].plot(f, W[:,1], label='Comp 2', color='C1')
    ax[0][1].legend(loc='upper right')
    ax[0][1].set_xlim([0,1000])
    ax[0][1].set_title('Matrix W')
    ax[0][1].set_xlabel('Frequency [Hz]')
    
    t = np.linspace(0, len(signal_in), H.shape[1])
    ax[1][1].plot(t, H[0], label='Comp 1', color='C0')
    ax[1][1].plot(t, H[1], label='Comp 2', color='C1')
    ax[1][1].legend(loc='upper right')
    ax[1][1].set_title('Matrix H')
    ax[1][1].set_xlabel('Samples')
    
    # Definición del título
    # fig.suptitle(f'NMF decomposition')
    
    # Se guarda la figura
    fig.savefig(f'{filepath_to_save}/Signal plot.png') 
    
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
            return _plot_nmf(signal_in, samplerate, comps, W, H, 
                             filepath_to_save, assign_method)
        else:
            return to_return 


def _plot_segments_nmf(signal_in, samplerate, comps, W, H, N_fade, lower, upper, 
                       num, filepath_to_save, assign_method):
    '''Función auxiliar que permite realizar los gráficos en la función
    "nmf_applied_interest_segments".
    '''
    # Definición de la clase que usará el bóton para regular las funciones
    class Boton_seleccion:
        ''' Clase que permitirá controlar la variable de interés y los métodos
        para los botones implementados.
        
        Aclaración: La correspondencia es la siguiente.
        [None, 1, 2, 3, 'up', 'down']
        {None, MouseButton.LEFT, MouseButton.MIDDLE, MouseButton.RIGHT, 'up', 'down'}
        '''
        def __init__(self):
            self.value = None
        
        def componente_1(self, event):
            if event.button == 1:
                self.value = 0
                plt.close()
        
        def componente_2(self, event):
            if event.button == 1:
                self.value = 1
                plt.close()
    
    # Definición del backend para maximizar la ventana. Dependiendo del SO
    # puede variar. Para revisar usar comando matplotlib.get_backend()
    plt.switch_backend('TkAgg')
    
    # Creación del plot
    fig, ax = plt.subplots(1, 3, figsize=(17,7))
    
    # Plots
    t = np.linspace(lower - N_fade, upper + N_fade, len(comps[0]))
    ax[0].plot(t, signal_in[lower-N_fade:upper+N_fade], label='Original', color='C2')
    ax[0].plot(t, comps[0], label='Comp 1', color='C0')
    ax[0].plot(t, comps[1], label='Comp 2', color='C1')
    ax[0].set_xlabel('Samples')
    ax[0].legend(loc='upper right')
    ax[0].set_title('Signals')
    ax[0].tick_params(axis='x', labelrotation=45)
    
    f = np.linspace(0, samplerate // 2, W.shape[0])
    ax[1].plot(f, W[:,0], label='Comp 1', color='C0')
    ax[1].plot(f, W[:,1], label='Comp 2', color='C1')
    ax[1].legend(loc='upper right')
    ax[1].set_xlim([0,1000])
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_title('Matrix W')
    
    t = np.linspace(lower - N_fade, upper + N_fade, H.shape[1])
    ax[2].plot(t, H[0], label='Comp 1', color='C0')
    ax[2].plot(t, H[1], label='Comp 2', color='C1')
    ax[2].set_xlabel('Samples')
    ax[2].legend(loc='upper right')
    ax[2].set_title('Matrix H')
    ax[2].tick_params(axis='x', labelrotation=45)
    
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
            return _plot_segments_nmf(signal_in, samplerate, comps, W, H, 
                                      N_fade, lower, upper, 
                                      num, filepath_to_save, assign_method)
        else:
            return to_return 


def _plot_segments_nmf_kmore(signal_in, heart_comps, resp_comps, N_fade, 
                             lower, upper, num, filepath_to_save):
    '''Función auxiliar que permite realizar los gráficos en la función
    "nmf_applied_interest_segments".
    '''
    # Creación de la figura
    plt.figure(figsize=(17,9))
    
     # Graficando cada segmento de la señal
    plt.plot(range(lower - N_fade, upper + N_fade), 
             signal_in[lower - N_fade:upper + N_fade],
             linewidth=3, zorder=1, color='C2', label='Original')
    plt.plot(range(lower - N_fade, upper + N_fade), heart_comps, 
             color='C0', zorder=2, label='Heart')
    plt.plot(range(lower - N_fade,upper + N_fade), resp_comps, 
             color='C1', zorder=2, label='Respiration')
    
    # Labels y títulos
    plt.xlabel('Samples')
    plt.ylabel('Signals')
    plt.legend(loc='upper right')
    plt.suptitle(f'Segment #{num}')
    plt.savefig(f'{filepath_to_save}/Segment #{num} clustering.png')
    plt.close()


def _plot_masked_nmf(signal_in, samplerate, comps, W, H, filepath_to_save, assign_method):
    '''Función auxiliar que permite realizar los gráficos en la función
    "nmf_applied_masked_segments".
    '''
    # Definición de la clase que usará el bóton para regular las funciones
    class Boton_seleccion:
        ''' Clase que permitirá controlar la variable de interés y los métodos
        para los botones implementados.
        
        Aclaración: La correspondencia es la siguiente.
        [None, 1, 2, 3, 'up', 'down']
        {None, MouseButton.LEFT, MouseButton.MIDDLE, MouseButton.RIGHT, 'up', 'down'}
        '''
        def __init__(self):
            self.value = None
        
        def componente_1(self, event):
            if event.button == 1:
                self.value = 0
                plt.close()
        
        def componente_2(self, event):
            if event.button == 1:
                self.value = 1
                plt.close()
    
    # Definición del backend para maximizar la ventana. Dependiendo del SO
    # puede variar. Para revisar usar comando matplotlib.get_backend()
    plt.switch_backend('TkAgg')
    
    # Creación del plot
    fig, ax = plt.subplots(2, 2, figsize=(17,10))
    
    # Plots
    ax[0][0].plot(signal_in, label='Original')
    ax[0][0].legend(loc='upper right')
    ax[0][0].set_title('Original signal')
    ax[0][0].set_xlabel('Samples')
    
    ax[1][0].plot(comps[0], label='Comp 1', color='C0')
    ax[1][0].plot(comps[1], label='Comp 2', color='C1')
    ax[1][0].legend(loc='upper right')
    ax[1][0].set_title('NMF components')
    ax[1][0].set_xlabel('Samples')
    
    f = np.linspace(0, samplerate // 2, W.shape[0])
    ax[0][1].plot(f, W[:,0], label='Comp 1', color='C0')
    ax[0][1].plot(f, W[:,1], label='Comp 2', color='C1')
    ax[0][1].legend(loc='upper right')
    ax[0][1].set_xlim([0,1000])
    ax[0][1].set_title('Matrix W')
    ax[0][1].set_xlabel('Frequency [Hz]')
    
    t = np.linspace(0, len(signal_in), H.shape[1])
    ax[1][1].plot(t, H[0], label='Comp 1', color='C0')
    ax[1][1].plot(t, H[1], label='Comp 2', color='C1')
    ax[1][1].legend(loc='upper right')
    ax[1][1].set_title('Matrix H')
    ax[1][1].set_xlabel('Samples')
    
    # Definición del título
    # fig.suptitle(f'Signal plots')
    
    # Se guarda la figura
    fig.savefig(f'{filepath_to_save}/Signal plot.png') 
    
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
            return _plot_masked_nmf(signal_in, samplerate, comps, W, H, 
                                    filepath_to_save, assign_method)
        else:
            return to_return 


def _plot_masked_segments_nmf(signal_in, comps, heart_decision, resp_decision, 
                              lower, upper, N_fade, num, filepath_to_save):
    '''Función auxiliar que permite guardar las imágenes de cada segmento en la 
    función "nmf_applied_masked_segments".
    '''
    # Creación de la figura
    plt.figure(figsize=(17,9))
    
     # Graficando cada segmento de la señal
    plt.plot(range(lower - N_fade, upper + N_fade), 
             signal_in[lower - N_fade:upper + N_fade],
             linewidth=3, zorder=1, color='C2', label='Original')
    plt.plot(range(lower - N_fade, upper + N_fade), 
             comps[heart_decision][lower - N_fade:upper + N_fade], 
             color='C0', zorder=2, label='Heart')
    plt.plot(range(lower - N_fade,upper + N_fade), 
             comps[resp_decision][lower - N_fade:upper + N_fade], 
             color='C1', zorder=2, label='Respiration')
    
    # Labels y títulos
    plt.xlabel('Muestras')
    plt.ylabel('Señales')
    plt.legend(loc='upper right')
    plt.suptitle(f'Segment #{num}')
    plt.savefig(f'{filepath_to_save}/Segment #{num}.png')
    plt.close()


def _plot_masked_segments_nmf_kmore(signal_in, heart_comps, resp_comps, lower, upper, 
                                    N_fade, num, filepath_to_save):
    '''Función auxiliar que permite guardar las imágenes de cada segmento en la 
    función "nmf_applied_masked_segments" cuando k >= 2.
    '''
    # Creación de la figura
    plt.figure(figsize=(17,9))
    
     # Graficando cada segmento de la señal
    plt.plot(range(lower - N_fade, upper + N_fade), 
             signal_in[lower - N_fade:upper + N_fade],
             linewidth=3, zorder=1, color='C2', label='Original')
    plt.plot(range(lower - N_fade, upper + N_fade), 
             heart_comps[lower - N_fade:upper + N_fade], 
             color='C0', zorder=2, label='Heart')
    plt.plot(range(lower - N_fade,upper + N_fade), 
             resp_comps[lower - N_fade:upper + N_fade], 
             color='C1', zorder=2, label='Respiration')
    
    # Labels y títulos
    plt.xlabel('Samples')
    plt.ylabel('Signals')
    plt.legend(loc='upper right')
    plt.suptitle(f'Segment #{num}')
    plt.savefig(f'{filepath_to_save}/Segment #{num} clustering.png')
    plt.close()


def _plot_clustering_points(filepath, spec_coef, cent_coef, temp_coef, 
                            measure_temp_crit):
    # Propiedades de etiquetas
    if measure_temp_crit == 'q_equal':
        text_temp_crit = 'Equal Percentage (%)'
    elif measure_temp_crit == 'correlation':
        text_temp_crit = 'Correlation'
    
    plt.figure()
    plt.plot(spec_coef, 'bx')
    plt.axhline(np.mean(spec_coef), color='r')
    plt.xlabel('Component (i)')
    plt.ylabel('Correlation')
    plt.title('Spectral criteria')
    plt.savefig(f'{filepath}/Spectral criteria.png')
    plt.close()
    
    plt.figure()
    plt.plot(cent_coef, 'bx')
    plt.axhline(np.mean(cent_coef), color='r')
    plt.xlabel('Component (i)')
    plt.ylabel('Centroid')
    plt.title('Centroid criteria')
    plt.savefig(f'{filepath}/Centroid criteria.png')
    plt.close()
    
    plt.figure()
    plt.plot(temp_coef, 'bx')
    plt.axhline(np.mean(temp_coef), color='r')
    plt.xlabel('Component (i)')
    plt.ylabel(f'{text_temp_crit}')
    plt.title('Temporal criteria')
    plt.savefig(f'{filepath}/Temporal criteria.png')
    plt.close()


def _manage_registerdata_nmf(dict_simulation, filepath_to_save, in_func=False):
    '''Rutina que permite realizar la gestión del registro de las simulaciones 
    realizadas. Se realiza el guardado los datos, corroboración de simulaciones
    ya existentes (preguntando si es que se quieren realizar nuevamente).
    
    Se asigna una etiqueta a cada señal nueva contando la cantidad de líneas 
    utilizadas anteriormente. Es decir, el id corresponde al número de la línea
    en el archivo.
    
    Destacar que esta sub rutina solo sirve para la función separación total. Además,
    esta función es la que realiza el guardado en este registro.
    
    Parameters
    ----------
    dict_simulation : dict
        Diccionario de parámetros de la simulación.
    filepath_to_save : str
        Carpeta donde se revisa el registro de la simulación.
    in_func : bool, optional
        Usar en True solo para aplicar esta función cuando se está adentro de las 
        funciones "nmf_applied_interest_segments" y "nmf_applied_masked_segments".
        Por defecto es False.
    decided : bool, optional

    
    Returns
    -------
    dict_simulation : dict
        Diccionario actualizado con el id
    continue_dec : bool
        Indica si se sigue trabajando con la señal.
    '''
    def Mbox(title, text, style):
        '''Función que permite crear una ventana para realizar consultas'''
        return ctypes.windll.user32.MessageBoxW(0, text, title, style)
    
    
    def _continue_dec(dict_simulation):
        '''Función auxiliar para decidir continuar con la simulación'''
        # Definición de valores
        title = 'Advertencia (!)'
        text = f'La simulación que quieres hacer ya fue realizada (en id: '\
               f'{dict_simulation["id"]}), ¿deseas hacerla nuevamente?'

        # Preguntar si es que se quiere hacer la simulación de nuevo en caso de que 
        # ya exista
        continue_decision = Mbox(title, text, 1)
            
        if continue_decision == 1:
            return True
            
        elif continue_decision == 2:
            print('Función terminada.\n')
            return False
    
    
    # Definición del contador de líneas (que servirá como id de cada simulación)
    id_count = 1
    
    # Definición de un booleano para controlar si se asignó un id
    id_assigned = False
    
    # Se intenta porque puede ser que el archivo no esté creado
    try:
        # Se obtienen las componetes a partir de las etiquetas generadas
        with open(f'{filepath_to_save}/Simulation register.txt', 'r', encoding='utf8') as data:
            for line in data:
                # Obtención del diccionario con el contexto
                dict_line = literal_eval(line.strip())
                
                # Se obtienen los componentes tentativos
                possible_id = dict_line['id']
                
                # Y se eliminan los labels para luego realizar una comparación
                del dict_line['id']
                
                # Y se comparan los parámetros utilizados
                if dict_simulation == dict_line:
                    # Se agregan los valores al diccionario de la simulación
                    dict_id = {'id': possible_id}
                    
                    # Y se concatenan
                    dict_id.update(dict_simulation)
                    dict_simulation = dict_id
                    
                    # Dado que se asigna, se cambia el booleano
                    id_assigned = True
                    break
                
                # Se aumenta el contador por cada línea leída
                id_count += 1
    
    # En caso de que no exista
    except:
        # Definitivamente debe ingresarse manualmente
        print('Se creará el archivo de registro de simulaciones...\n')
    
    if not id_assigned:
        # Se agregan los valores al diccionario de la simulación
        dict_id = {'id': id_count}
        
        # Y se concatenan
        dict_id.update(dict_simulation)
        dict_simulation = dict_id
        
        # Se escribe de manera directa solo si es que no es la función masked (es decir,
        # la función nmf_applied_all)
        if not in_func:
            # Se escribe sobre el archivo solo si es que no ha sido asignado previanente
            with open(f'{filepath_to_save}/Simulation register.txt', 'a', encoding='utf8') as data:
                data.write(f'{dict_simulation}\n')
        
        return dict_simulation, True
        
    else:
        if in_func:
            # Si ya está decidido previamente (en get_HR_components), entonces simplemente 
            # se continúa
            continue_dec = True
        else:
            # Decisión para seguir con la simulación
            continue_dec =  _continue_dec(dict_simulation)
        
        if not continue_dec:
            return dict_simulation, False
        else:
            return dict_simulation, True


def _binary_masking(signal_in, W, H, S, k, N, noverlap, padding, repeat, window, whole,
                    same_outshape=True, get_inverse=True):
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
        
        # Agregando a la lista de componentes
        Y_list.append(Yi)
        
        if get_inverse:
            # Y posteriormente la transformada inversa
            yi = get_inverse_spectrogram(Yi * np.exp(1j * np.angle(S)), 
                                         N=N, noverlap=noverlap, window=window, 
                                         padding=padding, repeat=repeat, whole=whole)

            if same_outshape:
                yi = yi[:len(signal_in)]

            # Agregando a la lista de componentes
            components.append(np.real(yi))
        
        
    return components, Y_list


def _wiener_masking(signal_in, W, H, S, k, N, noverlap, padding, repeat, window, whole, 
                    alpha_wiener, same_outshape=True, get_inverse=True):
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
        
        # Agregando a la lista de componentes
        Y_list.append(Yi)
        
        if get_inverse:
            # Y posteriormente la transformada inversa
            yi = get_inverse_spectrogram(Yi, N=N, noverlap=noverlap, window=window, 
                                         padding=padding, repeat=repeat, whole=whole)

            if same_outshape:
                yi = yi[:len(signal_in)]

            # Agregando a la lista de componentes
            components.append(np.real(yi))
    
    return components, Y_list


def _no_masking(signal_in, W, H, S, k, N, noverlap, padding, repeat, window, whole, 
                scale, same_outshape=True, get_inverse=True):
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
        
        # Agregando a la lista de componentes
        Y_list.append(Yi)
        
        if get_inverse:
            # Y posteriormente la transformada inversa
            yi = get_inverse_spectrogram(Yi, N=N, noverlap=noverlap, window=window, 
                                         padding=padding, repeat=repeat, whole=whole)

            if same_outshape:
                yi = yi[:len(signal_in)]

            # Agregando a la lista de componentes
            components.append(np.real(yi))
            
    return components, Y_list


# Module testing
'''
filepath = 'Database_manufacturing/db_HR/Source Separation/Seed-0 - x - 1_Heart 1_Resp 0_White noise'
dir_file = f'{filepath}/HR 122_2b2_Al_mc_LittC2SE Seed[2732]_S1[59]_S2[60].wav'
filepath_to_save = f'{filepath}/Components/Separation to all'

filepath_to_save_id = f'{filepath_to_save}/id 1'
nmf_applied_all(dir_file, filepath_to_save_id, clustering='auto', n_components=2, N=2048, noverlap=1536, 
                    padding=0, window='hamming', whole=False, alpha_wiener=1, filter_out='wiener', 
                    init='random', solver='cd', beta=2, tol=1e-4, max_iter=200, 
                    alpha_nmf=0, l1_ratio=0, random_state=0, W_0=None, H_0=None, scale='abs')

#comparison_components_nmf_ground_truth(filepath, id_rev=1, sep_type='masked segments', 
#                                       plot_signals=True, plot_show=False)
'''

'''
check_auto_criteria(filepath, sep_type='masked segments', id_rev=4, 
                    criteria='centroid')


get_components_HR_sounds(filepath, sep_type='masked segments', assign_method='auto_centroid',  
                        n_components=2, N=512, N_lax=0, N_fade=500, 
                        noverlap=256, padding=0, window='hann', whole=False, 
                        alpha_wiener=1, wiener_filt=True, init='random', 
                        solver='cd', beta=2, tol=1e-4, max_iter=200, 
                        alpha_nmf=0, l1_ratio=0, random_state=0, 
                        W_0=None, H_0=None, plot_segments=False)

print()

dir_file = f'{filepath}/HR 122_2b2_Al_mc_LittC2SE Seed[2732]_S1[59]_S2[60].wav'
nmf_applied_masked_segments(dir_file, n_components=2, N=2048, N_lax=0, N_fade=500,
                            noverlap=1024, padding=0,
                            window='hann', whole=False, alpha_wiener=1, 
                            wiener_filt=True, init='random', solver='cd', beta=2,
                            tol=1e-4, max_iter=200, alpha_nmf=0, l1_ratio=0,
                            random_state=0, W_0=None, H_0=None, 
                            plot_segments=True)


nmf_applied_interest_segments(dir_file, assign_method='manual', n_components=2, 
                                N=2048, N_lax=0, N_fade=500, noverlap=1024, padding=0,
                                window='hamming', whole=False, alpha_wiener=1, 
                                wiener_filt=True, init='random', solver='cd', beta=2,
                                tol=1e-4, max_iter=200, alpha_nmf=0, l1_ratio=0,
                                random_state=0, W_0=None, H_0=None, 
                                plot_segments=False)



comparison_components_nmf_ground_truth(filepath, id_rev=1, sep_type='on segments', 
                                       plot_signals=True, plot_show=True)
'''