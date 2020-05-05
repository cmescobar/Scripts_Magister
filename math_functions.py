import numpy as np
import matplotlib.pyplot as plt


def hamming_window(N):
    # Definición de la ventana hamming de modo que se pueda generar para un
    # largo de ventana definido
    return np.asarray([0.53836 - 0.46164*np.cos((2*np.pi*i)/N)
                       for i in range(int(N))])


def hann_window(N):
    # Definición de la ventana hamming de modo que se pueda generar para un
    # largo de ventana definido
    return np.asarray([0.5 - 0.5*np.cos((2*np.pi*i)/N)
                       for i in range(int(N))])


def recognize_peaks_by_derivates(x, signal, peak_type='min', tol_dx=0.01,
    tol_d2x=1e-2, lookup=1500, plot=False):
    # Se definen las derivadas 
    dx = np.gradient(signal, x)
    d2x = np.gradient(dx, x)
    
    # Buscando los puntos donde la derivada se vuelve cero
    der_vect_0 = [i for i in range(len(dx)) if abs(dx[i]) <= tol_dx]
    
    # Y definiendo si estos puntos corresponden a mínimos o máximos se realiza
    if peak_type == 'min':
        sel_indexes = [i for i in der_vect_0 if d2x[i] >= tol_d2x]
    elif peak_type == 'max':
        sel_indexes = [i for i in der_vect_0 if d2x[i] <= - tol_d2x]
    elif peak_type == 'all':
        sel_indexes = der_vect_0
    else:
        raise ValueError('La opcion de eleccion de peak utilizada no es valida.')
    
    # Seleccionando un punto característico de la región (ya que
    # muchos de los "puntos" aparecen agrupados en más puntos). En primer lugar,
    # se obtiene un vector de diferencias para conocer los puntos en los que se
    # pasa de un cluster a otro
    dif_indexes = [i + 1 for i in range(len(sel_indexes) - 1)
                   if sel_indexes[i + 1] - sel_indexes[i] > 1] + \
                  [len(sel_indexes) + 1]

    # Separando los clusters de puntos y encontrando el índice representativo de
    # cada uno
    begin = 0
    out_indexes = []
    for i in dif_indexes:
        # Definición del punto posible. Se hace round en caso de que sea un
        # decimal, e int para pasarlo si o si a un elemento tipo "int" para
        # indexar 
        possible_point = int(round(np.mean(sel_indexes[begin:i])))
        
        # Finalmente, se debe reconocer si este punto es realmente un mínimo o
        # un  máximo y no un punto de inflexión. Para ello se revisará en un
        # rango de 'lookup' alrededor de este punto. Definiendo los puntos a
        # revisar 
        look_before = signal[possible_point - lookup] \
            if possible_point - lookup >= 0 else signal[0]
        look_after  = signal[possible_point + lookup] \
            if possible_point + lookup <= len(signal) else signal[len(signal)-1]

        # Luego, realizando la comparación
        if peak_type == 'min':
            # Corroborando que alrededor de este punto se forma un "valle"
            if (look_after > signal[possible_point] and 
                look_before > signal[possible_point]):
                out_indexes.append(possible_point)

        elif peak_type == 'max':
            # Corroborando que alrededor de este punto se forma una "cueva"
            if (look_after < signal[possible_point] and 
                look_before < signal[possible_point]):
                out_indexes.append(possible_point)
        
        elif peak_type == 'all':
            # Corroborando alguno de los 2 casos anteriores
            if (look_after > signal[possible_point] and 
                look_before > signal[possible_point]) or \
               (look_after < signal[possible_point] and 
                look_before < signal[possible_point]):
                out_indexes.append(possible_point)

        # Redefiniendo el comienzo del análisis
        begin = i
    
    # Graficando para corroborar visualmente
    if plot:
        plt.subplot(3,1,1)
        plt.plot(signal)
        plt.plot(out_indexes, [signal[i] for i in out_indexes], 'rx')

        plt.subplot(3,1,2)
        plt.plot(dx)

        plt.subplot(3,1,3)
        plt.plot(d2x)

        plt.show()

    return out_indexes


def wiener_filter(V, WiHi, W, H, alpha=1):
    # El filtro de Wiener permite generar una máscara que obtenga información
    # del espectrograma original a partir de la proporción obtenida mediante la
    # multiplicación de las matrices W y H (estimación de la señal original)
    
    # Obteniendo la máscara
    mask = np.divide(WiHi ** alpha, np.matmul(W, H) + 1e-15)
    
    # Aplicando la máscara al espectrograma original, se obtiene el resultado
    # final del proceso de separación de fuentes
    return mask * V


def SNR(signal_in, signal_denoised, snr_type='db'):
    if snr_type == 'db':
        return 10 * np.log10(sum(signal_in ** 2) / 
                             sum((signal_in - signal_denoised) ** 2))


def moving_average(signal_in, Lf):
    # Definición de N
    N = len(signal_in)
    # Creación del vector del resultado
    result = np.zeros(N)
    
    # Se hace el promedio para cada segmento
    for n in range(N):
        if 0 <= n <= Lf - 1:
            result[n] = np.divide(sum(signal_in[:n+Lf+1]), Lf + n + 1)
        elif Lf <= n <= N - Lf - 1:
            result[n] = np.divide(sum(signal_in[n-Lf:n+Lf+1]), 2*Lf + 1)
        elif N - Lf <= n <= N - 1:
            result[n] = np.divide(sum(signal_in[n-Lf:N]), Lf + N - 1)
            
    return result


def raised_cosine_modified(N, beta):
    '''Creación de una ventana tipo pulso coseno elevado.
    
    Parameters
    ----------
    N : int
        Cantidad de puntos de la ventana.
    beta : float
        Parámetro de la función coseno elevado para la apertura de la ventana.
        
    Returns
    -------
    rc_out : ndarray
        Ventana pulso coseno elevado de N puntos con el valor de beta ingresado
    '''
    # Definición de la frecuencia f
    f = np.linspace(-1/2, 1/2, N)
    
    # Control de parámetro para beta
    if beta <= 0:
        beta = 0
    elif beta >= 1:
        beta = 1
    
    # Definición del vector de salida
    rc_out = np.array([])
    
    # Para punto f
    for i in f:
        if abs(i) <= (1 - beta)/2:
            rc_out = np.concatenate((rc_out, [1]))
        elif (1 - beta)/2 < abs(i) <= (1 + beta)/2:
            to_append =  np.cos(np.pi / beta * (abs(i) - (1 - beta)/2))
            rc_out = np.concatenate((rc_out, [to_append]))
        else:
            rc_out = np.concatenate((rc_out, [0]))
            
    return rc_out


def raised_cosine_fading(N, beta, side='right'):
    ''' Creacion de una ventana de desvanecimiento basada en coseno elevado.
    
    Parameters
    ----------
    N : int
        Cantidad de puntos de la ventana.
    beta : float
        Parámetro de la función coseno elevado para la apertura de la ventana.
    side : {'left', 'right'}, optional
        Dirección en la cual se puede usará la ventana. Se recomienda 'right' para
        el final de la señal y 'left' para el comienzo. Por defecto es 'right'.
    
    Returns
    -------
    vanish_window : ndarray
        Ventana de desvanecimiento de N puntos.
    '''    
    # Definición de la frecuencia f
    f = np.linspace(-1, 1, 2*N)
    
    # Control de parámetro para beta
    if beta <= 0:
        beta = 0
    elif beta >= 1:
        beta = 1
    
    # Definición del vector de salida
    rc_out = np.array([])
    
    # Para punto f
    for i in f:
        if abs(i) <= (1 - beta)/2:
            rc_out = np.concatenate((rc_out, [1]))
        elif (1 - beta)/2 < abs(i) <= (1 + beta)/2:
            to_append =  1/2 * (1 + np.cos(np.pi / beta * (abs(i) - (1 - beta)/2)))
            rc_out = np.concatenate((rc_out, [to_append]))
        else:
            rc_out = np.concatenate((rc_out, [0]))
    
    # Selección del lado
    if side == 'right':
        vanish_window = rc_out[N:]
    elif side == 'left':
        vanish_window = 1 - rc_out[N:]
    
    return vanish_window



def db_coef(db):
    '''Función que obitene el coeficiente por el cual se debe multiplicar un arreglo
    para obtener el valor de decibel deseado (relativo).

    Parameters
    ----------
    db : float
        Valor de dB deseado para realizar una transformación.

    Returns
    -------
    db_value : float
        Valor por el que se debe multiplicar un arreglo para obtener el decibel 
        deseado.
    '''
    return 10 ** (db/20)


def db_attenuation(signal_in, db):
    '''Función que permite atenuar una señal a partir de su valor en dB

    Parameters
    ----------
    signal_in : ndarray
        Señal a atenuar.
    dB : float
        Valor de atenuación en dB (positivo para atenuar).

    Returns
    -------
    signal_attenuated : ndarray
        Señal atenuada en db dB.
    '''
    return signal_in * db_coef(-db)