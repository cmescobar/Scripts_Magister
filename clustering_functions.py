import numpy as np
from math_functions import correlation, cosine_similarity


def spectral_correlation_criteria(W_i, W_dic, threshold):
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
    threshold : float, optional
        Valor del umbral de decisión de la para clasificar una componente como
        sonido cardíaco.
    
    Referencias
    -----------
    [1] Canadas-Quesada, et. al (2017). A non-negative matrix factorization 
        approach  based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier. Chapter 3.2.1.
    '''
    # Cantidad de comparaciones
    N = W_dic.shape[0]
    
    # Definición de la lista de valores a guardar
    cosine_similarities = list()
    
    # Similaridad coseno para cada caso
    for j in range(N):
        SC_ij = cosine_similarity(W_i, W_dic[j])
        cosine_similarities.append(SC_ij)
    
    # Selección del máximo
    S_i = max(cosine_similarities)
    
    return S_i >= threshold


def roll_off_criteria(X, f0, percentage=0.85):
    '''Función que retorna el valor de verdad de la pertenencia de una componente
    X_i al cluster de sonidos cardíacos, utilizando el criterio de la comparación 
    de energía bajo una frecuencia f0 con respecto a la energía total de sí misma. 
    Retorna True si es que corresponde a sonido cardíaco y False si es que no.
    
    Parameters
    ----------
    X : ndarray
        Espectrograma de la componente a clasificar.
    f0 : float
        Valor de la frecuencia de corte en bins.
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
    ro = np.sum(X[:f0,:])
    
    # Definición de la energía del espectrograma
    er = np.sum(X)
    
    # Finalmente, se retorna la cualidad de verdad del criterio
    return ro >= percentage * er


def temporal_correlation_criteria(H_i, P, threshold=0):
    '''Función que retorna el valor de verdad de la pertenencia de una componente
    X_i al cluster de sonidos cardíacos, utilizando el criterio de la correlación
    temporal entre la información de la matriz H de la componente i, y el heart 
    rate del sonido cardiorespiratorio original. Retorna True si es que corresponde 
    a sonido cardíaco y False si es que no. 
    
    H_i : ndarray
        Información temporal de la componente i a partir de la matriz H.
    P : ndarray
        Heart rate de la señal cardiorespiratoria.
    threshold : float, optional
        Valor del umbral de decisión de la para clasificar una componente como
        sonido cardíaco. Por defecto es 0.
    
    Referencias
    -----------
    [1] Canadas-Quesada, et. al (2017). A non-negative matrix factorization 
        approach  based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier. Chapter 3.2.3.
    '''
    # Obtener el promedio de la señal
    H_i_mean =  np.mean(H_i)
    
    # Preprocesando la matriz H_i
    H_binary = np.where(H_i >= H_i_mean, 1, 0)
    
    # Calculando la correlación
    TC_i = correlation(P, H_binary)
    
    return TC_i >= threshold
