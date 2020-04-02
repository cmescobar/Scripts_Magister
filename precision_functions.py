import numpy as np


def find_closest_point(reference_list, compare_list, distance_limit=44100):
    '''Función que revisa punto a punto una lista de referencia para encontrar el
    punto más cercano en la lista a comparar
    
    Parameters
    ----------
    reference_list : list
        Lista de etiquetas manualmente realizadas sobre el elemento a revisar
    compare_list : list
        Lista de etiquetas detectadas mediante algún algoritmo implementado
    distance_limit : int, optional
        Umbral del muestras máximo a considerar para un acierto (Por defecto es 44100)
        
    Returns
    -------
    list
        Retorna una lista de información en orden:
        1. Lista de puntos en reference_list que fueron correctamente asignados (en orden)
        2. Lista de puntos en compare_list que son más cercanos a reference_list (en orden) 
    '''
    # Definición de la lista de puntos detectados de la lista de referencia
    detected_correspond = list()
    # Definición de la lista de referencia actualizada (en caso de que algunos no
    # entren debido a )
    referenced_correspond = list()

    # Para cada punto en la lista de referencia, se buscará el punto más cercano en
    # las detecciones, considerando como límite máximo de detección el parámetro 
    # "distance_limit"
    for n_point in reference_list:
        # Vector de diferencias entre el punto en particular y el la lista a comparar
        diference_vect = abs(np.array(compare_list) - n_point)
        
        # Corroborar el límite superior de cercanía
        if min(diference_vect) <= distance_limit:
            # Índice del punto más cercano (considerando el límite superior)
            index_closest_point = np.argmin(abs(np.array(compare_list) - n_point))

            # Valor del punto más cercano
            closest_point = compare_list[index_closest_point]

            # Agregando a la listas
            referenced_correspond.append(n_point)
            detected_correspond.append(closest_point)
            
    return referenced_correspond, detected_correspond


def get_unclasified_points(reference_list, compare_list, correspond_list):
    '''Función que permite entregar los puntos que no son adecuadamente clasificados
    
    Parameters
    ----------
    reference_list : list or array
        Lista de etiquetas manualmente realizadas sobre el elemento a revisar
    compare_list : list or array
        Lista de etiquetas detectadas mediante algún algoritmo implementado
    correspond_list : list or array
        Lista de tuplas/arreglos de etiquetas y puntos a comparar relacionados
    '''
    # Definición de la lista de puntos no asignados de la lista de referencia
    unclasified_references_point = list()
    
    # Definición de la lista de puntos no asignados de la lista de comparación
    unclasified_compare_point = list()
    
    for i in reference_list:
        if i not in correspond_list[:, 0]:
            unclasified_references_point.append(i)
    
    for i in compare_list:
        if i not in correspond_list[:, 1]:
            unclasified_compare_point.append(i)
            
    return unclasified_references_point, unclasified_compare_point
    

def get_precision_info(reference_list, compare_list, clean_repeated=True,
                       distance_limit=44100):
    '''Función que permite entregar información acerca de la precisión de una lista de
    puntos de detección con respecto a una lista de etiquetas realizadas manualmente.
    
    Parameters
    ----------
    reference_list : list
        Lista de etiquetas manualmente realizadas sobre el elemento a revisar
    compare_list : list
        Lista de etiquetas detectadas mediante algún algoritmo implementado
    clean_repeated : bool
        Booleano que indica si se hace una limpieza de puntos repetidos en la detección.
        Se recomienda hacer ya que en el proceso de asignación 1 a 1, hay puntos en la
        detección que corresponden a más de un punto en la lista de referencia.
    distance_limit : int, optional
        Umbral del muestras máximo a considerar para un acierto (Por defecto es 44100)
    
    Returns
    -------
    list
        Retorna una lista de información en orden:
        1. Accuracy (# detecciones correctas / # ptos. de referencia)
        2. Lista de puntos correspondientes 1 a 1
        3. Información de precisión (media, desv. estándar y rango)
        4. Cantidad de detecciones correctamente realizadas
        5. Puntos de la referencia que no se asocian a un punto de comparación
        6. Puntos de comparación que no se asocian a un punto ground truth
    '''
    # Se encuentran los puntos más cercanos en "compare list" correspondientes a la 
    # "reference_list"
    referenced_correspond, compare_correspond =\
        find_closest_point(reference_list, compare_list, distance_limit=distance_limit)
    
    # Rutina de eliminación de puntos repetidos
    if clean_repeated:
        # Primero, dejar todos los puntos correspondientes obtenidos solo una vez en 
        # la lista
        compare_unique = list(set(compare_correspond))
        # Ordenando
        compare_unique.sort()
        
        # Inversamente, se buscan los puntos más cercanos a la lista de detecciones 
        # únicas, la cual forzará la inyectividad (1 a 1) de ambas listas
        compare_unique, reference_unique =\
            find_closest_point(compare_unique, referenced_correspond, 
                               distance_limit=distance_limit)
        
        # Generar la lista de puntos correspondientes
        correspond_list = [(reference_unique[i], compare_unique[i]) 
                           for i in range(len(reference_unique))] 
        
    else:
        # Definición del largo de la lista
        len_list = len(compare_correspond)
        
        # Generar la lista de puntos correspondientes
        correspond_list = [(referenced_correspond[i], compare_correspond[i]) 
                           for i in range(len_list)] 
    
    # Pasando a arreglo
    correspond_list = np.asarray(correspond_list)
    
    # Información de interés
    ## Cantidad de detecciones correctamente realizadas
    q_classified_ok = correspond_list.shape[0]
    
    ## Accuracy
    accuracy = q_classified_ok / len(reference_list)
    
    ## Precisiones
    distances = abs(correspond_list[:, 0] - correspond_list[:, 1])
    
    mean_precision = np.mean(distances)
    sd_precision = np.std(distances)
    rank_precision = np.max(distances)
    
    ### Definición del pack precisión
    pack_precision = (mean_precision, sd_precision, rank_precision)

    ## Puntos sin clasificar
    unclasified_references_point, unclasified_compare_point =\
        get_unclasified_points(reference_list, compare_list, correspond_list)
    
    return (accuracy, correspond_list, pack_precision, q_classified_ok,
            unclasified_references_point, unclasified_compare_point)


# Módulo de testeo
'''import soundfile as sf
from file_management import get_heartbeat_points
from heart_sound_detection import get_upsampled_thresholded_wavelets, get_zero_points
filename = 'normal__201105011626'
file = f'Heartbeat sounds/Generated/normal_a/{filename}'
audio, samplerate = sf.read(f'{file}.wav')

to_sust = [4,5]
wavelets = get_upsampled_thresholded_wavelets(audio, samplerate, freq_pass=950, freq_stop=1000, 
                                               method='lowpass', lp_method='fir', 
                                               fir_method='kaiser', gpass=1, gstop=80, 
                                               plot_filter=False, levels_to_get=to_sust, 
                                               levels_to_decompose=6, wavelet='db4', mode='periodization', 
                                               threshold_criteria='hard', threshold_delta='universal',
                                               min_percentage=None, print_delta=False,
                                               plot_wavelets=False, normalize=True)

wavelet_final = sum(wavelets)

heart_beats_points = get_heartbeat_points(filename)
zero_center = get_zero_points(wavelet_final, complement=True, to_return='center')

info = get_precision_info(heart_beats_points, zero_center, clean_repeated=True)
print(info)'''