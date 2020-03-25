import numpy as np
import matplotlib.pyplot as plt
from wavelet_functions import get_wavelet_levels, upsample_signal_list
from thresholding_functions import wavelet_thresholding
from filter_and_sampling import upsampling_signal, downsampling_signal


def get_upsampled_thresholded_wavelets(signal_in, samplerate, freq_pass=950, freq_stop=1000, 
                                       method='lowpass', lp_method='fir', 
                                       fir_method='kaiser', gpass=1, gstop=80, 
                                       plot_filter=False, levels_to_get=[3,4,5], 
                                       levels_to_decompose=6, wavelet='db4', mode='periodization', 
                                       threshold_criteria='hard', threshold_delta='universal',
                                       min_percentage=None, print_delta=False,
                                       plot_wavelets=False, normalize=True):
    '''Función que permite ingresar una señal para recuperar los niveles de interés
    sampleados a la tasa de muestreo original, obtenidos a partir de una descomposición en 
    transformada de wavelet discreta
    
    Parámetros
    - signal_in: Señal a submuestrear
    - samplerate: Tasa de muestreo de la señal "signal_in"
    - freq_pass: Frecuencia de corte de la pasa banda
    - freq_stop: Frecuencia de corte de la rechaza banda. Esta es
                 la que se toma en cuenta al momento de hacer el 
                 último corte (por ende, si busca samplear a 2kHz,
                 seleccione este parámetro en 1kHz)
    - method: Método de submuestreo
        - [lowpass]: Se aplica un filtro pasabajos para evitar
                     aliasing de la señal. Luego se submuestrea
        - [cut]: Simplemente se corta en la frecuencia de interés
        - ['resample']:Se aplica la función resample de scipy
        - ['resample_poly']:Se aplica la función resample_poly de scipy
    - lp_method: Método de filtrado para elección lowpass
        - [fir]: se implementa un filtro FIR
        - [iir]: se implementa un filtro IIR
    - fir_method: Método de construcción del filtro FIR  en caso 
                  de seleccionar el método lowpass con filtro FIR
        - ['window']: Construcción por método de la ventana
        - ['kaiser']: Construcción por método de ventana kaiser
        - ['remez']: Construcción por algoritmo remez
    - gpass: Ganancia en dB de la magnitud de la pasa banda
    - gstop: Ganancia en dB de la magnitud de la rechaza banda
    - levels_to_get: Niveels de los Wavelet a recuperar
        - ['all']: Se recuperan los "levels_to_decompose" niveles
        - [lista]: Se puede ingresar un arreglo de niveles de interés
    - wavelet: Wavelet utilizado para el proceso de dwt. Revisar en 
               pywt.families(kind='discrete')
    - mode: Tipo de descomposición en wavelets (revisar wavelets del 
            paquete pywt)
    - threshold_criteria: Criterio de aplicación de umbral, entre "hard" y "soft"
    - threshold_delta: Selección del criterio de cálculo de umbral. Opciones:
        - ["mad"]: Median Absolute Deviation
        - ["universal"]: universal
        - ["sureshrink"]: Aplicando SURE
        - ["percentage"]: Aplicación del porcentage en relación al máximo
    - min_percentage: Valor del porcentaje con respecto al máximo en la opción
                      "percentage" de la variable "threshold_delta
    - print_delta: Booleano para indicar si se imprime el valor de delta
    - plot_wavelets: Booleano para indicar si se grafican los wavelets
    - normalize: Normalización de la señal de salida
    '''
    # Aplicando un downsampling a la señal para disminuir la cantidad de puntos a 
    # procesar
    new_rate, dwns_signal = downsampling_signal(signal_in, samplerate, 
                                                freq_pass, freq_stop, 
                                                method=method, 
                                                lp_method=lp_method, 
                                                fir_method=fir_method, 
                                                gpass=gpass, gstop=gstop, 
                                                plot_filter=plot_filter, 
                                                normalize=normalize)
    
    # Se obtienen los wavelets que interesan
    interest_wavelets = get_wavelet_levels(dwns_signal, 
                                           levels_to_get=levels_to_get,
                                           levels_to_decompose=levels_to_decompose, 
                                           wavelet=wavelet, mode=mode, 
                                           threshold_criteria=threshold_criteria, 
                                           threshold_delta=threshold_delta, 
                                           min_percentage=min_percentage, 
                                           print_delta=print_delta, 
                                           plot_wavelets=plot_wavelets)
    
    # Finalmente, upsampleando
    upsampled_wavelets = upsample_signal_list(interest_wavelets, new_rate, samplerate, 
                                              levels_to_get, len(signal_in), 
                                              method=method, 
                                              trans_width=abs(freq_stop - freq_pass), 
                                              lp_method=lp_method, 
                                              fir_method=fir_method, 
                                              gpass=gpass, gstop=gstop, 
                                              plot_filter=False, 
                                              plot_signals=False,
                                              plot_wavelets=plot_wavelets, 
                                              normalize=normalize)
    
    return upsampled_wavelets


def find_potential_onset_offset(signal_in, det_type):
    '''Función que permite encontrar los potenciales onsets basado en la
    revisión de puntos adyacentes. Se revisa si es que el punto actual es 
    cero y el punto siguiente es distinto de cero. Si se cumple, es porque
    se está en presencia de un "onset".
    Referencias: 
    - Qingshu Liu, et.al. An automatic segmentation method for heart sounds.
      2018. Biomedical Engineering.
    
    Parámetros
    - signal_in: Señal de entrada
    - det_type: Opción para el tipo de salida
        - [onset]: Retorna los onset (inicio de sonidos cardíacos)
        - [offset]: Retorna los offset (fin de sonidos cardíacos)
        - [all]: Retorna tanto onsets como offsets
    '''
    if det_type == 'onset':
        return [i for i in range(len(signal_in)-1) 
                if signal_in[i] == 0 and signal_in[i+1] != 0]
    elif det_type == 'offset':
        return [i for i in range(1, len(signal_in)) 
                if signal_in[i-1] != 0 and signal_in[i] == 0]
    elif det_type == 'all':
        all_points = [i for i in range(len(signal_in)-1) 
                      if signal_in[i] == 0 and signal_in[i+1] != 0] + \
                     [i for i in range(1, len(signal_in)) 
                      if signal_in[i-1] != 0 and signal_in[i] == 0]
        # Ordenando
        all_points.sort()
        return all_points
    
    else:
        raise Exception('Opción seleccionada no es válida. Ingrese un tipo '
                        'de salida disponible en las opciones.')


def get_zero_points(signal_in, complement=False, tol=1e-12, 
                    to_return='all'):
    '''Función que permite encontrar los ceros de la señal
    
    Parámetros
    - signal_in: Señal de entrada
    - complement: Booleano que indica si se entregan los puntos de 
                  la señal donde se encuentran los ceros. Si es "True"
                  se entregan todos los puntos distintos de cero. Si es
                  "False", se entregan todos los puntos en cero. Por
                  defecto se deja en "False"
    - tol: Rango de tolerancia para los valores "cero"
    - to_return: Lista de puntos que se entregará
        - ['all']: Se retornan directamente todos los puntos encontrados
        - ['center']: Se retornan los puntos centrales de cada uno de los
                      clusters obtenidos
    '''
    # Selección de tipo de puntos con respecto a los ceros
    if complement:
        point_list = [i for i in range(len(signal_in)) 
                      if abs(signal_in[i]) >= tol]
    else:
        point_list = [i for i in range(len(signal_in)) 
                      if abs(signal_in[i]) <= tol]
        
    if to_return == 'all':
        return point_list
    elif to_return == 'center':
        # Seleccionando un punto característico de la región (ya que muchos
        # de los "puntos" aparecen agrupados en más puntos). En primer lugar,
        # se obtiene un vector de diferencias para conocer los puntos en los
        # que se pasa de un cluster a otro
        dif_indexes = [i + 1 for i in range(len(point_list) - 1)
                       if point_list[i + 1] - point_list[i] > 1] + \
                      [len(point_list) + 1]
        
        # Separando los clusters de puntos y encontrando el índice representativo de
        # cada uno
        begin = 0
        out_indexes = []
        for i in dif_indexes:
            # Definición del punto posible. Se hace round en caso de que sea un
            # decimal, e int para pasarlo si o si a un elemento tipo "int" para
            # indexar 
            out_indexes.append(int(round(np.mean(point_list[begin:i]))))
            # Redefiniendo el comienzo del análisis
            begin = i
        
        return out_indexes


# Testing module
'''import os
import soundfile as sf

# Parámetros de descomposición
levels_to_get = [3,4,5]
levels_to_decompose = 6
heart_quality = 4
wavelet = 'db4'

filepath = f'Interest_Audios/Heart_sound_files/Level {heart_quality}'
filenames = [i for i in os.listdir(filepath) if i.endswith('.wav')]

for i in filenames:
    print(f'Getting wavelets of {i}...')
    # Cargando los archivos
    audio_file, samplerate = sf.read(f'{filepath}/{i}')
    
    ups_wav = get_upsampled_thresholded_wavelets(audio_file, samplerate, freq_pass=950,
                                                 freq_stop=1000,
                                                 method='lowpass', lp_method='fir',
                                                 fir_method='kaiser', gpass=1, gstop=80,
                                                 plot_filter=False, levels_to_get=levels_to_get,
                                                 levels_to_decompose=levels_to_decompose,
                                                 wavelet=wavelet,
                                                 mode='periodization',
                                                 threshold_criteria='hard',
                                                 threshold_delta='universal',
                                                 min_percentage=None, print_delta=False,
                                                 plot_wavelets=False, normalize=True)

    dir_to_paste = f'{filepath}/{wavelet}'
    
    # Preguntar si es que la carpeta que almacenará los sonidos se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(dir_to_paste):
        os.makedirs(dir_to_paste)
    
    for n in range(len(ups_wav)):
        # Definición del nivel a recuperar
        level = levels_to_get[n]
        # Definición del sonido a recuperar
        to_rec = ups_wav[n]
        
        # Grabando
        sf.write(f'{dir_to_paste}/{i.strip(".wav")} - wavelet level {level}.wav', 
                 to_rec, samplerate)
    
    print('Completed!\n')
'''