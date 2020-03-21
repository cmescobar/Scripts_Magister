import numpy as np
import soundfile as sf
import pywt
import wavio
import matplotlib.pyplot as plt
from thresholding_functions import wavelet_thresholding
from filter_and_sampling import upsampling_signal


def dwt_decomposition(signal_in, wavelet='db4', mode='periodization',
                      levels='all', return_concatenated=False):
    '''Esta función permite descomponer una señal de entrada en todos los
    posibles niveles de wavelet producto de la aplicación de la "Discrete
    Wavelet Transform" (DWT).
    
    Parámetros
    - signal_in: Señal de entrada
    - wavelet: Tipo de wavelet a utilizar (revisar wavelets del paquete
                pywt)
    - mode: Tipo de descomposición en wavelets (revisar wavelets del 
                paquete pywt)
    - levels: Niveles de descomposición para la aplicación de la
                transformada en wavelets.
        - ['all']: Realiza el proceso de disminución a la mitad hasta
                   llegar al final del proceso
        - [(int)#]: Es posible entregar la cantidad de niveles a 
                    descomponer
    - return_concatenated: Booleano que pregunta si es que la salida se
                    entregará concatenada. Al aplicar 'True', se entregará
                    un arreglo con todas las etapas concatenadas. En caso
                    contrario, se entregará una lista donde cada uno de los
                    N + 1 elementos estarán dado por los N coeficientes de
                    detalle, y el último será el/los último/s coeficiente/s 
                    de aproximación
    '''
    # Definición de la señal a descomponer
    to_decompose = signal_in
    
    # Rutina para salida concatenada
    if return_concatenated:
        # Definición del vector wavelet de salida
        wavelet_out = np.array([])
        
        # Para descomposición en todos los niveles posibles
        if levels == 'all':
            while len(to_decompose) > 1:
                # Descomposición de wavelet
                (to_decompose, cD) = pywt.dwt(to_decompose, wavelet=wavelet, 
                                              mode=mode)
                
                # Agregando el detalle al final del vector de salida
                wavelet_out = np.append(cD, wavelet_out)
        
        # Para selección de niveles
        elif isinstance(levels, int):
            # Descomponiendo en niveles
            for _ in range(levels):
                # Descomposición de wavelet
                (to_decompose, cD) = pywt.dwt(to_decompose, wavelet=wavelet, 
                                              mode=mode)

                # Agregando el detalle al final del vector de salida
                wavelet_out = np.append(cD, wavelet_out)

        return np.append(to_decompose, wavelet_out)
        
    # Rutina para salida no concatenada
    else:
        # Definición de la lista de wavelets de salida
        wavelets_out = list()
        
        if levels == 'all':
            while len(to_decompose) > 1:
                # Descomposición de wavelet
                (to_decompose, cD) = pywt.dwt(to_decompose, wavelet=wavelet, 
                                              mode=mode)

                # Agregando el detalle a la lista
                wavelets_out.append(cD)
        
        elif isinstance(levels, int):
            # Descomponiendo en niveles
            for _ in range(levels):
                # Descomposición de wavelet
                (to_decompose, cD) = pywt.dwt(to_decompose, wavelet=wavelet, 
                                              mode=mode)

                # Agregando el detalle al final del vector de salida. Se agregan,
                # considerando que son N niveles de descomposición, de la forma
                # d1, d2, ... , dN 
                wavelets_out.append(cD)

        # Y finalmente se guarda la aproximación
        wavelets_out.append(to_decompose)

        return wavelets_out


def dwt_recomposition(signal_in, wavelet='db4', mode='periodization',
                      levels='all', is_concatenated=False):
    '''Esta función permite recomponer una señal de entrada en todos los
    posibles niveles de wavelet producto de la aplicación de la "Discrete
    Wavelet Transform" (DWT).
    
    Parámetros
    - signal_in: Señal de entrada
    - wavelet: Tipo de wavelet a utilizar (revisar wavelets del paquete
                pywt)
    - mode: Tipo de descomposición en wavelets (revisar wavelets del 
                paquete pywt)
    - levels: Niveles de descomposición para la aplicación de la
                transformada en wavelets.
        - ['all']: Realiza el proceso de disminución a la mitad hasta
                   llegar al final del proceso
        - [(int)#]: Es posible entregar la cantidad de niveles a 
                    recomponer
    - is_concatenated: Booleano que pregunta si es que la entrada se
                    entregará concatenada. Al aplicar 'True', se procesará
                    un arreglo con todas las etapas concatenadas. En caso
                    contrario, se entregará una lista donde cada uno de los
                    N + 1 elementos estarán dado por los N coeficientes de
                    detalle, y el último será el/los último/s coeficiente/s 
                    de aproximación
    '''
    # Rutina para entrada concatenada
    if is_concatenated:
        if levels == 'all':
            # Definición de los niveles de reconstrucción
            N = int(np.log2(len(signal_in)))

            # Definición de la señal a recomponer
            cA = np.array([signal_in[0]])

            for i in range(N):
                to_look = 2 ** (i+1)

                # Definición del cD y cA de turno
                cD = signal_in[to_look//2:to_look]
                cA = pywt.idwt(cA, cD, wavelet=wavelet, 
                               mode=mode)

            return cA
    
    else:
        # Definición de los niveles de reconstrucción (el "-1" se debe a
        # que el último elemento es el de aproximación)
        N = len(signal_in) - 1
        
        # Definición de la primera componente de aproximación
        cA = signal_in[-1]
        
        # Iteraciones hasta la recomposición
        for i in reversed(range(N)):
            # Definición del cA de turno (se ajusta el tamaño de
            # cA al tamaño de la señal original ingresada)
            cA = pywt.idwt(cA[:len(signal_in[i])], signal_in[i], 
                           wavelet=wavelet, mode=mode)
            
        return cA


def wavelet_packet_decomposition(signal_in, wavelet='db4', mode='periodization',
                                 levels=3):
    '''Esta función permite descomponer una señal de entrada en todos los
    posibles niveles de wavelet producto de la aplicación de la "Discrete
    Wavelet Transform" (DWT).
    
    Parámetros
    - signal_in: Señal de entrada
    - wavelet: Tipo de wavelet a utilizar (revisar wavelets del paquete
                pywt)
    - mode: Tipo de descomposición en wavelets (revisar wavelets del 
                paquete pywt)
    - levels: Número de niveles de descomposición para la aplicación de la
                transformada en wavelets.
                
    Referencias
    (1) S.M. Debbal. Computerized Heart Sounds Analysis. Department of 
        electronic. Faculty of science engineering, University Aboubekr 
        Belkaid. 2011. Algeria.
    '''
    # Definición de la señal a descomponer
    to_decompose = [signal_in]
    
    # Descomponiendo en niveles
    for _ in range(levels):
        # Definición de una lista en las que se almacenarán las 
        # descomposiciones en cada nivel. Nótese que para cada nuevo nivel 
        # se tiene que vaciar para almacenar en orden las descomposiciones 
        wavelets_out = list()
        
        for s_in in to_decompose:
            # Descomposición de wavelet
            (cA, cD) = pywt.dwt(s_in, wavelet=wavelet, mode=mode)

            # Se agregan las señales a lista que mantiene las 
            # descomposiciones ordenadas para cada nivel
            wavelets_out.append(cA)
            wavelets_out.append(cD)
        
        # Una vez terminadas las descomposiciones de cada nivel, se genera
        # este arreglo para aplicar la descomposición de cada uno en el 
        # siguiente nivel
        to_decompose = [i for i in wavelets_out]

    return wavelets_out

    
def wavelet_packet_recomposition(signal_in, wavelet='db4', mode='periodization'):
    '''Esta función permite recomponer una señal de entrada en todos los
    posibles niveles de wavelet producto de la aplicación de la "Discrete
    Wavelet Transform" (DWT).
    
    Parámetros
    - signal_in: Señal de entrada
    - wavelet: Tipo de wavelet a utilizar (revisar wavelets del paquete
                pywt)
    - mode: Tipo de descomposición en wavelets (revisar wavelets del 
                paquete pywt)
    '''
    # Definición de los niveles de reconstrucción
    N = int(np.log2(len(signal_in)))
    
    # Definición de la capa a reconstruir
    layer_to_reconstruct = signal_in

    # Iteraciones hasta la recomposición (por capas)
    for _ in range(N):
        # Definición de una lista en las que se almacenarán las 
        # descomposiciones en cada nivel. Nótese que para cada nuevo nivel 
        # se tiene que vaciar para almacenar en orden las descomposiciones 
        signal_out = list()
        
        # Iteraciones para cada par en cada capa
        for i in range(len(layer_to_reconstruct)// 2):
            # Definición del cA de turno (se ajusta el tamaño de
            # cA al tamaño de la señal original ingresada)
            to_append = pywt.idwt(layer_to_reconstruct[2*i], 
                                  layer_to_reconstruct[2*i+1], 
                                  wavelet=wavelet, mode=mode)
            
            # Se agrega las señales a lista que mantiene las 
            # recomposiciones ordenadas para cada nivel
            signal_out.append(to_append)
        
        # Una vez terminadas las descomposiciones de cada nivel, se genera
        # este arreglo auxiliar para aplicar la descomposición de cada uno  
        # en el siguiente nivel
        layer_to_reconstruct = [i for i in signal_out]
        
    print(len(signal_out))
    return signal_out[0]


def get_wav_of_dwt_level(filename, level_to_get, levels,
                         wavelet='db4', thresholded=True,
                         delta=None, threshold_criteria='hard',
                         threshold_delta='universal',
                         min_percentage=None, print_delta=True):
    '''Creación de sonidos en formato .wav a partir de wavelets de obtenidas
    recuperando el wavelet de un nivel en particular
    
    Parámetros
    - filename: Nombre del archivo a procesar
    - level_to_get: Wavelet del nivel a recuperar
    - levels: Cantidad de niveles en las que se descompondrá la señal
    - delta: Definición de umbral de corte en caso de aplicar thresholding
    - threshold_criteria: Criterio de aplicación de umbral, entre "hard" y "soft"
    - threshold_delta: Selección del criterio de cálculo de umbral. Opciones:
        - ["mad"]: Median Absolute Deviation
        - ["universal"]: universal (4)
        - ["sureshrink"]: Aplicando SURE (4)
        - ["percentage"]: Aplicación del porcentage en relación al máximo
    - min_percentage: Valor del porcentaje con respecto al máximo en la opción
                      "percentage" de la variable "threshold_delta
    - print_delta: Booleano para indicar si se imprime el valor de delta
    '''
    # Cargando señal a procesar
    signal_in, samplerate = sf.read(f'{filename}.wav')
    
    # Probando DWT
    dwt_values = dwt_decomposition(signal_in, wavelet=wavelet, 
                                   mode='periodization',
                                   levels=levels, 
                                   return_concatenated=False)
    
    # Definición de la señal a recuperar según el orden del nivel
    n = level_to_get - 1
    
    if thresholded:
        signal_out = wavelet_thresholding(dwt_values[n], delta=delta, 
                                          threshold_criteria=threshold_criteria,
                                          threshold_delta=threshold_delta,
                                          min_percentage=min_percentage,
                                          print_delta=print_delta)
    else:
        # Señal a obtener
        signal_out = dwt_values[n]
    
    # Samplerate de la señal a recuperar
    sr_out = samplerate // (2 ** level_to_get)
    
    # Generando el archivo de audio
    wavio.write(f"{filename}_DWT_level_{level_to_get}_SR{sr_out}.wav", 
                signal_out, sr_out, sampwidth=3)


def get_wavelet_levels(signal_in, levels_to_decompose=6, levels_to_get='all', wavelet='db4',
                       mode='periodization', threshold_criteria='hard', 
                       threshold_delta='universal', min_percentage=None, 
                       print_delta=False, plot_wavelets=False, plot_show=False,
                       plot_save=(False, None)):
    '''
    Parámetros
    - signal_in: Señal de entrada
    - levels_to_decompose: Cantidad de niveles en las que se descompondrá la señal
    - level_to_get: Wavelet del nivel a recuperar.
        - ['all']: Se recuperan los "levels_to_decompose" niveles
        - [lista]: Se puede ingresar un arreglo de niveles de interés
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
    - plot_show: Booleano para indicar si se muestran estas gráficas
    - plot_save: Tupla que acepta un booleano para indicar si se muestran estas 
                 gráficas (1), y una dirección de almacenamiento en string (2)
    '''
    # Obteniendo la descomposición en wavelets
    dwt_values = dwt_decomposition(signal_in, wavelet=wavelet, mode=mode,
                                   levels=levels_to_decompose, 
                                   return_concatenated=False)
    
    # Definición de la lista de wavelets a retornar
    wavelets_out = []

    if levels_to_get == 'all':
        for interest_signal in dwt_values:
            # Aplicando thresholding
            thresh_signal = wavelet_thresholding(interest_signal, delta=None, 
                                                threshold_criteria=threshold_criteria,
                                                threshold_delta=threshold_delta, 
                                                min_percentage=min_percentage, 
                                                print_delta=print_delta)
            # Agregando a la lista
            wavelets_out.append(thresh_signal)
    else:
        for i in range(len(levels_to_get)):
            # Obtención de la señal a procesar
            interest_signal = dwt_values[levels_to_get[i] - 1]

            # Aplicando thresholding
            thresh_signal = wavelet_thresholding(interest_signal, delta=None, 
                                                threshold_criteria=threshold_criteria,
                                                threshold_delta=threshold_delta, 
                                                min_percentage=min_percentage, 
                                                print_delta=print_delta)
            # Agregando a la lista
            wavelets_out.append(thresh_signal)

    if plot_wavelets:
        # Graficando para los niveles de interés
        plt.figure(figsize=(13,9))
        
        if levels_to_get == 'all':
            # Graficando todos los coeficientes de detalle
            for i in range(len(dwt_values) - 1):
                plt.subplot(len(dwt_values) - 1, 2, 2*i+1)
                plt.plot(dwt_values[i])
                plt.ylabel(f"Nivel {i+1}")

                plt.subplot(len(dwt_values) - 1, 2, 2*i+2)
                plt.plot(wavelets_out[i])
        else:
            # Graficando los coeficientes de detalle especificados
            for i in range(len(levels_to_get)):
                plt.subplot(len(levels_to_get), 2, 2*i+1)
                plt.plot(dwt_values[levels_to_get[i] - 1])
                plt.ylabel(f"Nivel {levels_to_get[i]}")

                plt.subplot(len(levels_to_get), 2, 2*i+2)
                plt.plot(wavelets_out[i])
        
        if plot_show:
            # Mostrando la imagen
            plt.show()
            
        if plot_save[0]:
            # Guardando la imagen
            plt.savefig(plot_save[1])

        # Cerrando la figura
        plt.close()
    
    return wavelets_out


def upsample_wavelets(wavelet_list, samplerate, new_rate, levels_to_get, 
                      N_desired, method='lowpass', trans_width=50, 
                      lp_method='fir', fir_method='kaiser', gpass=1, 
                      gstop=80, plot_filter=False, plot_signals=False,
                      plot_wavelets=True, normalize=True):
    # Definición de la lista donde se almacenarán los wavelets
    upsampled_wavelets = []

    for i in range(len(wavelet_list)):
        # Aplicando un upsampling
        resampled_signal = upsampling_signal(wavelet_list[i], 
                                             samplerate / (2 ** (levels_to_get[i])), 
                                             new_rate, N_desired=N_desired, 
                                             method=method,
                                             trans_width=trans_width, 
                                             lp_method=lp_method, 
                                             fir_method=fir_method, 
                                             gpass=gpass, gstop=gstop, 
                                             plot_filter=plot_filter, 
                                             plot_signals=plot_signals,
                                             normalize=normalize)
        # Guardando
        upsampled_wavelets.append(resampled_signal)
    
    if plot_wavelets:
        # Creando el plot de grillas
        gridsize = (len(wavelet_list), 2)
        plt.figure(figsize=(9, 6))
        
        # Graficando los componentes a la izquierda
        for i in range(len(wavelet_list)):
            ax = plt.subplot2grid(gridsize, (i, 0))
            ax.plot(upsampled_wavelets[i])
        
        # Y graficando la suma a la derecha
        ax = plt.subplot2grid(gridsize, (0, 1), colspan=1, 
                              rowspan=len(wavelet_list))
        
        # Suma de wavelets
        wavelet_final = sum(upsampled_wavelets)
        ax.plot(wavelet_final)
        plt.show()
    
    return upsampled_wavelets