import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os
from breath_onset_detection import boundary_cycle_method
from heart_sound_detection import get_upsampled_thresholded_wavelets, get_zero_points


def put_clicks_respiratory_cycle(filepath, fmin=80, fmax=1000, k_audio=5, k_click=2.5):
    '''Función que permite agregar clicks en los puntos donde se comienza/termina un 
    ciclo respiratorio en base al método de detección de bordes por ciclo
    
    Parámetro
    - filepath: Directorio donde se encuenta el archivo de audio
    - fmin: Frecuencia mínima a considerar para el espectro de energía
    - fmax: Frecuencia máxima a considerar para el espectro de energía
    - k_audio: Parámetro de ponderación relativo para la señal de entrada
               Tiene relación con el volumen del audio sobre la salida
    - k_click: Parámetro de ponderación relativo para la señal de click
    
    Referencias:
    - Aras, Selim & OZTURK, Mehmet & Gangal, Ali. (2018). Automatic detection of the respiratory
    cycle from recorded, single-channel sounds from lungs. Turkish Journal of Electrical
    Engineering and Computer Sciences. Disponible en: 
    http://journals.tubitak.gov.tr/elektrik/issues/elk-18-26-1/elk-26-1-2-1705-16.pdf 
    '''
    # Apertura del archivo de audio
    audio_data, samplerate = sf.read(filepath)
    # Apertura del archivo del click
    audio_click, _ = sf.read('useful_audio/click.wav')

    # Detección de peaks para la señal de audio ingresada. En caso de que no
    # funcione, se registrarán los audios con problema para revisarlos
    # posteriormente 
    try:
        peaks = boundary_cycle_method(audio_data, samplerate, fmin=fmin, 
                                      fmax=fmax, overlap=0)
    except:
        print(f"Error en audio {filepath}")
        with open('Interest_Audios/error_data.csv', 'a', encoding='utf8') \
            as file:
            file.write(f"{filepath}\n")
        return

    # Regulación del volumen de los audios para el sonido final
    audio_mast = audio_data * k_audio
    click_mast = audio_click * k_click

    # Sumando el click en las posiciones donde ocurre el fin/comienzo de un
    # ciclo respiratorio
    audio_out = np.asarray([i for i in audio_mast])

    # Agregando el click a cada elemento
    for peak in peaks:
        for i in range(len(audio_click)):
            audio_out[peak + i] += click_mast[i]
    
    # Normalizando el audio
    audio_out /= max(audio_out)

    # Definición de la carpeta dónde se guardará el archivo
    dir_list = filepath.split('/')
    folder_data = f"{'/'.join(dir_list[:2])}/Segmented_by_resp" 
    
    # Se crea una carpeta en caso de que no exista la carpeta del archivo a
    # guardar
    if not os.path.isdir(folder_data):
        os.mkdir(folder_data)

    # Definiendo el path del archivo de audio
    folder_to_save = f"{folder_data}/{dir_list[-1]}"
    sf.write(folder_to_save, audio_out, samplerate)


    '''list_cycles = [41728, 129024, 206592, 296960, 377216, 450432,
                  521856, 608256, 684288, 777600, 872064]
    plt.plot(audio_data)
    plt.plot(peaks, [0 for i in peaks], 'rx')
    plt.plot(list_cycles, [0 for i in list_cycles], 'go')
    
    plt.show()'''


def get_audio_respiratory_clicks_by_symptom(symptom, fmin=80, fmax=1000):
    '''Función que permite agregar clicks en los puntos donde se comienza/termina un 
    ciclo respiratorio en base al método de detección de bordes por ciclo, eligiendo
    todos los archivos de un síntoma en específico
    
    Parámetros
    - symptom: Síntoma a procesar
    - fmin: Frecuencia mínima a considerar para el espectro de energía
    - fmax: Frecuencia máxima a considerar para el espectro de energía
    '''
    # Definición de la carpeta a revisar
    folder_data = f'Interest_Audios/{symptom}'
    # Obtener todos los archivos de la carpeta de archivos ya normalizados
    files = os.listdir(folder_data)
    # Luego, la dirección de cada uno de los archivos por síntoma es
    symptom_dirs = [f"{folder_data}/{i}" for i in files
                    if i.endswith('.wav')]
    
    # Para cada uno de los archivos se crea un audio con clicks
    for i in symptom_dirs:
        print(f"Creating clicked audio of {i.split('/')[-1]}...")
        put_clicks_respiratory_cycle(i, fmin=fmin, fmax=fmax)
        print("¡Complete!")
    

def put_clicks_on_audio(signal_in, samplerate, points_to_put,
                        f_sound=2000, n_sound=1000, k_audio=5, 
                        k_click=2, click_window=None, 
                        normalize=True):
    ''' Función que permite poner clicks de sonido sobre un audio
    
    Parámetros
    - signal_in: Señal de entrada
    - samplerate: Tasa de muestreo de la señal
    - points_to_put: Lista de puntos para poner clicks sobre la señal de 
                     entrada
    - f_sound: Frecuencia a la que sonará el click
    - n_sound: Cantidad de puntos que tendrá el click
    - k_audio: Parámetro de ponderación relativo para la señal de entrada
               Tiene relación con el volumen del audio sobre la salida
    - k_click: Parámetro de ponderación relativo para la señal de click
               Tiene relación con el volumen del click sobre la salida
    - click_windowed: Parámetro que indica si el click es ventaneado
        - [None]: No se ventanea
        - ['hamming']: Se ventanea por una ventana hamming
    - normalize: Booleano para normalización de la señal
    
    Referencias:
    - Elaboración propia
    '''
    # Definición del click
    n = np.arange(n_sound)
    audio_click = np.sin(2 * np.pi * f_sound / samplerate * n)
    
    # Aplicación de ventana
    if click_window == 'hamming':
        audio_click *= np.hamming(len(audio_click))
    elif click_window == 'hann':
        audio_click *= np.hanning(len(audio_click))
    elif click_window == 'blackman':
        audio_click *= np.blackman(len(audio_click))
    elif click_window == 'bartlett':
        audio_click *= np.bartlett(len(audio_click))
    
    # Regulación del volumen de los audios para el sonido final
    audio_mast = signal_in * k_audio
    click_mast = audio_click * k_click

    # Sumando el click en las posiciones donde ocurre el fin/comienzo de un
    # ciclo respiratorio
    audio_out = np.array([i for i in audio_mast])
    
    # Definición del parámetro de largo de la mitad del click
    half_click = len(click_mast) // 2

    # Agregando el click a cada elemento
    for point in points_to_put:
        audio_out[point - half_click:point + half_click] += click_mast
            
    if normalize:
        return audio_out / max(audio_out)
    else:
        return audio_out


def get_audio_sum_wavelets(filepath, freq_pass=950, freq_stop=1000,
                           method='lowpass', lp_method='fir',
                           fir_method='kaiser', gpass=1, gstop=80,
                           levels_to_get=[3,4,5],
                           levels_to_decompose=6, wavelet='db4', mode='periodization',
                           threshold_criteria='hard', threshold_delta='universal',
                           min_percentage=None, print_delta=False,
                           normalize=True):
    '''Función que permite generar archivos de audio en formato .wav de los wavelets
    de los archivos que se encuentran en el directorio de interés mediante el proceso
    de descomposición dwt
    
    Parámetros
    - filepath: Directorio donde se encuenta el set de audios
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
    - levels_to_decompose: Cantidad de niveles en las que se descompondrá la señal
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
    - normalize: Booleano para normalización de la señal
    
    Referencias:
    - Elaboración propia
    '''
    # Lista de los sonidos cardíacos
    filenames = [i for i in os.listdir(filepath) if i.endswith('.wav')]
    
    # Definición de la carpeta a almacenar
    filepath_to_save = f'{filepath}/{wavelet}'
    
    # Preguntar si es que la carpeta que almacenará las imágenes se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(filepath_to_save):
        os.makedirs(filepath_to_save)
    
    for i in filenames:
        print(f"Getting wavelet sum audio of {i}...")
        # Cargando los archivos
        audio_file, samplerate = sf.read(f'{filepath}/{i}')
        
        # Definición de la dirección dónde se almacenará la imagen
        filesave = f'{filepath_to_save}/{i.strip(".wav")} Wavelet_sum levels'\
                   f'{levels_to_get}.wav'
        
        # Obteniendo los wavelets de interés (upsampleados)
        ups_wavelets = \
            get_upsampled_thresholded_wavelets(audio_file, samplerate, 
                                               freq_pass=freq_pass, freq_stop=freq_stop, 
                                               method=method, lp_method=lp_method, 
                                               fir_method=fir_method, 
                                               gpass=gpass, gstop=gstop, 
                                               plot_filter=False, levels_to_get=levels_to_get, 
                                               levels_to_decompose=levels_to_decompose,
                                               wavelet=wavelet, 
                                               mode=mode, 
                                               threshold_criteria=threshold_criteria, threshold_delta=threshold_delta,
                                               min_percentage=min_percentage, 
                                               print_delta=print_delta,
                                               plot_wavelets=False, normalize=normalize)
        
        # Sumando estos wavelets
        wavelet_sum = sum(ups_wavelets)
        
        # Normalizando
        wavelet_sum = wavelet_sum / max(abs(wavelet_sum))
        
        # Guardando
        sf.write(filesave, wavelet_sum, samplerate)
        
        print('Completed!\n')


def get_audios_heart_clicked(filepath, freq_pass=950, freq_stop=1000,
                             method='lowpass', lp_method='fir',
                             fir_method='kaiser', gpass=1, gstop=80,
                             levels_to_get=[3,4,5],
                             levels_to_decompose=6, wavelet='db4', mode='periodization',
                             threshold_criteria='hard', threshold_delta='universal',
                             min_percentage=None, print_delta=False,
                             f_sound=2000, n_sound=1000, k_audio=5, 
                             k_click=2, click_window=None,normalize=True):
    '''Función que permite ingresar una dirección donde se encuentre un set de audios
    para agregarles un click donde haya presencia de un sonido cardíaco mediante el
    método de detección con wavelets de cierto nivel
    
    Parámetros
    - filepath: Directorio donde se encuenta el set de audios
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
    - levels_to_decompose: Cantidad de niveles en las que se descompondrá la señal
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
    - f_sound: Frecuencia a la que sonará el click
    - n_sound: Cantidad de puntos que tendrá el click
    - k_audio: Parámetro de ponderación relativo para la señal de entrada
               Tiene relación con el volumen del audio sobre la salida
    - k_click: Parámetro de ponderación relativo para la señal de click
               Tiene relación con el volumen del click sobre la salida
    - click_windowed: Parámetro que indica si el click es ventaneado
        - [None]: No se ventanea
        - ['hamming']: Se ventanea por una ventana hamming
    - normalize: Booleano para normalización de la señal
    
    Referencias:
    - Elaboración propia
    '''   
    # Lista de los sonidos cardíacos
    filenames = [i for i in os.listdir(filepath) if i.endswith('.wav')]
    
    # Definición de la carpeta a almacenar
    filepath_to_save = f'{filepath}/{wavelet}'
    
    # Preguntar si es que la carpeta que almacenará las imágenes se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(filepath_to_save):
        os.makedirs(filepath_to_save)
    
    for i in filenames:
        print(f"Getting clicked audio of {i}...")
        # Cargando los archivos
        audio_file, samplerate = sf.read(f'{filepath}/{i}')
        
        # Definición de la dirección dónde se almacenará la imagen
        filesave = f'{filepath_to_save}/{i.strip(".wav")} clicked levels {levels_to_get}.wav'
        
        # Obteniendo los wavelets de interés (upsampleados)
        ups_wavelets = \
            get_upsampled_thresholded_wavelets(audio_file, samplerate, 
                                               freq_pass=freq_pass, freq_stop=freq_stop, 
                                               method=method, lp_method=lp_method, 
                                               fir_method=fir_method, 
                                               gpass=gpass, gstop=gstop, 
                                               plot_filter=False, levels_to_get=levels_to_get, 
                                               levels_to_decompose=levels_to_decompose,
                                               wavelet=wavelet, 
                                               mode=mode, 
                                               threshold_criteria=threshold_criteria, threshold_delta=threshold_delta,
                                               min_percentage=min_percentage, 
                                               print_delta=print_delta,
                                               plot_wavelets=False, normalize=normalize)
        
        # Sumando estos wavelets
        wavelet_sum = sum(ups_wavelets)
        
        # Obteniendo los puntos donde hay un sonido cardíaco
        heart_sound_pos = get_zero_points(abs(wavelet_sum), complement=True, to_return='center')
        
        # Insertando los clicks en la señal
        audio_with_clicks = put_clicks_on_audio(audio_file, samplerate, heart_sound_pos,
                                                f_sound=f_sound, n_sound=n_sound, 
                                                k_audio=k_audio, k_click=k_click, 
                                                click_window=click_window, 
                                                normalize=normalize)

        # Guardando el archivo de audio
        sf.write(filesave, audio_with_clicks, samplerate)
        
        print('Completed!\n')


# Module test
#put_clicks_respiratory_cycle('Interest_Audios/Healthy/'
#                            '125_1b1_Tc_sc_Meditron.wav')

# get_audio_respiratory_clicks_by_symptom('Healthy', fmin=120)

# Nivel de importancia
'''heart_quality = 4
filepath = f'Interest_Audios/Heart_sound_files/Level {heart_quality}'

get_audio_sum_wavelets(filepath, freq_pass=950, freq_stop=1000,
                       method='lowpass', lp_method='fir',
                       fir_method='kaiser', gpass=1, gstop=80,
                       levels_to_get=[4,5],
                       levels_to_decompose=6, wavelet='db4', mode='periodization',
                       threshold_criteria='hard', threshold_delta='universal',
                       min_percentage=None, print_delta=False,
                       normalize=True)

get_audios_heart_clicked(filepath, freq_pass=950, freq_stop=1000,
                        method='lowpass', lp_method='fir',
                        fir_method='kaiser', gpass=1, gstop=80,
                        levels_to_get=[3,4,5],
                        levels_to_decompose=6, wavelet='db4', mode='periodization',
                        threshold_criteria='hard', threshold_delta='universal',
                        min_percentage=None, print_delta=False,
                        f_sound=2000, n_sound=1000, k_audio=5, 
                        k_click=2, click_window=None,normalize=True)'''