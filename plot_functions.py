import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import descriptor_functions as df
from tqdm import tqdm
from file_management import get_patient_by_symptom, get_dir_audio_by_id, get_heartbeat_points,\
    get_heartbeat_points_created_db
from heart_sound_detection import get_wavelet_levels, get_upsampled_thresholded_wavelets,\
    get_zero_points
from filter_and_sampling import downsampling_signal, bandpass_filter
from precision_functions import get_precision_info


def get_symptom_images_by_frame(symptom, func_to_apply="normal",
                                display_time=True):
    '''Función que permite graficar los archivos de audio utilizando una función
    de interés para un síntoma en específico
    
    Parámetros
    - symptom: Síntoma a procesar
    - func_to_apply: Función a aplicar sobre los archivos de audio del síntoma
    - display_time: Si el plot muestra como eje x al tiempo (y no a las muestras)
    '''
    # Consultar si es que la función está como string
    if isinstance(func_to_apply, str):
        func_name = func_to_apply
    else:
        func_name = func_to_apply.__name__
        # Si es que es función, consultar si es que la función está bien
        # definida
        if func_name not in dir(df):
            print('La función definida no está disponible en el set. Por favor,'
                  ' intente nuevamente')
            return

    # Obtener las id de los pacientes
    symptom_list = get_patient_by_symptom(symptom)

    # Luego obtener las direcciones de los archivos de audio para cada caso
    symptom_dirs = get_dir_audio_by_id(symptom_list)

    # Preguntar si es que la carpeta que almacenará las imágenes se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(f'Results/{symptom}'):
        os.mkdir(f'Results/{symptom}')

    # Una vez que se haya corroborado la creación de la carpeta, se preguntará
    # si es que la carpeta que almacenará las imágenes para cada función se
    # ha creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(f'Results/{symptom}/{func_name}'):
        os.mkdir(f'Results/{symptom}/{func_name}')

    # Obteniendo la imagen de cada señal se tiene
    for i in symptom_dirs:
        # Para guardar la imagen y definir el título del plot, se debe obtener
        # el nombre del archivo de audio
        filename = i.split('/')[-1].strip('.wav')
        print(f"Plotting figure {filename} with function {func_name}...")

        # Lectura del audio
        audio, samplerate = sf.read(i)

        # Aplicación de la función
        if func_name == "normal":
            time = [i / samplerate for i in range(len(audio))]
            to_plot = audio
        else:
            time, to_plot = df.apply_function_to_audio(func_to_apply, audio,
                                                       samplerate)

        # Gráfico
        plt.figure(figsize=[12, 6.5])
        if display_time:
            plt.plot(time, to_plot)
            plt.xlabel('Tiempo [s]')
        else:
            plt.plot(to_plot)
            plt.xlabel('Muestras')
        plt.ylabel('Amplitud')
        plt.title(f'Plot {filename}')

        # Guardando la imagen
        plt.savefig(f"Results/{symptom}/{func_name}/{filename}.png")

        # Cerrando la figura
        plt.close()

        print("Plot Complete!\n")


def get_symptom_images_at_all(symptom, func_to_apply, N=1, display_time=False):
    '''Función que permite generar una carpeta de plots aplicando una función de 
    interés sobre los archivos de audio de un síntoma en particular
    
    Parámetros
    - symptom: Síntoma a procesar
    - func_to_apply: Función a aplicar sobre los archivos de audio del síntoma
    - N: Coeficiente utilizado para la repetición de la función "abs_fourier_shift"
    - display_time: Si el plot muestra como eje x al tiempo (y no a las muestras)
    '''
    # Obtener las id de los pacientes
    symptom_list = get_patient_by_symptom(symptom)

    # Luego obtener las direcciones de los archivos de audio para cada caso
    pneumonia_dirs = get_dir_audio_by_id(symptom_list)

    # Preguntar si es que la carpeta que almacenará las imágenes se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(f'Results/{symptom}'):
        os.mkdir(f'Results/{symptom}')

    # Una vez que se haya corroborado la creación de la carpeta, se preguntará
    # si es que la carpeta que almacenará las imágenes para cada función se
    # ha creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(f'Results/{symptom}/{func_to_apply.__name__}'):
        os.mkdir(f'Results/{symptom}/{func_to_apply.__name__}')

    for i in pneumonia_dirs:
        # Para guardar la imagen y definir el título del plot, se debe obtener
        # el nombre del archivo de audio
        filename = i.split('/')[-1].strip('.wav')
        print(f"Plotting figure {filename} with function "
              f"{func_to_apply.__name__}...")

        # Lectura del audio
        audio, samplerate = sf.read(i)

        # Aplicando la función ingresada
        if func_to_apply.__name__ == "get_spectrogram":
            if display_time:
                plt.subplot(2, 1, 1)
                # Creación de un vector de tiempo
                time = [i / samplerate for i in range(len(audio))]
                plt.plot(time, audio)
                plt.xlim([0, time[-1]])
                plt.ylabel('Amplitud')

                plt.subplot(2, 1, 2)

            plt.specgram(audio, Fs=samplerate)
            plt.xlabel('Tiempo [seg]')
            plt.ylabel('Frecuencia [Hz]')
            plt.suptitle(f'Plot {filename}')

        else:
            freq, to_plot = func_to_apply(audio, samplerate, N)

            # Gráficando
            plt.plot(freq, to_plot)
            plt.xlabel('Frecuencia [Hz]')
            plt.ylabel('Amplitud')
            plt.title(f'Plot {filename}')

        # Guardando la imagen
        plt.savefig(f"Results/{symptom}/{func_to_apply.__name__}/"
                    f"{filename}.png")

        # Cerrando la figura
        plt.close()

        print("Plot Complete!\n")


def get_wavelets_images_of_heart_sounds(filepath, freq_pass=950, freq_stop=1000, 
                                        freqs_bp=[], method='lowpass', lp_method='fir',
                                        fir_method='kaiser', gpass=1, gstop=80,
                                        levels_to_get=[3,4,5],
                                        levels_to_decompose=6, wavelet='db4', 
                                        mode='periodization',
                                        threshold_criteria='hard', 
                                        threshold_delta='universal',
                                        min_percentage=None, print_delta=False,
                                        normalize=True):
    '''Función que permite obtener un gráfico de la señal original y los niveles
    de interés obtenidos a partir de la descomposición en wavelets
    
    Parámetros
    - filepath: Directorio donde se encuenta el set de audios
    - freq_pass: Frecuencia de corte de la pasa banda
    - freq_stop: Frecuencia de corte de la rechaza banda. Esta es
                 la que se toma en cuenta al momento de hacer el 
                 último corte (por ende, si busca samplear a 2kHz,
                 seleccione este parámetro en 1kHz)
    freqs_bp : list, optional
        Frecuencias de la aplicación del filtro pasabanda (en orden). En caso de estar vacía
        o no cumplir con las 4 frecuencias necesarias, no se aplicará filtro. Por defecto es [].
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
    filepath_to_save = f'{filepath}/{wavelet}/Wavelet decompositon'
    
    # Preguntar si es que la carpeta que almacenará las imágenes se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(filepath_to_save):
        os.makedirs(filepath_to_save)

    for i in tqdm(filenames, desc='Wavelets', ncols=70):
        #print(f"Plotting wavelets of {i}...")
        # Cargando los archivos
        audio_file, samplerate = sf.read(f'{filepath}/{i}')
        
        # Definición del texto bandpassed
        bandpassed_txt = ''
        
        # Aplicación de filtro pasa banda si es que se define una lista de frecuencias
        if freqs_bp:
            try:
                audio_to_wav = bandpass_filter(audio_file, samplerate, 
                                               freq_stop_1=freqs_bp[0], 
                                               freq_pass_1=freqs_bp[1],
                                               freq_pass_2=freqs_bp[2], 
                                               freq_stop_2=freqs_bp[3], 
                                               bp_method='scipy_fir', 
                                               lp_method='fir', hp_method='fir', 
                                               lp_process='manual_time_design',
                                               fir_method='kaiser', gpass=gpass, gstop=gstop, 
                                               plot_filter=False, correct_by_gd=True, 
                                               gd_padding='periodic', normalize=True)
                
                # Definición de la dirección dónde se almacenará la imagen
                filesave = f'{filepath_to_save}/{i.strip(".wav")} Wavelets bandpassed '\
                           f'{freqs_bp}.png'
                
                # Redefinición del texto bandpassed
                bandpassed_txt = f' bp {freqs_bp}'

            except:
                raise Exception('Frecuencias de pasa banda no están bien definidas. Por favor, ' 
                                'intente nuevamente.')
        
        else:
            # Definición de la dirección dónde se almacenará la imagen
            filesave = f'{filepath_to_save}/{i.strip(".wav")} Wavelets.png'
            
            # Definición del archivo a procesar
            audio_to_wav = audio_file
        
        # Aplicando un downsampling a la señal para disminuir la cantidad de puntos a 
        # procesar
        _, dwns_signal = downsampling_signal(audio_to_wav, samplerate, 
                                            freq_pass, freq_stop, 
                                            method=method, 
                                            lp_method=lp_method, 
                                            fir_method=fir_method, 
                                            gpass=gpass, gstop=gstop, 
                                            plot_filter=False, 
                                            normalize=normalize)
        
        # Se obtienen los wavelets que interesan
        #print(f"Wavelets of {i} completed!\n")
        _ = get_wavelet_levels(dwns_signal, 
                               levels_to_get=levels_to_get,
                               levels_to_decompose=levels_to_decompose, 
                               wavelet=wavelet, mode=mode, 
                               threshold_criteria=threshold_criteria, 
                               threshold_delta=threshold_delta, 
                               min_percentage=min_percentage, 
                               print_delta=print_delta, 
                               plot_wavelets=True,
                               plot_show=False,
                               plot_save=(True, filesave))


def get_sum_wavelets_vs_audio(filepath, freq_pass=950, freq_stop=1000,
                              freqs_bp=[], method='lowpass', lp_method='fir',
                              fir_method='kaiser', gpass=1, gstop=80,
                              levels_to_get=[3,4,5],
                              levels_to_decompose=6, wavelet='db4', mode='periodization',
                              threshold_criteria='hard', threshold_delta='universal',
                              min_percentage=None, print_delta=False,
                              plot_show=False, normalize=True):
    '''Funnción que permite graficaar la suma de los wavelets de ciertos niveles de interés
    a la tasa de muestreo de la señal original, y la señal de audio en el mismo cuadro
    de modo que pueda realizarse una comparación visual
    
    Parameters
    ----------
    - filepath: Directorio donde se encuenta el set de audios.
    - freq_pass: Frecuencia de corte de la pasa banda
    - freq_stop: Frecuencia de corte de la rechaza banda. Esta es
                 la que se toma en cuenta al momento de hacer el 
                 último corte (por ende, si busca samplear a 2kHz,
                 seleccione este parámetro en 1kHz)
    freqs_bp : list, optional
        Frecuencias de la aplicación del filtro pasabanda (en orden). En caso de estar vacía
        o no cumplir con las 4 frecuencias necesarias, no se aplicará filtro. Por defecto es [].
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
    - plot_show: Booleano que sirve para mostrar mientras se corre el programa
                 el gráfico construido
    - normalize: Booleano para normalización de la señal
    
    Referencias:
    - Elaboración propia
    '''
    # Lista de los sonidos cardíacos
    filenames = [i for i in os.listdir(filepath) if i.endswith('.wav')]
    
    # Definición de la carpeta a almacenar
    filepath_to_save = f'{filepath}/{wavelet}/Sum Wavelet vs audio'
    
    # Preguntar si es que la carpeta que almacenará las imágenes se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(filepath_to_save):
        os.makedirs(filepath_to_save)
    
    for i in tqdm(filenames, desc='Wavelets', ncols=70):
        # print(f"Plotting wavelet sum of {i}...")
        # Cargando los archivos
        audio_file, samplerate = sf.read(f'{filepath}/{i}')
        
        # Definición del texto bandpassed
        bandpassed_txt = ''
        
        # Aplicación de filtro pasa banda si es que se define una lista de frecuencias
        if freqs_bp:
            try:
                audio_to_wav = bandpass_filter(audio_file, samplerate, 
                                               freq_stop_1=freqs_bp[0], 
                                               freq_pass_1=freqs_bp[1],
                                               freq_pass_2=freqs_bp[2], 
                                               freq_stop_2=freqs_bp[3], 
                                               bp_method='scipy_fir', 
                                               lp_method='fir', hp_method='fir', 
                                               lp_process='manual_time_design',
                                               fir_method='kaiser', gpass=gpass, gstop=gstop, 
                                               plot_filter=False, correct_by_gd=True, 
                                               gd_padding='periodic', normalize=True)
                
                # Definición de la dirección dónde se almacenará la imagen
                filesave = f'{filepath_to_save}/Bandpassed {freqs_bp} {i.strip(".wav")} '\
                           f'Levels {levels_to_get}.png'
                
                # Redefinición del texto bandpassed
                bandpassed_txt = f' bp {freqs_bp}'

            except:
                raise Exception('Frecuencias de pasa banda no están bien definidas. Por favor, ' 
                                'intente nuevamente.')
        
        else:
            # Definición de la dirección dónde se almacenará la imagen
            filesave = f'{filepath_to_save}/{i.strip(".wav")} Levels {levels_to_get}.png'
            
            # Definición del archivo a procesar
            audio_to_wav = audio_file
        
        # Obteniendo los wavelets de interés (upsampleados)
        wavelets = \
            get_upsampled_thresholded_wavelets(audio_to_wav, samplerate, 
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
        wavelet_sum = sum(wavelets)
        
        # Normalizando
        wavelet_sum = wavelet_sum / max(abs(wavelet_sum))
        
        # Obteniendo los puntos donde hay un sonido cardíaco
        heart_sound_pos = get_zero_points(abs(wavelet_sum), complement=True, to_return='center')
        
        # Finalmente, graficando
        plt.figure(figsize=(13,9))
        
        plt.subplot(2,1,1)
        #plt.plot(heart_sound_pos, [0] * len(heart_sound_pos), 'gx')
        for x_pos in heart_sound_pos:
            plt.axvline(x=x_pos, color='lime')
        plt.plot(audio_to_wav)
        plt.plot(abs(wavelet_sum))
        plt.ylabel('Audio\nOriginal')
        
        plt.subplot(2,1,2)
        plt.plot(wavelet_sum)
        plt.ylabel('Suma\nWavelets')
        
        plt.suptitle(f'Original vs Wavelets {i.strip(".wav")} levels {levels_to_get}'
                     f' {bandpassed_txt}')
        
        # Guardando
        plt.savefig(f'{filesave}')
        
        # Opción mostrar
        if plot_show:
            plt.show()
        
        #print(f"Wavelets of {i} completed!\n")
        # Se cierra el plot
        plt.close()


def get_detection_vs_labels_heartbeats_db(filepath, freq_pass=950, freq_stop=1000,
                                          freqs_bp=[], method='lowpass', lp_method='fir',
                                          fir_method='kaiser', gpass=1, gstop=80,
                                          levels_to_get=[3,4,5],
                                          levels_to_decompose=6, wavelet='db4', 
                                          mode='periodization',
                                          threshold_criteria='hard', threshold_delta='universal',
                                          min_percentage=None, print_delta=False,
                                          plot_show=False, plot_precision_info=True, 
                                          clean_repeated=True, distance_limit=44100,
                                          normalize=True):
    '''Función que permite generar los plots de detección de las señales PCG
    de la base de datos normal del set A en "Heartbeat sounds", incluyendo la
    señal original, la magnitud de la suma de los wavelets en niveles de interés
    y los puntos originales con los detectados por la implementación. También es 
    posible usarla para la base de datos creada en "Database_manufacturing"
    
    Parameters
    ----------
    filepath : str
        Directorio donde se encuenta el set de audios.
    freq_pass : float, optional
        Frecuencia de corte de la pasa banda (en Hz). Por defecto es 950.
    freq_stop : float, optional
        Frecuencia de corte de la rechaza banda (en Hz). Esta es la que se toma en cuenta al
        momento de hacer el último corte (por ende, si busca samplear a 2kHz, seleccione 
        este parámetro en 1kHz). Por defecto es 1000.
    freqs_bp : list, optional
        Frecuencias de la aplicación del filtro pasabanda (en orden). En caso de estar vacía
        o no cumplir con las 4 frecuencias necesarias, no se aplicará filtro. Por defecto es [].
    method : {'lowpass', 'cut', 'resample', 'resample poly'}, optional
        Método utilizado para submuestreo. Para 'lowpass', se aplica un filtro pasabajos 
        para evitar aliasing de la señal, luego se submuestrea. Para 'cut', se corta en la 
        frecuencia de interés. Para 'resample', se aplica la función resample de scipy. Y
        para 'resample_poly', se aplica la función resample_poly de scipy. Por defecto es
        'lowpass'.
    lp_method : {'fir', 'iir'}, optional
        Método de filtrado para elección lowpass. Para 'fir' se implementa un filtro FIR.
        Para 'iir' se implementa un filtro IIR. Por defecto es 'fir'.
    fir_method : {'window', 'kaiser', 'remez'}, optional
        Método de construcción del filtro FIR en caso de seleccionar el método lowpass con 
        filtro FIR. Para 'window', se usa construye por método de la ventana. Para 'kaiser',
        se cosntruye por método de ventana kaiser. Para 'remez', se construye por algoritmo 
        remez. Por defecto se usa 'kaiser'.
    gpass : float, optional
        Ganancia en dB de la magnitud de la pasa banda. Por defecto es 1 (dB).
    gstop : float, optional 
        Ganancia en dB de la magnitud de la rechaza banda. Por defecto es 80 (dB).
    levels_to_get : {'all', list}, optional
        Niveles de los Wavelet a recuperar. Para 'all' se recuperan los "levels_to_decompose" 
        niveles. Además, es posible usar una lista de valores enteros con los niveles de 
        interés. Por defecto es [3, 4, 5].
    levels_to_decompose : int, optional
        Cantidad de niveles en las que se descompondrá la señal. Por defecto es 6.
    wavelet : str, optional
        Wavelet utilizado para el proceso de dwt. Revisar en pywt.families(kind='discrete').
        Por defecto es 'db4' (Daubechies 4).
    mode : str, optional
        Tipo de descomposición en wavelets (revisar wavelets del paquete pywt). Por defecto
        es 'periodization'.
    threshold_criteria : {'hard', 'soft'}, optional
        Criterio de aplicación de umbral, entre "hard" y "soft". Por defecto es 'hard'.
    threshold_delta : {'mad', 'universal', 'sureshrink', 'percentage'} 
        Selección del criterio de cálculo de umbral. Para 'mad' se usa Median Absolute Deviation.
        Para 'universal' se usa criterio universal (internet). Para 'sureshrink' se aplica
        algoritmo SURE. Y para 'percentage', se establece un umbral de porcentaje en relación al 
        máximo. Por defecto es 'universal'.
    min_percentage : int, optional
        Valor del porcentaje con respecto al máximo en la opción "percentage" de la variable 
        "threshold_delta". Por defecto es None.
    print_delta : bool, optional
        Indicar si se imprime el valor de delta (umbral de threshold). Por defecto es False.
    plot_show : bool,optional
        Mostrar gráficos mientras se corre el programa. Por defecto es False.
    plot_precision_info : bool, optional
        Mostrar adicionalmente la información del análisis de precisión en el plot. Por defecto
        es True.
    clean_repeated : bool, optional
        Indica si se hace una limpieza de puntos repetidos en la detección (ver función
        "get_precision_info" en precisions_functions.py). Por defecto es True.
    distance_limit : int, optional
        Umbral del muestras máximo a considerar para un acierto (Por defecto es 44100)
    normalize : bool, optional
        Normalización de la señal. Por defecto es True.
    '''
    # Definición de la carpeta a buscar
    # filepath = 'Heartbeat sounds/Generated/normal_a_labeled'
    # filepath = 'Database_manufacturing/db_HR/Seed-0 - 1_Heart 1_Resp 0_White noise'
    
    # Lista de los sonidos cardíacos a procesar
    filenames = [i for i in os.listdir(filepath) if i.endswith('.wav')]
    
    # Definición de la carpeta a almacenar
    filepath_to_save = f'{filepath}/{wavelet}/Heart sound labels'
    
    # Preguntar si es que la carpeta que almacenará las imágenes se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(filepath_to_save):
        os.makedirs(filepath_to_save)
    
    for audio_name in tqdm(filenames, desc='Sounds', ncols=70):
        # print(f'Plotting heart sound detection of {audio_name}...')
        # Dirección del archivo en la carpeta madre. Este archivo es el que se copiará
        dir_to_copy = f"{filepath}/{audio_name}"
        
        # Lectura del archivo
        audio_file, samplerate = sf.read(dir_to_copy)
        
        # Definición del texto bandpassed
        bandpassed_txt = ''
        
        # Aplicación de filtro pasa banda si es que se define una lista de frecuencias
        if freqs_bp:
            try:
                audio_to_wav = bandpass_filter(audio_file, samplerate, 
                                               freq_stop_1=freqs_bp[0], 
                                               freq_pass_1=freqs_bp[1],
                                               freq_pass_2=freqs_bp[2], 
                                               freq_stop_2=freqs_bp[3], 
                                               bp_method='scipy_fir', 
                                               lp_method='fir', hp_method='fir', 
                                               lp_process='manual_time_design',
                                               fir_method='kaiser', gpass=gpass, gstop=gstop, 
                                               plot_filter=False, correct_by_gd=True, 
                                               gd_padding='periodic', normalize=True)
                
                # Definición de la dirección dónde se almacenará la imagen
                filesave = f'{filepath_to_save}/Bandpassed {freqs_bp} '\
                           f'{audio_name.strip(".wav")} Levels {levels_to_get}'
                
                # Redefinición del texto bandpassed
                bandpassed_txt = f' bp {freqs_bp}'

            except:
                raise Exception('Frecuencias de pasa banda no están bien definidas. Por favor, ' 
                                'intente nuevamente.')
        
        else:
            audio_to_wav = audio_file
            
            # Definición de la dirección dónde se almacenará la imagen
            filesave = f'{filepath_to_save}/{audio_name.strip(".wav")} Levels {levels_to_get}'
        
        # Obteniendo los wavelets de interés (upsampleados)
        wavelets = \
            get_upsampled_thresholded_wavelets(audio_to_wav, samplerate, 
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
        
        # Definición de la suma de wavelets
        sum_wavelets = abs(sum(wavelets))
        
        # Obtenición de los puntos estimados
        detected_points = get_zero_points(sum_wavelets, complement=True, tol=1e-12, 
                                          to_return='center')
        
        # Obtención de los puntos etiquetados
        if 'Database_manufacturing' in filepath:
            # Definición del nombre del archivo de audio del corazón en la base de datos creada
            audio_name_heart = audio_name.split(' ')[2].strip('.wav')
            
            # Buscando
            labeled_points = get_heartbeat_points_created_db(audio_name_heart)
        else:
            # Buscando
            labeled_points = get_heartbeat_points(audio_name)
        
        # Graficando las señales
        plt.figure(figsize=(15,7))
        plt.plot(audio_file)
        plt.plot(sum_wavelets)
        
        if plot_precision_info:
            # Dirección en la cual se almacenará este nuevo archivo
            dir_to_paste = f"{filesave} detection and info.png"
            
            # Obteniendo la información de precisión de la detección realizada
            info = get_precision_info(labeled_points, detected_points, 
                                      clean_repeated=clean_repeated,
                                      distance_limit=distance_limit)
            
            # Y añadiendo las etiquetas de detección (que logran el match)
            plt.plot(info[1][:,0], [0] * len(info[1][:,0]), color='red', 
                     marker='o', ls='', label='Detected')
            plt.plot(info[1][:,1], [0] * len(info[1][:,1]), color='lime', 
                     marker='X', ls='', label='Labels')
            
            # Y las que no logran el match
            plt.plot(info[5], [0] * len(info[5]), color='magenta',
                     marker='o', ls='', label='Detection unlabeled')
            plt.plot(info[4], [0] * len(info[4]), color='cyan',
                     marker='X', ls='', label='Labels undetected')
            
            # Se añade una leyenda al gráfico realizado
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=8)
            
            # Finalmente se añaden tablas resumen
            table_pos = plt.table(cellText=info[1],
                        rowLoc='center', colLoc='center',
                        colColours=['lime', 'red'],
                        colLabels=["Labels pos", "Detected pos"],
                        bbox=[1.01, 0.06, 0.23, 0.94],)
            
            table_und = plt.table(cellText=[[len(info[4]), len(info[5])]],
                                  rowLoc='center', colLoc='center',
                                  colColours=['cyan', 'magenta'],
                                  colLabels=["Labels und", "Detected unl"],
                                  bbox=[1.01, 0, 0.23, 0.05],)
            
            # Se setean sus fuentes
            table_pos.set_fontsize(6)
            table_und.set_fontsize(6)
            
            # Y se ajusta el plot principal para que quepa la tabla
            plt.subplots_adjust(right=0.77)
        
        else:
            # Dirección en la cual se almacenará este nuevo archivo
            dir_to_paste = f"{filesave}.png"
            
            # Graficando las etiquetas no procesadas
            plt.plot(labeled_points, [0] * len(labeled_points), color='lime', 
                    marker='o', linestyle='', label='Etiquetas')
            plt.plot(detected_points, [0] * len(detected_points), color='red', 
                    marker='x', linestyle='', label='Detecciones')
            
            # Se añade una leyenda al gráfico realizado
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=8)
        
        # Agregar titulo
        plt.suptitle(f'Detection points of {audio_name} {bandpassed_txt}')
        
        # Guardando
        plt.savefig(dir_to_paste)
        
        # Mostrando
        if plot_show:
            plt.show()
        #print('Complete!\n')
        plt.close()


# Módulo de testeo

filepath = 'Database_manufacturing/db_HR/Seed-0 - 1_Heart 10_Resp 0_White noise'

filepaths = ['Database_manufacturing/db_HR/Seed-0 - 1_Heart 1_Resp 0_White noise',
             'Database_manufacturing/db_HR/Seed-0 - 1_Heart 2_Resp 0_White noise',
             'Database_manufacturing/db_HR/Seed-0 - 1_Heart 3_Resp 0_White noise',
             'Database_manufacturing/db_HR/Seed-0 - 1_Heart 5_Resp 0_White noise',
             'Database_manufacturing/db_HR/Seed-0 - 1_Heart 10_Resp 0_White noise']

freqs_bp = [50,100,250,300]

for i in filepaths:
    print(i)
    get_detection_vs_labels_heartbeats_db(i, freq_pass=950, freq_stop=1000,
                                        freqs_bp=freqs_bp,
                                        method='lowpass', lp_method='fir',
                                        fir_method='kaiser', gpass=1, gstop=80,
                                        levels_to_get=[3,4,5],
                                        levels_to_decompose=6, wavelet='db4', 
                                        mode='periodization',
                                        threshold_criteria='hard', threshold_delta='universal',
                                        min_percentage=None, print_delta=False,
                                        plot_show=False, plot_precision_info=True, 
                                        clean_repeated=True, normalize=True)



'''
get_wavelets_images_of_heart_sounds(i, freq_pass=950, freq_stop=1000, 
                                        freqs_bp=freqs_bp, method='lowpass', 
                                        lp_method='fir',
                                        fir_method='kaiser', gpass=1, gstop=80,
                                        levels_to_get='all',
                                        levels_to_decompose=6, wavelet='db4', 
                                        mode='periodization',
                                        threshold_criteria='hard', 
                                        threshold_delta='universal',
                                        min_percentage=None, print_delta=False,
                                        normalize=True)

get_sum_wavelets_vs_audio(filepath, freq_pass=950, freq_stop=1000,
                          freqs_bp=freqs_bp,
                          method='lowpass', lp_method='fir',
                          fir_method='kaiser', gpass=1, gstop=80,
                          levels_to_get=[3,4,5],
                          levels_to_decompose=6, wavelet='db4', mode='periodization',
                          threshold_criteria='hard', threshold_delta='universal',
                          min_percentage=None, print_delta=False,
                          normalize=True)


'''


'''
# Nivel de importancia
heart_quality = 4
filepath = f'Interest_Audios/Heart_sound_files/Level {heart_quality}'

get_sum_wavelets_vs_audio(filepath, freq_pass=950, freq_stop=1000,
                          method='lowpass', lp_method='fir',
                          fir_method='kaiser', gpass=1, gstop=80,
                          levels_to_get=[3,4,5],
                          levels_to_decompose=6, wavelet='db4', mode='periodization',
                          threshold_criteria='hard', threshold_delta='universal',
                          min_percentage=None, print_delta=False,
                          normalize=True)


import pywt
wavelets_of_interest = pywt.wavelist(kind='discrete')

# Nivel de importancia
heart_quality = 4
filepath = f'Interest_Audios/Heart_sound_files/Level {heart_quality}'

get_sum_wavelets_vs_audio(filepath, freq_pass=950, freq_stop=1000,
                          method='lowpass', lp_method='fir',
                          fir_method='kaiser', gpass=1, gstop=80,
                          levels_to_get=[4,5],
                          levels_to_decompose=6, wavelet='db4', mode='periodization',
                          threshold_criteria='hard', threshold_delta='universal',
                          min_percentage=None, print_delta=False,
                          normalize=True)'''

'''get_wavelets_images_of_heart_sounds(filepath, freq_pass=950, freq_stop=1000,
                                    method='lowpass', lp_method='fir',
                                    fir_method='kaiser', gpass=1, gstop=80,
                                    levels_to_get='all',
                                    levels_to_decompose=6, wavelet='db4',
                                    mode='periodization',
                                    threshold_criteria='hard',
                                    threshold_delta='universal',
                                    min_percentage=None, print_delta=False,
                                    normalize=True)'''

'''filename = 'Interest_Audios/Heart_sound_files/Level 4/136_1b1_Ar_sc_Meditron'
audio, samplerate = sf.read(f'{filename}.wav')
to_sust = [3,4,5]
wavelets = get_upsampled_thresholded_wavelets(audio, samplerate, freq_pass=950, freq_stop=1000, 
                                     method='lowpass', lp_method='fir', 
                                     fir_method='kaiser', gpass=1, gstop=80, 
                                     plot_filter=False, levels_to_get=to_sust, 
                                     levels_to_decompose=6, wavelet='db4', mode='periodization', 
                                     threshold_criteria='hard', threshold_delta='universal',
                                     min_percentage=None, print_delta=False,
                                     plot_wavelets=False, normalize=True)

wavelet_final = sum(wavelets)
plt.plot(wavelet_final)
plt.show()
'''