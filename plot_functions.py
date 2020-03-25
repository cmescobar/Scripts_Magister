from file_management import get_patient_by_symptom, get_dir_audio_by_id
from heart_sound_detection import get_wavelet_levels, get_upsampled_thresholded_wavelets,\
    get_zero_points
from filter_and_sampling import downsampling_signal
import numpy as np
import soundfile as sf
import os
import descriptor_functions as df
import matplotlib.pyplot as plt


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
                                        method='lowpass', lp_method='fir',
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
        print(f"Plotting wavelets of {i}...")
        # Cargando los archivos
        audio_file, samplerate = sf.read(f'{filepath}/{i}')
        
        # Definición de la dirección dónde se almacenará la imagen
        filesave = f'{filepath_to_save}/{i.strip(".wav")} Wavelets.png'
        
        # Aplicando un downsampling a la señal para disminuir la cantidad de puntos a 
        # procesar
        _, dwns_signal = downsampling_signal(audio_file, samplerate, 
                                            freq_pass, freq_stop, 
                                            method=method, 
                                            lp_method=lp_method, 
                                            fir_method=fir_method, 
                                            gpass=gpass, gstop=gstop, 
                                            plot_filter=False, 
                                            normalize=normalize)
        
        # Se obtienen los wavelets que interesan
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
        print(f"Wavelets of {i} completed!\n")


def get_sum_wavelets_vs_audio(filepath, freq_pass=950, freq_stop=1000,
                              method='lowpass', lp_method='fir',
                              fir_method='kaiser', gpass=1, gstop=80,
                              levels_to_get=[3,4,5],
                              levels_to_decompose=6, wavelet='db4', mode='periodization',
                              threshold_criteria='hard', threshold_delta='universal',
                              min_percentage=None, print_delta=False,
                              plot_show=False, normalize=True):
    '''Funnción que permite graficaar la suma de los wavelets de ciertos niveles de interés
    a la tasa de muestreo de la señal original, y la señal de audio en el mismo cuadro
    de modo que pueda realizarse una comparación visual
    
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
    - plot_show: Booleano que sirve para mostrar mientras se corre el programa
                 el gráfico construido
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
        print(f"Plotting wavelet sum of {i}...")
        # Cargando los archivos
        audio_file, samplerate = sf.read(f'{filepath}/{i}')
        
        # Definición de la dirección dónde se almacenará la imagen
        filesave = f'{filepath_to_save}/{i.strip(".wav")} Wavelet_sum levels {levels_to_get}.svg'
        
        # Obteniendo los wavelets de interés (upsampleados)
        wavelets = \
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
        wavelet_sum = sum(wavelets)
        
        # Normalizando
        wavelet_sum = wavelet_sum / max(abs(wavelet_sum))
        
        # Obteniendo los puntos donde hay un sonido cardíaco
        heart_sound_pos = get_zero_points(abs(wavelet_sum), complement=True, to_return='center')
        
        # Finalmente, graficando
        plt.figure(figsize=(13,9))
        
        plt.subplot(2,1,1)
        plt.plot(audio_file)
        plt.plot(abs(wavelet_sum))
        #plt.plot(heart_sound_pos, [0] * len(heart_sound_pos), 'gx')
        for x_pos in heart_sound_pos:
            plt.axvline(x=x_pos, color='lime')
        plt.ylabel('Audio\nOriginal')
        
        plt.subplot(2,1,2)
        plt.plot(wavelet_sum)
        plt.ylabel('Suma\nWavelets')
        
        plt.suptitle(f'Original vs Wavelets {i.strip(".wav")} levels {levels_to_get}')
        
        # Guardando
        plt.savefig(f'{filesave}')
        
        # Opción mostrar
        if plot_show:
            plt.show()
        
        # Se cierra el plot
        plt.close()
        
        print(f"Wavelets of {i} completed!\n")



# Módulo de testeo
'''import pywt
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