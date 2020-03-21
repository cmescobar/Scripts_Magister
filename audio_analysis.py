import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os
from breath_onset_detection import boundary_cycle_method


def put_clicks_respiratory_cycle(file_dir, fmin=80, fmax=1000, k_audio=5, k_click=2.5):
    # Apertura del archivo de audio
    audio_data, samplerate = sf.read(file_dir)
    # Apertura del archivo del click
    audio_click, _ = sf.read('useful_audio/click.wav')

    # Detección de peaks para la señal de audio ingresada. En caso de que no
    # funcione, se registrarán los audios con problema para revisarlos
    # posteriormente 
    try:
        peaks = boundary_cycle_method(audio_data, samplerate, fmin=fmin, 
                                      fmax=fmax, overlap=0)
    except:
        print(f"Error en audio {file_dir}")
        with open('Interest_Audios/error_data.csv', 'a', encoding='utf8') \
            as file:
            file.write(f"{file_dir}\n")
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
    dir_list = file_dir.split('/')
    folder_data = f"{'/'.join(dir_list[:2])}/Segmented_by_resp" 
    
    # Se crea una carpeta en caso de que no exista la carpeta del archivo a
    # guardar
    if not os.path.isdir(folder_data):
        os.mkdir(folder_data)

    # Definiendo el path del archivo de audio
    folder_to_save = f"{folder_data}/{dir_list[-1]}"
    sf.write(folder_to_save, audio_out, samplerate, 'PCM_24')


    '''list_cycles = [41728, 129024, 206592, 296960, 377216, 450432,
                  521856, 608256, 684288, 777600, 872064]
    plt.plot(audio_data)
    plt.plot(peaks, [0 for i in peaks], 'rx')
    plt.plot(list_cycles, [0 for i in list_cycles], 'go')
    
    plt.show()'''


def get_audio_cycles_by_symptom(symptom, fmin=80, fmax=1000):
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
    - k_audio: Parámetro de ponderación relativo para la señal de entrada.
               Tiene relación con el volumen del audio sobre la salida
    - k_click: Parámetro de ponderación relativo para la señal de click.
               Tiene relación con el volumen del click sobre la salida
    - click_windowed: Parámetro que indica si el click es ventaneado
        - [None]: No se ventanea
        - ['hamming']: Se ventanea por una ventana hamming
    - normalize: Booleano para normalización de la señal
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


#put_clicks_respiratory_cycle('Interest_Audios/Healthy/'
#                            '125_1b1_Tc_sc_Meditron.wav')

# get_audio_cycles_by_symptom('Healthy', fmin=120)
