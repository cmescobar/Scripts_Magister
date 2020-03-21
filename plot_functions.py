from file_management import get_patient_by_symptom, get_dir_audio_by_id
from heart_sound_detection import get_wavelet_levels
from filter_and_sampling import downsampling_signal
import numpy as np
import pywt
import soundfile as sf
import os
import descriptor_functions as df
import matplotlib.pyplot as plt


def get_symptom_images_by_frame(symptom, func_to_apply="normal",
                                display_time=True):
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
        
        
        
def get_wavelets_images_of_heart_sounds(freq_pass=950, freq_stop=1000, 
                                        method='lowpass', lp_method='fir',
                                        fir_method='kaiser', gpass=1, gstop=80,
                                        levels_to_get=[3,4,5],
                                        levels_to_decompose=6, wavelet='db4', 
                                        mode='periodization',
                                        threshold_criteria='hard', 
                                        threshold_delta='universal',
                                        min_percentage=None, print_delta=False,
                                        normalize=True):
    # Dirección de los sonidos cardíacos
    filepath = 'Interest_Audios/Heart_sound_files'
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
        filesave = f'{filepath_to_save}/{i.strip(".wav")}.png'
        
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



# Módulo de testeo
wavelets_of_interest = pywt.wavelist(kind='discrete')
get_wavelets_images_of_heart_sounds(freq_pass=950, freq_stop=1000, 
                                    method='lowpass', lp_method='fir',
                                    fir_method='kaiser', gpass=1, gstop=80,
                                    levels_to_get='all',
                                    levels_to_decompose=6, wavelet='db4', 
                                    mode='periodization',
                                    threshold_criteria='hard', 
                                    threshold_delta='universal',
                                    min_percentage=None, print_delta=False,
                                    normalize=True)