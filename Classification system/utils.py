import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from heart_sound_segmentation.filter_and_sampling import downsampling_signal
from heart_sound_segmentation.evaluation_functions import eval_sound_model, \
    eval_sound_model_db, class_representations


def signal_segmentation_db(symptom, model_name, ausc_pos='toracic', priority=1,
                           lowpass_params=None, plot_outputs=False):
    '''Función que segmenta un archivo de audio disponible en la
    base de datos de sonidos cardiorrespiratorios en base a síntoma,
    posición de auscultación, prioridad y nombre de la red.
    
    Parameters
    ----------
    symptom : {"Healthy", "Pneumonia"}
        Síntoma a seleccionar.
    model_name : str
        Nombre del modelo de la red en la dirección 
        "heart_sound_segmentation/models".
    ausc_pos : {"toracic", "trachea", "all"}
        Posición de auscultación de los sonidos a estudiar. 
        Por defecto es "toracic".
    priority : {1, 2, 3}
        Prioridad de la base de datos a revisar. Por defecto
        es 1.
    lowpass_params : dict or None
        Diccionario que contiene la información del filtro pasa 
        bajos en la salida de la red. Si es None, no se utiliza. 
        Por defecto es None.
    plot_outputs : bool
        Booleano para indicar si se realizan gráficos. Por defecto es 
        False.
        
    Returns
    -------
    y_hat : ndarray
        Salidas de la red indicando la probabilidad de ocurrencia 
        de cada clase.
    y_out3 : ndarray
        Salida de la red indicando las 3 posibles clases.
    y_out4 : ndarray
        Salida de la red indicando las 4 posibles clases.
    '''
    # Función auxiliar para desplegar los archivos a seleccionar
    def _file_selection(filenames):
        print('Seleccione el archivo que desea descomponer:')
        for num, name in enumerate(filenames):
            print(f'[{num + 1}] {name}')
            
        # Definición de la selección
        selection = int(input('Selección: '))
        
        # Se retorna
        try:
            return filenames[selection-1].strip('.wav')
        except:
            raise Exception('No ha seleccionado un archivo válido.')
    
    
    # Criterios de control
    if symptom not in ["Healthy", "Pneumonia"]:
        raise Exception('Error al elegir el síntoma.')
    
    if priority not in [1, 2, 3]:
        raise Exception('Error al elegir la prioridad.')
    
    # Carpeta de ubicación de la base de datos
    db_folder = 'cardiorespiratory_database'
        
    # Definición de la carpeta a revisar
    filepath = f'{db_folder}/{symptom}/{ausc_pos}/Priority_{priority}'
    
    # Definición de la dirección del modelo a utilizar
    model_to = f'heart_sound_segmentation/models/{model_name}.h5'
    
    # Definición del archivo a revisar
    filenames = [i for i in os.listdir(filepath) if i.endswith('.wav')]
    
    # Obtención del archivo
    filename = f'{filepath}/{_file_selection(filenames)}'
    
    # Salida de la red
    y_hat, y_out3, y_out4 = \
                    eval_sound_model_db(filename, model_to,
                                        lowpass_params=lowpass_params,
                                        plot_outputs=plot_outputs)
    
    return y_hat, y_out3, y_out4


def find_and_open_audio(symptom, ausc_pos='toracic', priority=1):
    '''Función que permite la apertura de archivos de audio en la base
    de datos de la carpeta "cardiorespiratory_database".
    
    Parameters
    ----------
    symptom : {"Healthy", "Pneumonia"}
        Síntoma a seleccionar.
    ausc_pos : {"toracic", "trachea", "all"}
        Posición de auscultación de los sonidos a estudiar. 
        Por defecto es "toracic".
    priority : {1, 2, 3}
        Prioridad de la base de datos a revisar. Por defecto
        es 1.
        
    Returns
    -------
    audio : ndarray
        Señal de audio de interés.
    samplerate : int or float
        Tasa de muestreo de la señal.    
    '''
    def _file_selection(filenames):
        print('Seleccione el archivo que desea descomponer:')
        for num, name in enumerate(filenames):
            print(f'[{num + 1}] {name}')
            
        # Definición de la selección
        selection = int(input('Selección: '))
        
        # Se retorna
        try:
            return filenames[selection-1].strip('.wav')
        except:
            raise Exception('No ha seleccionado un archivo válido.')
    
    
    def _open_file(filename):
        # Obtención del archivo de audio .wav
        try:
            samplerate, audio = wavfile.read(f'{filename}.wav')
        except:
            audio, samplerate = sf.read(f'{filename}.wav')
            
        return audio, samplerate
    
    
    # Criterios de control
    if symptom not in ["Healthy", "Pneumonia"]:
        raise Exception('Error al elegir el síntoma.')
    
    if priority not in [1, 2, 3]:
        raise Exception('Error al elegir la prioridad.')
    
    # Carpeta de ubicación de la base de datos
    db_folder = 'cardiorespiratory_database'
        
    # Definición de la carpeta a revisar
    filepath = f'{db_folder}/{symptom}/{ausc_pos}/Priority_{priority}'
    
    # Definición del archivo a revisar
    filenames = [i for i in os.listdir(filepath) if i.endswith('.wav')]

    # Definición de la ubicación del archivo
    filename = f'{filepath}/{_file_selection(filenames)}'
    
    # Retornando
    return _open_file(filename)
    

def signal_segmentation(signal_in, samplerate, model_name,
                        length_desired, lowpass_params=None, 
                        plot_outputs=False):
    '''Función que segmenta un archivo de audio disponible en la
    base de datos de sonidos cardiorrespiratorios en base a síntoma,
    posición de auscultación, prioridad y nombre de la red.
    
    Parameters
    ----------
    signal_in : ndarray
        Señal de entrada.
    samplerate : int or float
        Tasa de muestreo de la señal de entrada.
    model_name : str
        Nombre del modelo de la red en la dirección 
        "heart_sound_segmentation/models".
    length_desired : int or float
        Largo deseado de la señal.
    lowpass_params : dict or None
        Diccionario que contiene la información del filtro pasa 
        bajos en la salida de la red. Si es None, no se utiliza. 
        Por defecto es None.
    plot_outputs : bool
        Booleano para indicar si se realizan gráficos. Por defecto 
        es False.
        
    Returns
    -------
    y_hat : ndarray
        Salidas de la red indicando la probabilidad de ocurrencia 
        de cada clase.
    (y_out2, y_out3, y_out4) : list of ndarray
        Salida de la red indicando las 2, 3 y 4 posibles clases.
    '''
    # Salida de la red
    _audio, y_hat = eval_sound_model(signal_in, samplerate, model_name,
                                     lowpass_params=lowpass_params)
    
    # Definición del largo deseado ajustado a y_hat
    length_desired_to = round(len(y_hat[0,:,0]) / len(_audio) * \
                              length_desired)
    
    # Definición de las probabilidades resampleadas
    y_hat_to = np.zeros((1, length_desired_to, 3))
    
    # Para cada una de las salidas, se aplica un resample
    for i in range(3):
        y_hat_to[0, :, i] = \
            segments_redimension(y_hat[0, :, i], 
                                 length_desired=length_desired_to,
                                 kind='cubic')
    
    # Representación en clases
    y_out2, y_out3, y_out4 = \
        class_representations(y_hat_to, plot_outputs=plot_outputs,
                              audio_data=None)
    
    return y_hat, y_hat_to, (y_out2, y_out3, y_out4)


def segments_redimension(signal_in, length_desired, kind='linear'):
    '''Función que redimensiona la salida y_hat de las redes para
    dejarlo en función de un largo deseado.
    
    Parameters
    ----------
    signal_in : ndarray
        Señal de entrada.
    length_desired : int
        Largo deseado de la señal.
    kind : str
        Opción kind de la función "scipy.interpolate.interp1d".
        Por defecto es "linear".
    
    Returns 
    -------
    signal_out : ndarray
        Señal resampleada.
    '''
    # Definición del eje temporal de la señal
    x = np.linspace(0, length_desired - 1, len(signal_in)) 
    
    # Función de interpolación en base a los datos de la señal
    f = interp1d(x, signal_in, kind=kind)
    
    # Definición de la nueva cantidad de puntos
    x_new = np.arange(length_desired)
        
    # Interpolando finalmente
    return f(x_new)



# Módulo de testeo
if __name__ == '__main__':
    print("Testeo de función en utils\n")
    
    # Definición de la función a testear
    test_func = 'signal_segmentation'
    
    
    # Aplicación de la función
    if test_func == 'signal_segmentation_db':
        # Parámetros del filtro pasa bajos a la salida de la red
        lowpass_params = {'freq_pass': 140, 'freq_stop': 150}
        model_name = 'segnet_based_12_10'
        
        signal_segmentation_db(symptom='Healthy', model_name=model_name, 
                               ausc_pos='toracic', priority=1, 
                               lowpass_params=lowpass_params, 
                               plot_outputs=True)
    
       
    elif test_func == 'resample_example':
        # Archivo de audio
        audio, samplerate = \
            sf.read('cardiorespiratory_database/Healthy/toracic/Priority_1/'
                    '123_1b1_Al_sc_Meditron.wav')
        
        # Downsampling
        new_rate, audio_to = downsampling_signal(audio, samplerate, 
                                                 freq_pass=450, 
                                                 freq_stop=500)
        
        # Interpolando
        x = np.linspace(0, len(audio) - 1, len(audio_to))
        f = interp1d(x, audio_to, kind='cubic')
        x_new = np.arange(len(audio))
        
        print(x_new[-1])
        print(x_new.shape)
        print(len(audio))
        
        
        plt.subplot(2,1,1)
        plt.plot(audio)
        # plt.plot(x, audio_to)
        # plt.plot(f(np.arange(len(audio))))
        
        plt.subplot(2,1,2)
        plt.plot(abs(audio - f(np.arange(len(audio)))))
        plt.show()


    elif test_func == 'find_and_open_audio':
        model_name = 'segnet_based_12_10'
        audio, sr = find_and_open_audio(symptom='Healthy', ausc_pos='toracic', 
                                        priority=1)
        
        print(sr)
        plt.plot(audio)
        plt.show()

    
    elif test_func == 'signal_segmentation':
        # Definición de la frecuencia de muestreo deseada para separación de fuentes
        samplerate_des = 11025  # Hz
        
        # Cargando el archivo de audio 
        audio, samplerate = find_and_open_audio(symptom='Healthy', ausc_pos='toracic', 
                                                priority=1)
        
        # Realizando un downsampling para obtener la tasa de muestreo
        # fs = 11025 Hz utilizada en la separación de fuentes
        new_rate, audio_dwns = downsampling_signal(audio, samplerate, 
                                                   samplerate_des//2-100, 
                                                   samplerate_des//2)
                        
        # Parámetros del filtro pasa bajos a la salida de la red
        lowpass_params = {'freq_pass': 140, 'freq_stop': 150}
        model_name = 'segnet_based_12_10'
        
        # Definición del largo deseado de la salida
        
        # Obteniendo la salida de la red
        y_hat, y_hat_to, (y_out2, y_out3, y_out4) = \
                signal_segmentation(audio, samplerate, model_name,
                                    length_desired=len(audio_dwns),
                                    lowpass_params=lowpass_params,
                                    plot_outputs=False)
        
        # plt.subplot(2,1,1)
        # plt.plot(np.linspace(0, len(y_hat_res) - 1, len(y_hat[0,:,0])), y_hat[0,:,0])
        # plt.subplot(2,1,2)
        # plt.plot(y_hat_res)
        
        # plt.subplot(3,1,1)
        # plt.plot(y_outs[0])
        # plt.subplot(3,1,2)
        # plt.plot(y_outs[1])
        # plt.subplot(3,1,3)
        # plt.plot(y_outs[2])
        
        
        # plt.plot(audio_dwns)
        # plt.plot(np.linspace(0, len(audio_dwns) - 1, len(y_hat[0,:,0])), y_hat[0,:,0])
        # plt.plot(y_out2)
        
        
        # plt.plot(y_hat_to[0,:,0])
        # plt.plot(np.linspace(0, len(y_hat_to[0,:,0]) - 1, len(y_hat[0,:,0])), y_hat[0,:,0])
        
        
        # fig, axs = plt.subplots(2, 1) # , sharex=True)
        # new_rate, audio_data = downsampling_signal(audio, samplerate, 
        #                                              freq_pass=450, 
        #                                              freq_stop=500)
        # audio_data_plot = 0.5 * audio_data / max(abs(audio_data))
        
        # axs[0].plot(np.linspace(0, len(audio_data) - 1, len(audio_data_plot)),
        #             audio_data_plot - min(audio_data_plot), label=r'$s(n)$', 
        #             color='silver', zorder=0)
        # axs[0].plot(y_hat[0,:,0], label=r'$S_0$', color='limegreen', zorder=2)
        # axs[0].plot(y_hat[0,:,1], label=r'$S_1$', color='red', zorder=1)
        # axs[0].plot(y_hat[0,:,2], label= r'$S_2$', color='blue', zorder=1)
        # axs[0].legend(loc='lower right')
        # axs[0].set_yticks([0, 0.5, 1])
        # axs[0].set_ylabel(r'$P(y(n) = k | X)$')
        
        # audio_data_plot = 0.5 * audio_dwns / max(abs(audio_dwns))
        # axs[1].plot(audio_data_plot - min(audio_data_plot), label=r'$s(n)$', 
        #             color='silver', zorder=0)
        # axs[1].plot(# np.linspace(0, len(audio_data) - 1, len(y_hat_to[0,:,0])),
        #             y_hat_to[0,:,0], label=r'$S_0$', color='limegreen', zorder=2)
        # axs[1].plot(# np.linspace(0, len(audio_data) - 1, len(y_hat_to[0,:,1])),
        #             y_hat_to[0,:,1], label=r'$S_1$', color='red', zorder=1)
        # axs[1].plot(# np.linspace(0, len(audio_data) - 1, len(y_hat_to[0,:,2])),
        #             y_hat_to[0,:,2], label=r'$S_2$', color='blue', zorder=1)
        
        # # fig.subplots_adjust(wspace=0.1, hspace=0)
        
        # plt.show()
