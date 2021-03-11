import os
import matplotlib.pyplot as plt
from heart_sound_segmentation.filter_and_sampling import downsampling_signal
from heart_sound_segmentation.evaluation_functions import eval_sound_model
from utils import signal_segmentation_db, find_and_open_audio, signal_segmentation
from source_separation.source_separation import nmf_applied_masked_segments
from testing_functions import test_hss


def preprocessing_audio(model_name, lowpass_params, symptom, 
                        ausc_pos='toracic', priority=1,
                        plot_segmentation=False):
    '''
    
    Parameters
    ----------
    model_name : str
        Nombre del modelo de la red en la dirección 
        "heart_sound_segmentation/models".
    lowpass_params : dict or None
        Diccionario que contiene la información del filtro pasa 
        bajos en la salida de la red. Si es None, no se utiliza. 
        Por defecto es None.
    symptom : {"Healthy", "Pneumonia"}
        Síntoma a seleccionar.
    ausc_pos : {"toracic", "trachea", "all"}
        Posición de auscultación de los sonidos a estudiar. 
        Por defecto es "toracic".
    priority : {1, 2, 3}
        Prioridad de la base de datos a revisar. Por defecto 
        es 1.
    plot_segmentation : bool, optional
        Booleano que indica si es que se grafica el proceso de 
        segmentación. Por defecto es False.

    '''
    # Definición de la frecuencia de muestreo deseada para 
    # separación de fuentes
    samplerate_des = 11025  # Hz
    
    # Cargando el archivo de audio 
    audio, samplerate = find_and_open_audio(symptom='Healthy', 
                                            ausc_pos='toracic', 
                                            priority=1)
    
    # Realizando un downsampling para obtener la tasa de muestreo
    # fs = 11025 Hz utilizada en la separación de fuentes
    new_rate, audio_dwns = downsampling_signal(audio, samplerate, 
                                               samplerate_des//2-100, 
                                               samplerate_des//2)
        
    # Obteniendo la salida de la red
    _, y_hat_to, (y_out2, _, _) = \
            signal_segmentation(audio, samplerate, model_name,
                                length_desired=len(audio_dwns),
                                lowpass_params=lowpass_params,
                                plot_outputs=False)

    # Aplicando la separación de fuentes
    
    
    
    # Graficando la segmentación
    if plot_segmentation:
        audio_data_plot = 0.5 * audio_dwns / max(abs(audio_dwns))
        plt.plot(audio_data_plot - min(audio_data_plot), label=r'$s(n)$', 
                 color='silver', zorder=0)
        plt.plot(y_hat_to[0,:,0], label=r'$S_0$', color='limegreen', zorder=2)
        plt.plot(y_hat_to[0,:,1], label=r'$S_1$', color='red', zorder=1)
        plt.plot(y_hat_to[0,:,2], label=r'$S_2$', color='blue', zorder=1)
        plt.legend(loc='lower right')
        plt.yticks([0, 0.5, 1])
        plt.ylabel(r'$P(y(n) = k | X)$')
        plt.show()
        
    


if __name__ == '__main__':
    # Definición de la función a revisar
    test_func = 'test_hss'
    
    if test_func == 'test_hss':
        test_hss()
        
    elif test_func == 'preprocessing_audio':
        # Parámetros de la función
        lowpass_params = {'freq_pass': 140, 'freq_stop': 150}
        model_name = 'segnet_based_12_10'
        
        # Parámetros base de datos
        symptom = 'Healthy'
        priority = 1
        ausc_pos='toracic'
        
        # Aplicando la rutina
        preprocessing_audio(model_name, lowpass_params, symptom=symptom, 
                            ausc_pos=ausc_pos, priority=priority,
                            plot_segmentation=True)


