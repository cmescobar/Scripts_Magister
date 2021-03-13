import soundfile as sf
import matplotlib.pyplot as plt
from utils import find_and_open_audio, signal_segmentation,\
    find_segments_limits
from source_separation.nmf_decompositions import nmf_to_all, nmf_on_segments, \
    nmf_masked_segments
from heart_sound_segmentation.filter_and_sampling import downsampling_signal


def preprocessing_audio(model_name, lowpass_params, symptom,
                        nmf_parameters, ausc_pos='toracic', 
                        priority=1, nmf_method='masked_segments',
                        plot_segmentation=False,
                        plot_separation=False):
    '''Función que permite hacer un preprocesamiento de la señal
    auscultada en la base de datos de interés.
    
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
    nmf_method : {'to_all', 'on_segments', 'masked_segments'}, optional
        Método de descomposición NMF a aplicar en la separación
        de fuentes. Por defecto es "masked_segments".
    plot_segmentation : bool, optional
        Booleano que indica si es que se grafica el proceso de 
        segmentación. Por defecto es False.
    plot_separation : bool, optional
        Booleano que indica si es que se grafica el proceso de 
        separación de fuentes. Por defecto es False.

    Returns
    -------
    resp_signal : ndarray
        Señal respiratoria obtenida mediante la descomposición.
    heart_signal : ndarray
        Señal cardíaca obtenida mediante la descomposición.
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
    _, audio_dwns = downsampling_signal(audio, samplerate, 
                                        samplerate_des//2-100, 
                                        samplerate_des//2)
    
    # Obteniendo la salida de la red
    _, y_hat_to, (y_out2, _, _) = \
            signal_segmentation(audio, samplerate, model_name,
                                length_desired=len(audio_dwns),
                                lowpass_params=lowpass_params,
                                plot_outputs=False)

    # Definiendo los intervalos para realizar la separación de fuentes
    interval_list = find_segments_limits(y_out2, segments_return='Heart')
    
    # Print de sanidad
    print(f'Aplicando separación de fuentes {nmf_method}...')
    
    
    # Aplicando la separación de fuentes
    if nmf_method == 'to_all':
        (resp_signal, heart_signal), _ = \
            nmf_to_all(audio_dwns, samplerate_des, hs_pos=y_out2, 
                       interval_list=interval_list, 
                       n_components=nmf_parameters['n_components'], 
                       N=nmf_parameters['N'], N_lax=nmf_parameters['N_lax'], 
                       noverlap=nmf_parameters['noverlap'], 
                       repeat=nmf_parameters['repeat'], 
                       padding=nmf_parameters['padding'], 
                       window=nmf_parameters['window'],
                       init=nmf_parameters['init'], 
                       solver=nmf_parameters['solver'], 
                       beta=nmf_parameters['beta'], tol=nmf_parameters['tol'], 
                       max_iter=nmf_parameters['max_iter'],
                       alpha_nmf=nmf_parameters['alpha_nmf'], 
                       l1_ratio=nmf_parameters['l1_ratio'], 
                       random_state=nmf_parameters['random_state'],
                       dec_criteria=nmf_parameters['dec_criteria'])
    
    
    elif nmf_method == 'on_segments':
        resp_signal, heart_signal = \
            nmf_on_segments(audio_dwns, samplerate_des, 
                            interval_list=interval_list, 
                            n_components=nmf_parameters['n_components'],
                            N=nmf_parameters['N'], N_lax=nmf_parameters['N_lax'],  
                            N_fade=nmf_parameters['N_fade'], 
                            noverlap=nmf_parameters['noverlap'], 
                            repeat=nmf_parameters['repeat'], 
                            padding=nmf_parameters['padding'], 
                            window=nmf_parameters['window'],
                            init=nmf_parameters['init'], 
                            solver=nmf_parameters['solver'], 
                            beta=nmf_parameters['beta'], tol=nmf_parameters['tol'], 
                            max_iter=nmf_parameters['max_iter'],
                            alpha_nmf=nmf_parameters['alpha_nmf'], 
                            l1_ratio=nmf_parameters['l1_ratio'], 
                            random_state=nmf_parameters['random_state'],
                            dec_criteria=nmf_parameters['dec_criteria'])
    
    
    elif nmf_method == 'masked_segments':
        (resp_signal, heart_signal), _ = \
            nmf_masked_segments(audio_dwns, samplerate_des, hs_pos=y_out2, 
                                interval_list=interval_list, 
                                n_components=nmf_parameters['n_components'],
                                N=nmf_parameters['N'], N_lax=nmf_parameters['N_lax'],  
                                N_fade=nmf_parameters['N_fade'], 
                                noverlap=nmf_parameters['noverlap'], 
                                repeat=nmf_parameters['repeat'], 
                                padding=nmf_parameters['padding'], 
                                window=nmf_parameters['window'],
                                init=nmf_parameters['init'], 
                                solver=nmf_parameters['solver'], 
                                beta=nmf_parameters['beta'], tol=nmf_parameters['tol'], 
                                max_iter=nmf_parameters['max_iter'],
                                alpha_nmf=nmf_parameters['alpha_nmf'], 
                                l1_ratio=nmf_parameters['l1_ratio'], 
                                random_state=nmf_parameters['random_state'],
                                dec_criteria=nmf_parameters['dec_criteria'])
    
    
    else:
        raise Exception('Opción para el método de descomposición no '
                        'válida.')
    
    
    print('Separación de fuentes completada')
    
    # Graficando la segmentación
    if plot_segmentation:
        audio_data_plot = 0.5 * audio_dwns / max(abs(audio_dwns))
        plt.plot(audio_data_plot - min(audio_data_plot), label=r'$s(n)$', 
                 color='silver', zorder=0)
        plt.plot(y_hat_to[0,:,0], label=r'$S_0$', color='limegreen', zorder=2)
        plt.plot(y_hat_to[0,:,1], label=r'$S_1$', color='red', zorder=1)
        plt.plot(y_hat_to[0,:,2], label=r'$S_2$', color='blue', zorder=1)
        for i in interval_list:
            plt.axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        plt.legend(loc='lower right')
        plt.yticks([0, 0.5, 1])
        plt.ylabel(r'$P(y(n) = k | X)$')
        plt.show()
    
    
    # Graficando la separación de fuentes
    if plot_separation:
        fig, ax = plt.subplots(3, 1, figsize=(15,7), sharex=True)
        
        ax[0].plot(audio_dwns)
        ax[0].set_ylabel('Señal\noriginal')
        for i in interval_list:
            ax[0].axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        
        ax[1].plot(resp_signal)
        ax[1].set_ylabel('Señal\nRespiratoria')
        for i in interval_list:
            ax[1].axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        
        ax[2].plot(heart_signal)
        ax[2].set_ylabel('Señal\nCardiaca')
        for i in interval_list:
            ax[2].axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        
        # Ajustando las etiquetas del eje
        fig.align_ylabels(ax[:])
        # Quitando el espacio entre gráficos
        fig.subplots_adjust(wspace=0.1, hspace=0)

        plt.suptitle('Separación de fuentes')
        plt.show()
    
    
    # Grabando archivos temporales
    sf.write('_temp_files/resp_signal_temp.wav', 
             resp_signal, samplerate=samplerate_des)
    sf.write('_temp_files/heart_signal_temp.wav', 
             heart_signal, samplerate=samplerate_des)
    
    return resp_signal, heart_signal


# Módulo de testeo
if __name__ == '__main__':
    pass
