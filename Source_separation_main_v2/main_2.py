import os, gc
import numpy as np
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt
import source_separation.evaluation_metrics as ev_met
from ast import literal_eval
from numba import cuda
from utils import signal_segmentation, find_segments_limits, nmf_process
from process_functions import preprocessing_audio
from heart_sound_segmentation.filter_and_sampling import downsampling_signal, \
    upsampling_signal, lowpass_filter, highpass_filter
from source_separation.nmf_decompositions import nmf_to_all, nmf_on_segments, \
    nmf_masked_segments, nmf_replacing_segments


# Variable global
samplerate_des = 11025
model_name = 'definitive_segnet_based'
lowpass_params = {'freq_pass': 140, 'freq_stop': 150}


# Definiendo la GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_mixed_signal(signal_in_1, signal_in_2, snr_expected,
                     plot_signals=False, print_snr=False, normalize=True):
    '''Función que permite mezclar 2 señales en base a una especificación 
    SNR en decibeles.
    
    Parameters
    ----------
    signal_in : ndarray or list
        Señal de entrada.
    snr_expected : float
        Relación SNR deseada para la relación entre la señal 1 y la 
        señal 2.
    plot_signal : bool, optional
        Booleano para preguntar si es que se grafica la señal original en 
        conjunto con el ruido blanco generado. Por defecto es False.
    normalize : bool, optional
        Booleano para normalizar la señal de salida. Por defecto es True.
        
    Returns
    -------
    signal_out : ndarray
        Señales mezcladas según la relación "snr_expected" en dB.
    '''
    # Largo final de la señal será el largo de la señal más corta
    len_to = min(len(signal_in_1), len(signal_in_2))
    
    # Modificando la señal
    signal_in_1 = signal_in_1[:len_to]
    signal_in_2 = signal_in_2[:len_to]
    
    # Calcular la energía de las señales de entrada
    e_signal_1 = np.sum(signal_in_1 ** 2)
    e_signal_2 = np.sum(signal_in_2 ** 2)
    
    # Calculando el coeficiente necesario para que la energía del ruido
    # cumpla con la SNR especificada
    k = 10 ** (snr_expected / 10)
        
    # Se define pondera la primera señal. A partir de la relación
    # k = e1/e2, se espera que la energía de la señal sea e1 = k * e2.
    # Dado que la energía de la señal 1 es e1, entonces al dividir
    # la energía e1 de la señal por e1, y multiplicando por k * e2 
    # se obtiene la ponderación real. El sqrt(.) es del paso de 
    # energía a señal.
    signal_in_1 = np.sqrt(k * e_signal_2 / e_signal_1) * signal_in_1
    
    # Mostrar el SNR obtenido
    if print_snr:
        snr_obt = 10 * np.log10(np.sum(signal_in_1 ** 2) / \
                                np.sum(signal_in_2 ** 2))
        print(f'SNR obtained = {snr_obt} dB')
    
    # Finalmente se agrega la señal de entrada
    signal_out = signal_in_1 + signal_in_2
    
    # Normalizando
    if normalize:
        signal_out = signal_out / max(abs(signal_out))
        
    # Graficando
    if plot_signals:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(signal_out)
        plt.ylabel('Señal de salida')
        
        plt.subplot(2,1,2)
        plt.plot(signal_in_1)
        plt.plot(signal_in_2)
        plt.ylabel('Señales individuales')
        plt.show()
        
    return signal_out, signal_in_1, signal_in_2


def conditioning_signal(signal_in, samplerate, samplerate_des, bandwidth=100):
    if samplerate > samplerate_des:
        _, signal_out = \
            downsampling_signal(signal_in, samplerate, 
                                freq_pass=samplerate_des//2 - bandwidth, 
                                freq_stop=samplerate_des//2)
    
    elif samplerate < samplerate_des:
        signal_out = \
            upsampling_signal(signal_in, samplerate, samplerate_des)
        
    else:
        signal_out = signal_in
    
    return signal_out


def nmf_process_OLD(signal_in, samplerate, hs_pos, interval_list, nmf_parameters,
                nmf_method='masked_segments'):
    '''Proceso de descomposición NMF y evaluación en comparación con las 
    señales originales.
    
    Parameters
    ----------
    signal_in : ndarray or list
        Señal a descomponer.
    samplerate : int
        Tasa de muestreo de la señal.
    hs_pos : ndarray
        Señal binaria que indica las posiciones de los sonidos cardiacos.
    interval_list : list
        Lista con los intervalos donde se encuentran los sonidos cardiacos.
    nmf_parameters : dict
        Diccionario que contiene los parámetros de las funciones de 
        descomposición NMF.
    nmf_method : {'to_all', 'on_segments', 'masked_segments'}, optional
        Método de descomposición NMF a aplicar en la separación
        de fuentes. Por defecto es "masked_segments".
    
    Returns
    -------
    resp_signal : ndarray
        Señal respiratoria obtenida mediante la descomposición.
    heart_signal : ndarray
        Señal cardíaca obtenida mediante la descomposición.
    '''
    # Aplicando la separación de fuentes
    if nmf_method == 'to_all':
        (resp_signal, heart_signal), _ = \
            nmf_to_all(signal_in, samplerate, hs_pos=hs_pos, 
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
            nmf_on_segments(signal_in, samplerate, interval_list=interval_list, 
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
            nmf_masked_segments(signal_in, samplerate, hs_pos=hs_pos, 
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
    
    return resp_signal, heart_signal


def preprocess_study(signal_index, nmf_parameters, snr_expected, filter_parameters, 
                     nmf_method='masked_segments', print_metrics=False,
                     base_factor= 0.05, N_wind=100, plot_segmentation=False, 
                     plot_separation=False):
    def _adjust_lenghts(signal_1, signal_2):
        # Asegurando de que los largos de las señales sean los mismos
        len_max = max(len(signal_1), len(signal_2))
        
        if len(signal_1) != len_max:
            signal_1 = np.concatenate((signal_1, [0] * (len_max - len(signal_1))))
        elif len(signal_2) != len_max:
            signal_2 = np.concatenate((signal_2, [0] * (len_max - len(signal_2))))
            
        return signal_1, signal_2
    
    
    ######### Import de archivos de audio #########
    
    # A partir de este índice, obtener el nombre del sonido cardiaco
    for i in zip(heart_filenames, resp_filenames):
        if int(i[0].split(' ')[0]) == signal_index:
            heart_name = i[0]
            resp_name = i[1]
            break
    
    # Cargar archivo de sonido respiratorio
    resp_signal, resp_sr = sf.read(f'{db_resp}/{resp_name}')
    # Cargar archivo de sonido cardiaco
    heart_signal, heart_sr = sf.read(f'{db_heart}/{heart_name}') 
    
    # Acondicionando el sonido respiratorio y cardiaco
    resp_to = conditioning_signal(resp_signal, resp_sr, samplerate_des, 
                                  bandwidth=100)
    heart_to = conditioning_signal(heart_signal, heart_sr, samplerate_des, 
                                   bandwidth=100)
       
    # Realizando la mezcla, solo para cortar en caso de que sea necesario
    mixed_to, resp_to, heart_to = \
            get_mixed_signal(resp_to, heart_to, snr_expected=snr_expected, 
                             plot_signals=False, print_snr=True, 
                             normalize=False)
    
    print(resp_name, heart_name)
    sf.write(f'_beta_cardiorespiratory_database/Heart_Resp_Sounds/{signal_index}.wav',
             mixed_to, samplerate_des)
    
    ######### Posición de los sonidos cardiacos #########
    
    # Clasificando
    _, y_hat_to, (y_out2, _, _) = \
        signal_segmentation(heart_to, samplerate_des, model_name,
                            length_desired=len(mixed_to), 
                            lowpass_params=lowpass_params, 
                            plot_outputs=False)
    
    # Definiendo la lista de intervalos
    interval_list = find_segments_limits(y_out2, segments_return='Heart')
    
    if plot_segmentation:
        fig, ax = plt.subplots(2, 1, figsize=(9,5), sharex=True)
        mixed_to_plot = mixed_to / max(abs(mixed_to))
        audio_data_plot = 0.5 * mixed_to_plot / max(abs(mixed_to_plot))
        ax[0].plot(audio_data_plot - min(audio_data_plot), label=r'$s(n)$', 
                 color='silver', zorder=0)
        ax[0].plot(y_hat_to[0,:,0], label=r'$S_0$', color='limegreen', zorder=2)
        ax[0].plot(y_hat_to[0,:,1], label=r'$S_1$', color='red', zorder=1)
        ax[0].plot(y_hat_to[0,:,2], label=r'$S_2$', color='blue', zorder=1)
        ax[0].legend(loc='lower right')
        ax[0].set_yticks([0, 0.5, 1])
        ax[0].set_ylabel(r'$P(y(n) = k | X)$')

        ax[1].plot(y_out2)

        plt.show()
        
    # Suprimir ruido en las bandas que no interesan. Se obtiene una representación
    # suavizada para el fade
    conv_sign = np.convolve(y_out2, np.hamming(N_wind), 'same')
    conv_sign = conv_sign / max(abs(conv_sign)) * (1 - base_factor) + base_factor
    
    # Redefiniendo
    heart_to = heart_to * conv_sign
    mixed_to = resp_to + heart_to
    
    # Interactivo
    # fig, ax = plt.subplots(2,1, figsize=(7,7), sharex=True)
    # ax[0].plot(heart_to, label='Señal cardiaca')
    # ax[0].plot(conv_sign)
    
    # ax[1].plot(heart_to * conv_sign)
    # ax[1].set_xlim([0, 14000])
    # plt.show()
    
    # sf.write('_temp/heart.wav', heart_to / max(abs(heart_to)), samplerate_des)
    # sf.write('_temp/mixed_to.wav', mixed_to / max(abs(mixed_to)), samplerate_des)

    ######### Separación de fuentes #########
    # Separando
    resp_pred, heart_pred = \
                nmf_process(mixed_to, samplerate_des, hs_pos=y_out2, 
                            interval_list=interval_list, 
                            nmf_parameters=nmf_parameters,
                            filter_parameters=filter_parameters, 
                            nmf_method=nmf_method)
    
    if plot_separation:
        fig, ax = plt.subplots(3, 1, figsize=(15,7), sharex=True)

        ax[0].plot(mixed_to)
        ax[0].set_ylabel('Señal\noriginal')
        for i in interval_list:
            ax[0].axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)

        ax[1].plot(resp_to, color='C0', linewidth=2)
        ax[1].plot(resp_pred, color='C1')
        ax[1].set_ylabel('Señal\nRespiratoria')
        for i in interval_list:
            ax[1].axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)

        ax[2].plot(heart_to, color='C0', linewidth=2)
        ax[2].plot(heart_pred, color='C1')
        ax[2].set_ylabel('Señal\nCardiaca')
        for i in interval_list:
            ax[2].axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)

        # Ajustando las etiquetas del eje
        fig.align_ylabels(ax[:])
        # Quitando el espacio entre gráficos
        fig.subplots_adjust(wspace=0.1, hspace=0)
        
        plt.suptitle('Separación de fuentes')
        plt.show()
    
    # plt.figure(figsize=(15,7))
    # plt.subplot(2,1,1)
    # plt.plot(mixed_to, linewidth=3)
    # plt.plot(resp_pred + heart_pred)
    # for i in interval_list:
    #     plt.axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        
    # plt.subplot(2,1,2)
    # plt.plot(abs(mixed_to - (resp_pred + heart_pred)))
    # plt.show()
    
    ######### Análisis de la separación #########
    
    # Ajustando los largos
    resp_to, resp_pred = _adjust_lenghts(resp_to, resp_pred)
    
    # Calculando las métricas
    temp_corr = ev_met.get_correlation(resp_to, resp_pred)
    spec_corr = ev_met.psd_correlation(resp_to, resp_pred, samplerate_des, 
                                       window='hann',
                                       N=nmf_parameters['N'], 
                                       noverlap=nmf_parameters['noverlap'])
    mse = ev_met.MSE(resp_to, resp_pred)
    sdr = ev_met.SDR(resp_to, resp_pred)
    
    if print_metrics:
        print(f'Temp_correlation = {temp_corr}')
        print(f'Spec_correlation = {spec_corr}')
        print(f'MSE = {mse}')
        print(f'SDR = {sdr}')

    return ((resp_to, resp_pred), (heart_to, heart_pred), mixed_to,
            (temp_corr, spec_corr, mse, sdr))


def preprocess_study_regbased(signal_index, nmf_parameters, snr_expected, filter_parameters, 
                              nmf_method='masked_segments', print_metrics=False,
                              base_factor=0.05, N_wind=100, plot_separation=False,
                              N_expand=30):
    def _adjust_lenghts(signal_1, signal_2):
        # Asegurando de que los largos de las señales sean los mismos
        len_max = max(len(signal_1), len(signal_2))
        
        if len(signal_1) != len_max:
            signal_1 = np.concatenate((signal_1, [0] * (len_max - len(signal_1))))
        elif len(signal_2) != len_max:
            signal_2 = np.concatenate((signal_2, [0] * (len_max - len(signal_2))))
            
        return signal_1, signal_2
    
    
    def _interval_to_signal(_signal_in, interval_list, N_exp):
        # Definición del heart_out de salida
        y_out = np.zeros(len(_signal_in))
        interval_new = list()
                
        # Intervalo
        for interval in interval_list:
            # Definición de los límites
            lower = interval[0] - N_exp
            upper = interval[1] + N_exp
            
            # Condiciones de borde
            if lower <= 0:
                lower = 0
            if upper > len(_signal_in):
                upper = len(_signal_in) - 1
            
            # Definiendo los límites
            y_out[lower:upper] = 1
            # Lista
            interval_new.append([lower, upper])
            
        return y_out, interval_new
    
    
    ######### Import de archivos de audio #########
    
    # A partir de este índice, obtener el nombre del sonido cardiaco
    for i in zip(heart_filenames, resp_filenames):
        if int(i[0].split(' ')[0]) == signal_index and \
           int(i[1].split(' ')[0]) == signal_index:
            heart_name = i[0]
            resp_name = i[1]
            break
        
    print(heart_name, resp_name)
    
    # Cargar archivo de sonido respiratorio
    resp_signal, resp_sr = sf.read(f'{db_resp}/{resp_name}')
    # Cargar archivo de sonido cardiaco
    heart_signal, heart_sr = sf.read(f'{db_heart}/{heart_name}') 
    
    # Acondicionando el sonido respiratorio y cardiaco
    resp_to = conditioning_signal(resp_signal, resp_sr, samplerate_des, 
                                  bandwidth=100)
    heart_to = conditioning_signal(heart_signal, heart_sr, samplerate_des, 
                                   bandwidth=100)
       
    # Realizando la mezcla, solo para cortar en caso de que sea necesario
    mixed_to, resp_to, heart_to = \
            get_mixed_signal(resp_to, heart_to, snr_expected=snr_expected, 
                             plot_signals=False, print_snr=True, 
                             normalize=False)
    
    
    ######### Posición de los sonidos cardiacos #########
    
    # Lecutra de la base de datos
    with open(f'{db_heart}/Interval_list_corrected.txt', 'r', encoding='utf8') as file:
        # Lectura del diccionario
        dict_to_rev = literal_eval(file.readline().strip())
    
    # Definición de la lista de intervalos
    interval_list = dict_to_rev[signal_index]
    
    # Definiendo la salida binaria
    y_out2, interval_list = _interval_to_signal(heart_to, interval_list,
                                                N_exp=N_expand)
    
    
    # Suprimir ruido en las bandas que no interesan. Se obtiene una representación
    # suavizada para el fade
    conv_sign = np.convolve(y_out2, np.hamming(N_wind), 'same')
    conv_sign = conv_sign / max(abs(conv_sign)) * (1 - base_factor) + base_factor
    
    # Redefiniendo
    heart_to = heart_to * conv_sign
    mixed_to = resp_to + heart_to
    
    # y_out2_original, _a = _interval_to_signal(heart_to, interval_list, N_exp=0)
    
    # print(_a)
    # plt.plot(heart_to)
    # plt.plot(y_out2_original)
    # plt.plot(y_out2)
    # plt.show()
    
    # Interactivo
    # fig, ax = plt.subplots(2,1, figsize=(7,7), sharex=True)
    # ax[0].plot(heart_to, label='Señal cardiaca')
    # ax[0].plot(conv_sign)
    
    # ax[1].plot(heart_to * conv_sign)
    # ax[1].set_xlim([0, 14000])
    # plt.show()
    
    # sf.write('_temp/heart.wav', heart_to / max(abs(heart_to)), samplerate_des)
    # sf.write('_temp/mixed_to.wav', mixed_to / max(abs(mixed_to)), samplerate_des)

    ######### Separación de fuentes #########
    # Separando
    resp_pred, heart_pred = \
                nmf_process(mixed_to, samplerate_des, hs_pos=y_out2, 
                            interval_list=interval_list, 
                            nmf_parameters=nmf_parameters,
                            filter_parameters=filter_parameters, 
                            nmf_method=nmf_method)
    
    if plot_separation:
        fig, ax = plt.subplots(3, 1, figsize=(15,7), sharex=True)

        ax[0].plot(mixed_to)
        # ax[0].set_ylabel('Señal\noriginal')
        ax[0].set_ylabel('Original\nsignal')
        for i in interval_list:
            ax[0].axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        ax[0].set_ylim([-1.2, 1.2])

        # ax[1].plot(resp_to, color='C0', linewidth=2)
        ax[1].plot(resp_pred, color='C0')
        # ax[1].set_ylabel('Señal\nRespiratoria')
        ax[1].set_ylabel('Lung\nsignal')
        for i in interval_list:
            ax[1].axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        ax[1].set_ylim([-1.2, 1.2])

        # ax[2].plot(heart_to, color='C0', linewidth=2)
        ax[2].plot(heart_pred, color='C0')
        # ax[2].set_ylabel('Señal\nCardiaca')
        ax[2].set_ylabel('Heart\nsignal')
        for i in interval_list:
            ax[2].axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        ax[2].set_ylim([-1.2, 1.2])

        # Ajustando las etiquetas del eje
        fig.align_ylabels(ax[:])
        # Quitando el espacio entre gráficos
        fig.subplots_adjust(wspace=0.1, hspace=0)
        
        ax[2].set_xlim([0, 35000])
        ax[2].set_xlabel('Samples')
        # ax[2].set_xlabel('Muestras')
        
        # plt.savefig('Images/Original_signal_and_components_SPA.pdf', transparent=True)
        plt.savefig('Images/Original_signal_and_components.pdf', transparent=True)
        # plt.suptitle('Separación de fuentes')
        plt.suptitle('Source separation')
        plt.show()
    
    # plt.figure(figsize=(15,7))
    # plt.subplot(2,2,1)
    # plt.plot(mixed_to, linewidth=3)
    # plt.plot(resp_pred + heart_pred)
    # for i in interval_list:
    #     plt.axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        
    # plt.subplot(2,2,3)
    # plt.plot(abs(mixed_to - (resp_pred + heart_pred)))
    # for i in interval_list:
    #     plt.axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
    
    # plt.subplot(2,2,2)
    # plt.plot(resp_to)
    # plt.plot(resp_pred)
    
    # plt.subplot(2,2,4)
    # plt.plot(abs(resp_to - resp_pred))
    
    # plt.show()
    
    ######### Análisis de la separación #########
    
    # Ajustando los largos
    resp_to, resp_pred = _adjust_lenghts(resp_to, resp_pred)
    
    # Calculando las métricas
    temp_corr = ev_met.get_correlation(resp_to, resp_pred)
    spec_corr = ev_met.psd_correlation(resp_to, resp_pred, samplerate_des, 
                                       window='hann',
                                       N=nmf_parameters['N'], 
                                       noverlap=nmf_parameters['noverlap'])
    mse = ev_met.MSE(resp_to, resp_pred)
    sdr = ev_met.SDR(resp_to, resp_pred)
    
    if print_metrics:
        print(f'Temp_correlation = {temp_corr}')
        print(f'Spec_correlation = {spec_corr}')
        print(f'MSE = {mse}')
        print(f'SDR = {sdr}')

    return ((resp_to, resp_pred), (heart_to, heart_pred), mixed_to,
            (temp_corr, spec_corr, mse, sdr))


def eval_db(nmf_method, snr_expected, nmf_parameters, filter_parameters, 
            file_mode='a', base_factor=0.05, N_wind=100, save_res=True):
    #### Rutina ####
    
    # Definición de las listas que almacenarán los resultados
    temp_list = list()
    spec_list = list()
    mse_list = list()
    sdr_list = list()
    
    for signal_index in range(1, 13):
        # Aplicando
        print(f'Señal con índice {signal_index}')
        _, _, _, metrics = \
            preprocess_study(signal_index, nmf_parameters, snr_expected,
                             filter_parameters=filter_parameters,
                             nmf_method=nmf_method, plot_segmentation=False, 
                             plot_separation=False, base_factor= base_factor,
                             N_wind=N_wind)
        
        # Agregando las métricas a cada lista
        temp_list.append(metrics[0])
        spec_list.append(metrics[1])
        mse_list.append(metrics[2])
        sdr_list.append(metrics[3])
    
    print()
    print(f'Results {nmf_method}')
    print('---------------------')
    print(f'Temporal correlation = {round(np.mean(temp_list), 4)} $\\pm$ {round(np.std(temp_list), 4)}')
    print(f'Spectral correlation = {round(np.mean(spec_list), 4)} $\\pm$ {round(np.std(spec_list), 4)}')
    print(f'MSE = {round(np.mean(mse_list), 4)} $\\pm$ {round(np.std(mse_list), 4)}')
    print(f'SDR = {round(np.mean(sdr_list), 4)} $\\pm$ {round(np.std(sdr_list), 4)}')
    
    # Agregando los resultados al diccionario
    nmf_parameters['Temp'] = temp_list
    nmf_parameters['Spec'] = spec_list
    nmf_parameters['MSE'] = mse_list
    nmf_parameters['SDR'] = sdr_list
    
    if save_res:
        with open(f'Results/Register_{nmf_method}.txt', file_mode, encoding='utf8') as file:
            file.write(f'{nmf_parameters}\n')


def eval_db_regbased(nmf_method, snr_expected, nmf_parameters, filter_parameters, 
                     file_mode='a', base_factor=0.05, N_wind=100, save_res=True,
                     plot_separation=False):
    #### Rutina ####
    
    # Definición de las listas que almacenarán los resultados
    temp_list = list()
    spec_list = list()
    mse_list = list()
    sdr_list = list()
    
    for signal_index in range(1, 13):
        # Aplicando
        print(f'Señal con índice {signal_index}')
        _, _, _, metrics = \
            preprocess_study_regbased(signal_index, nmf_parameters, snr_expected,
                                      filter_parameters=filter_parameters,
                                      nmf_method=nmf_method, base_factor= base_factor,  
                                      plot_separation=plot_separation, 
                                      N_wind=N_wind)
        
        # Agregando las métricas a cada lista
        temp_list.append(metrics[0])
        spec_list.append(metrics[1])
        mse_list.append(metrics[2])
        sdr_list.append(metrics[3])
    
    print()
    print(f'Results {nmf_method}')
    print('---------------------')
    print(f'Temporal correlation = {round(np.mean(temp_list), 4)} $\\pm$ {round(np.std(temp_list), 4)}')
    print(f'Spectral correlation = {round(np.mean(spec_list), 4)} $\\pm$ {round(np.std(spec_list), 4)}')
    print(f'MSE = {round(np.mean(mse_list), 4)} $\\pm$ {round(np.std(mse_list), 4)}')
    print(f'SDR = {round(np.mean(sdr_list), 4)} $\\pm$ {round(np.std(sdr_list), 4)}')
    
    # Agregando los resultados al diccionario
    nmf_parameters['Temp'] = temp_list
    nmf_parameters['Spec'] = spec_list
    nmf_parameters['MSE'] = mse_list
    nmf_parameters['SDR'] = sdr_list
    
    if save_res:
        with open(f'Results/Register_{nmf_method}.txt', file_mode, encoding='utf8') as file:
            file.write(f'{nmf_parameters}\n')
  

def simulations_control(nmf_method, nmf_parameters):
    # Definición de un booleano que indique si se realiza o no
    to_do = True
    
    try:
        # Consultando el archivo de interés de las simulaciones
        with open(f'Results/Register_{nmf_method}.txt', 'r', encoding='utf8') as file:
            
            for num, line in enumerate(file):
                # Definición del diccionario en la línea
                dict_to_rev = literal_eval(line.strip())
                
                # Si es que los parámetros de interésdel diccionario a revisar coinciden, 
                # entonces se informa, y se rompe el loop
                if (nmf_parameters['beta'] == dict_to_rev['beta'] and
                    nmf_parameters['n_components'] == dict_to_rev['n_components'] and
                    nmf_parameters['N'] == dict_to_rev['N'] and
                    nmf_parameters['noverlap'] == dict_to_rev['noverlap'] and
                    nmf_parameters['dec_criteria'] == dict_to_rev['dec_criteria'] and
                    nmf_parameters['filter_parameters'] == dict_to_rev['filter_parameters']):
                    
                    print(f'Simulación registrada en la línea {num + 1} del archivo.')
                    to_do = False
                    break
    
    except:
        print('El archivo no ha sido creado todavía. Trabajando...\n')
    
    return to_do



if __name__ == '__main__':
    # Definición de la dirección de la base de datos de sonidos resp
    db_resp =  '_beta_cardiorespiratory_database/Resp_Sounds'
    # Definición de la dirección de la base de datos de sonidos cardiacos
    db_heart = '_beta_cardiorespiratory_database/Heart_Sounds'

    # Archivos .wav de la carpeta
    resp_filenames = [i for i in os.listdir(db_resp) if i.endswith('.wav')]
    heart_filenames = [i for i in os.listdir(db_heart) if i.endswith('.wav')]

    # Ordenando
    resp_filenames.sort()
    heart_filenames.sort()
    
    # Definición de la rutina a ejecutar
    func_to = 'complete_regbased'
    
    for i in range(len(resp_filenames)):
        print(resp_filenames[i], heart_filenames[i])
    
    
    if func_to == 'complete':
        #### Parámetros ####
        # Beta
        # N
        # noverlap
        # n_comps
        
        snr_expected = -10
        filter_parameters = {'bool': True , 'freq_pass': 980, 'freq_stop': 1000}
        nmf_parameters = {'n_components': 5, 'N': 1024, 'N_lax': 100, 
                        'N_fade': 100, 'noverlap': 768, 'repeat': 0, 
                        'padding': 0, 'window': 'hamming', 'init': 'random',
                        'solver': 'mu', 'beta': 1, 'tol': 1e-4, 
                        'max_iter': 200, 'alpha_nmf': 0, 'l1_ratio': 0, 
                        'random_state': 0, 'dec_criteria': 'energy_criterion',
                        'filter_parameters': filter_parameters}
        
            
        nmf_method = 'to_all'
        eval_db(nmf_method, snr_expected, nmf_parameters, 
                filter_parameters=filter_parameters,
                file_mode='a', base_factor=0, N_wind=100,
                save_res=False)
    
    
    elif func_to == 'complete_regbased':
        #### Parámetros ####
        # Beta
        # N
        # noverlap
        # n_comps
        
        snr_expected = 0
        filter_parameters = {'bool': False , 'freq_pass': 980, 'freq_stop': 1000}
        nmf_parameters = {'n_components': 10, 'N': 1024, 'N_lax': 100, 
                        'N_fade': 100, 'noverlap': 768, 'repeat': 0, 
                        'padding': 0, 'window': 'hamming', 'init': 'random',
                        'solver': 'mu', 'beta': 1, 'tol': 1e-4, 
                        'max_iter': 200, 'alpha_nmf': 0, 'l1_ratio': 0, 
                        'random_state': 0, 'dec_criteria': 'vote',
                        'filter_parameters': filter_parameters}
        
            
        nmf_method = 'replace_segments'
        eval_db_regbased(nmf_method, snr_expected, nmf_parameters, 
                         filter_parameters=filter_parameters,
                         file_mode='a', base_factor=0, N_wind=100,
                         save_res=False, plot_separation=True)
    
     
    elif func_to == 'single':
        filter_parameters = {'bool': True , 'freq_pass': 980, 'freq_stop': 1000}
        nmf_parameters = {'n_components': 5, 'N': 1024, 'N_lax': 100, 
                          'N_fade': 100, 'noverlap': 768, 'repeat': 0, 
                          'padding': 0, 'window': 'hamming', 'init': 'random',
                          'solver': 'mu', 'beta': 1, 'tol': 1e-4, 
                          'max_iter': 200, 'alpha_nmf': 0, 'l1_ratio': 0, 
                          'random_state': 0, 'dec_criteria': 'vote'}

        signal_index = 7
        a = preprocess_study(signal_index, nmf_parameters, 
                             snr_expected=0, filter_parameters=filter_parameters,
                             nmf_method='to_all', plot_segmentation=True,
                             plot_separation=True, base_factor=0, N_wind=100)
    
    
    elif func_to == 'single_regbased':
        filter_parameters = {'bool': False , 'freq_pass': 980, 'freq_stop': 1000}
        nmf_parameters = {'n_components': 10, 'N': 2048, 'N_lax': 100, 
                          'N_fade': 100, 'noverlap': int(0.9 * 2048), 'repeat': 0, 
                          'padding': 0, 'window': 'hamming', 'init': 'random',
                          'solver': 'mu', 'beta': 1, 'tol': 1e-4, 
                          'max_iter': 500, 'alpha_nmf': 0, 'l1_ratio': 0, 
                          'random_state': 0, 'dec_criteria': 'temp_criterion'}

        signal_index = 10
        a = preprocess_study_regbased(signal_index, nmf_parameters, 
                                      snr_expected=0, filter_parameters=filter_parameters, 
                                      nmf_method='replace_segments', plot_separation=True, 
                                      base_factor=0, N_wind=100)
    

    elif func_to == 'iterations':
        # Definición del tipo de NMF a realizar
        nmf_method = 'to_all'
        snr_expected = 0
        base_factor = 0
        N_wind = 100
        
        # Tuplas de opciones
        dec_crit_opts = ('vote', 'spec_criterion', 'energy_criterion', 'temp_criterion')
        bool_filt_opts = (False, True)
        beta_opts = (1, 2)
        n_comps_opts = (2, 3, 5, 7, 10, 15, 20, 30, 50)
        N_opts = (512, 1024, 2048, 4096)
        
        # Mensaje informativo
        print(f'Corriendo {nmf_method}...')
        # Contador de simulaciones
        cont = 1
        
        for bool_filt in bool_filt_opts:
            for dec_crit in dec_crit_opts:
                for beta in beta_opts:
                    for n_comps in n_comps_opts:
                        for N in N_opts:
                            # Definición de las opciones de noverlap
                            noverlap_opts = [int(0.5 * N), int(0.75 * N), int(0.9 * N)]
                            for noverlap in noverlap_opts:
                                # Diccionario de filtro sobre la señal descomposición
                                filter_parameters = {'bool': bool_filt , 'freq_pass': 980, 'freq_stop': 1000}
                                
                                # Definición del diccionario
                                nmf_parameters = {'n_components': n_comps, 'N': N, 'N_lax': 100, 
                                                  'N_fade': 100, 'noverlap': noverlap, 'repeat': 0, 
                                                  'padding': 0, 'window': 'hamming', 'init': 'random',
                                                  'solver': 'mu', 'beta': beta, 'tol': 1e-4, 
                                                  'max_iter': 500, 'alpha_nmf': 0, 'l1_ratio': 0, 
                                                  'random_state': 0, 'dec_criteria': dec_crit, 
                                                  'filter_parameters': filter_parameters}

                                # Control de simulaciones
                                to_sim = simulations_control(nmf_method, nmf_parameters)
                                
                                # Si es que no está registrado, se realiza
                                if to_sim:
                                    # Realizando la evaluación
                                    eval_db(nmf_method, snr_expected, nmf_parameters, 
                                            filter_parameters=filter_parameters, file_mode='a',
                                            base_factor=base_factor, N_wind=N_wind)
                                    print(f'Simulación #{cont}\n')
                                    cont += 1
                                                        
                                    # Eliminando las variables registradas que no se referencian en memoria
                                    print("\nRecolectando registros de memoria sin uso...")
                                    n = gc.collect()
                                    print("Número de objetos inalcanzables recolectados por el GC:", n)
                                    print("Basura incoleccionable:", gc.garbage)
                                    
                                    # Liberando memoria de la GPU
                                    print('Liberando memoria de GPU...\n')
                                    tf.keras.backend.clear_session()

    
    elif func_to == 'iterations_regbased':
        # Definición del tipo de NMF a realizar
        nmf_method = 'replace_segments'
        snr_expected = 0
        base_factor = 0.05
        N_wind = 100
        
        # Tuplas de opciones
        dec_crit_opts = ('vote', 'spec_criterion', 'energy_criterion', 'temp_criterion')
        bool_filt_opts = [False] # (False, True)
        beta_opts = (1, 2)
        n_comps_opts = (2, 3, 5, 7, 10, 15, 20, 30)
        N_opts = (512, 1024, 2048, 4096)
        
        # Mensaje informativo
        print(f'Corriendo {nmf_method}...')
        # Contador de simulaciones
        cont = 1
        
        for bool_filt in bool_filt_opts:
            for dec_crit in dec_crit_opts:
                for beta in beta_opts:
                    for n_comps in n_comps_opts:
                        for N in N_opts:
                            # Definición de las opciones de noverlap
                            noverlap_opts = [int(0.5 * N), int(0.75 * N), int(0.9 * N)]
                            for noverlap in noverlap_opts:
                                # Diccionario de filtro sobre la señal descomposición
                                filter_parameters = {'bool': bool_filt , 'freq_pass': 980, 'freq_stop': 1000}
                                
                                print(f'Simulación # {cont}\n')
                                print('-----------------')
                                
                                # Definición del diccionario
                                nmf_parameters = {'n_components': n_comps, 'N': N, 'N_lax': 100, 
                                                  'N_fade': 100, 'noverlap': noverlap, 'repeat': 0, 
                                                  'padding': 0, 'window': 'hamming', 'init': 'random',
                                                  'solver': 'mu', 'beta': beta, 'tol': 1e-4, 
                                                  'max_iter': 500, 'alpha_nmf': 0, 'l1_ratio': 0, 
                                                  'random_state': 0, 'dec_criteria': dec_crit, 
                                                  'filter_parameters': filter_parameters}

                                # Control de simulaciones
                                to_sim = simulations_control(nmf_method, nmf_parameters)
                                
                                # Si es que no está registrado, se realiza
                                if to_sim:
                                    # Realizando la evaluación
                                    eval_db_regbased(nmf_method, snr_expected, nmf_parameters, 
                                                     filter_parameters=filter_parameters, file_mode='a',
                                                     base_factor=base_factor, N_wind=N_wind, 
                                                     plot_separation=False)
                                                        
                                    # Eliminando las variables registradas que no se referencian en memoria
                                    print("\nRecolectando registros de memoria sin uso...")
                                    n = gc.collect()
                                    print("Número de objetos inalcanzables recolectados por el GC:", n)
                                    print("Basura incoleccionable:", gc.garbage)
                                    
                                    # Liberando memoria de la GPU
                                    print('Liberando memoria de GPU...\n')
                                    tf.keras.backend.clear_session()
                                
                                print(f'Fin de simulación # {cont}\n')
                                cont += 1
