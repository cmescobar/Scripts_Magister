import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
from ast import literal_eval
from scipy.io import wavfile, loadmat
from scipy import signal
from descriptor_functions import get_inverse_windowed_signal
from filter_and_sampling import bandpass_filter
from heart_sound_physionet_management import get_windows_and_labels
from envelope_functions import get_envelope_pack, stationary_multiscale_wavelets, \
    stationary_wavelets_decomposition, modified_spectral_tracking, spectral_energy_bands


def get_test_filenames(model_name, db_folder):
    # Definición de los índices de la lista de archivos de salida
    index_list = list()
    
    # Revisión de registro de base de datos usados para testear
    with open(f'Trained_models/{model_name}_db.txt', 'r', encoding='utf8') as file:
        for line in file:
            # Leyendo la línea
            dict_line = literal_eval(line.strip())
            
            # Y agregando a la lista de test de salida
            index_list.extend(dict_line['test_indexes'])
    
    # Una vez definidos los índices se obtienen las direcciones de los archivos de
    # audio
    filenames = [f'{db_folder}/{i[:-4]}' for i in os.listdir(db_folder) 
                 if i.endswith('.wav')]
    
    # Filtrando por los índices de interés
    filenames = [filename for num, filename in enumerate(filenames) if num in index_list]
    
    return filenames


def get_windowed_signal(model_name, filename):
    # Obtener los parámetros utilizados para obtener la señal ventaneada
    # y sus etiquetas
    with open(f'Trained_models/{model_name}-get_model_data_params.txt', 
              'r', encoding='utf8') as file:
        # Definición del diccionario de los parámetros de ventaneo
        data_dict = literal_eval(file.readline().strip())

    # Obteniendo el archivo de audio
    signal_wind, s1_wind, s2_wind = \
        get_windows_and_labels(filename, N=data_dict['N'], 
                               noverlap=data_dict['noverlap'], 
                               padding_value=data_dict['padding_value'], 
                               activation_percentage=0.5, append_audio=True, 
                               append_envelopes=data_dict['append_envelopes'], 
                               apply_bpfilter=data_dict['apply_bpfilter'],
                               bp_parameters=data_dict['bp_parameters'], 
                               apply_noise=False, snr_expected=0, seed_snr=None, 
                               homomorphic_dict=data_dict['homomorphic_dict'], 
                               hilbert_bool=data_dict['hilbert_bool'], 
                               simplicity_dict=data_dict['simplicity_dict'], 
                               vfd_dict=data_dict['vfd_dict'], 
                               wavelet_dict=data_dict['wavelet_dict'], 
                               spec_track_dict=data_dict['spec_track_dict'],
                               append_fft=data_dict['append_fft'])
        
    return signal_wind, s1_wind, s2_wind


def test_heart_sound(model_name, filename, db_folder):
    # Obtención del sonido cardiaco ventaneado y sus etiquetas
    signal_wind, s1_lab, s2_lab = get_windowed_signal(model_name, filename)
    
    # Cargar el modelo de interés
    model = tf.keras.models.load_model(f'Trained_models/{model_name}.h5')
    
    if model_name in ['Model_5_2_3', 'Model_5_2_4', 'Model_5_2_4_1', 'Model_5_2_5', 
                      'Model_5_2_6', 'Model_5_2_7', 'Model_5_2_8', 'Model_5_2_9']:
        # Evaluándolo
        s1_pred, s2_pred = model.predict([signal_wind[:, :, i] 
                                          for i in range(signal_wind.shape[2]) ])
    
    plt.subplot(2,1,1)
    plt.plot(s1_lab)
    plt.plot(s1_pred)
    
    plt.subplot(2,1,2)
    plt.plot(s2_lab)
    plt.plot(s2_pred)
    plt.show()
    
    # print(model.summary())


def test_envelope_images():
    # Obtención de los archivos de testeo
    heart_db = 'PhysioNet 2016 CINC Heart Sound Database'
    # db_ultimate = 'PhysioNet 2016 CINC Heart Sound Database'
    wav_files = [f'{heart_db}/{i}' for i in os.listdir(heart_db) 
                 if i.endswith('.wav')]
    
    # Cargando el archivo de audio
    samplerate, audio = wavfile.read(wav_files[10])

    # Se normaliza
    audio = audio / max(abs(audio))

    # Se filtra
    audio_bp = bandpass_filter(audio, samplerate, 
                            freq_stop_1=20, freq_pass_1=30, 
                            freq_pass_2=180, freq_stop_2=200, 
                            bp_method='scipy_fir', normalize=True)

    # Parámetros
    N_env = 128
    step_env = 16

    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = {'N': N_env, 'noverlap': N_env - step_env, 'kmin': 4, 'kmax': 4, 
                'step_size_method': 'unit', 'inverse': True}
    wavelet_dict = {'wavelet': 'db4', 'levels': [3,4], 'start_level': 1, 'end_level': 4}
    spec_track_dict = {'freq_obj': [100, 150], 'N': N_env, 'noverlap': N_env - step_env, 
                    'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [40, 200], 'alpha': 1, 'N': N_env, 
                        'noverlap': N_env - step_env, 'padding': 0, 'repeat': 0 , 
                        'window': 'hann'}
    append_fft = False
    
    envelopes = get_envelope_pack(audio_bp, samplerate, 
                                  homomorphic_dict=homomorphic_dict,
                                  hilbert_dict=hilbert_dict, 
                                  simplicity_dict=simplicity_dict,
                                  vfd_dict=vfd_dict, wavelet_dict=wavelet_dict,
                                  spec_track_dict=spec_track_dict, 
                                  spec_energy_dict=spec_energy_dict,
                                  norm_type='minmax')
    
    plt.figure(figsize=(9,4))
    plt.plot(audio_bp, label='Señal original')
    plt.plot(envelopes[:,0], label='Filtro homomórfico')
    plt.title(r'Filtro homomórfico con $f_c = 10 Hz$ y $\Delta f = 5$ Hz')
    plt.xlabel('Tiempo [ms]')
    plt.legend(loc='lower right')

    # plt.figure(figsize=(9,4))
    # plt.plot(audio_bp, label='Señal original')
    # plt.plot(abs(hilbert_representation(audio_bp, samplerate)[0]), label=r'$A(t)$')
    # plt.title(r'Magnitud de la transformada de Hilbert')
    # plt.xlabel('Tiempo [ms]')
    # plt.legend(loc='lower right')

    plt.figure(figsize=(9,4))
    plt.plot(audio_bp, label='Señal original')
    plt.plot(envelopes[:,1], label=r'$A(t)$')
    plt.title(r'Magnitud de la transformada de Hilbert modificada')
    plt.xlabel('Tiempo [ms]')
    plt.legend(loc='lower right')

    plt.figure(figsize=(9,4))
    plt.plot(audio_bp, label='Señal original')
    plt.plot(envelopes[:,2], label='VFD')
    plt.title(r'Variance Fractal Dimension')
    plt.xlabel('Tiempo [ms]')
    plt.legend(loc='lower right')

    plt.figure(figsize=(9,4))
    plt.plot(audio_bp, label='Señal original')
    plt.plot(envelopes[:,3], label='MWP')
    plt.title(r'Multiscale Wavelet Product de niveles de detalle 3 y 4, con Wavelet db4')
    plt.xlabel('Tiempo [ms]')
    plt.legend(loc='lower right')

    plt.figure(figsize=(9,4))
    plt.plot(audio_bp, label='Señal original')
    plt.plot(envelopes[:,4], label=r'f = 100 Hz')
    plt.plot(envelopes[:,5], label=r'f = 150 Hz')
    plt.title(r'Envolventes de spectral tracking')
    plt.xlabel('Tiempo [ms]')
    plt.legend(loc='lower right')

    plt.figure(figsize=(9,4))
    plt.plot(audio_bp, label='Señal original')
    plt.plot(envelopes[:,6], label='Energía por bandas')
    plt.title(r'Envolvente de energía por bandas $f \in [40, 200] Hz$')
    plt.xlabel('Tiempo [ms]')
    plt.legend(loc='lower right')
    plt.show()


def test_correlate_functions():
    def _correlation(a, b):
        '''Función de correlación entre 2 series temporales.
        
        Parameters
        ----------
        a , b : ndarray
            Series de entrada.
        
        Returns
        -------
        r : float
            Correlación entre las 2 entradas, dadas por:
            1 / (N - 1) * np.sum((a - mu_a) * (b - mu_b)) / (sig_a * sig_b)
            
        Referencias
        -----------
        [1] https://en.wikipedia.org/wiki/Correlation_and_dependence
        '''
        # Definición de la cantidad de puntos
        N = len(a)
        
        # Cálculo de la media de ambas series
        mu_a = np.mean(a)
        mu_b = np.mean(b)
        
        # Cálculo de la desviación estándar de ambas series
        sig_a = np.std(a)
        sig_b = np.std(b)
        
        # Definición de correlación
        r =  1 / (N - 1) * np.sum((a - mu_a) * (b - mu_b)) / (sig_a * sig_b)
        
        # Propiedad de límite para r    
        r = r if r <= 1.0 else 1.0
        r = r if r >= -1.0 else -1.0

        return r


    def _norm_01(x, resample=False):
        if resample:
            x = signal.resample(x, len(signal_in))
        x = x - min(x)
        return x / max(abs(x))


    def spec_track_test(signal_in, samplerate, spec_track_dict):
        track_list = modified_spectral_tracking(signal_in, samplerate, 
                                                freq_obj=spec_track_dict['freq_obj'], 
                                                N=spec_track_dict['N'], 
                                                noverlap=spec_track_dict['noverlap'], 
                                                padding=spec_track_dict['padding'], 
                                                repeat=spec_track_dict['repeat'], 
                                                window=spec_track_dict['window'])
        # Creación del vector de envolventes
        envelope_out = np.zeros((len(signal_in), 0))    
        
        # Normalizando y concatenando
        for track in track_list:
            # Resampleando
            track_res = get_inverse_windowed_signal(track, N=spec_track_dict['N'], 
                                                    noverlap=spec_track_dict['noverlap'])

            # Recortando para el ajuste con la señal
            N_cut = spec_track_dict['N'] // 2

            # Normalización
            track_norm = norm_func(track_res[N_cut:N_cut + len(signal_in)], 
                                resample=False)

            # Concatenando
            track_norm = np.expand_dims(track_norm, -1)
            envelope_out = np.concatenate((envelope_out, track_norm), axis=1)

        return envelope_out


    def stat_wav_test(signal_in, wavelet_dict):
        wav_mult, _ = \
            stationary_multiscale_wavelets(signal_in, wavelet=wavelet_dict['wavelet'], 
                                        levels=wavelet_dict['levels'], 
                                        start_level=wavelet_dict['start_level'], 
                                        end_level=wavelet_dict['end_level'])
            
        # Normalizando
        wav_mult = norm_func(abs(wav_mult))
        
        return wav_mult


    def wav_test(signal_in, wavelet_dict):
        wav_coeffs = \
            stationary_wavelets_decomposition(signal_in, wavelet=wavelet_dict['wavelet'], 
                                            levels=wavelet_dict['levels'],
                                            start_level=wavelet_dict['start_level'], 
                                            end_level=wavelet_dict['end_level'], 
                                            erase_pad=True)
        
        # Creación del vector de envolventes
        envelope_out = np.zeros((len(signal_in), 0))    
        
        # Normalizando y concatenando
        for i in range(wav_coeffs.shape[1]):
            # Normalización
            wavelet_norm = norm_func(abs(wav_coeffs[:,i]), resample=False)

            # Concatenando
            wavelet_norm = np.expand_dims(wavelet_norm, -1)
            envelope_out = np.concatenate((envelope_out, wavelet_norm), axis=1)
        
        return envelope_out
    
    
    def spec_energy_test(signal_in, samplerate, spec_energy_dict):
        energy_env = spectral_energy_bands(signal_in, samplerate, 
                                        band_limits=spec_energy_dict['band_limits'], 
                                        alpha=spec_energy_dict['alpha'],
                                        N=spec_energy_dict['N'],  
                                        noverlap=spec_energy_dict['noverlap'], 
                                        padding=spec_energy_dict['padding'],
                                        repeat=spec_energy_dict['repeat'], 
                                        window=spec_energy_dict['window'])
            
        # Resampleando
        energy_env_res = \
                    get_inverse_windowed_signal(energy_env, N=spec_energy_dict['N'], 
                                                noverlap=spec_energy_dict['noverlap'])

        # Recortando para el ajuste con la señal
        N_cut = spec_energy_dict['N'] // 2

        # Normalización
        energy_env_norm = norm_func(energy_env_res[N_cut:N_cut + len(signal_in)], 
                                    resample=False)
        
        return energy_env_norm

        
    # Obtención de los archivos de testeo
    heart_db = 'PhysioNet 2016 CINC Heart Sound Database'
    # db_ultimate = 'PhysioNet 2016 CINC Heart Sound Database'
    wav_files = [f'{heart_db}/{i}' for i in os.listdir(heart_db) if i.endswith('.wav')]

    # Definición del diccionario de registros
    reg_dict = defaultdict(list)

    # Parámetros
    N_env = 128
    step_env = 16
    
    # Definición de función de normalización
    norm_func = _norm_01

    for wav_file in wav_files:
        # Cargando el archivo de audio
        samplerate, audio = wavfile.read(wav_file)

        # Se normaliza
        audio = audio / max(abs(audio))

        # Se filtra
        audio_bp = bandpass_filter(audio, samplerate, 
                                freq_stop_1=20, freq_pass_1=30, 
                                freq_pass_2=180, freq_stop_2=200, 
                                bp_method='scipy_fir', normalize=True)
        
        # Obtención del archivo de las etiquetas .mat
        data_info = loadmat(f'{wav_file[:-4]}.mat')
            
        # Etiquetas a 50 Hz de samplerate
        labels = data_info['PCG_states']
        
        # Pasando a 1000 Hz
        labels_adj = np.repeat(labels, 20)
        
        # Recuperación de las etiquetas de S1
        s1_labels = (labels_adj == 1)
        s2_labels = (labels_adj == 3)
        
        # Finalmente...
        heart_labels = s1_labels + s2_labels
        
        
        #### Simulaciones ####
        
        # Multiscale Wavelet Product
        wavelet_dict = {'wavelet': 'db4', 'levels': [1,2,3,4], 'start_level': 0, 'end_level': 4}
        swp_1 = stat_wav_test(audio_bp, wavelet_dict)

        wavelet_dict = {'wavelet': 'db4', 'levels': [2,3,4], 'start_level': 0, 'end_level': 4}
        swp_2 = stat_wav_test(audio_bp, wavelet_dict)

        wavelet_dict = {'wavelet': 'db4', 'levels': [3,4], 'start_level': 0, 'end_level': 4}
        swp_3 = stat_wav_test(audio_bp, wavelet_dict)

        wavelet_dict = {'wavelet': 'db4', 'levels': [2,3], 'start_level': 0, 'end_level': 4}
        swp_4 = stat_wav_test(audio_bp, wavelet_dict)

        wavelet_dict = {'wavelet': 'db4', 'levels': [1,2,3], 'start_level': 0, 'end_level': 4}
        swp_5 = stat_wav_test(audio_bp, wavelet_dict)

        wavelet_dict = {'wavelet': 'db4', 'levels': [3,4,5], 'start_level': 0, 'end_level': 5}
        swp_6 = stat_wav_test(audio_bp, wavelet_dict)
        
        wavelet_dict = {'wavelet': 'db4', 'levels': [4,5], 'start_level': 0, 'end_level': 5}
        swp_7 = stat_wav_test(audio_bp, wavelet_dict)
        
        
        
        # Spectral tracking
        spec_track_dict = {'freq_obj': [30, 40, 50, 60, 80, 90, 100, 120, 150], 'N': N_env, 
                        'noverlap': N_env - step_env, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
        spec_track = spec_track_test(audio_bp, samplerate, spec_track_dict)
        
        
        
        # Wavelets
        wavelet_dict = {'wavelet': 'db4', 'levels': [1,2,3,4,5,6], 'start_level': 0, 'end_level': 6}
        wav_coeffs = wav_test(audio_bp, wavelet_dict)
        
        
        
        # Energy bands
        spec_energy_dict = {'band_limits': [30, 180], 'alpha': 1, 'N': N_env, 
                            'noverlap': N_env - step_env, 'padding': 0, 'repeat': 0 , 
                            'window': 'hann'}
        energy_1 = spec_energy_test(audio_bp, samplerate, spec_energy_dict)

        spec_energy_dict = {'band_limits': [30, 100], 'alpha': 1, 'N': N_env, 
                            'noverlap': N_env - step_env, 'padding': 0, 'repeat': 0 , 
                            'window': 'hann'}
        energy_2 = spec_energy_test(audio_bp, samplerate, spec_energy_dict)

        spec_energy_dict = {'band_limits': [30, 60], 'alpha': 1, 'N': N_env, 
                            'noverlap': N_env - step_env, 'padding': 0, 'repeat': 0 , 
                            'window': 'hann'}
        energy_3 = spec_energy_test(audio_bp, samplerate, spec_energy_dict)
        
        
        
        # Registrando los valores de las correlaciones
        reg_dict['swp_1'].append(_correlation(swp_1, heart_labels))
        reg_dict['swp_2'].append(_correlation(swp_2, heart_labels))
        reg_dict['swp_3'].append(_correlation(swp_3, heart_labels))
        reg_dict['swp_4'].append(_correlation(swp_4, heart_labels))
        reg_dict['swp_5'].append(_correlation(swp_5, heart_labels))
        reg_dict['swp_6'].append(_correlation(swp_6, heart_labels))
        reg_dict['swp_7'].append(_correlation(swp_7, heart_labels))
        
        reg_dict[f'spec_track_{spec_track_dict["freq_obj"][0]}'].append(_correlation(heart_labels, spec_track[:,0]))
        reg_dict[f'spec_track_{spec_track_dict["freq_obj"][1]}'].append(_correlation(heart_labels, spec_track[:,1]))
        reg_dict[f'spec_track_{spec_track_dict["freq_obj"][2]}'].append(_correlation(heart_labels, spec_track[:,2]))
        reg_dict[f'spec_track_{spec_track_dict["freq_obj"][3]}'].append(_correlation(heart_labels, spec_track[:,3]))
        reg_dict[f'spec_track_{spec_track_dict["freq_obj"][4]}'].append(_correlation(heart_labels, spec_track[:,4]))
        reg_dict[f'spec_track_{spec_track_dict["freq_obj"][5]}'].append(_correlation(heart_labels, spec_track[:,5]))
        reg_dict[f'spec_track_{spec_track_dict["freq_obj"][6]}'].append(_correlation(heart_labels, spec_track[:,6]))
        reg_dict[f'spec_track_{spec_track_dict["freq_obj"][7]}'].append(_correlation(heart_labels, spec_track[:,7]))
        reg_dict[f'spec_track_{spec_track_dict["freq_obj"][8]}'].append(_correlation(heart_labels, spec_track[:,8]))
        
        reg_dict['wavelet_1'].append(_correlation(wav_coeffs[:,0], heart_labels))
        reg_dict['wavelet_2'].append(_correlation(wav_coeffs[:,1], heart_labels))
        reg_dict['wavelet_3'].append(_correlation(wav_coeffs[:,2], heart_labels))
        reg_dict['wavelet_4'].append(_correlation(wav_coeffs[:,3], heart_labels))
        reg_dict['wavelet_5'].append(_correlation(wav_coeffs[:,4], heart_labels))
        reg_dict['wavelet_6'].append(_correlation(wav_coeffs[:,5], heart_labels))
                
        reg_dict['energy_1'].append(_correlation(energy_1, heart_labels))
        reg_dict['energy_2'].append(_correlation(energy_2, heart_labels))
        reg_dict['energy_3'].append(_correlation(energy_3, heart_labels))
        
        # Calculando la correlación de pearson
        print(wav_file)
    
    print('swp_1: ', np.mean(reg_dict['swp_1']), '+-', np.std(reg_dict['swp_1']))
    print('swp_2: ', np.mean(reg_dict['swp_2']), '+-', np.std(reg_dict['swp_2']))
    print('swp_3: ', np.mean(reg_dict['swp_3']), '+-', np.std(reg_dict['swp_3']))
    print('swp_4: ', np.mean(reg_dict['swp_4']), '+-', np.std(reg_dict['swp_4']))
    print('swp_5: ', np.mean(reg_dict['swp_5']), '+-', np.std(reg_dict['swp_5']))
    print('swp_6: ', np.mean(reg_dict['swp_6']), '+-', np.std(reg_dict['swp_6']))
    print('swp_7: ', np.mean(reg_dict['swp_7']), '+-', np.std(reg_dict['swp_7']))
    print()

    print(f'spec_track_{spec_track_dict["freq_obj"][0]}: ', 
        np.mean(reg_dict[f'spec_track_{spec_track_dict["freq_obj"][0]}']))
    print(f'spec_track_{spec_track_dict["freq_obj"][1]}: ', 
        np.mean(reg_dict[f'spec_track_{spec_track_dict["freq_obj"][1]}']))
    print(f'spec_track_{spec_track_dict["freq_obj"][2]}: ', 
        np.mean(reg_dict[f'spec_track_{spec_track_dict["freq_obj"][2]}']))
    print(f'spec_track_{spec_track_dict["freq_obj"][3]}: ', 
        np.mean(reg_dict[f'spec_track_{spec_track_dict["freq_obj"][3]}']))
    print(f'spec_track_{spec_track_dict["freq_obj"][4]}: ', 
        np.mean(reg_dict[f'spec_track_{spec_track_dict["freq_obj"][4]}']))
    print(f'spec_track_{spec_track_dict["freq_obj"][5]}: ', 
        np.mean(reg_dict[f'spec_track_{spec_track_dict["freq_obj"][5]}']))
    print(f'spec_track_{spec_track_dict["freq_obj"][6]}: ', 
        np.mean(reg_dict[f'spec_track_{spec_track_dict["freq_obj"][6]}']))
    print(f'spec_track_{spec_track_dict["freq_obj"][7]}: ', 
        np.mean(reg_dict[f'spec_track_{spec_track_dict["freq_obj"][7]}']))
    print()
        
    print('wavelet_1: ', np.mean(reg_dict['wavelet_1']), '+-', np.std(reg_dict['wavelet_1']))
    print('wavelet_2: ', np.mean(reg_dict['wavelet_2']), '+-', np.std(reg_dict['wavelet_2']))
    print('wavelet_3: ', np.mean(reg_dict['wavelet_3']), '+-', np.std(reg_dict['wavelet_3']))
    print('wavelet_4: ', np.mean(reg_dict['wavelet_4']), '+-', np.std(reg_dict['wavelet_4']))
    print('wavelet_5: ', np.mean(reg_dict['wavelet_5']), '+-', np.std(reg_dict['wavelet_5']))
    print('wavelet_6: ', np.mean(reg_dict['wavelet_6']), '+-', np.std(reg_dict['wavelet_6']))
    print()
        
    print('energy_1: ', np.mean(reg_dict['energy_1']), '+-', np.std(reg_dict['energy_1']))
    print('energy_2: ', np.mean(reg_dict['energy_2']), '+-', np.std(reg_dict['energy_2']))
    print('energy_3: ', np.mean(reg_dict['energy_3']), '+-', np.std(reg_dict['energy_3']))


def test_cnn_all_input():
    import os
    import pywt
    import pydot
    import numpy as np
    import soundfile as sf
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from ast import literal_eval
    from scipy import signal, stats
    from scipy.interpolate import interp1d
    from scipy.io import wavfile, loadmat
    from collections import defaultdict
    
    
    def segnet_based_1_1_all(input_shape, padding_value, name=None):
        '''CNN basada en arquitectura encoder-decoder basada en SegNet.
        Se utiliza el mismo canal para todas las envolventes.
        
        Envolventes usadas:
        - Filtros homomórficos
        - Envolventes de Hilbert
        - DWT
        - Energy envolve
        
        Salida de 3 etiquetas:
        - S1
        - S2
        - None
        
        References
        ----------
        [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
            Segnet: A deep convolutional encoder-decoder architecture for 
            image segmentation. IEEE transactions on pattern analysis and 
            machine intelligence, 39(12), 2481-2495.
        [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
            encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
        '''
        def _conv_bn_act_layer(input_layer, filters, kernel_size, padding,
                            kernel_initializer, name):
            '''Función auxiliar que modela las capas azules conv + batchnorm +
            Activation ReLU para realizar el ENCODING.'''
            # Aplicando la concatenación de capas
            x_conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                            kernel_initializer=kernel_initializer,
                                            padding=padding, 
                                            name=f'Conv_{name}')(input_layer)
            x_conv = \
                tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_conv)
            x_conv = \
                tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_conv)

            return x_conv
        
        
        def _encoding_layer(input_layer, n_layers_conv, layer_params):
            '''Función auxiliar que permite modelar "n_layers_conv" capas CNN seguida de 
            una capa de Maxpooling, tal como se puede ver en la figura 2 de [1].  
            '''
            # Definición de la salida de este bloque
            x_enc = input_layer
            
            # Aplicando "n_layers_conv" capas convolucionales de codificación
            for i in range(n_layers_conv):
                x_enc = _conv_bn_act_layer(x_enc, filters=layer_params['filters'], 
                                        kernel_size=layer_params['kernel_size'], 
                                        padding=layer_params['padding'],
                                        kernel_initializer=layer_params['kernel_initializer'], 
                                        name=f"{layer_params['name']}_{i}")

            # Finalmente la capa de MaxPooling
            x_enc = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, 
                                                padding='valid',
                                                name=f"MaxPool_Conv_{layer_params['name']}")(x_enc)
            return x_enc
        
        
        def _decoding_layer(input_layer, n_layers_conv, layer_params):
            '''Función auxiliar que permite modelar una capa de upsampling seguido de 
            "n_layers_conv" capas CNN, tal como se puede ver en la figura 2 de [1].  
            '''
            # Capa de upsampling
            x_dec = tf.keras.layers.UpSampling1D(size=2, name=f"Upsampling_"\
                                                            f"{layer_params['name']}")(input_layer)
            
            # Aplicando "n_layers_conv" capas convolucionales de decodificación
            for i in range(n_layers_conv):
                x_dec = _conv_bn_act_layer(x_dec, filters=layer_params['filters'], 
                                        kernel_size=layer_params['kernel_size'], 
                                        padding=layer_params['padding'],
                                        kernel_initializer=layer_params['kernel_initializer'], 
                                        name=f"{layer_params['name']}_{i}")

            return x_dec
        
        
        # Definición de la entrada
        x_in = tf.keras.Input(shape=input_shape, dtype='float32')

        # Definición de la capa de máscara
        x_masked = tf.keras.layers.Masking(mask_value=padding_value)(x_in)

        ############        Definición de las capas convolucionales        ############
        
        ### Encoding ###
        
        # Primera capa de encoding
        layer_params_1 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': 'enc1'}
        x_enc1 = _encoding_layer(x_masked, n_layers_conv=2, layer_params=layer_params_1)
        
        # Segunda capa de encoding
        layer_params_2 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': 'enc2'}
        x_enc2 = _encoding_layer(x_enc1, n_layers_conv=2, layer_params=layer_params_2)
        
        # Tercera capa de encoding
        layer_params_3 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': 'enc3'}
        x_enc3 = _encoding_layer(x_enc2, n_layers_conv=3, layer_params=layer_params_3)
        
        # Cuarta capa de encoding
        layer_params_4 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': 'enc4'}
        x_enc4 = _encoding_layer(x_enc3, n_layers_conv=3, layer_params=layer_params_4)
        
        
        ### Decoding ###
        
        # Cuarta capa de salida del decoding
        layer_params_4 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': 'dec4'}
        x_dec4 = _decoding_layer(x_enc4, n_layers_conv=3, layer_params=layer_params_4)
        
        # Tercera capa de salida del decoding
        layer_params_3 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': 'dec3'}
        x_dec3 = _decoding_layer(x_dec4, n_layers_conv=3, layer_params=layer_params_3)
        
        # Segunda capa de salida del decoding
        layer_params_2 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': 'dec2'}
        x_dec2 = _decoding_layer(x_dec3, n_layers_conv=2, layer_params=layer_params_2)
        
        # Primera capa de salida del decoding
        layer_params_1 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': 'dec1'}
        x_dec1 = _decoding_layer(x_dec2, n_layers_conv=2, layer_params=layer_params_1)
                                        
        
        # Aplicando reshape
        # x_reshaped = tf.keras.layers.Reshape((input_shape[0], input_shape[1] * 2))(x_dec1)
        
        # Definición de la capa de salida
        x_out = tf.keras.layers.Dense(3, activation='softmax', kernel_initializer='he_normal',
                                    name='softmax_out')(x_dec1)
        
        # Definición del modelo
        model = tf.keras.Model(inputs=x_in, outputs=x_out, name=name)
        
        return model
        
        
    # Definición de la GPU con la que se trabajará
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Obtención de los archivos de testeo
    heart_db = 'PhysioNet 2016 CINC Heart Sound Database'
    # db_ultimate = 'PhysioNet 2016 CINC Heart Sound Database'
    wav_files = [f'{heart_db}/{i}' for i in os.listdir(heart_db) if i.endswith('.wav')]

    # Parámetros de cantidad de audio
    q_audio = 10

    # Definición de los largos de cada ventana
    N_env_vfd = 64
    step_env_vfd = 8
    N_env_spec = 64
    step_env_spec = 8
    N_env_energy = 128
    step_env_energy = 16

    # Definición de los diccionarios
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [3,4], 'start_level': 0, 'end_level': 4}
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}


    # Audios
    audio_list = list()

    for wav_file in tqdm(wav_files[:q_audio], desc='Database', ncols=70):
        # Cargando el archivo de audio
        samplerate, audio = wavfile.read(wav_file)
        
        audio_list.append(audio / max(abs(audio)))
    
    
    # Definición de la carpeta con la base de datos
    db_folder = 'PhysioNet 2016 CINC Heart Sound Database'
    # Obtener todos los archivos .mat de la base de datos
    mat_files = [i for i in os.listdir(db_folder) if i.endswith('.mat')]

    # Definición de las etiquetas a considerar
    s1_labels_list = list()
    s2_labels_list = list()
    s0_labels_list = list() 

    for mat_name in mat_files[:q_audio]:
        # Obtención de los datos del archivo .mat
        data_info = loadmat(f'{db_folder}/{mat_name}')
        
        # Etiquetas a 50 Hz de samplerate
        labels = data_info['PCG_states']
        
        # Pasando a 1000 Hz
        labels_adj = np.repeat(labels, 20)
        
        # Agregando a la lista
        s1_labels_list.append(labels_adj == 1)
        s2_labels_list.append(labels_adj == 3)
        s0_labels_list.append(np.ones(len(labels_adj)) - (labels_adj == 1) - (labels_adj == 3))
        
        
        
    audios_padded = tf.keras.preprocessing.sequence.pad_sequences(audio_list, padding='post', value=2, dtype='float32')
    s1_labels_padded = \
        tf.keras.preprocessing.sequence.pad_sequences(s1_labels_list, padding='post', value=2, dtype='float32')
    s2_labels_padded = \
        tf.keras.preprocessing.sequence.pad_sequences(s2_labels_list, padding='post', value=2, dtype='float32')
    s0_labels_padded = \
        tf.keras.preprocessing.sequence.pad_sequences(s0_labels_list, padding='post', value=2, dtype='float32')

    audios_padded = np.expand_dims(audios_padded, -1)
    y1 = np.expand_dims(s1_labels_padded, -1)
    y2 = np.expand_dims(s2_labels_padded, -1)
    y0 = np.expand_dims(s0_labels_padded, -1)

    y_to = np.concatenate((y0, y1, y2), axis=-1)
        
    model = segnet_based_1_1_all(input_shape=(audios_padded.shape[1], audios_padded.shape[2]), 
                             padding_value=10)
    model.summary()
    
    optimizer = 'Adam'
    loss_func = 'categorical_crossentropy'
    metrics = ['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    loss_weights = None # [1., 1.]
    model.compile(optimizer=optimizer, loss=loss_func,
                metrics=metrics, loss_weights=loss_weights)
    
    model.fit(x=audios_padded, y=y_to, epochs=10, batch_size=70, verbose=1, validation_split=0.1)
    
    


## Módulo de testeo ##
if __name__ == '__main__':
    # Definición de la función a aplciar
    name_func = 'test_model'
    
    
    if name_func == 'test_heart_sound':
        db_folder = 'PhysioNet 2016 CINC Heart Sound Database'
        model_name = 'Model_5_2_9'

        # test_heart_sound(model_name)
        filenames = get_test_filenames(model_name, db_folder)
        test_heart_sound(model_name, filenames[2], db_folder)


    elif name_func == 'test_envelope_images':
        test_envelope_images()
        
        
    elif name_func == 'test_correlate_functions':
        test_correlate_functions()

    elif name_func == 'test_cnn_all_input':
        test_cnn_all_input()
        
    elif name_func == 'test_model':
        model = tf.keras.models.load_model('Paper_models/2 - Descriptors/segnet_based_2_16/segnet_based_2_16.h5')
        print(model.summary())