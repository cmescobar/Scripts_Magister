import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from ast import literal_eval
from sklearn.decomposition import NMF
from math_functions import wiener_filter
from filter_and_sampling import resampling_by_points
from descriptor_functions import get_spectrogram, get_inverse_spectrogram


def nmf_decomposition(signal_in, samplerate, n_components=2, N=2048, overlap=0.75,
                      padding=0, window='hamming', whole=False, alpha=1,
                      wiener_filt=True):
    '''Función que permite separar una señal utilizando la descomposición NMF,
    la cual usa como entrada al sistema el espectrograma de la señal de audio.
    
    Parameters
    ----------
    signal_in : ndarray or list
        Señal a descomponer.
    samplerate : int
        Tasa de muestreo de la señal.
    n_components : int, optional
        Cantidad de componentes a separar la señal. Por defecto es 2.
    N : int, optional
        Cantidad de puntos utilizados en cada ventana de la STFT. Por defecto es 2048.
    overlap : float, optional
        Nivel de traslape que existe entre una ventana y la siguiente al calcular la 
        STFT. Por defecto es 0.75 (75%).
    padding : int, optional
        Cantidad de ceros añadidos al final para aplicar zero padding. Por defecto es 0.
    window : {None, 'hamming', 'hann', 'nutall', 'tukey'}, optional
        Opciones para las ventanas a utilizar en el cálculo de cada segmento del STFT.
        En caso de elegir None, se asume la ventana rectangular. Por defecto es 'hamming'.
    whole : bool, optional
        Indica si se retorna todo el espectro de frecuencia de la STFT o solo la mitad 
        (por redundancia). True lo entrega completo, False la mtiad. Por defecto es False.
    alpha : int, optional
        Exponente alpha del filtro de Wiener. Por defecto es 1.
    wiener_filt : bool, optional
        Indica si se aplica el filtro de wiener una vez separado ambas componentes.
        Por defecto es True. 
    
    Returns
    -------
    components : list
        Lista que contiene las componentes en el dominio del tiempo.
    Y_list : list
        Lista que contiene las componentes en espectrogramas.
    X : ndarray
        Magnitud del spectrograma de la señal de entrada (entrada NMF).
    W : ndarray
        Matriz W (plantillas espectrales) de la descomposición NMF.
    H : ndarray
        Matriz H (plantillas temporales) de la descomposición NMF.
    
    References
    ----------
    - Tutorial: https://ccrma.stanford.edu/~njb/teaching/sstutorial/part2.pdf 
    '''
    # Definición de una lista que almacene las componentes
    components = []
    # Listas de valores de interés
    Y_list = []
    
    # Obteniendo el espectrograma
    _, _, S = get_spectrogram(signal_in, samplerate, N=N, padding=padding, 
                              overlap=overlap, window=window, whole=whole)
    
    # Definiendo la magnitud
    X = np.abs(S)
    
    # Definiendo el modelo de NMF
    model = NMF(n_components=n_components)#, beta_loss='itakura-saito', solver='mu')
    
    # Ajustando
    W = model.fit_transform(X)
    H = model.components_
    
    # Se define la función de transformación para Yi
    if wiener_filt:
        filt = lambda source_i: wiener_filter(X, source_i, W, H, alpha=alpha)
    else:
        filt = lambda source_i: source_i
    
    # Obteniendo las fuentes
    for i in range(n_components):
        source_i = np.outer(W[:,i], H[i])
        
        # Aplicando el filtro
        Yi = filt(source_i) * np.exp(1j * np.angle(S))
        
        # Y posteriormente la transformada inversa
        yi = get_inverse_spectrogram(Yi, overlap=overlap, window=window, 
                                     whole=whole)
        
        # Agregando a la lista de componentes
        components.append(np.real(yi))
        Y_list.append(Yi)
        
    return components, Y_list, X, W, H

        
def get_components_HR_sounds(filepath, sep_type='to all', n_components=2, N=2048, overlap=0.75, 
                             padding=0, window='hamming', whole=False, alpha=1, 
                             wiener_filt=True, plot_segments=False):
    # Lista de los archivos de la carpeta
    filenames = [i for i in os.listdir(filepath) if i.endswith('.wav')]
    
    # Definición de la carpeta a almacenar
    if sep_type == 'to_all':
        filepath_to_save = f'{filepath}/Components/Separation to all'
    elif sep_type == 'on segments':
        filepath_to_save = f'{filepath}/Components/Separation on segments'
    else:
        raise Exception('Opción no válida para tipo de separación.')
    
    
    # Preguntar si es que la carpeta que almacenará las imágenes se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(filepath_to_save):
        os.makedirs(filepath_to_save)
    
    for audio_name in tqdm(filenames, desc='NMF decomp', ncols=70):
        # Dirección del archivo en la carpeta madre. Este archivo es el que se descompondrá
        dir_to_dec = f"{filepath}/{audio_name}"
        
        if sep_type == 'to_all':            
            # Lectura del archivo
            audio_file, samplerate = sf.read(dir_to_dec)
            
            # Aplicando la descomposición
            comps, _, _, _, _ = nmf_decomposition(audio_file, samplerate, 
                                                n_components=n_components, N=N, 
                                                overlap=overlap, padding=padding, 
                                                window=window, whole=whole, alpha=alpha,
                                                wiener_filt=wiener_filt)
            
            # Definiendo el nombre de los archivos
            name_to_save = f'{filepath_to_save}/{audio_name.strip(".wav")}'
            
            # Grabando cada componente
            sf.write(f'{name_to_save} Comp 1.wav', comps[0], samplerate)
            sf.write(f'{name_to_save} Comp 2.wav', comps[1], samplerate)
            
        elif sep_type == 'on segments':
            # Dirección del archivo en la carpeta madre. Este archivo es el que se descompondrá
            dir_to_dec = f"{filepath}/{audio_name}"
            
            nmf_applied_interest_segments(dir_to_dec, n_components=n_components, 
                                          N=N, overlap=overlap, padding=padding, 
                                          window=window, whole=whole, alpha=alpha,
                                          wiener_filt=wiener_filt,
                                          plot_segments=plot_segments)


def nmf_applied_interest_segments(dir_file, assign_method='manual', n_components=2, 
                                  N=2048, overlap=0.75, padding=0, window='hamming', 
                                  whole=False, alpha=1, wiener_filt=True, 
                                  plot_segments=False):
    '''Parameters
    ----------
    dir_file : str
        Dirección del archivo de audio a segmentar.
        
    interval_list : list
        Lista de todos los intervalos donde está detectado el sonido cardíaco.
        
    n_components : int, optional
        Cantidad de componentes a separar la señal. Por defecto es 2.
    N : int, optional
        Cantidad de puntos utilizados en cada ventana de la STFT. Por defecto es 2048.
    overlap : float, optional
        Nivel de traslape que existe entre una ventana y la siguiente al calcular la 
        STFT. Por defecto es 0.75 (75%).
    padding : int, optional
        Cantidad de ceros añadidos al final para aplicar zero padding. Por defecto es 0.
    window : {None, 'hamming', 'hann', 'nutall', 'tukey'}, optional
        Opciones para las ventanas a utilizar en el cálculo de cada segmento del STFT.
        En caso de elegir None, se asume la ventana rectangular. Por defecto es 'hamming'.
    whole : bool, optional
        Indica si se retorna todo el espectro de frecuencia de la STFT o solo la mitad 
        (por redundancia). True lo entrega completo, False la mtiad. Por defecto es False.
    alpha : int, optional
        Exponente alpha del filtro de Wiener. Por defecto es 1.
    wiener_filt : bool, optional
        Indica si se aplica el filtro de wiener una vez separado ambas componentes.
        Por defecto es True. 
    
    Returns
    -------
    signal_out : ndarray

    '''
    # Abriendo el archivo de sonido
    signal_in, samplerate = sf.read(f'{dir_file}')
    
    # Definición de la carpeta donde se ubica
    filepath = '/'.join(dir_file.split('/')[:-1])
    
    # Definición del nombre del archivo
    filename = dir_file.strip('.wav').split('/')[-1]
        
    # Definición de la carpeta a guardar los segmentos
    filepath_to_save = f'{filepath}/Components/Separation on segments/{filename}'
    
    # Preguntar si es que la carpeta que almacenará las imágenes se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(filepath_to_save) and plot_segments:
        os.makedirs(filepath_to_save)
    
    # A partir del nombre del archivo es posible obtener también su lista de intervalos.
    ## Primero se obtiene el nombre del sonido cardíaco a revisar
    file_heart = filename.split(' ')[-1].strip('.wav')
    
    ## Luego se define la dirección del archivo de segmentos
    segment_folder = f'Database_manufacturing/db_heart/Manual Combinations'
    
    try:
        ## Y se retorna el archivo de segmentos correspondiente al nombre
        name_file_segment = [i for i in os.listdir(segment_folder) 
                             if i.endswith('.txt') and file_heart in i][0]
        
        ## Se abre el archivo y se obtiene la lista de intervalos
        with open(f'{segment_folder}/{name_file_segment}', 'r', encoding='utf8') as data:
            interval_list = literal_eval(data.readline())
        
    except:
        raise Exception(f'No se logra encontrar el archivo con intervalos cardíacos de '
                        f'{filename}')
    
    # Definición de la señal respiratoria de salida
    respiratory_signal = np.copy(signal_in)
    # Definición de la señal cardíaca de salida
    heart_signal = np.zeros(len(signal_in))
    
    # Método de trabajo
    if assign_method == 'auto':
        # Se obtienen las componetes a partir de las etiquetas generadas
        with open(f'{filepath_to_save}/Heart comp labels.txt', 'r', encoding='utf8') as data:
            comps_choice = literal_eval(data.readline())
    
    elif assign_method == 'manual':
        # Si se usa el modo manual, se crea una lista que vaya guardando las etiquetas
        heart_comp_labels = list()
    
    # Aplicando NMF en cada segmento de interés
    for num, interval in enumerate(interval_list, 1):
        segment = signal_in[interval[0]:interval[1]]
        
        # Aplicando NMF 
        comps, _, _, W, H =nmf_decomposition(segment, samplerate, n_components=n_components, 
                                             N=N, overlap=overlap, padding=padding,
                                             window=window, whole=whole, alpha=alpha,
                                             wiener_filt=wiener_filt)
        
        # Graficos
        if plot_segments:
            # Creación del plot
            plt.figure(figsize=(17,7))
            # Definición del backend para  maximizar la ventana. Dependiendo del SO
            # puede variar. Para revisar usar comando matplotlib.get_backend()
            # plt.switch_backend('TkAgg')
            # manager = plt.get_current_fig_manager()
            # manager.window.state('zoomed')
            
            plt.subplot(1,3,1)
            plt.plot(comps[0], label='Comp 1', )
            plt.plot(comps[1], label='Comp 2')
            plt.plot(signal_in[interval[0]:interval[1]], label='Original')
            plt.legend(loc='upper right')
            plt.title('Señales')
            
            plt.subplot(1,3,2)
            plt.plot(W[:,0], label='Comp 1')
            plt.plot(W[:,1], label='Comp 2')
            plt.legend(loc='upper right')
            plt.xlim([0,100])
            plt.title('Matriz W')
            
            plt.subplot(1,3,3)
            plt.plot(H[0], label='Comp 1')
            plt.plot(H[1], label='Comp 2')
            plt.legend(loc='upper right')
            plt.title('Matriz H')
            
            plt.suptitle(f'Segment #{num}')
            
            if assign_method == 'manual':
                # Se muestra el gráfico
                plt.show()
            
            # Se guarda la figura
            plt.savefig(f'{filepath_to_save}/Segment {num}.png') 
            
            # Se cierra la figura
            plt.close()
            
        if assign_method == 'auto':
            # Se obtiene la decisión etiquetada como sonido cardíaco
            heart_decision = comps_choice[num - 1]
        
        elif assign_method == 'manual':
            # Se pregunta para la decisión del sonido cardíaco
            heart_decision = decision_question_in_comps(n_components)
            
            # Se agrega la decisión a la lista de labels
            heart_comp_labels.append(heart_decision)
        
        # Y se complementa para el sonido respiratorio
        resp_decision = 0 if heart_decision == 1 else 1
        
        # Se redefinen las señales cardíaca
        heart_signal[interval[0]:interval[1]] =comps[heart_decision][:len(segment)]
        # Y respiratoria
        respiratory_signal[interval[0]:interval[1]] = comps[resp_decision][:len(segment)]
        
        
    if assign_method == 'manual' and plot_segments:
        # Finalmente, se escribe las etiquetas de las componentes
        with open(f'{filepath_to_save}/Heart comp labels.txt', 'w', encoding='utf8') as data:
            data.write(str(heart_comp_labels))
    
    return respiratory_signal, heart_signal


def comparison_components_nmf_ground_truth(filepath, sep_type='to all', plot_signals=False):
    '''
    
    Parameters
    ----------
    sep_type : {'to all', 'on segments'}, optional
        Selección del tipo de la base generada por la separación mediante NMF. 'to all' 
        usa la base que separa toda la señal con NMF y 'on_segments' usa la base que
        solamente separa con NMF en segmentos de interés. Por defecto es 'to all'. 
    '''
    # Lista de los archivos de la carpeta
    filenames = [i for i in os.listdir(filepath) if i.endswith('.wav')]
    
    # Definición de la carpeta de componentes
    if sep_type == 'to all':
        filepath_comps = f'{filepath}/Components/Separation to all'
    elif sep_type == 'on segments':
        filepath_comps = f'{filepath}/Components/Separation on segments'
    else:
        raise Exception('Opción no válida para tipo de separación.')
    
    # Definición de carpeta donde se encuentran los archivos de sonido cardíaco
    heart_dir = 'Database_manufacturing/db_heart/Manual combinations'
    
    # Definición de carpeta donde se encuentran los archivos de sonido cardíaco
    resp_dir = 'Database_manufacturing/db_respiratory/Adapted'
    
    for audio_name in tqdm(filenames, desc='Comp. analysis', ncols=70):
        # Separando el nombre para lo que corresponda según la codificación utilizada
        _, resp_name, heart_name = audio_name.strip('.wav').split(' ')
        
        # Obtención del sonido respiratorio
        audio_resp, _ = sf.read(f'{resp_dir}/{resp_name}.wav')
        
        # Obtención del sonido cardíaco
        audio_heart, _ = sf.read(f'{heart_dir}/{heart_name}.wav')
        
        # Obtención de las componentes
        comp_1, _ = sf.read(f'{filepath_comps}/{audio_name.strip(".wav")} Comp 1.wav')
        comp_2, _ = sf.read(f'{filepath_comps}/{audio_name.strip(".wav")} Comp 2.wav')
        
        # Resampleando para igual número de muestras
        comp_1 = resampling_by_points(comp_1, 44100, len(audio_resp), trans_width=50,
                                      resample_method='interp1d', lp_method='fir', 
                                      fir_method='kaiser', gpass=1, gstop=80, 
                                      correct_by_gd=True, gd_padding='periodic',
                                      plot_filter=False, normalize=True)
        comp_2 = resampling_by_points(comp_2, 44100, len(audio_resp), trans_width=50,
                                      resample_method='interp1d', lp_method='fir', 
                                      fir_method='kaiser', gpass=1, gstop=80, 
                                      correct_by_gd=True, gd_padding='periodic',
                                      plot_filter=False, normalize=True)
        
        # Definición Heart comp
        heart_index = np.argmin((sum(abs(audio_heart - comp_1)),
                                 sum(abs(audio_heart - comp_2))))
        
        heart_comp = comp_1 if heart_index == 0 else comp_2
        
        # Definición de señal respiratoria como el complemento del resultado anterior
        resp_comp = comp_2 if heart_index == 0 else comp_1
        
        if plot_signals:
            # Ploteando las diferencias
            plt.figure(figsize=(12,5))
            plt.subplot(2,1,1)
            plt.plot(audio_resp, label='Original')
            plt.plot(resp_comp, label='Componente')
            plt.ylabel('Signals')
            
            plt.subplot(2,1,2)
            plt.plot(abs(audio_resp - resp_comp))
            plt.ylabel('Error')
            
            plt.suptitle(f'{audio_name} Respiratory')
            plt.savefig(f'{filepath_comps}/{audio_name.strip(".wav")} Resp.png')
            plt.close()
            
            plt.figure(figsize=(12,5))
            plt.subplot(2,1,1)
            plt.plot(audio_heart, label='Original')
            plt.plot(heart_comp, label='Componente')
            plt.ylabel('Signals')
            
            plt.subplot(2,1,2)
            plt.plot(abs(audio_heart - heart_comp))
            plt.ylabel('Error')
            
            plt.suptitle(f'{audio_name} Heart')
            plt.savefig(f'{filepath_comps}/{audio_name.strip(".wav")} Heart.png')
            plt.close()


def decision_question_in_comps(n_components):
    '''Función que plantea la pregunta que permite decidir la componente que corresponde al
    sonido cardíaco.
    
    Parameters
    ----------
    n_components : int
        Cantidad de componentes a decidir para la señal.
    
    Returns
    -------
    decision : int
        Índice de la componente de la decisión tomada.
    '''
    while True:
        # Se pregunta
        decision = int(input('Seleccione el componente que corresponde al corazón: ')) - 1
        
        # Se asegura de que la respuesta sea correcta
        if decision in [i for i in range(n_components)]:
            break
        else:
            print('La componente seleccionada no existe. Por favor, seleccione nuevamente.\n')
            
    return decision

# Module testing
filepath = 'Database_manufacturing/db_HR/Source Separation/Seed-0 - 1_Heart 1_Resp 0_White noise'
dir_file = f'{filepath}/HR 122_2b2_Al_mc_LittC2SE Seed[2732]_S1[59]_S2[60].wav'

respiratory_signal, heart_signal =\
    nmf_applied_interest_segments(dir_file, assign_method='auto', n_components=2, 
                                  N=2048, overlap=0.75, padding=0, window='hamming', 
                                  whole=False, alpha=1, wiener_filt=True, 
                                  plot_segments=True)

sf.write(f'{filepath}/Components/Separation on segments/HR 122_2b2_Al_mc_LittC2SE Seed[2732]_S1[59]_S2[60]/Respiratory signal.wav', respiratory_signal, 44100)
sf.write(f'{filepath}/Components/Separation on segments/HR 122_2b2_Al_mc_LittC2SE Seed[2732]_S1[59]_S2[60]/Heart signal.wav', heart_signal, 44100)

#comparison_components_nmf_ground_truth(filepath)

'''
get_components_HR_sounds(filepath, sep_type='on segments', n_components=2, N=2048, overlap=0.75, 
                             padding=0, window='hamming', whole=False, alpha=1, 
                             wiener_filt=True, plot_segments=True)
                             
filename = 'NMF_design/159_1b1_Al_sc_Meditron'
audio, samplerate = sf.read(f'{filename}.wav')

comps, _, X, W, H = nmf_decomposition(audio, samplerate, n_components=2, N=4096, overlap=0.75,
                                       padding=0, window='hamming', whole=False, alpha=1,
                                       wiener_filt=True)


plt.figure(figsize=(9,5))
plt.subplot(2,1,1)
plt.plot(comps[0], label='Comp 1')
plt.plot(comps[1], label='Comp 2')
plt.ylabel('Componentes')
plt.legend()

plt.subplot(2,1,2)
plt.plot(audio)
plt.ylabel('Señal original')
plt.show()
'''