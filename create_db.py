import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from matplotlib.lines import Line2D
from math_functions import raised_cosine_modified
from filter_and_sampling import resampling_by_points


def create_visual_lookups(N_periods, lens_sig, lens_sil):
    '''Función que permite crear una representación temporal de las activaciones
    de las señales de interés
    
    Parameters
    ----------
    N_periods : int
        Cantidad de perídos de activación que tendrá la señal.
    lens_sig : ndarray
        Arreglo de largos que tendrán las señales a replicar.
    lens_sil : ndarray
        Arreglo de largos que tendrán los silencios a replicar.
        
    Returns
    -------
    sawtooth_arr : ndarray
        Señal de diente de sierra de la activación de las señales.
    square_arr : ndarray
        Señal cuadrada de la activación de las señales.
    '''
    # Definición del arreglo donde se almacenará la forma del diente de sierra
    sawtooth_arr = np.array([])
    # Definición del arreglo donde se almacenará la forma de la señal cuadrada
    square_arr = np.array([])
    
    for i in range(N_periods):
        # Creación del vector de valores cero
        zeros_to_append = np.zeros(lens_sil[i])
        
        # Creación del vector de diente de sierra a revisar
        sawtooth_to_append = np.linspace(0, 1, lens_sig[i])
        
        # Creación del vector cuadrado a revisar
        square_to_append = np.ones(lens_sig[i])
        
        # Agregando los valores generados
        sawtooth_arr = np.concatenate((sawtooth_arr, zeros_to_append, 
                                       sawtooth_to_append))
        square_arr = np.concatenate((square_arr, zeros_to_append, 
                                     square_to_append))
    
    
    # Agregando los ceros finales del diente de sierra y cuadrado 
    # respectivamente
    sawtooth_arr = np.concatenate((sawtooth_arr, np.zeros(lens_sil[-1])))
    square_arr = np.concatenate((square_arr, np.zeros(lens_sil[-1])))
    
    return sawtooth_arr, square_arr


def create_signal_lookups(signals_in, N_periods, lens_sig, lens_sil, beta=0.2,
                          trans_width=50, resample_method='interp1d', lp_method='fir', 
                          fir_method='kaiser', gpass=1, gstop=80, 
                          correct_by_gd=True, gd_padding='periodic',
                          plot_filter=False, normalize=True):
    '''Función que permite crear una representación temporal de las activaciones
    de las señales de interés.
    
    Parameters
    ----------
    signals_in : list
        Lista de señales para la generación del sonido cardíaco. El primer elemento [0]
        contiene una lista de los sonidos S1. El segundo elemento [1] contiene una lista
        de los sonidos S2. Los sonidos, a su vez, contienen el audio y la tasa de muestreo.
    N_periods : int
        Cantidad de perídos de activación que tendrá la señal.
    lens_sig : ndarray
        Arreglo de largos que tendrán las señales a replicar.
    lens_sil : ndarray
        Arreglo de largos que tendrán los silencios a replicar.
    beta : float, optional
        Parámetro de la función coseno elevado utilizada para asegurar continuidad en los bordes
        de los sonidos cardíacos. Por defecto es 0.2.
    trans_width : int, optional
        Banda de transición entre la frecuencia de corte de la señal original (que representa 
        la frecuencia de corte del rechaza banda) y la pasa banda del filtro aplicado para 
        eliminar las repeticiones [1]. Por defecto es 50.
    resample_method : {'resample', 'resample poly', 'interp1d', 'stretching'}, optional
        Método usado para resamplear. Para 'resample', se aplica la función resample de scipy.
        Para 'resample_poly', se aplica la función resample_poly de scipy. Para 'interp1d',
        se aplica la función 'interp1d' de scipy. Y para 'stretching' se realiza el 
        estiramiento a la señal por un parámetro "N_st" obtenido automáticamente. Por defecto 
        es 'interp1d'.
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
    correct_by_gd : bool, optional. 
        Booleano que permite corregir la salida del filtro FIR  por su retraso de grupo.
        Por defecto es True.
    gd_padding : {None, 'zero', 'periodic', 'constant'}, optional
        Formato para el padding de la señal de entrada. Si se escoge None, la señal de entrada del
        filtro no es paddeada. Si se escoge "zero", se hace padding con "len(signal_in)" ceros a
        la izquierda y derecha de la señal. Si se escoge "periodic", se hace padding copiando la
        señal 3 veces. Si es "constant", se hace padding copiando el primer y el último valor
        para el caso de la izquierda y la derecha respectivamente. Por defecto es "periodic".
    plot_filter : bool, optional
        Activar ploteo del filtro aplicado. Por defecto es False.
    normalize : bool, optional
        Normalización de la señal. Por defecto es True.
        
    Returns
    -------
    signal_out : ndarray
        Señal con el sonido de interés repetido a intervalos dados por los 
        parámetros.
    '''
    # Definición del arreglo donde se almacenará la señal de salida
    signal_out = np.array([])
    
    # Definición del arreglo donde se almacenarán las etiquetas de cada
    # señal utilizada durante el proceso
    labels_out = list()
    
    for i in range(N_periods):
        # Creación del vector de valores cero
        zeros_to_append = np.zeros(lens_sil[i])
        
        # Sonidos pares son S1, e impares S2
        if i % 2 == 0:
            signal_set = signals_in[0]
        else:
            signal_set = signals_in[1]
            
        # Selección de la señal dentro del set
        signal_in, label = signal_set[np.random.randint(len(signal_set))]
        
        # Agregando la etiqueta a la lista de etiquetas para el registro
        labels_out.append(label)
        
        # Aplicando una ventana que permita eliminar los problemas de borde
        window = raised_cosine_modified(len(signal_in[0]), beta=beta)
        
        # Señal de entrada
        sig_in = signal_in[0] * window

        # Creación del vector de diente de sierra a revisar
        signal_to_append = resampling_by_points(signal_in=sig_in, 
                                                samplerate=signal_in[1], 
                                                N_desired=lens_sig[i], 
                                                trans_width=trans_width, 
                                                resample_method=resample_method, 
                                                lp_method=lp_method, 
                                                fir_method=fir_method, 
                                                gpass=gpass, gstop=gstop, 
                                                correct_by_gd=correct_by_gd, 
                                                gd_padding=gd_padding,
                                                plot_filter=plot_filter, 
                                                normalize=normalize)
        
        # Agregando los valores generados
        signal_out = np.concatenate((signal_out, zeros_to_append, 
                                     signal_to_append))

    # Agregando los ceros finales del diente de sierra y cuadrado 
    # respectivamente
    signal_out = np.concatenate((signal_out, np.zeros(lens_sil[-1])))
    
    return labels_out, signal_out


def lookup_table_function(signals_in, N_periods, params_signal, 
                          params_silence, beta=0.2, gen_type='normal', 
                          trans_width=50, resample_method='interp1d', lp_method='fir', 
                          fir_method='kaiser', gpass=1, gstop=80, 
                          correct_by_gd=True, gd_padding='periodic',
                          plot_filter=False, normalize=True):
    '''Función que permite crear un sonido cardíaco utilizando el método de lookup tables 
    (wave tables) a partir de un set de señales de entrada, creando también las señales de
    referencia que permitirán recorrer el wave table.
    
    Parameters
    ----------
    signals_in : list
        Lista de señales para la generación del sonido cardíaco. El primer elemento [0]
        contiene una lista de los sonidos S1. El segundo elemento [1] contiene una lista
        de los sonidos S2. Los sonidos, a su vez, contienen el audio y la tasa de muestreo.
    N_periods : int
        Cantidad de perídos de activación que tendrá la señal.
    params_signal : list or tuple, optional
        Parámetros que dependerán del formato de generación de números aleatorios elegido. Si es
        "normal", el índice 0 será mu y el 1 será sigma. Si es "uniforme", el índice 0 será el
        mínimo valor del largo posible y el índice 1 será el máximo valor del largo posible.
        Por defecto son: (5000,500).
    params_silence : list or tuple, optional
        Sigue la misma lógica que params_signal, pero aplicado a los intervalos en los que habrá
        silencio. Por defecto son: (15000,2000).
    beta : float, optional
        Parámetro de la función coseno elevado utilizada para asegurar continuidad en los bordes
        de los sonidos cardíacos. Por defecto es 0.2.
    gen_type : {'normal', 'uniform'}, optional
        Método de generación de números aleatorios. Al escoger "normal" se crearán mediante un
        proceso de generación vía dist. normal (0,1). Si se escoge "uniform", se crearán mediante
        un proceso de generación vía distribución uniforme. Por defecto es 'normal'.
    trans_width : int, optional
        Banda de transición entre la frecuencia de corte de la señal original (que representa 
        la frecuencia de corte del rechaza banda) y la pasa banda del filtro aplicado para 
        eliminar las repeticiones [1]. Por defecto es 50.
    resample_method : {'resample', 'resample poly', 'interp1d', 'stretching'}, optional
        Método usado para resamplear. Para 'resample', se aplica la función resample de scipy.
        Para 'resample_poly', se aplica la función resample_poly de scipy. Para 'interp1d',
        se aplica la función 'interp1d' de scipy. Y para 'stretching' se realiza el 
        estiramiento a la señal por un parámetro "N_st" obtenido automáticamente. Por defecto 
        es 'interp1d'.
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
    correct_by_gd : bool, optional. 
        Booleano que permite corregir la salida del filtro FIR  por su retraso de grupo.
        Por defecto es True.
    gd_padding : {None, 'zero', 'periodic', 'constant'}, optional
        Formato para el padding de la señal de entrada. Si se escoge None, la señal de entrada del
        filtro no es paddeada. Si se escoge "zero", se hace padding con "len(signal_in)" ceros a
        la izquierda y derecha de la señal. Si se escoge "periodic", se hace padding copiando la
        señal 3 veces. Si es "constant", se hace padding copiando el primer y el último valor
        para el caso de la izquierda y la derecha respectivamente. Por defecto es "periodic".
    plot_filter : bool, optional
        Activar ploteo del filtro aplicado. Por defecto es False.
    normalize : bool, optional
        Normalización de la señal. Por defecto es True.
    
    Returns
    -------
    tuple
        Donde los argumentos de salida son:
        1. Señal diente de sierra de referencia.
        2. Señal cuadrada de referencia.
        3. Una lista de los intervalos en los cuales se encuentran los sonidos cardíacos.
        4. Señal de sonido cardíaco generada.
    '''
    # Creación de los largos de cada intervalo
    if gen_type == 'normal':
        # Generación de números aleatorios
        rand_sig = np.random.randn(N_periods)
        rand_sil = np.random.randn(N_periods + 1)
        
        # Escalando y redoneando a la escala de interés
        lens_sig = np.rint(params_signal[1] * rand_sig + params_signal[0])
        lens_sil = np.rint(params_silence[1] * rand_sil + params_silence[0])
        
        # Pasando a valores enteros
        lens_sig = lens_sig.astype('int32')
        lens_sil = lens_sil.astype('int32')
        
    elif gen_type == 'uniform':
        # Generación de números aleatorios
        lens_sig = np.random.randint(low=params_signal[0],
                                     high=params_signal[1],
                                     size=N_periods)
        lens_sil = np.random.randint(low=params_silence[0],
                                     high=params_silence[1],
                                     size=N_periods + 1)
    else:
        raise Exception('Opción no válida para parámetro "gen_type". '
                        'Por favor, intente nuevamente con una opción'
                        ' válida.')
    
    # Definición de la lista para almacenar los intervalos de interés donde 
    # habrá sonido cardíaco
    intervals = list()
    
    for i in range(N_periods):
        # Intervalo donde el sonido se hace presente
        to_append = (sum(lens_sil[:i+1]) + sum(lens_sig[:i]), 
                     sum(lens_sil[:i+1]) + sum(lens_sig[:i+1]))
        intervals.append(to_append)
    
    # Creando las señales a rev
    sawtooth_arr, square_arr = create_visual_lookups(N_periods, lens_sig, lens_sil)
    
    # Creando la señal de interés
    signal_info = create_signal_lookups(signals_in, N_periods, lens_sig, lens_sil,
                                        beta=beta, trans_width=trans_width, 
                                        resample_method=resample_method,
                                        lp_method=lp_method, 
                                        fir_method=fir_method, 
                                        gpass=gpass, gstop=gstop, 
                                        correct_by_gd=correct_by_gd, 
                                        gd_padding=gd_padding,
                                        plot_filter=plot_filter, 
                                        normalize=normalize)
    
    return sawtooth_arr, square_arr, intervals, signal_info
    

def create_artificial_heart_sounds(filepath, N_periods, choose=1, seed=0, 
                                   beta=0.2, segments='Good_segments',
                                   params_signal=(5000,500), params_silence=(15000,2000),
                                   s1_selection=None, s2_selection=None, gen_type='normal',
                                   trans_width=50, resample_method='interp1d', lp_method='fir', 
                                   fir_method='kaiser', gpass=1, gstop=80, 
                                   correct_by_gd=True, gd_padding='periodic',
                                   plot_filter=False, plot_show=False,
                                   save_wav=True, normalize_res=False, 
                                   normalize_out=True):
    ''' Función que permite generar un archivo de audio de sonidos cardíacos y su plot 
    correspondientemente etiquetado (en base a los sonidos utilizados, ).
    
    Parameters
    ----------
    filepath : str
        Dirección donde se encuentra la carpeta madre de los archivos.
    N_periods : int
        Cantidad de perídos de activación que tendrá la señal.
    choose : int, optional
        Cantidad de señales base a elegir para el set de creación de sonidos cardíacos. Por
        defecto es 1.
    seed : int, optional
        Valor de la semilla a utilizar para el experimento. Por defecto es 0.
    beta : float, optional
        Parámetro de la función coseno elevado utilizada para asegurar continuidad en los bordes
        de los sonidos cardíacos. Por defecto es 0.2.
    segments : {'Good_segments', 'Heart_segments'}, optional
        Base de datos (carpeta) utilizada para generar los sonidos. Por defecto es 'Good_segments'.
    params_signal : list or tuple, optional
        Parámetros que dependerán del formato de generación de números aleatorios elegido. Si es
        "normal", el índice 0 será mu y el 1 será sigma. Si es "uniforme", el índice 0 será el
        mínimo valor del largo posible y el índice 1 será el máximo valor del largo posible.
        Por defecto son: (5000,500).
    params_silence : list or tuple, optional
        Sigue la misma lógica que params_signal, pero aplicado a los intervalos en los que habrá
        silencio. Por defecto son: (15000,2000).
    s1_selection : Nonetype or int, optional
        Sonido S1 seleccionado para la generación de sonido cardíaco. Por defecto es None.
    s2_selection : Nonetype or int, optional
        Sonido S2 seleccionado para la generación de sonido cardíaco. Por defecto es None.
    gen_type : {'normal', 'uniform'}, optional
        Método de generación de números aleatorios. Al escoger "normal" se crearán mediante un
        proceso de generación vía dist. normal (0,1). Si se escoge "uniform", se crearán mediante
        un proceso de generación vía distribución uniforme. Por defecto es 'normal'.
    trans_width : int, optional
        Banda de transición entre la frecuencia de corte de la señal original (que representa 
        la frecuencia de corte del rechaza banda) y la pasa banda del filtro aplicado para 
        eliminar las repeticiones [1]. Por defecto es 50.
    resample_method : {'resample', 'resample poly', 'interp1d', 'stretching'}, optional
        Método usado para resamplear. Para 'resample', se aplica la función resample de scipy.
        Para 'resample_poly', se aplica la función resample_poly de scipy. Para 'interp1d',
        se aplica la función 'interp1d' de scipy. Y para 'stretching' se realiza el 
        estiramiento a la señal por un parámetro "N_st" obtenido automáticamente. Por defecto 
        es 'interp1d'.
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
    correct_by_gd : bool, optional. 
        Booleano que permite corregir la salida del filtro FIR  por su retraso de grupo.
        Por defecto es True.
    gd_padding : {None, 'zero', 'periodic', 'constant'}, optional
        Formato para el padding de la señal de entrada. Si se escoge None, la señal de entrada del
        filtro no es paddeada. Si se escoge "zero", se hace padding con "len(signal_in)" ceros a
        la izquierda y derecha de la señal. Si se escoge "periodic", se hace padding copiando la
        señal 3 veces. Si es "constant", se hace padding copiando el primer y el último valor
        para el caso de la izquierda y la derecha respectivamente. Por defecto es "periodic".
    plot_filter : bool, optional
        Activar ploteo del filtro aplicado. Por defecto es False.
    plot_show : bool, optional.
        Mostrar el plot. Por defecto es False.
    save_wav : bool, optional
        Grabar el archivo de audio. Por defecto es True.
    normalize_res : bool, optional
        Normalización de la lectura de las señales para generar (resample). Por defecto es False.
    normalize_out : bool, optional
        Normalización de la señal de sonido cardíaco para la salida. Por defecto es True.
    '''
    # Plantando la semilla
    np.random.seed(seed) 
    
    # Definición de lista de sonidos para usar
    s1_to_use = list()
    s2_to_use = list()
    
    # Definición de la carpeta a revisar para los archivos
    filepath_heart_sounds = f'{filepath}/{segments}' 
    
    if (s1_selection is not None) and (s2_selection is not None):
        # Largo de choose
        choose = 1
        
        # Lista con los nombres de los sonidos cardíacos
        heart_sounds = [i for i in os.listdir(filepath_heart_sounds) if i.endswith('.wav')]
        
        # Dado que en este caso se entregan los índices se obtiene el primer y segundo 
        # sonido cardíaco
        s1_choice = [i for i in heart_sounds if i.split('-')[0] == str(s1_selection)][0]
        s2_choice = [i for i in heart_sounds if i.split('-')[0] == str(s2_selection)][0]
        
        # Abriendo los sonidos correspondientes a la elección
        s1 = sf.read(f'{filepath_heart_sounds}/{s1_choice}')
        s2 = sf.read(f'{filepath_heart_sounds}/{s2_choice}')

        # Se agrega a la lista tanto el sonido como la etiqueta
        s1_to_use.append((s1, str(s1_selection)))
        s2_to_use.append((s2, str(s2_selection)))
        
        # Definición del nombre del archivo de salida
        filename = f'Seed[{seed}]_S1[{s1_selection}]_S2[{s2_selection}]'

        # Creación de la carpeta a almacenar la información
        filepath_to_save = f'{filepath}/Manual combinations'
    
    else:
        # Lista con los nombres de los sonidos cardíacos
        heart_sounds = [i for i in os.listdir(filepath_heart_sounds) if i.endswith('.wav')]

        # Separando en primer y segundo sonido cardíaco
        s1_sounds = [i for i in heart_sounds if "S1" in i]
        s2_sounds = [i for i in heart_sounds if "S2" in i]

        # Selección de sonidos a utilizar para crear la señal respiratoria
        s1_selection = np.random.randint(0, len(s1_sounds), size=choose)
        s2_selection = np.random.randint(0, len(s2_sounds), size=choose)

        # Definición de una lista de etiquetas
        s1_labels = list()
        s2_labels = list()
        
        for i in range(choose):
            # Sonido aleatorio elegido
            s1_choice = s1_sounds[s1_selection[i]]
            s2_choice = s2_sounds[s2_selection[i]]

            # Etiqueta de cada sonido
            s1_label = s1_choice.split('-')[0]
            s2_label = s2_choice.split('-')[0]

            # Abriendo los sonidos correspondientes a la elección
            s1 = sf.read(f'{filepath_heart_sounds}/{s1_choice}')
            s2 = sf.read(f'{filepath_heart_sounds}/{s2_choice}')

            # Se agrega a la lista tanto el sonido como la etiqueta
            s1_to_use.append((s1, s1_label))
            s2_to_use.append((s2, s2_label))
            
            # Agregando las etiquetas para el nombre
            s1_labels.append(s1_label)
            s2_labels.append(s2_label)
            
        # Definición del nombre del archivo de salida
        filename = f'Seed[{seed}]_S1{s1_labels}_S2{s2_labels}'

        # Creación de la carpeta a almacenar la información
        filepath_to_save = f'{filepath}/choose {choose}'
    
    # Creación de N_periods ciclos de S1-S2
    signals_in = (s1_to_use, s2_to_use)
        
    # Obteniendo las señales de salida
    sawtooth, square, intervals, signal_info =\
        lookup_table_function(signals_in, N_periods, 
                              params_signal=params_signal, 
                              params_silence=params_silence,
                              beta=beta,
                              gen_type=gen_type, 
                              trans_width=trans_width, 
                              resample_method=resample_method, 
                              lp_method=lp_method, 
                              fir_method=fir_method, 
                              gpass=gpass, gstop=gstop, 
                              correct_by_gd=correct_by_gd,
                              gd_padding=gd_padding, 
                              plot_filter=plot_filter, 
                              normalize=normalize_res)
    
    # Definición del nombre de salida del archivo, el cual estará codificado
    # como el "[# de la semilla]_S1[lista de S1]_S2[lista de S2].wav"
    s1_labels = str([i[1] for i in s1_to_use]).replace("'", "")
    s2_labels = str([i[1] for i in s2_to_use]).replace("'", "")
    
    # Nombre del archivo a guardar (por poner extensión)
    dir_to_save = f'{filepath_to_save}/{filename}'
    
    # Preguntar si es que la carpeta que almacenará las imágenes se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(filepath_to_save):
        os.makedirs(filepath_to_save)
    
    plt.figure(figsize=(15,7))
    
    # Subplot de señales de referencia
    plt.subplot(2,1,1)
    plt.plot(square)
    plt.plot(sawtooth)
    plt.ylabel('Señales de\nreferencia')
    
    # Subplot de señal de salida
    plt.subplot(2,1,2)
    plt.plot(signal_info[1], linewidth=0.9)
    plt.xlabel('Muestras')
    plt.ylabel('Sonido\ncardíaco')
    
    for num, i in enumerate(intervals):
        # Definición de los colores utilizados para cada sonido cardíaco
        if num % 2 == 0:
            c_plot = 'r'
            c_span = 'orange'
        else:
            c_plot = 'limegreen'
            c_span = 'lime'
        
        # Rango en el que ploteará la señal
        x = np.arange(i[0], i[1])
        plt.plot(x, signal_info[1][x], c=c_plot)
        
        # Destacando toda la sección
        plt.axvspan(i[0], i[1], facecolor=c_span, alpha=0.3)
        
        # Añadiendo las etiquetas
        plt.text(i[0], min(signal_info[1]), signal_info[0][num], fontsize='small',
                    bbox=dict(boxstyle='circle', facecolor=c_span, alpha=0.2))
    
    # Creación de la leyenda
    plt.legend(handles=[Line2D([0], [0], color='r', lw=2, label='S1'),
                        Line2D([0], [0], color='limegreen', lw=2, label='S2')],
                loc='upper right')
    
    # Título del gráfico
    plt.suptitle(f'Sonido cardíaco: Semilla={seed}, N={N_periods}')
    
    # Mostrando
    if plot_show:
        plt.show()
    
    # Guardando
    plt.savefig(f'{dir_to_save}.png')
    plt.close()
    
    # Opción de normalizar
    if normalize_out:
        signal_out = signal_info[1] / max(abs(signal_info[1]))
    else:
        signal_out = signal_info[1]
    
    # Opción de grabar sonido
    if save_wav:
        # Grabar el archivo .wav
        sf.write(f'{dir_to_save}.wav', signal_out, 44100)
    
    # Registrar los segmentos donde se encuentran los sonidos cardíacos
    with open(f'{dir_to_save} - segments.txt', 'w', encoding='utf8') as file:
        file.write(str(intervals))
    
    return signal_out


def get_heart_sounds_by_name(name, filepath, N_periods=30, seed=0, 
                             beta=0.2, segments='Good_segments'):
    ''' Rutina que permite generar archivos de sonido cardíaco utilizando los sonidos
    extraídos de pistas en particular. En este caso, no se mezclan distintas pistas de 
    la base de datos original.
    
    Parameters
    ----------
    name : str
        Nombre del archivo a partir del cual se obtienen los sonidos cardíacos
    filepath : str
        Dirección donde se encuentra la carpeta madre de los archivos.
    N_periods : int
        Cantidad de perídos de activación que tendrá la señal.
    seed : int, optional
        Valor de la semilla a utilizar para el experimento. Por defecto es 0.
    beta : float, optional
        Parámetro de la función coseno elevado utilizada para asegurar continuidad en los bordes
        de los sonidos cardíacos. Por defecto es 0.2.
    segments : {'Good_segments', 'Heart_segments'}, optional
        Base de datos (carpeta) utilizada para generar los sonidos. Por defecto es 'Good_segments'.
    '''
    # Plantando esta semilla para la generación de valores aleatorios
    np.random.seed(seed)
    
    # Definición de la carpeta a revisar para los archivos
    filepath_heart_sounds = f'{filepath}/{segments}' 
    
    # Archivos a trabajar
    heart_sounds = [i for i in os.listdir(filepath_heart_sounds) 
                    if i.endswith('.wav') and name in i]
    
    # Archivos para S1 y S2
    s1_indexes = [int(i.split('-')[0]) for i in heart_sounds if "S1" in i]
    s2_indexes = [int(i.split('-')[0]) for i in heart_sounds if "S2" in i]
    
    # Se obtienen todas las posibles combinaciones
    combinations = list(product(s1_indexes, s2_indexes))

    # Entonces para cada combinación se obtiene el sonido deseado
    for comb in combinations:
        seed_gen = np.random.randint(2**16)
        print(f'Generating files of S1 = {comb[0]} and S2 = {comb[1]} with seed {seed_gen}...')
        _ = \
            create_artificial_heart_sounds(filepath, N_periods=N_periods, 
                                           choose=1, seed=seed_gen, 
                                           beta=beta, segments=segments,
                                           params_signal=(5000,500), 
                                           params_silence=(15000,2000),
                                           s1_selection=comb[0], s2_selection=comb[1], 
                                           gen_type='normal',
                                           trans_width=50, resample_method='interp1d', 
                                           lp_method='fir', fir_method='kaiser', 
                                           gpass=1, gstop=80, correct_by_gd=True,
                                           gd_padding='periodic', plot_filter=False,
                                           plot_show=False, save_wav=True, 
                                           normalize_res=False, normalize_out=True)
        print('Completed!\n')


def get_heart_respiratory_sounds(dir_to_heart, a_heart=1, a_resp=1, a_noise=0, seed=0):
    '''Rutina que permite crear los sonidos cardíacos + respiratorios en una carpeta, tomando como
    base los archivos disponibles en "dir_to_heart" y "db_respiratory/Original".
    
    Parameters
    ----------
    dir_to_heart : str
        Dirección de los sonidos cardíacos a utilizar
    a_heart : float, optional
        Ponderación del sonido cardíaco en la mezcla. Por defecto es 1.
    a_resp : float, optional
        Ponderación del sonido respiratorio en la mezcla. Por defecto es 1.
    a_noise : float, optional
        Ponderación de un ruido blanco en la mezcla. Por defecto es 0.
    seed : int
        Semilla a utilizar para la creación del ruido blanco
    '''
    # Plantando esta semilla para la generación de valores aleatorios del ruido blanco
    np.random.seed(seed)
       
    # Definición del directorio donde se guardará
    folder_save = f'Database_manufacturing/db_HR/'\
                  f'Seed-{seed} - {a_heart}_Heart {a_resp}_Resp {a_noise}_White noise'
    
    # Preguntar si es que la carpeta que almacenará los sonidos se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(folder_save):
        os.makedirs(folder_save)
    
    # Definición del directorio a revisar para sonidos respiratorios
    dir_to_resp = 'Database_manufacturing/db_respiratory/Original'
    # Definición del directorio a guardar para sonidos respiratorios cortados
    dir_to_resp_adapted = 'Database_manufacturing/db_respiratory/Adapted'
    
    # Lista de sonidos cardiacos
    heart_sounds = [i for i in os.listdir(dir_to_heart) if i.endswith('.wav')]
    
    # Lista de sonidos respiratorios
    resp_sounds = [i for i in os.listdir(dir_to_resp) if i.endswith('.wav')] 

    # Definición de las posibles combinaciones entre sonidos
    combinations = tuple(product(resp_sounds, heart_sounds))
    
    for comb in tqdm(combinations, desc='Sounds', ncols=70):
        #print(f'Recording {comb[0]} + {comb[1]} sound...')
        # Abriendo los archivos de audio
        resp_audio, _ = sf.read(f'{dir_to_resp}/{comb[0]}')
        heart_audio, _ = sf.read(f'{dir_to_heart}/{comb[1]}')
        
        # Definición del largo máximo
        if len(resp_audio) < len(heart_audio):
            length = len(resp_audio)
        else:
            length = len(heart_audio)
        
        # Cortando el audio más largo hasta el largo del mínimo
        heart_to_sum = heart_audio[:length]
        resp_to_sum = resp_audio[:length]
        
        # Creación del ruido blanco
        white_noise = np.random.normal(0, 1, length)
        # Normalizando este sonido
        white_noise = white_noise / max(abs(white_noise))

        # Sonido cardíaco + respiratorio + ruido
        hr_sound = a_resp * resp_to_sum + a_heart * heart_to_sum + a_noise * white_noise
        
        # Noramlizando el sonido
        hr_sound = hr_sound / max(abs(hr_sound))
        
        # Definición del nombre de los archivos
        hr_filename = f'HR {comb[0][:-4]} {comb[1][:-4]}'
        
        # Dirección donde se guardará el archivo de audio
        dir_to_save = f'{folder_save}/{hr_filename}.wav'
        
        # Grabando el archivo de sonido cardíaco + respiratorio
        sf.write(dir_to_save, hr_sound, samplerate=44100)
        
        # Y grabando el sonido modificado del sonido respiratorio
        sf.write(f'{dir_to_resp_adapted}/{comb[0]}', resp_to_sum, samplerate=44100)

# Módulo de testeo

# Opciones de panel
dir_to_heart = 'Database_manufacturing/db_heart/Manual combinations'
get_heart_respiratory_sounds(dir_to_heart, a_heart=1, a_resp=1, a_noise=10, seed=0)

'''
filepath = 'Database_manufacturing/db_heart'
s1_sel = 59
s2_sel = 60
name = 'normal__201108011118'

get_heart_sounds_by_name(name, filepath, N_periods=30, 
                         beta=0.2, segments='Good_segments')

# Señal de salida
signal_out = \
    create_artificial_heart_sounds(filepath, N_periods=30, choose=1, seed=0, 
                                   beta=0.2, segments='Good_segments',
                                   params_signal=(5000,500), params_silence=(15000,2000),
                                   s1_selection=s1_sel, s2_selection=s2_sel, gen_type='normal',
                                   trans_width=50, resample_method='interp1d', lp_method='fir', 
                                   fir_method='kaiser', gpass=1, gstop=80, 
                                   correct_by_gd=True, gd_padding='periodic',
                                   plot_filter=False, plot_show=False,
                                   save_wav=True, normalize_res=False, 
                                   normalize_out=True)
'''