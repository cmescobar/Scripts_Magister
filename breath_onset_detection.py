import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import copy
import os
from ast import literal_eval
from scipy.signal import find_peaks
from smooth_functions import robust_smooth
from math_functions import hamming_window, recognize_peaks_by_derivates
from scipy.interpolate import interp1d
from file_management import get_dir_audiotxt_by_symptom


def variance_fractal_dimension_method(audio, samplerate, overlap=0.5, NT=1024,
                                      nk=4, smooth=False, plot=False):
    '''Variance fractal dimension está dada por la expresión:
        D_o = 2 - H

    Donde D_E corresponde a la dimensión del problema a resolver (por
    ejemplo, en el caso de una curva D_E = 1, para un plano D_E = 2 y para
    el espacio D_E = 3) y donde:
        H = lim_{dt -> 0} log(var(ds))/(2*log(dt))
    
    En el que 's' es la señal muestreada y 'ds' la variación entre 2 puntos. Asi
    mismo, 'dt' es la diferencia entre 2 puntos.

    Referencia: APPLICATIONS OF VARIANCE FRACTAL DIMENSION: A SURVEY
    '''

    # Definición del vector d_sigma
    d_sigma = []

    # Definición del vector de tiempo
    time = []
    t = 0

    # Copia de un audio para obtener el plot
    audio_plot = copy.copy(audio)

    while audio.any():
        # Se corta la cantidad de muestras que se necesite, o bien, las que se
        # puedan cortar
        if len(audio) >= NT:
            q_samples = NT
            avance = int(NT * (1 - overlap))
        else:
            q_samples = avance = len(audio)

        # Recorte en la cantidad de muestras
        audio_frame = audio[:q_samples]
        audio = audio[avance:]
        
        # Una vez definido el largo de este bloque N_T se procede a calcular
        # el valor de H, el cual está definido por la suma de todos los
        # posibles sub-bloques con diferencias
        try:
            var_dx = variance_delta_block(audio_frame, NT, nk)

            # Definición de delta_t
            delta_t = len(audio_frame)/samplerate

            # Con este valor, es posible obtener el exponente de Hurst H
            h = 1/2 * np.log(var_dx) / np.log(delta_t)

            # Con lo cual es posible obtener d_sigma
            d_sigma.append(2 - h)

            # Generar vector de tiempo
            time.append(t)
            t += avance / samplerate

        except IndexError:
            print("Without final frame")

    # Si se escoge aplicar suavizamiento, se tiene que...
    if smooth:
        print("Paso!")
        # Realizando un suavizamiento de la señal mediante el método presentado
        d_sigma = np.asarray(d_sigma)
        fractal_smoothed = robust_smooth(d_sigma, iters=50, tol_limit=1e-5)
        # Y una interpolación cuadrática para obtener la función cuadrática que
        # la representa para poder obtener una mayor cantidad de puntos
        f_fractal=interp1d([i for i in range(len(fractal_smoothed))],
                            fractal_smoothed, 'quadratic')
        print("Paso 2")

        # Reescribiendo para obtener más puntos
        x=np.arange(0, len(fractal_smoothed) - 1, 0.0005)
        fractal_sm_int=f_fractal(x)
            
        # Y obteniendo los peaks requeridos    
        '''peaks = recognize_peaks_by_derivates(x, energy_sm_int, peak_type='min',
            plot=plot_peak_det)'''
        
        # Extrapolando el los puntos obtenidos a tiempo muestral con overlap, y
        # luego recuperando el esquema de tiempo original (multiplicando por los
        # saltos 'N' entre cada bloque)
        '''peaks_in_time = [int(round(x[i])) * N * (1-overlap) for i in peaks]'''


    if plot:
        plt.subplot(2, 1, 1)
        plt.plot(fractal_sm_int)

        plt.subplot(2, 1, 2)
        plt.plot(d_sigma)

        d_sigma = np.asarray(d_sigma)
        peaks, _ = find_peaks(d_sigma, distance=128, height=1)
        plt.plot(peaks, d_sigma[peaks], 'rx')
        plt.show()

    return time, d_sigma


def variance_delta_block(block, NT, nk):
    # Definición de la cantidad de sub-ventanas Nk a partir de los pasos nk
    # entregados
    Nk = int(NT/nk)
    delta_x = np.asarray([block[i*nk - 1] - block[(i - 1)*nk] for i in
                          range(1, Nk + 1)])
    # Con lo cual se obtiene la varianza
    return 1/(Nk - 1) * (sum(delta_x**2) - 1/Nk * (sum(delta_x))**2)


def boundary_cycle_method(audio, samplerate, N=4410, overlap=0.5,
                          fmin=80, fmax=1000, plot_peak_det=False,
                          return_smooth=False):
    '''Método de detección de ciclos de respiración aplicados a pulmones.
    Basado en: Automatic detection of the respiratory cycle from recorded,
    single-channel sounds from lungs
    '''

    # Paso 2.2) Preprocesamiento

    # Aplicando transformada de Fourier con una a una sección con ventana
    # hamming se tiene la representación de un espectrograma. Representando
    # la ventana hamming
    hamm = hamming_window(N)

    # Definición de la matriz que contendrá el espectrograma
    spectrogram_list = []
    # Definición de la matriz que contendrá la energía promedio
    mean_energy = []

    # Definición del vector de tiempos
    time = []
    t = 0
    # Y de muestras
    samples = []
    sample = 0

    # Definición de los índices de las frecuencias de interés
    freqs = [i for i in range(N) if fmin <= (i * samplerate / N) <= fmax]

    # Iteración sobre el audio
    while audio.any():
        # Se corta la cantidad de muestras que se necesite, o bien, las que se
        # puedan cortar
        if len(audio) >= N:
            q_samples = N
            step = int(N * (1 - overlap))
        else:
            q_samples = step = len(audio)

        # Recorte en la cantidad de muestras
        audio_frame = audio[:q_samples]
        audio = audio[step:]

        # Una vez obtendido el fragmento del audio, se ventanea este fragmento
        try:
            # Caso normal
            x_windowed = audio_frame * hamm
        except ValueError:
            # En el caso del final del audio, dado que puede ser de un largo
            # menor, se completa con ceros el audio hasta llegar al largo
            # deseado
            audio_frame = np.append(audio_frame, [0] * (N - len(audio_frame)))
            x_windowed = audio_frame * hamm

        # Y se aplica la transformada  de Fourier a esta ventana para obtener
        # los vectores del espectrograma
        spec_nk = np.fft.fft(x_windowed)
        spectrogram_list.append(spec_nk[:len(spec_nk//2)])

        # Obteniendo las muestras solo de la banda de interés
        spec_nk_band = np.asarray([spec_nk[i] for i in freqs])

        # Se calcula la energía promedio entre las bandas de frecuencia fmin
        # y fmax
        p_n = sum(abs(spec_nk_band ** 2)) / (fmax - fmin)

        # Y se agrega al vector de energía promedio
        mean_energy.append(p_n)

        # Generar vector de tiempo
        time.append(t)
        t += step / samplerate
        # Y de muestras
        samples.append(sample)
        sample += step

    # Una vez obtenido el vector de energía media por cada frame,
    # se suavizará esta señal utilizando el algoritmo de mínimos cuadrados
    # penalizados
    mean_energy_db = 20 * np.log10(mean_energy)
    
    # Realizando un suavizamiento de la señal mediante el método presentado
    energy_smoothed = robust_smooth(mean_energy_db, iters=20, tol_limit=1e-5)
    # Y una interpolación cuadrática para obtener la función cuadrática que la
    # representa para poder obtener una mayor cantidad de puntos
    f_energy=interp1d([i for i in range(len(energy_smoothed))],
                       energy_smoothed, 'quadratic')

    # Reescribiendo para obtener más puntos
    x=np.arange(0, len(energy_smoothed) - 1, 0.0005)
    energy_sm_int=f_energy(x)
        
    # Y obteniendo los peaks requeridos    
    peaks = recognize_peaks_by_derivates(x, energy_sm_int, peak_type='max',
        plot=plot_peak_det, lookup=10000)
    
    # Extrapolando el los puntos obtenidos a tiempo muestral con overlap, y
    # luego recuperando el esquema de tiempo original (multiplicando por los
    # saltos 'N' entre cada bloque)
    peaks_in_time = [int(round(x[i])) * N * (1-overlap) for i in peaks]

    if return_smooth:
        return peaks, peaks_in_time, energy_sm_int
    else:
        return peaks_in_time


def plot_cycles_by_symptom(symptom, sep_type='all', preproc='raw', show=False,
    method=1, img_format='png'):
    # Corroborar la opción de separación
    if sep_type not in ['all', 'tracheal', 'toracic']:
        print('The option "sep_type" is not valid. Please try again.')
        return

    # Definición de la carpeta de trabajo
    foldpath = f'Interest_Audios/{symptom}/{sep_type}'

    # Obtener la carpeta a partir del síntoma
    if preproc is 'raw':
        filepath = foldpath
    elif preproc in os.listdir(f'{foldpath}/Source_Separation'):
        filepath = f'{foldpath}/Source_Separation/{preproc}'
    else:
        print('The option "preproc" is not valid. Please try again.')
        return

    # Obtener todos los archivos de audio del filepath entregado
    list_files = os.listdir(filepath)
    # Recuperar todos los archivos .wav
    list_files = [i for i in list_files if i.endswith('.wav')]


    with open(f'Results/{symptom}/{symptom}_resp_original_points.txt', 'r',
                encoding='utf8') as file:
        for i in file:
            # Eliminar el salto de línea
            filename, point_list = i.strip().split(';')

            if filename in list_files:
                # Transformarlo a diccionario
                points = literal_eval(point_list)
                
                # Abriendo el archivo de audio
                audio, sr = sf.read(f"{filepath}/{filename}")
                
                if method == 1:
                    # Obtenieniendo su energía suavizada, y sus peaks
                    try:
                        peaks, peaks_in_time, e_smooth = \
                            boundary_cycle_method(audio, sr, overlap=0,
                                return_smooth=True, fmin=150, fmax=1000)
                    except:
                        continue

                    print(f"Plotting respiratory cycles points of {filename}")

                    # Graficando
                    plt.figure(figsize=(12,6))
                    plt.subplot(3, 1, 1)
                    plt.specgram(audio, Fs=sr, NFFT=1024)
                    plt.xticks([])
                    plt.ylim(0, 3000)
                    plt.ylabel('Frecuencia [Hz]')

                    plt.subplot(3, 1, 2)
                    plt.plot(e_smooth)
                    plt.plot(peaks, [e_smooth[i] for i in peaks], 'rx')
                    plt.xticks([])
                    plt.xlim(0, len(e_smooth))
                    plt.ylabel('Energia/banda [dB/KHz]')

                    plt.subplot(3, 1, 3)
                    time = [i / sr for i in range(len(audio))]
                    points = [i / sr for i in points]
                    peaks_in_time = [i / sr for i in peaks_in_time] 
                    plt.plot(time, audio, label='Audio')
                    plt.plot(points, [0] * len(points), color='#00FF00',
                            marker='o', label='Marcas', linestyle='')
                    plt.plot(peaks_in_time, [0] * len(peaks_in_time), 'rx',
                            label='Detecciones')
                    plt.xlim(0, time[-1])
                    plt.xlabel('Tiempo [s]')
                    plt.ylabel('Amplitud [V]')
                    plt.legend(loc='upper right', bbox_to_anchor=(1, 1),
                        fontsize=8)

                    plt.suptitle(f"{filename}")

                    # Creación de la carpeta en caso de no existir
                    savefig_path = f"{filepath}/Imgs/smooth"

                    # Si la carpeta para guardar no se ha creado, se crea una
                    if not os.path.isdir(savefig_path):
                        os.makedirs(savefig_path)

                    # Guardando finalmente la figura...
                    plt.savefig(f'{savefig_path}/smooth_{filename[:-3]}'
                                f'{img_format}')
                                    
                    
                elif method == 2:
                    # Obtenieniendo su varinza fractal, y sus peaks
                    try:
                        t, d_sig = variance_fractal_dimension_method(audio, sr)
                    except:
                        continue
                    
                    print(f"Plotting respiratory cycles points of {filename}")

                    # Graficando
                    plt.figure(figsize=(12,6))
                    plt.subplot(2, 1, 1)
                    plt.specgram(audio, Fs=sr, NFFT=1024)
                    plt.xticks([])
                    plt.ylim(0, 3000)
                    plt.ylabel('Frecuencia [Hz]')

                    plt.subplot(2, 1, 2)
                    plt.plot(t, d_sig)
                    plt.xlim(0, t[-1])
                    plt.ylabel('Fractal dimension')
                    
                    plt.suptitle(f"{filename}")

                    savefig_path = f"{filepath}/Imgs/frac/frac_"\
                                    f"{filename[:-3]}{img_format}"

                    # Si la carpeta para guardar no se ha creado, se crea una
                    if not os.path.isdir(savefig_path):
                        os.makedirs(savefig_path)

                    # Guardando finalmente la figura...
                    plt.savefig(savefig_path)
                
                print('¡Completed!\n')

                if show:
                    plt.show()
                else:
                    plt.close()


def generate_resp_points_by_symptom(symptom):
    # Obtener la lista de todos los archivos de texto de los audios
    # correspondientes al síntoma entregado
    list_audiotxt = get_dir_audiotxt_by_symptom(symptom, wav=False)

    # Definición del filepath donde se guardará el archivo
    filepath_to_save = f"Results/{symptom}/{symptom}_resp_original_points.txt"
    
    # Para cada uno de los archivos en esta lista, se abre para obtener los
    # puntos de interés
    for txt_audio in list_audiotxt:
        namefile_sound = txt_audio.replace('.txt', '.wav')
        _, sr = sf.read(namefile_sound)

        with open(txt_audio, 'r', encoding='utf8') as file:
            # Creación del almacen de los puntos (se hace con set para evitar
            # que se repitan los puntos)
            points_to_app = set()
            for line in file:
                p1, p2, _, _ = line.strip().split('\t')
                points_to_app.add(int(float(p1) * sr))
                points_to_app.add(int(float(p2) * sr))
            
        # Una vez obtenidos todos los puntos, se pasa el set a list, y se
        # ordena
        points_to_app = list(points_to_app)
        points_to_app.sort()

        # Se recupera el nombre del archivo
        namefile_sound = namefile_sound.split('/')[-1]
        
        # Creación del archivo de almacenamiento de puntos de ciclo respiratorio
        with open(filepath_to_save, 'a', encoding='utf8') as file:
            file.write(f"{namefile_sound};{points_to_app}\n")
    



# Test module
'''
generate_resp_points_by_symptom("Pneumonia")

audio_data, samplerate = sf.read('Interest_Audios/Healthy/'
                            '125_1b1_Tc_sc_Meditron.wav')
#d_sigma = variance_fractal_dimension_method(audio_data, samplerate, plot=True)
boundary_cycle_method(audio_data, samplerate, N=44100//10, overlap=0)
'''
# plot_manually_cycles('Interest_Audios/Healthy', method=2)
'''audio_data, samplerate = sf.read('Interest_Audios/Healthy/'
                            '127_1b1_Ar_sc_Meditron.wav')
d_sigma = variance_fractal_dimension_method(audio_data, samplerate, smooth=True,
        plot=True)



symptom = 'Healthy'
sep_type = 'toracic'
for i in os.listdir(f'Interest_Audios/{symptom}/{sep_type}/Source_Separation'):
    plot_cycles_by_symptom(symptom, sep_type=sep_type, preproc=i, show=False,
        method=1, img_format='png')

plot_cycles_by_symptom(symptom, sep_type=sep_type, preproc='raw', show=False,
    method=1, img_format='png')
'''