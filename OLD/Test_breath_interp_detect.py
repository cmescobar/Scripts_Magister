import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import copy
from scipy.signal import find_peaks
from smooth_functions import robust_smooth
from math_functions import hamming_window, recognize_peaks_by_derivates
from scipy.interpolate import interp1d


def variance_fractal_dimension_method(audio, samplerate, overlap=0.5, NT=1024,
                                      nk=4, plot=False):
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

    if plot:
        plt.subplot(2, 1, 1)
        plt.plot(audio_plot)

        plt.subplot(2, 1, 2)
        plt.plot(d_sigma)

        d_sigma = np.asarray(d_sigma)
        peaks, _ = find_peaks(d_sigma, distance=128, height=1)
        plt.plot(peaks, d_sigma[peaks], 'rx')
        plt.show()

    return d_sigma


def variance_delta_block(block, NT, nk):
    # Definición de la cantidad de sub-ventanas Nk a partir de los pasos nk
    # entregados
    Nk = int(NT/nk)
    delta_x = np.asarray([block[i*nk - 1] - block[(i - 1)*nk] for i in
                          range(1, Nk + 1)])
    # Con lo cual se obtiene la varianza
    return 1/(Nk - 1) * (sum(delta_x**2) - 1/Nk * (sum(delta_x))**2)


def boundary_cycle_method(audio, samplerate, N=2048, overlap=0.5, fmin=80,
                          fmax=1000):
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
    x = np.arange(0, len(energy_smoothed) - 1, 0.005)
    energy_sm_int = f_energy(x)
    
    '''plt.subplot(2,1,1)
    plt.plot(mean_energy_db)
    plt.subplot(2,1,2)
    plt.plot(energy_smoothed)
    plt.plot(x, energy_sm_int)
    plt.show()'''
    
    peaks = recognize_peaks_by_derivates(energy_sm_int, peak_type='min')




audio_data, samplerate = sf.read('Interest_Audios/Healthy/'
                            '125_1b1_Tc_sc_Meditron.wav')
# d_sigma = variance_fractal_dimension_method(audio, samplerate, plot=True)
boundary_cycle_method(audio_data, samplerate, N=44100//10, overlap=0)

