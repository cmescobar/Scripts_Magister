import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import mfcc
from librosa.feature import mfcc as mfcc_in_segm
from file_management import get_segmentation_points_by_filename
from math_functions import hamming_window
from scipy.signal import tukey


# Descriptores temporales
def centroide(audio):
    return sum([i*abs(audio[i]) for i in range(len(audio))]) / sum(abs(audio))


def promedio_aritmetico(audio):
    return sum(audio) / len(audio)


def varianza(audio):
    return sum((audio - promedio_aritmetico(audio)) ** 2) / len(audio)


def skewness(audio):
    return sum((audio - promedio_aritmetico(audio)) ** 3) / \
           (len(audio) * varianza(audio) ** (3 / 2))


def kurtosis(audio):
    return sum((audio - promedio_aritmetico(audio)) ** 4) / \
           (len(audio) * varianza(audio) ** 2) - 3


def rms(audio):
    return np.sqrt(sum(audio ** 2) / len(audio))


def max_amp(audio):
    return max(abs(audio))


def zero_cross_rate(audio):
    return sum([abs(np.sign(audio[i]) - np.sign(audio[i + 1]))
                for i in range(len(audio) - 1)]) / (2 * len(audio))


def centroide_espectral(audio):
    # Aplicando fft
    fft_audio = np.fft.fft(audio)
    # Luego se hace el calculo del centroide
    return sum([i * abs(fft_audio[i])**2
                for i in range(round(len(fft_audio) / 2))]) / \
           sum(abs(fft_audio) ** 2)


# Descriptores frecuenciales
def pendiente_espectral(audio):
    # Aplicando fft
    fft_audio = np.fft.fft(audio)
    # Definición de kappa
    K = len(fft_audio)
    # Luego realizando el cálculo del flujo espectral
    a = sum([i * abs(fft_audio[i]) for i in range(round(K / 2))])
    b = sum([i for i in range(round(K / 2))]) * \
        sum([abs(fft_audio[i]) for i in range(round(K / 2))])
    c = sum([i ** 2 for i in range(round(K / 2))])
    d = sum([i for i in range(round(K / 2))])
    return (K/2*a - b) / (K/2*c - d**2)


def flujo_espectral(audio, audio_before):
    # Aplicando fft
    fft_1 = np.fft.fft(audio)
    fft_2 = np.fft.fft(audio_before)
    # Definición de kappa
    K = len(fft_1)
    # Luego realizando el cálculo del flujo espectral
    return np.sqrt(sum([(abs(fft_1[i]) - abs(fft_2[i])) ** 2
                        for i in range(round(K / 2))])) / (K / 2)


def spectral_flatness(audio):
    # Aplicando fft
    fft_audio = np.fft.fft(audio)
    # Definición de kappa
    K = round(len(fft_audio)/2)
    # Luego realizando el cálculo de la planitud espectral
    return np.exp(1 / K * sum([np.log(abs(fft_audio[i])) for i in range(K)])) /\
           (1 / K * sum([abs(fft_audio[i]) for i in range(K)]))


def abs_fourier_shift(audio, samplerate, N_rep):
    audio_rep = np.tile(audio, N_rep)
    fourier = np.fft.fft(audio_rep)
    frec = np.fft.fftfreq(len(audio_rep), 1/samplerate)
    fourier_shift = abs(fourier)
    return frec, fourier_shift


def get_spectrogram(audio, samplerate, N=512, padding=512, overlap=0,
                    window='tukey', spect_type='abs', plot=False):
    # Lista donde se almacenará los valores del espectrograma
    spect_mag = []
    spect_pha = []
    # Lista de tiempo
    times = []
    
    # Variables auxiliares
    t = 0   # Tiempo
    
    # Seleccionar ventana
    if window == 'tukey':
        wind_mask = tukey(N)
    elif window == 'hamming':
        wind_mask = hamming_window(N)
        
    # Seleccionar tipo de espectrograma
    if spect_type == 'abs':
        mag_func = lambda audio: 1/N * abs(audio) #** 2
    elif spect_type == 'dB':
        mag_func = lambda audio: 20*np.log10(abs(audio)*1/N) 
    
    # Iteración sobre el audio
    while audio.any():
        # Se corta la cantidad de muestras que se necesite, o bien, las que se
        # puedan cortar
        if len(audio) >= N:
            q_samples = N
            step = int(N * (1 - overlap))
        else:
            break
            # q_samples = step = len(audio)
            
        # Recorte en la cantidad de muestras
        audio_frame = audio[:q_samples]
        audio = audio[step:]
               
        # Ventaneando
        audio_frame_wind = audio_frame * wind_mask
        
        # Aplicando padding
        audio_padded = np.append(audio_frame_wind, [0] * padding)
        
        # Aplicando transformada de fourier
        audio_fft = np.fft.fft(audio_padded)
        
        # Calculando la magntiud y fase 
        mag = mag_func(audio_fft)
        pha = np.angle(audio_fft)
               
        # Agregando a los vectores del espectro
        spect_mag.append(mag[0:int((N+padding)/2)])
        spect_pha.append(pha[0:int((N+padding)/2)])
        
        # Agregando al vector de tiempo
        times.append(t)
        t += step/samplerate
        
    # Generar el vector de frecuencias para cada ventana
    freqs = np.linspace(0, samplerate/2, (N+padding)/2)
        
    # Una vez obtenido el spect_mag y spect_pha, se pasa a matriz
    spect_mag = np.array(spect_mag)
    spect_pha = np.array(spect_pha)
    
    # Se plotea
    if plot:
        plt.pcolormesh(times, freqs, spect_mag.T)
        plt.colorbar()
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
    
    # Se retornan los valores que permiten construir el 
    return times, freqs, spect_mag.T, spect_pha.T


def get_inverse_spectrogram(X, window='tukey'):
    # Obtener la dimensión de la matriz
    rows, cols = X.shape
    
    # Seleccionar ventana
    if window == 'tukey':
        wind_mask = tukey(rows)
    elif window == 'hamming':
        wind_mask = hamming_window(rows) 
    
    # Definición de una lista en la que se almacena la transformada inversa
    inv_spect = []
    
    # Transformando columna a columna
    for i in range(cols):
        inv_spect += list(wind_mask * np.fft.ifft(X[:, i]))
        
    return inv_spect
        

def get_mfcc(audio, samplerate, nfft=2048):
    mfcc_vect = mfcc(audio, samplerate, nfft=nfft)
    return mfcc_vect.mean(0)


def get_mfcc_by_respiratory_segments(audio, samplerate, symptom, filename):
    # Se obtiene la lista de puntos de segmentación 
    list_points = get_segmentation_points_by_filename(symptom, filename)

    # Agregando la primera y la última para tener los intervalos
    list_points = [0] + list_points + [len(audio)]
    
    mfcc_vect = []
    # Separando por tramos
    for i in range(1, len(list_points)):
        # Límites a considerar del audio
        begin = list_points[i - 1]
        endls = list_points[i]
        
        # Ciclo respiratorio
        audio_segment = audio[begin:endls]
        
        # Obteniendo el MFCC a partir de este segmento
        mfcc_segment = mfcc_in_segm(audio_segment, sr=samplerate, n_mfcc=13)
        print(mfcc_segment)
        exit()
        
        # Agregando a la lista
        mfcc_vect.append(mfcc_segment)
        
    # Transformando la lista en matriz
    print(len(mfcc_vect))
    print(len(mfcc_vect[0]))
    exit()
        

def normalize(X):
    mf = X.mean(0)
    sf = X.std(0)
    a = 1 / sf
    b = - mf / sf
    return X * a + b, a, b


def apply_function_to_audio(func, audio, samplerate, separation=1024,
                            overlap=0):
    # Definir los límites del traslape
    if overlap >= 100:
        overlap = 99
    elif overlap < 0:
        overlap = 0

    # Vectores de valores para cada descriptor
    out = []
    time_list = []
    t = 0

    # Audio auxiliar que guarda el valor anterior para el flujo espectral
    audio_before = None

    while audio.any():
        # Se corta la cantidad de muestras que se necesite, o bien, las que se
        # puedan cortar
        if len(audio) >= separation:
            q_samples = separation
            avance = int(separation * (1 - overlap))
        else:
            q_samples = len(audio)
            if len(audio >= separation * (1 - overlap)):
                avance = int(separation * (1 - overlap))
            else:
                avance = len(audio)

        # Recorte en la cantidad de muestras
        audio_frame = audio[:q_samples]
        audio = audio[avance:]

        # Obtención del descriptor para cada frame. En caso de que se trabaje
        # con flujo espectral se debe hacer un proceso más particular
        if func.__name__ == "flujo_espectral":
            if audio_before is not None:
                out.append(func(audio_frame, audio_before))
            else:
                out.append(1)
        else:
            out.append(func(audio_frame))
        time_list.append(t)
        t += q_samples / samplerate

        # Actualizar el valor de audio antiguo con el valor que se deja
        audio_before = audio_frame

    return time_list, out
