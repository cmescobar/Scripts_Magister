import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
from sklearn.decomposition import NMF
from math_functions import hamming_window
from file_management import get_dir_audiotxt_by_symptom
from scipy import signal


def blind_source_separation_nmf(audio, samplerate, namefile, dir_foldersave,
    overlap=0, N=1024, plot_results=False, plot_norm=False, img_format='png',
    normalized=False):
    """Basado en el método mencionado en el paper: Lin, C., & Hasting, E. (2013.
    Blind source separation of heart and lung sounds based on nonnegative matrix
    factorization. 2013 International Symposium on Intelligent Signal 
    Processing and Communication Systems. Referencia tutorial:
    https://ccrma.stanford.edu/~njb/teaching/sstutorial/part2.pdf """

    # Almacen de datos para separación de sonidos
    recuperated_sound_1 = np.asarray([0] * len(audio), dtype=np.complex64)
    recuperated_sound_2 = np.asarray([0] * len(audio), dtype=np.complex64)
    
    audio_original = copy.copy(audio)

    # Definición del índice para agregar los datos
    ind_to_app = 0
    # Definición de la ventana hamming
    hamm = hamming_window(N)

    # Vector de tiempo para la señal original
    time= [i/samplerate for i in range(len(audio))]

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
            # menor, se define una ventana hamming auxiliar de menor tamaño
            x_windowed = audio_frame * hamming_window(len(audio_frame))

        # STFT pedida para la variable V
        fft_frame = np.fft.fft(x_windowed)
        abs_frame = np.abs(fft_frame)
        pha_frame = np.angle(fft_frame)

        # A este frame ventaneado se le aplica NMF
        model = NMF(n_components=2, init='nndsvda', random_state=0)
        W = model.fit_transform(np.array([abs_frame]))
        H = model.components_

        # Obteniendo el valor de V_estimado
        v_est1 = W[:,0] * H[0,:]
        v_est2 = W[:,1] * H[1,:]
        
        # Obteniendo el vector de ponderaciones (masking) para la proyección
        v_est1_norm = abs(v_est1)/(abs(v_est1) + abs(v_est2))
        v_est2_norm = abs(v_est2)/(abs(v_est1) + abs(v_est2))

        # Proyectando sobre el audio original (masking)
        sound_1_absfft = v_est1_norm * abs_frame
        sound_2_absfft = v_est2_norm * abs_frame
        
        # Y aplicando la transformada inversa de fourier
        frame_sound_1 = np.fft.ifft(sound_1_absfft * np.exp(1j * pha_frame))
        frame_sound_2 = np.fft.ifft(sound_2_absfft * np.exp(1j * pha_frame))
        
        # Finalmente guardando sobre el archivo de almacenamiento definido
        # anteriormente 
        recuperated_sound_1[ind_to_app:ind_to_app + q_samples] += \
            frame_sound_1
        recuperated_sound_2[ind_to_app:ind_to_app + q_samples] += \
            frame_sound_2
        
        # Actualizando el valor del índice
        ind_to_app += step

    # Corroborar si es que se busca normalizar
    if normalized:
        # Finalmente, formateando para grabar en el rango [-1,1]
        bss_sound_1 = recuperated_sound_1/max(abs(recuperated_sound_1.real))
        bss_sound_2 = recuperated_sound_2/max(abs(recuperated_sound_2.real))
    else:
        bss_sound_1 = recuperated_sound_1
        bss_sound_2 = recuperated_sound_2

    if plot_results:
        if plot_norm:
            plt.subplot(3, 1, 1)
            plt.plot(audio_original)
            plt.ylabel('Audio original')

            plt.subplot(3, 1, 2)
            plt.plot(bss_sound_1.real)
            plt.ylabel('Componente 1')

            plt.subplot(3, 1, 3)
            plt.plot(bss_sound_2.real, 'r')
            plt.ylabel('Componente 2')
        else:
            plt.figure(figsize=(12,6))
            plt.subplot(3, 1, 1)
            plt.plot(time, audio_original, color='b', label='Señal original',
                     lw=1.5)
            sum_norm = (recuperated_sound_1.real + recuperated_sound_2.real) / \
                max(abs(recuperated_sound_1.real + recuperated_sound_2.real))
            plt.plot(time, sum_norm, color='r', label='Suma de componentes',
                     lw=0.7)
            plt.xticks([])
            plt.ylabel('Amplitud')
            plt.legend()
            plt.title('Audio original')

            plt.subplot(3, 1, 2)
            plt.plot(time, recuperated_sound_1.real, 'g')
            plt.xticks([])
            plt.ylabel('Amplitud')
            plt.title('Componente 1')

            plt.subplot(3, 1, 3)
            plt.plot(time, recuperated_sound_2.real, 'g')
            plt.xlabel('Tiempo [s]')
            plt.ylabel('Amplitud')
            plt.title('Componente 2')
        
            plt.suptitle(f"BSS para {namefile}")
            plt.savefig(f"{dir_foldersave}/Imgs/{namefile}.{img_format}")
            plt.close()
         
    # Si la carpeta para guardar no se ha creado, se crea una
    if not os.path.isdir(dir_foldersave):
        os.makedirs(dir_foldersave)

    # Finalmente, grabando 
    sf.write(f"{dir_foldersave}/{namefile}_A.wav", bss_sound_1.real, samplerate)
    sf.write(f"{dir_foldersave}/{namefile}_B.wav", bss_sound_2.real, samplerate)


def highpass_filter_audio(audio, samplerate, crit_freq=1000, plot_filter=False,
                          gpass=1, gstop=80):
    # Realizando el diseño del filtro
    num, den = signal.iirdesign(wp=crit_freq / (samplerate / 2),
                                ws=crit_freq / samplerate,
                                gpass=gpass, gstop=gstop)

    # Y obteniendo la función de transferencia h
    w, h = signal.freqz(num, den)

    if plot_filter:
        _, ax1 = plt.subplots()
        ax1.set_title('Respuesta en frecuencia del filtro digital')
        magnitude = 20 * np.log10(abs(h))
        ax1.plot(w, magnitude, 'r')
        ax1.set_ylabel('Magnitude [dB]', color='r')
        ax1.set_xlabel('Frequencia [rad/sample]')
        ax1.set_ylim([min(magnitude), max(magnitude) + 10])
        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h))
        ax2.plot(w, angles, 'b')
        ax2.set_ylabel('Phase (radians)', color='b')
        ax2.axis('tight')
        ax2.set_ylim([min(angles), max(angles)])
        plt.show()

    # Para poder filtrar el audio
    return signal.lfilter(num, den, audio)



def apply_source_separation_to_symptom(symptom, sep_type='all',
    save_plot=False):
    # Corroborar la opción de separación
    if sep_type not in ['all', 'tracheal', 'toracic']:
        print('The option "sep_type" is not valid. Please try again.')
        return

    # Obtener todos los archivos de audio de un síntoma dado
    dirs_audio = get_dir_audiotxt_by_symptom(symptom)

    # Definición de la carpeta donde se guardará la información
    dir_folder = f'Interest_Audios/{symptom}/{sep_type}/Source_Separation/BSS'

    # Si la carpeta para guardar no se ha creado, se crea una
    if not os.path.isdir(dir_folder):
        os.makedirs(dir_folder)

    for i in dirs_audio:
        # Nombre del archivo
        namefile = i.split('/')[-1].split('.')[0]
        print(f'Separating {namefile}.wav...')
        
        # Cargando el archivo de audio
        audio, samplerate = sf.read(i)
        
        # Normalizando
        norm_audio = audio/max(abs(audio))

        # Aplicando el algoritmo de separación de fuentes
        blind_source_separation_nmf(norm_audio, samplerate, namefile,
            dir_folder, overlap=0.7, plot_results=False, img_format='png')
        
        print('¡Separation Completed!\n')


def apply_high_pass_filter_to_symptom(symptom, cutoff=100, sep_type='all'):
    # Definición de la carpeta de trabajo
    dir_work = f'Interest_Audios/{symptom}/{sep_type}'

    # Obtener todos los archivos de audio de un síntoma dado
    dirs_audio = os.listdir(dir_work)
    dirs_audio = [f'{dir_work}/{i}' for i in dirs_audio if i.endswith('.wav')]
    
    # Definición de la carpeta donde se guardará la información
    dir_foldersave = f'Interest_Audios/{symptom}/{sep_type}/Source_Separation/'\
                     f'{cutoff}_High_pass'

    # Si la carpeta para guardar no se ha creado, se crea una
    if not os.path.isdir(dir_foldersave):
        os.makedirs(dir_foldersave)

    for i in dirs_audio:
        # Nombre del archivo
        namefile = i.split('/')[-1].split('.')[0]
        print(f'Filtering {namefile}.wav...')
        
        # Cargando el archivo de audio
        audio, samplerate = sf.read(i)
        
        # Normalizando
        norm_audio = audio/max(abs(audio))

        # Aplicando el filtro
        high_pass_sound = highpass_filter_audio(norm_audio, samplerate, crit_freq=cutoff)

        # Finalmente, grabando 
        sf.write(f"{dir_foldersave}/{namefile}.wav", high_pass_sound,
            samplerate)

        print('¡Filtering Completed!\n')


# Test modules

# High pass module
'''
cfreqs = [200, 250, 300, 350, 400, 450, 500]
for i in cfreqs:
    for sep_type in ['all', 'tracheal', 'toracic']:
        apply_high_pass_filter_to_symptom("Healthy", cutoff=i,
            sep_type=sep_type)
        apply_high_pass_filter_to_symptom("Pneumonia", cutoff=i,
            sep_type=sep_type)
'''
# Blind source separation
for sep_type in ['all', 'tracheal', 'toracic']:
    apply_source_separation_to_symptom("Healthy", sep_type=sep_type)
    apply_source_separation_to_symptom("Pneumonia", sep_type=sep_type) 