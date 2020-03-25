import numpy as np
import os
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.decomposition import NMF
from descriptor_functions import get_spectrogram, get_inverse_spectrogram,\
    nmf_to_spectrogram
from math_functions import wiener_filter

# Definición del comentario de la corrida
phrase = 'Modificación de inicio de W y H'
comment = ' - ' + phrase


#####-------------- Opciones de simulación --------------#####

# Lectura de audio
track = 'Drums_zanark'
audio_mono, samplerate = sf.read(f'NMF_tests/{track}.wav')

# Opciones del espectrograma
N = 4096
overlap = 0.75
window = 'hamming'         # 'hann', 'tukey', 'hamming', 'nuttall', None
whole = False

# Opciones de nmf
comps_list = [3]#range(3, 51, 2)
tol = 1e-4
maxiter = 500
init_op = 'random_auto' # random_auto, random_man, custom_basic, custom_spect
seed_value = 100        # Para el random_man
beta = 2
solver = 'mu'
alpha = 0
l1_ratio = 0
random_state = 100

# Opciones de reconstrucción
apply_wiener = True
alpha_wie = 1

# Opciones de salida
plot_mult_wh = True
plot_after_nmf = True
save_audio = False
plot_components = False



#####-------------- Nombres carpetas y archivos --------------#####
# Definición del nombre del archivo
if apply_wiener:
    filename = f'[Wiener filter] N_{N} ov_{int(overlap*100)} window_{window} '\
               f'tol_{tol} maxiter_{maxiter} solver_{solver} alpha_{alpha} l1_'\
               f'{l1_ratio} beta_{beta}'
else:
    filename = f'N_{N} ov_{int(overlap*100)} window_{window} tol_{tol} '\
               f'maxiter_{maxiter} solver_{solver} alpha_{alpha} l1_{l1_ratio}'\
               f'beta_{beta}'

# Agregar valor de la semilla si es que es random manual
if init_op == 'random_man':
    filename += f' seed_{seed_value}'

# Nombres de carpetas
folder_path_wh = f'NMF_results/{track}/WH_comparison/{init_op}'
folder_path_anmf = f'NMF_results/{track}/after_nmf_comparison/{init_op}'
folder_path_comps = f'NMF_results/{track}/Components/{init_op}'
folder_path_audios = f'NMF_results/{track}/Audios/{init_op} {filename}'

# A cada una de las carpetas se le agrega también el comentario de cada corrida
# (para tener en consideración qué es lo que el investigador quizo hacer)
folder_path_wh += f' {comment}'
folder_path_anmf += f' {comment}'
folder_path_comps += f' {comment}'
folder_path_audios += f' {comment}'



#####-------------- Creación de directorios --------------#####

# Preguntar si es que la carpeta que almacenará los sonidos se ha
# creado. En caso de que no exista, se crea una carpeta
if plot_mult_wh:
    if not os.path.isdir(folder_path_wh):
        os.makedirs(folder_path_wh)
        
if plot_after_nmf:
    if not os.path.isdir(folder_path_anmf):
        os.makedirs(folder_path_anmf)

if plot_components:
    if not os.path.isdir(folder_path_comps):
        os.makedirs(folder_path_comps)


# Valores previos para funcionamiento de inicialización de rutina
t, f, S = get_spectrogram(audio_mono, samplerate, N=N, 
                          padding=0, overlap=overlap, window=window,
                          whole=False)
X = np.abs(S)

#####-------------- Rutina --------------#####

# Transformación de opción de inicio
if 'random_auto' != init_op:
    init = 'custom'
else:
    init = 'random'

# Aplicando NMF a cada una de las componentes
for n_nmf in comps_list:
    print(f'Getting NMF with {n_nmf} components...')

    if init_op == 'random_auto':
        # Se define el W_0 y H_0 como None porque no se utilizarán
        W_0 = H_0 = None
    
    elif init_op == 'random_man':
        # Se define de manera previa puntos de inicio
        np.random.seed(seed_value)
        # Cabe destacar que una vez invocada la semilla, tanto el W_0 y H_0 
        # tomarán siempre los mismos valores
        W_0 = np.random.rand(X.shape[0], n_nmf)
        H_0 = np.random.rand(n_nmf, X.shape[1])
    
    elif init_op == 'custom_basic':
        # Se definen de manera previa puntos de inicio (solo 1's)
        W_0 = np.ones((X.shape[0], n_nmf))
        H_0 = np.ones((n_nmf, X.shape[1]))
    
    elif init_op == 'custom_spect':
        # Se obtiene un promedio temporal del espectro a través del tiempo
        mean_spect = np.array([X.mean(axis=1)]).T
        
        # Luego se define W_0 como la repetición de este promedio
        W_0 = np.tile(mean_spect, n_nmf)
        H_0 = np.ones((n_nmf, X.shape[1]))
        
    else:
        print('Opción de inicio de NMF incorrecta. Por favor, intente '
              'nuevamente')
        exit()
    
    # Se aplica la separación por NMF
    comps, t, f, X, Y_list, W, H =\
    nmf_to_spectrogram(audio_mono, samplerate, N=N, overlap=overlap, padding=0,
                       window=window, wiener_filt=apply_wiener, 
                       alpha_wie=alpha_wie, n_components=n_nmf, init=init, 
                       solver=solver, beta=beta, tol=tol, max_iter=maxiter, 
                       alpha_nmf=alpha, l1_ratio=l1_ratio, 
                       random_state=random_state, W_0=W_0, H_0=H_0, whole=False)

    # Rutina que plotea la diferencia entre la estimación que se genera 
    # producto de la multiplicación entre W y H, y el valor original de la 
    # magnitud del espectrograma X
    if plot_mult_wh:
        print('Plotting WH...')
        # Multplicación de W y H
        mult_wh = np.matmul(W, H) + 1e-3
        
        # Creación de la figura
        plt.figure(figsize=(15, 7))
        
        plt.subplot(1,3,1)
        plt.pcolormesh(t, f, 20*np.log10(X), cmap='inferno')
        plt.colorbar()
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        
        plt.subplot(1,3,2)
        plt.pcolormesh(t, f, 20*np.log10(mult_wh), cmap='inferno')
        plt.colorbar()
        plt.xlabel('Time [sec]')
        
        plt.subplot(1,3,3)
        plt.pcolormesh(t, f, np.abs(20*np.log10(X) - 20*np.log10(mult_wh)),
                       cmap='inferno')
        plt.colorbar()
        plt.xlabel('Time [sec]')
        
        # Guardando el archivo
        plt.savefig(f'{folder_path_wh}/{n_nmf}_comps {filename}.png')
        plt.clf()
        plt.close()
        
        print('Plot WH complete!\n')
        
    if plot_after_nmf:
        print('Plotting after NMF...')
        # Espectrograma de la reconstrucción
        t, f, S1 = get_spectrogram(np.sum(comps, axis=0), samplerate, N=N, 
                                   padding=0, overlap=overlap, window=window, 
                                   whole=False)
        
        # Creación de la figura
        plt.figure(figsize=(15, 7))
        
        plt.subplot(1,3,1)
        plt.pcolormesh(t, f, 20*np.log10(X), cmap='inferno')
        plt.colorbar()
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        
        plt.subplot(1,3,2)
        plt.pcolormesh(t, f, 20*np.log10(np.abs(S1)), cmap='inferno')
        plt.colorbar()
        plt.xlabel('Time [sec]')
        
        plt.subplot(1,3,3)
        plt.pcolormesh(t, f, np.abs(20*np.log10(X) - 
                                    20*np.log10(np.abs(S1[:, :-1]))),
                       cmap='inferno')
        plt.colorbar()
        plt.xlabel('Time [sec]')
        plt.suptitle('')
        
        # Guardando el archivo
        plt.savefig(f'{folder_path_anmf}/{n_nmf}_comps {filename}.png')
        plt.clf()
        plt.close()
        
        print('Plot after NMF complete!\n')
    
    # Preguntar si se busca guardar archivos de audio
    if save_audio:
        print('Saving audio components...')
        # Ahora obteniendo la fuente por cada componente
        for i in range(n_nmf):
            # Pasando a 64 bits
            y_i = comps[i]
            
            # Creación de la carpeta donde se almacenará
            folder_to_rec = f'{folder_path_audios}/{n_nmf} Components'
            if not os.path.isdir(folder_to_rec):
                os.makedirs(folder_to_rec)
            
            # Guardando el archivo
            sf.write(f'{folder_to_rec}/Component_{i}.wav',
                     y_i, samplerate)
        
        print('Audio saving complete!\n')

    print(f'{"-" * 40}\n')
