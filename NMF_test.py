import numpy as np
import os
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.decomposition import NMF
from descriptor_functions import get_spectrogram, get_inverse_spectrogram
from math_functions import wiener_filter


# Opciones del script
plot_wh = False
plot_spect_1 = False
only_plot = False
comps_list = [2] #range(10, 500, 20)
N = 256
show = True

tol = 1e-4
maxiter = 500
init = 'random'   # random, custom_basic, custom_spect
beta = 1



#####-------------- Rutina --------------#####

# Lectura del audio
audio, samplerate = sf.read('useful_audio/Test_1.wav')

# Normalizar audio para ambos canales
audio[:,0] = audio[:,0]/max(abs(audio[:,0])) 
audio[:,1] = audio[:,1]/max(abs(audio[:,1]))

# Mezclando ambos canales para obtener una señal monoaural
audio_mono = 0.5 * (audio[:,0] + audio[:,1])

# Obteniendo el espectrograma
t, f, v, phase = get_spectrogram(audio_mono, samplerate, N=N, padding=N,
                                 plot=False, spect_type='abs')

# Guardar valores de la dimensión del espectrograma
row_dim, col_dim = v.shape


#####-------------- Opciones de ploteo --------------#####
###
if plot_wh:
    # Definición de la carpeta a guardar las imágenes
    folder_path = f'NMF_results/WH/{init} - beta_{beta}'\
                  f'_tol_{tol}_maxiter_{maxiter}'
elif plot_spect_1:
    # Definición de la carpeta a guardar las imágenes
    folder_path = f'NMF_results/Spectrogram/{init} - beta_{beta}'\
                  f'_tol_{tol}_maxiter_{maxiter}'
else:
    folder_path = None

# Definición de la carpeta a guardar los audios                 
audio_path = f'NMF_results/Audio/{init} - beta_{beta}'\
             f'_tol_{tol}_maxiter_{maxiter}'

# Preguntar si es que la carpeta que almacenará los sonidos se ha
# creado. En caso de que no exista, se crea una carpeta
if folder_path is not None:
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

# Preguntar si es que la carpeta que almacenará los sonidos se ha
# creado. En caso de que no exista, se crea una carpeta
if not os.path.isdir(audio_path):
    os.makedirs(audio_path)

# Iterando sobre la cantidad de componentes
for comp in comps_list:
    print(f'Getting NMF with {comp} components...')
    
    # Aplicando NMF para separar los canales
    if 'custom' in init:
        model = NMF(n_components=comp, init='custom', beta_loss=beta,
                    max_iter=maxiter, tol=tol, alpha=0, solver='mu')
    else:
        model = NMF(n_components=comp, init=init, beta_loss=beta,
                    max_iter=maxiter, tol=tol, alpha=0, solver='mu')
    
    # Definición del nombre del archivo
    filename = f'Comp_{comp}.png'

    if init == 'custom_basic':
        # Valores de inicialización
        W_0 = np.zeros((row_dim, comp)) + 5
        H_0 = np.zeros((comp, col_dim)) + 0.5

        # Recuperando las matrices
        W = model.fit_transform(v, W=W_0, H=H_0)
        
    elif init == 'custom_spect':
        # Se obtiene la media en el tiempo de la transformada de Fourier...
        init_value = np.array([v.mean(1)]).T
        
        # Para tener un valor inicial del espectrograma
        W_0 = np.tile(init_value, comp)
        
        # Mientras que la matriz de activaciones temporales se inicializa en 0.5
        H_0 = np.zeros((comp, col_dim)) + 0.5
        
        # Recuperando las matrices
        W = model.fit_transform(v, W=W_0, H=H_0) 
    
    elif init == 'random':
        # Recuperando las matrices
        W = model.fit_transform(v)
        
    H = model.components_

    if plot_wh:
        for i in range(comp):
            plt.subplot(comp, 2, 2*i+1)
            plt.plot(W[:,i].T)

            plt.subplot(comp, 2, 2*i+2)
            plt.plot(H[i,:])
        
        if show:
            plt.show()
            
        plt.savefig(f'{folder_path}/{filename}')
        plt.clf()

    if plot_spect_1:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.pcolormesh(t, f, 10*np.log10(v))
        plt.colorbar()
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        
        plt.subplot(1, 2, 2)
        plt.pcolormesh(t, f, 10*np.log10(np.matmul(W, H)))
        plt.colorbar()
        plt.xlabel('Time [sec]')
        
        if show:
            plt.show()
            
        plt.savefig(f'{folder_path}/{filename}')
        plt.clf()

    print(f'¡Plot successfull!\n')
    
    # Preguntar si es que solo se quiere plottear
    if only_plot:
        continue

    # Definición de una lista que almacene los resultados de la separación de
    # fuentes
    sources = []
    # Definición de una lista que almacene los audios de la separación de
    # fuentes
    audio = []

    # Para cada una de las fuentes
    for i in range(comp):
        # Repitiendo la máscara para ponderar por los índices correspondientes
        # de la matriz de activaciones temporales
        pattern_matrix = np.tile(np.array([W[:,i]]).T, col_dim)
        
        # Multiplicando por las activaciones temporales, se obtiene la
        # representación de la fuente i
        source_i = pattern_matrix * H[i,:]
        
        # Aplicando el filtro de wiener, se obtiene la salida de la fuente
        wiener_out = wiener_filter(v, source_i, W, H, alpha=1)
        
        # Aplicando la fase al espectrograma obtenido después de el filtro de
        # wiener 
        Yj = wiener_out * np.exp(1j*phase)
        Yj = source_i * np.exp(1j*phase)
        
        # Aplicando transformada inversa: OJOOOOO!!
        audio_j = get_inverse_spectrogram(Yj)
        
        # Pasando a float64
        audio_j = np.array(audio_j, dtype=np.float64)
        
        # Pasando a audio
        audio_filename = f'{audio_path}/Component_{i+1}.wav'
        print(audio_filename)
        sf.write(audio_filename, audio_j, samplerate)
        
        # Agregando la máscara a la lista
        sources.append(source_i)
        # Y el audio
        audio.append(audio_j)
        
        print(f'Componente {i} completa')

    