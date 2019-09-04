import numpy as np
import os
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.decomposition import NMF
from descriptor_functions import get_spectrogram


# Opciones del script
plot_wh = False
plot_spect_1 = True
only_plot = True
comps_list = range(10, 500, 20)
N = 256
show = False

tol = 1e-5
maxiter = 500
init = 'random'
beta = 'itakura-saito'



#####-------------- Rutina --------------#####

# Lectura del audio
audio, samplerate = sf.read('useful_audio/Test_1.wav')

# Normalizar audio para ambos canales
audio[:,0] = audio[:,0]/max(abs(audio[:,0])) 
audio[:,1] = audio[:,1]/max(abs(audio[:,1]))

# Mezclando ambos canales para obtener una señal monoaural
audio_mono = 0.5 * (audio[:,0] + audio[:,1])

# Obteniendo el espectrograma
t, f, v = get_spectrogram(audio_mono, samplerate, N=N, padding=N,
                          plot=False, spect_type='abs')

# Guardar valores de la dimensión del espectrograma
row_dim, col_dim = v.shape

if plot_wh:
    # Definición de la carpeta a guardar las imágenes
    folder_path = f'NMF_results/WH/{init} - beta_{beta}'\
                  f'_tol_{tol}_maxiter_{maxiter}'
elif plot_spect_1:
    # Definición de la carpeta a guardar las imágenes
    folder_path = f'NMF_results/Spectrogram/{init} - beta_{beta}'\
                  f'_tol_{tol}_maxiter_{maxiter}'

# Preguntar si es que la carpeta que almacenará los sonidos se ha
# creado. En caso de que no exista, se crea una carpeta
if not os.path.isdir(folder_path):
    os.makedirs(folder_path)

# Iterando sobre la cantidad de componentes
for comp in comps_list:
    print(f'Getting NMF with {comp} components...')
    
    # Aplicando NMF para separar los canales
    model = NMF(n_components=comp, init=init, beta_loss=beta, max_iter=maxiter,
                tol=tol, alpha=0, solver='mu')
    
    # Definición del nombre del archivo
    filename = f'Comp_{comp}.png'

    if init == 'custom':
        # Valores de inicialización
        W_0 = np.zeros((row_dim, comp)) + 5
        H_0 = np.zeros((comp, col_dim)) + 0.5

        # Recuperando las matrices
        W = model.fit_transform(v, W=W_0, H=H_0)
    
    elif init == 'random':
        # Recuperando las matrices
        W = model.fit_transform(v)
        
    H = model.components_

    # print(W.shape)
    # print(H.shape)

    if plot_wh:
        for i in range(comp):
            plt.subplot(comp, 2, 2*i+1)
            plt.plot(W[:,i].T)

            plt.subplot(comp, 2, 2*i+2)
            plt.plot(H[i,:])
        
        if show:
            plt.show()

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

    # Preguntar si es que solo se quiere plottear
    if only_plot:
        print(f'¡Plot successfull!\n')
        continue

    # Definiicón de una lista que almacene los resultados de la separación de
    # fuentes
    sources = []

    # Para cada una de las fuentes
    for i in range(comp):
        # Repitiendo la máscara para ponderar por los índices correspondientes
        # de la matriz de activaciones temporales
        pattern_matrix = np.tile(np.array([W[:,i]]).T, col_dim)
        # Multiplicando por las activaciones temporales, se obtiene la
        # representación de la fuente i
        source_i = pattern_matrix * H[i,:]
        
        # Obteniendo la mascara 
        mask_matrix_i = 0
        
        # Agregando la máscara a la lista
        sources.append(source_i)
    