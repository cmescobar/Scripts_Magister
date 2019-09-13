import numpy as np
import os
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.decomposition import NMF
from descriptor_functions import get_spectrogram, get_inverse_spectrogram
from math_functions import wiener_filter


#####-------------- Opciones de simulación --------------#####

# Opciones de audio
op = input('Seleccione el archivo de audio a leer\n'
           '[1] - Drums.wav\n'
           '[2] - Test_1.wav\n')

# Opciones del espectrograma
N = 2048
overlap = 0.75
window = 'hann'         # 'hann', 'tukey', 'hamming', 'nuttall', None
whole = False

# Opciones de nmf
comps_list = range(2, 100)
tol = 1e-4
maxiter = 500
init = 'random'         # random, custom_basic, custom_spect
beta = 2
solver = 'cd'

# Opciones de reconstrucción
apply_wiener = False
reconstruct = True

# Opciones de salida
plot_mult_wh = True
plot_components = False


#####-------------- Nombres carpetas y archivos --------------#####
if op == "1":
    track = 'Drums'
elif op == "2":
    track = "Test_1"


# Definición del nombre del archivo
if apply_wiener:
    filename = f'[Wiener filter] N_{N} ov_{int(overlap*100)} window_{window} '\
               f'tol_{tol} maxiter_{maxiter} solver_{solver}'
else:
    filename = f'N_{N} ov_{int(overlap*100)} window_{window} tol_{tol} '\
               f'maxiter_{maxiter} solver_{solver}'

# Nombres de carpetas
folder_path_wh = f'NMF_results/{track}/WH_comparison/{init} beta_{beta}'
folder_path_comps = f'NMF_results/{track}/Components/{init} beta_{beta}'
folder_path_audios = f'NMF_results/{track}/Audios/{init} beta_{beta}/'\
                     f'{filename}'



#####-------------- Creación de directorios --------------#####

# Preguntar si es que la carpeta que almacenará los sonidos se ha
# creado. En caso de que no exista, se crea una carpeta
if plot_mult_wh:
    if not os.path.isdir(folder_path_wh):
        os.makedirs(folder_path_wh)

if plot_components:
    if not os.path.isdir(folder_path_comps):
        os.makedirs(folder_path_comps)

if reconstruct:
    if not os.path.isdir(folder_path_audios):
        os.makedirs(folder_path_audios)



#####-------------- Rutina --------------#####

if op == '1':
    # Lectura del audio
    audio_mono, samplerate = sf.read('NMF_tests/Drums.wav')

elif op == '2':
    # Lectura del audio
    audio, samplerate = sf.read('NMF_tests/Test_1.wav')
    
    # Normalizar audio para ambos canales
    audio[:,0] = audio[:,0]/max(abs(audio[:,0])) 
    audio[:,1] = audio[:,1]/max(abs(audio[:,1]))

    # Mezclando ambos canales para obtener una señal monoaural
    audio_mono = 0.5 * (audio[:,0] + audio[:,1])
else:
    print('La opción seleccionada es inválida. Por favor, intente nuevamente.')
    exit()

# Obteniendo el espectrograma
t, f, S = get_spectrogram(audio_mono, samplerate, N=N, padding=0,
                          overlap=overlap, window=window, whole=whole)

# Trabajaremos con la magnitud del espectrograma
X = np.abs(S)
# Aplicando NMF a cada una de las componentes
for n_nmf in comps_list:
    print(f'Calculating NMF with {n_nmf} components...')
    
    model = NMF(n_components=n_nmf, init=init, solver=solver, beta_loss=beta,
                tol=tol, max_iter=maxiter)
    
    # Ajustando para obtener W y H
    if init == 'random':
        W = model.fit(X)
    elif init == 'custom':
        # Se definen de manera previa puntos de inicio
        W_0 = np.ones((X.shape[0], n_nmf))
        H_0 = np.ones((n_nmf, X.shape[1]))
        
        # Se transforma
        W = model.fit(X, W=W_0, H=H_0)
    
    # Se obtiene el W y H a partir de lo calculado anteriormente
    W = model.transform(X)
    H = model.components_
    
    print(f'NMF with {n_nmf} components is OK!\n')
    
    if plot_mult_wh:
        print('Plotting...')
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
        plt.pcolormesh(t, f, 20*np.log10(X) - 20*np.log10(mult_wh),
                       cmap='inferno')
        plt.colorbar()
        plt.xlabel('Time [sec]')
        
        # Guardando el archivo
        plt.savefig(f'{folder_path_wh}/{filename}.png')
        plt.clf()
        
        print('Plot complete!')
    
    # Preguntar si se busca reconstruir por componentes
    if reconstruct:
        print('Getting components...')
        # Ahora obteniendo la fuente por cada componente
        for i in range(n_nmf):
            # Se obtiene la fuente i
            source_i = np.outer(W[:,i], H[i])
            
            # Y se pregunta si se decide aplicar el filtro
            if apply_wiener:
                Y_i = wiener_filter(X, source_i, W, H, alpha=1) *\
                      np.exp(1j * np.angle(S))
            else:
                Y_i = source_i * np.exp(1j * np.angle(S))
                
            # Finalmente, se aplica la STFT inversa para obtener la señal
            # original
            y_i = get_inverse_spectrogram(Y_i, overlap=overlap, window=window, 
                                          whole=whole)
            
            # Pasando a 64 bits
            y_i = np.float64(np.real(y_i))
            
            # Guardando el archivo
            sf.write(f'{folder_path_audios}/Component_{i}.wav',
                     y_i, samplerate)
        
        print('Components complete!')
            