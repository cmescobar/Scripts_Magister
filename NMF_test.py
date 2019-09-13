import numpy as np
import os
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.decomposition import NMF
from descriptor_functions import get_spectrogram, get_inverse_spectrogram
from math_functions import wiener_filter

# Opciones de audio
op = input('Seleccione el archivo de audio a leer\n'
           '[1] - Drums.wav\n'
           '[2] - Test_1.wav\n')

# Opciones del espectrograma
N = 2048
show = True
overlap = 0.75
window = 'hann'         # 'hann', 'tukey', 'hamming', 'nuttall', None
whole = False

# Opciones de nmf
comps_list = range(2, 3, 2)
tol = 1e-4
maxiter = 500
init = 'random'         # random, custom_basic, custom_spect
beta = 2
solver = 'cd'

# Opciones de reconstucción
apply_wiener = False

# Opciones de salida
plot_mult_wh = True



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
    
    if plot_mult_wh:
        # Multplicación de W y H
        mult_wh = np.matmul(W, H) + 1e-3
        
        # Creación de la figura
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1,2,1)
        plt.pcolormesh(t, f, 20*np.log10(X), cmap='inferno')
        plt.colorbar()
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        
        plt.subplot(1,2,2)
        plt.pcolormesh(t, f, 20*np.log10(mult_wh), cmap='inferno')
        plt.colorbar()
        plt.xlabel('Time [sec]')
        
        plt.show()
        exit()
        
        plt.clf
        
    
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
            
        # Finalmente, se aplica la STFT inversa para obtener la señal original
        y_i = get_inverse_spectrogram(Y_i, overlap=overlap, window=window,
                                      whole=whole)
        
        
    