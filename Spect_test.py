import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
from scipy.signal import spectrogram
from descriptor_functions import get_spectrogram


# Lectura del audio
audio, samplerate = sf.read('useful_audio/Test_1.wav')

# Normalizar audio para ambos canales
audio[:,0] = audio[:,0]/max(abs(audio[:,0])) 
audio[:,1] = audio[:,1]/max(abs(audio[:,1]))

# Mezclando ambos canales para obtener una señal monoaural
audio_mono = 0.5 * (audio[:,0] + audio[:,1])

# Aplicando NMF para separar los canales
f, t, Sxx = spectrogram(audio[:,0], fs=samplerate, scaling='spectrum', 
                        mode='magnitude')


plt.subplot(3,1,1)
Sxx_2, freqs, times, _ = plt.specgram(audio[:,0], Fs=samplerate, scale='dB',
                                      mode='magnitude', noverlap=0)
plt.colorbar()
plt.xlabel('Tiempo [seg]')
plt.ylabel('Frecuencia [Hz]')

'''print(20*np.log10(Sxx))
print()
print(Sxx_2)'''

plt.subplot(3,1,2)
plt.pcolormesh(t, f, 20*np.log10(Sxx))
plt.colorbar()
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')


plt.subplot(3,1,3)
get_spectrogram(audio[:,0], samplerate, N=256, padding=256, plot=True)