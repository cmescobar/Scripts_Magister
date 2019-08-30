import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.decomposition import NMF
import os

# Lectura del audio
audio, samplerate = sf.read('Testing_modules/Test_1.wav')

# Mezclando ambos canales para obtener una señal monoaural
audio_mono = 0.5 * (audio[:,0] + audio[:,1])

# Aplicando NMF para separar los canales
