from descriptor_functions import get_mfcc
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os


def get_features_by_symptom(symptom, sep_type='all'):
    # Obtención de la lista de archivos
    filepath = f'Interest_Audios/{symptom}/{sep_type}'

    # Consulta de archivos wav
    files = os.listdir(filepath)

    # Definición de la lista de características
    features = []

    for file in files:
        if file.endswith('.wav'):
            print(f"Getting features of {file}...")

            # Lectura de archivos de audio
            audio, samplerate = sf.read(f"{filepath}/{file}")

            # Obteniendo las características
            mfcc_features = get_mfcc(audio, samplerate)

            # Y agregando a la lista de características
            features.append(mfcc_features)

            print('¡Completed!\n')
    
    # Transformando la lista a matriz
    features = np.array(features)

    # Guardando la información en un archivo .npz
    filename = f'Features_Extracted/{symptom}_{sep_type}_features.npz'
    np.savez(filename, features=features)


# Módulo de pruebas
# get_features_by_symptom('Healthy')
# get_features_by_symptom('Pneumonia')
