from descriptor_functions import get_mfcc, get_mfcc_by_respiratory_segments|
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os


def get_mfcc_by_symptom(symptom, sep_type='all', segmentation='frame'):
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

            if segmentation == 'frame':
                # Obteniendo las características
                mfcc_features = get_mfcc(audio, samplerate)
    
            elif segmentation == 'respiration':
                get_mfcc_by_respiratory_segments(audio, samplerate, 
                                                 symptom, file)
                
            # Y agregando a la lista de características
            #features.append(mfcc_features)

            print('¡Completed!\n')
    
    # Transformando la lista a matriz
    features = np.array(features)

    # Guardando la información en un archivo .npz
    filename = f'Features_Extracted/{symptom}_{sep_type}_features.npz'
    np.savez(filename, features=features)


def get_mfcc_with_segments_by_symptom(symptom, sep_type='all'):
    # Obtención de la lista de archivos
    filepath = f'Interest_Audios/{symptom}/{sep_type}'

    # Consulta de archivos wav
    files = os.listdir(filepath)

    # Definición de la lista de características
    features = []
    
    for file in files:
        if file.endswith('.wav'):
            print(f"Getting features of {file}...")
            
            
            
            print('¡Completed!\n')

# Módulo de pruebas
get_mfcc_by_symptom('Healthy', segmentation='respiration')
# get_mfcc_by_symptom('Pneumonia')
# get_mfcc_with_segments_by_symptom('Healthy')
