from plot_functions import get_symptom_images_by_frame, \
    get_symptom_images_at_all
from descriptor_functions import centroide, promedio_aritmetico, varianza, \
    skewness, kurtosis, rms, max_amp, zero_cross_rate, centroide_espectral, \
    pendiente_espectral, flujo_espectral, spectral_flatness, \
    abs_fourier_shift, get_spectrogram, abs_fourier_db_half
from file_management import get_audio_folder_by_symptom


op = input("Escoja alguna de las siguientes opciones\n"
        "[1] - Obtener características para cada cuadro\n"
        "[2] - Separar audios por cuadro\n")

disease_list = ["Healthy", "Pneumonia"]

if op == "1":
    for i in disease_list:
        print(i)
        '''get_symptom_images_by_frame(i, display_time=True)
        get_symptom_images_by_frame(i, centroide, display_time=True)
        get_symptom_images_by_frame(i, promedio_aritmetico, display_time=True)
        get_symptom_images_by_frame(i, varianza, display_time=True)
        get_symptom_images_by_frame(i, skewness, display_time=True)
        get_symptom_images_by_frame(i, kurtosis, display_time=True)
        get_symptom_images_by_frame(i, rms, display_time=True)
        get_symptom_images_by_frame(i, max_amp, display_time=True)
        get_symptom_images_by_frame(i, zero_cross_rate, display_time=True)
        get_symptom_images_by_frame(i, centroide_espectral, display_time=True)
        get_symptom_images_by_frame(i, pendiente_espectral, display_time=True)
        get_symptom_images_by_frame(i, flujo_espectral, display_time=True)
        get_symptom_images_by_frame(i, spectral_flatness, display_time=True)
        get_symptom_images_at_all(i, get_spectrogram, display_time = True) 
        get_symptom_images_at_all(i, abs_fourier_db_half, N=3, display_time=True)'''
        
elif op == "2":
    for i in disease_list:
        get_audio_folder_by_symptom(i)

else:
    print("Opción inválida. Por favor intente nuevamente.")
    exit()
