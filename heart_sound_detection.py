import matplotlib.pyplot as plt
from wavelet_functions import get_wavelet_levels, upsample_wavelets
from thresholding_functions import wavelet_thresholding
from filter_and_sampling import upsampling_signal, downsampling_signal


def get_upsampled_thresholded_wavelets(signal_in, samplerate, freq_pass=950, freq_stop=1000, 
                                       method='lowpass', lp_method='fir', 
                                       fir_method='kaiser', gpass=1, gstop=80, 
                                       plot_filter=False, levels_to_get=[3,4,5], 
                                       levels_to_decompose=6, wavelet='db4', mode='periodization', 
                                       threshold_criteria='hard', threshold_delta='universal',
                                       min_percentage=None, print_delta=False,
                                       plot_wavelets=False, normalize=True):
    '''
    '''
    # Aplicando un downsampling a la señal para disminuir la cantidad de puntos a 
    # procesar
    new_rate, dwns_signal = downsampling_signal(signal_in, samplerate, 
                                                freq_pass, freq_stop, 
                                                method=method, 
                                                lp_method=lp_method, 
                                                fir_method=fir_method, 
                                                gpass=gpass, gstop=gstop, 
                                                plot_filter=plot_filter, 
                                                normalize=normalize)
    
    # Se obtienen los wavelets que interesan
    interest_wavelets = get_wavelet_levels(dwns_signal, 
                                           levels_to_get=levels_to_get,
                                           levels_to_decompose=levels_to_decompose, 
                                           wavelet=wavelet, mode=mode, 
                                           threshold_criteria=threshold_criteria, 
                                           threshold_delta=threshold_delta, 
                                           min_percentage=min_percentage, 
                                           print_delta=print_delta, 
                                           plot_wavelets=plot_wavelets)
    
    # Finalmente, upsampleando
    upsampled_wavelets = upsample_wavelets(interest_wavelets, new_rate, samplerate, 
                                           levels_to_get, len(signal_in), 
                                           method=method, 
                                           trans_width=abs(freq_stop - freq_pass), 
                                           lp_method=lp_method, 
                                           fir_method=fir_method, 
                                           gpass=gpass, gstop=gstop, 
                                           plot_filter=False, 
                                           plot_signals=False,
                                           plot_wavelets=plot_wavelets, 
                                           normalize=normalize)
    
    return upsampled_wavelets


# Testing module
import os
import soundfile as sf

filepath = 'Interest_Audios/Heart_sound_files'
filenames = [i for i in os.listdir(filepath) if i.endswith('.wav')]

# Parámetros de descomposición
levels_to_get = [3,4,5]
levels_to_decompose = 6
wavelet = 'db4'

for i in filenames:
    print(f'Getting wavelets of {i}...')
    # Cargando los archivos
    audio_file, samplerate = sf.read(f'{filepath}/{i}')
    
    ups_wav = get_upsampled_thresholded_wavelets(audio_file, samplerate, freq_pass=950, 
                                                 freq_stop=1000, 
                                                 method='lowpass', lp_method='fir', 
                                                 fir_method='kaiser', gpass=1, gstop=80, 
                                                 plot_filter=False, levels_to_get=levels_to_get, 
                                                 levels_to_decompose=levels_to_decompose, 
                                                 wavelet=wavelet, 
                                                 mode='periodization', 
                                                 threshold_criteria='hard', 
                                                 threshold_delta='universal',
                                                 min_percentage=None, print_delta=False,
                                                 plot_wavelets=False, normalize=True)

    dir_to_paste = f'{filepath}/{wavelet}'
    for n in range(len(ups_wav)):
        # Definición del nivel a recuperar
        level = levels_to_get[n]
        # Definición del sonido a recuperar
        to_rec = ups_wav[n]
        
        # Grabando
        sf.write(f'{dir_to_paste}/{i.strip(".wav")} - wavelet level {level}.wav', 
                 to_rec, samplerate)
        
    print('Completed!\n')