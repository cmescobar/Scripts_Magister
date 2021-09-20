import warnings
import numpy as np
from numpy.core.shape_base import block
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import fftpack
import tqdm
from pybalu.feature_analysis.score import score
from heart_sound_segmentation.descriptor_functions import get_spectrogram


def pybalu_clean(features, tol=1e-8, corr_tol=0.99, show=False):
    n_features = features.shape[1]
    ip = np.ones(n_features, dtype=int)

    # cleaning correlated features
    warnings.filterwarnings('ignore')
    C = np.abs(np.corrcoef(features, rowvar=False))
    idxs = np.vstack(np.where(C > corr_tol))
    
    # remove pairs of same feature ( feature i will have a correlation of 1 whit itself )
    idxs = idxs[:, idxs[0,:] != idxs[1,:]]
    
    # remove correlated features
    if idxs.size > 0:
        ip[np.max(idxs, 0)] = 0
    
    # remove constant features
    s = features.std(axis=0, ddof=1)
    ip[s < tol] = 0
    p = np.where(ip.astype(bool))[0]

    if show:
        print(f'Clean: number of features reduced from {n_features} to {p.size}.')

    return p


def pybalu_sfs(features, classes, n_features, *, force=False, method='fisher', options=None, show=False):
    '''Función manipulada para que no solamente encuentre las N componentes que 
    mejoran la separabilibad, sino qu también decida cual es el valor de N
    '''
    N, M = features.shape
    remaining_feats = set(np.arange(M))
    selected = list()
    curr_feats = np.zeros((N, 0))
    j_values = list()
    
    if options is None:
        options = dict()

    def _calc_score(i):
        feats = np.hstack([curr_feats, features[:, i].reshape(-1, 1)])
        return score(feats, classes, method=method, **options)

    if show:
        _range = tqdm.trange(
            n_features, desc='Selecting Features', unit_scale=True, unit=' features')
    else:
        _range = range(n_features)
    
    for _ in _range:
        new_selected = max(remaining_feats, key=_calc_score)
        selected.append(new_selected)
        remaining_feats.remove(new_selected)
        
        # Agregando el valor de J
        j_values.append(_calc_score(new_selected))
        # Agregando al set de datos
        curr_feats = np.hstack(
            [curr_feats, features[:, new_selected].reshape(-1, 1)])

    # Encontrando el índice que maximiza la separación
    max_option = np.argmax(j_values)
    
    return np.array(selected[:max_option]), j_values


def get_filterbanks(N, samplerate, freq_lim, n_filters, norm_exp=1,
                    scale_type='mel', filter_type='triangular',
                    norm_filters=True, plot_filterbank=False):
    '''Función que permite obtener un banco de filtros linealmente
    espaciados o espaciados en frecuencia de mel para calcular
    coeficientes cepstrales.
    
    Parameters
    ----------
    N : ndarray
        Largo de la señal.
    samplerate : float
        Tasa de muestreo de la señal de entrada.
    freq_lim : float
        Frecuencia límite para calcular los coeficientes cepstrales.
    n_filters : int
        Cantidad de filtros a obtener.
    scale_type : {'mel', 'linear'}, optional
        Tipo de espaciado entre los bancos de filtros para el cálculo
        de los coeficientes cepstrales. Por defecto es 'mel' (MFCC). 
    filter_type : {'triangular', 'hanning', 'squared'}, optional
        Forma del filtro a utilizar para el cálculo de la energía en 
        cada banda. Por defecto es 'triangular'.
    inverse_func : {'dct', 'idft'}, optional
        Función a utilizar para obtener los coeficientes cepstrales.
        Por defecto es 'dct'.
    plot_filterbank : bool, optional
        Booleano que indica si se grafica el banco de filtros. Por 
        defecto es False.
    
    References
    ----------
    [1] http://practicalcryptography.com/miscellaneous/machine-learning/
        guide-mel-frequency-cepstral-coefficients-mfccs/
    [2] Xuedong Huang, Alex Acero, Hsiao-Wuen Hon - Spoken Language 
        Processing A Guide to Theory, Algorithm and System 
        Development-Prentice Hall PTR (2001)
    '''
    def _freq_to_bin(f):
        # Definición del bin correspondiente en la definición
        # del intervalo de cálculo. Se usa (N - 1) ya que los bins
        # se definen entre 0 y (N - 1) (largo N)
        return np.rint(f / samplerate * (N - 1)).astype(int)
    
    
    def _triangular_filter(bins_points):
        # Definición del banco de filtros
        filter_bank = np.zeros((n_filters, N))
        
        for i in range(1, n_filters + 1):
            # Tramo ascendente del filtro triangular
            filter_bank[i - 1][bins_points[i - 1]:bins_points[i] + 1] = \
                np.linspace(0, 1, abs(bins_points[i] - bins_points[i - 1] + 1))
            
            # Tramo descendente del filtro triangular
            filter_bank[i - 1][bins_points[i]:bins_points[i + 1] + 1] = \
                np.linspace(1, 0, abs(bins_points[i + 1] - bins_points[i] + 1))
            
        return filter_bank
    
    
    def _hanning_filter(bins_points):
        # Definición del banco de filtros
        filter_bank = np.zeros((n_filters, N))
        
        for i in range(1, n_filters + 1):
            # Tramo ascendente del filtro triangular
            filter_bank[i - 1][bins_points[i - 1]:bins_points[i + 1] + 1] = \
                np.hanning(abs(bins_points[i + 1] - bins_points[i - 1] + 1))
        
        return filter_bank
    
    
    def _squared_filter(bins_points):
        # Definición del banco de filtros
        filter_bank = np.zeros((n_filters, N))
        
        for i in range(1, n_filters + 1):
            # Tramo ascendente del filtro triangular
            filter_bank[i - 1][bins_points[i - 1]:bins_points[i + 1] + 1] = 1
        
        return filter_bank
    
    
    def _norm_filterbank(filter_bank):
        # Definición del banco de filtros de salida
        filter_bank_out = np.zeros((n_filters, N))
        
        # Normalizar los filtros a energía 1
        for i in range(n_filters):
            filter_bank_out[i] = filter_bank[i] / \
                                 sum(filter_bank[i] ** norm_exp)
            
        return filter_bank_out
    
    
    # Definición de los bines en base a las frecuencias de cada filtro
    if scale_type == 'linear':
        # Definición de las "n_filters" frecuencias equiespaciadas entre
        # 0 y freq_lim. Se le agregan 2 puntos (0 y el freq_lim) ya que se 
        # necesitan para definir los límites de los filtros.
        freqs = np.arange(0, (n_filters + 1) + 1) * freq_lim / (n_filters + 1)
    
    
    elif scale_type == 'mel':
        # Definición del límite en frecuencias de mel (para no pasarse del
        # freq_lim al devolverse)
        mel_freq_lim = 2595 * np.log10(1 + freq_lim / 700)
        
        # Definición de las "n_filters" frecuencias espaciadas en escala mel 
        # entre 0 y freq_lim. Se le agregan 2 puntos (0 y el freq_lim) ya 
        # que se necesitan para definir los límites de los filtros.
        mel_freqs = np.arange(0, (n_filters + 1) + 1) * mel_freq_lim / (n_filters + 1)
        
        # Transformando de intervalos equi espaciados usando la escala
        # de mel. Es necesario hacer la transformación inversa ya que
        # en este caso se dice que lo equi espaciado viene de mel
        freqs = 700 * (10 ** (mel_freqs / 2595) - 1)
    
    else:
        raise Exception('Opción de tipo de coeficiente cepstral no válido.')
    
    
    # Transformando a bins
    bins_to = _freq_to_bin(freqs)
    
    
    # Obtención del banco de filtros
    if filter_type == 'triangular':
        filter_bank = _triangular_filter(bins_to)
        
    if filter_type == 'hanning':
        filter_bank = _hanning_filter(bins_to)
    
    elif filter_type == 'squared':
        filter_bank = _squared_filter(bins_to)
    
    # Normalizar por la energía de la señal
    if norm_filters:
        filter_bank = _norm_filterbank(filter_bank)
    
    
    # Gráfico del banco de filtros
    if plot_filterbank:
        plt.figure()
        
        # Definición del vector de frecuencias
        # f_plot = np.arange(N) * samplerate / N
        
        for i in range(n_filters):
            plt.plot(filter_bank[i])
            # plt.plot(f_plot, filter_bank[i])

        for i in bins_to:
            # plt.axvline(i * samplerate / N, c='silver', linestyle=':')
            plt.axvline(i, c='silver', linestyle=':')
            
        # plt.xlim([0, freq_lim])
        plt.xlim([0, bins_to[-1]])
        plt.show()
    
    
    return filter_bank


def get_cepstral_coefficients(signal_in, samplerate, spectrogram_params,
                              freq_lim, n_filters, n_coefs, scale_type='mel', 
                              filter_type='triangular', inverse_func='dct', 
                              norm_filters=True, plot_filterbank=False, 
                              power=2):
    '''Función que permite obtener los coeficientes cepstrales a partir de 
    un banco de filtros.
    
    Parameters
    ----------
    signal_in : ndarray
        Señal de entrada.
    samplerate : float
        Tasa de muestreo de la señal de entrada.
    freq_lim : float
        Frecuencia límite para calcular los coeficientes cepstrales.
    n_coefs : int
        Cantidad de coeficientes a obtener.
    scale_type : {'mel', 'linear'}, optional
        Tipo de espaciado entre los bancos de filtros para el cálculo
        de los coeficientes cepstrales. Por defecto es 'mel' (MFCC). 
    filter_type : {'triangular', 'hanning', 'squared'}, optional
        Forma del filtro a utilizar para el cálculo de la energía en 
        cada banda. Por defecto es 'triangular'.
    inverse_func : {'dct', 'idft'}, optional
        Función a utilizar para obtener los coeficientes cepstrales.
        Por defecto es 'dct'.
    plot_filterbank : bool, optional
        Booleano que indica si se grafica el banco de filtros. Por 
        defecto es False.
    
    References
    ----------
    [1] http://practicalcryptography.com/miscellaneous/machine-learning/
        guide-mel-frequency-cepstral-coefficients-mfccs/
    [2] Xuedong Huang, Alex Acero, Hsiao-Wuen Hon - Spoken Language 
        Processing A Guide to Theory, Algorithm and System 
        Development-Prentice Hall PTR (2001)
    '''    
    # Definición de la cantidad de puntos a considerar
    filter_bank = get_filterbanks(spectrogram_params['N'], samplerate, 
                                  freq_lim=freq_lim, n_filters=n_filters, 
                                  scale_type=scale_type, 
                                  filter_type=filter_type,
                                  norm_filters=norm_filters, 
                                  plot_filterbank=plot_filterbank)
    
    # Obtener el espectrograma de la señal
    _, _, S = get_spectrogram(signal_in, samplerate, N=spectrogram_params['N'], 
                              padding=spectrogram_params['padding'], 
                              repeat=spectrogram_params['repeat'], 
                              noverlap=spectrogram_params['noverlap'], 
                              window=spectrogram_params['window'], 
                              whole=True)
    
    # Definición del espectro de la señal
    energy_spectrum = np.abs(S) ** power
    
    # Se aplica el banco de filtros sobre el espectro de la señal
    energy_coefs = np.dot(filter_bank, energy_spectrum)
    
    # Aplicando el logaritmo
    energy_coefs = np.log(energy_coefs + 1e-10)
    
    # Calculando los coeficientes cepstrales
    if inverse_func == 'dct':
        cepstral_coefs = fftpack.dct(energy_coefs, norm='ortho', axis=0)
    elif inverse_func == 'idft':
        cepstral_coefs = np.fft.ifft(energy_coefs, axis=-1).real
    else:
        raise Exception('Opción de tipo de función inversa no válida.')
    
    
    return cepstral_coefs[:n_coefs]


def get_bands_coefficients(signal_in, samplerate, spectrogram_params,
                           freq_lim, n_coefs, scale_type='mel', 
                           filter_type='triangular', norm_filters=True, 
                           plot_filterbank=False, 
                           power=2):
    '''Función que permite obtener la energía por bandas de frecuencia
    a partir de un banco de filtros.
    
    Parameters
    ----------
    signal_in : ndarray
        Señal de entrada.
    samplerate : float
        Tasa de muestreo de la señal de entrada.
    freq_lim : float
        Frecuencia límite para calcular los coeficientes cepstrales.
    n_coefs : int
        Cantidad de coeficientes a obtener.
    scale_type : {'mel', 'linear'}, optional
        Tipo de espaciado entre los bancos de filtros para el cálculo
        de los coeficientes cepstrales. Por defecto es 'mel' (MFCC). 
    filter_type : {'triangular', 'hanning', 'squared'}, optional
        Forma del filtro a utilizar para el cálculo de la energía en 
        cada banda. Por defecto es 'triangular'.
    inverse_func : {'dct', 'idft'}, optional
        Función a utilizar para obtener los coeficientes cepstrales.
        Por defecto es 'dct'.
    plot_filterbank : bool, optional
        Booleano que indica si se grafica el banco de filtros. Por 
        defecto es False.
    
    References
    ----------
    [1] http://practicalcryptography.com/miscellaneous/machine-learning/
        guide-mel-frequency-cepstral-coefficients-mfccs/
    [2] Xuedong Huang, Alex Acero, Hsiao-Wuen Hon - Spoken Language 
        Processing A Guide to Theory, Algorithm and System 
        Development-Prentice Hall PTR (2001)
    '''    
    # Definición de la cantidad de puntos a considerar
    filter_bank = get_filterbanks(spectrogram_params['N'], samplerate, 
                                  freq_lim=freq_lim, 
                                  n_filters=n_coefs, scale_type=scale_type, 
                                  filter_type=filter_type,
                                  norm_filters=norm_filters, 
                                  plot_filterbank=plot_filterbank)
    
    # Obtener el espectrograma de la señal
    _, _, S = get_spectrogram(signal_in, samplerate, N=spectrogram_params['N'], 
                              padding=spectrogram_params['padding'], 
                              repeat=spectrogram_params['repeat'], 
                              noverlap=spectrogram_params['noverlap'], 
                              window=spectrogram_params['window'], 
                              whole=True)
    
    # Definición del espectro de la señal
    energy_spectrum = np.abs(S) ** power
    
    # Se aplica el banco de filtros sobre el espectro de la señal
    energy_coefs = np.dot(filter_bank, energy_spectrum)
    
    return energy_coefs


def get_energy_bands(signal_in, samplerate, spectrogram_params, 
                     fmin=0, fmax=1000, fband=20, power=2):
    '''Función que permite definir un espectrograma en bandas de 
    energía.
    
    Parameters
    ----------
    signal_in : ndarray
        Señal de entrada.
    samplerate : float
        Tasa de muestreo de la señal de entrada.
    spectrogram_params : dict
        Parámetros del espectrograma.
    fmin : float, optional
        Frecuencia mínima a considerar en el intervalo de interés.
        Por defecto es 0.
    fmax : float, optional
        Frecuencia máxima a considerar en el intervalo de interés.
        Este valor no puede mayor a samplerate / 2. Por defecto 
        es 1000.
    fband : float, optional
        Ancho de cada banda de frecuencia entre fmin y fmax. Por 
        defecto es 20.
    power : float, optional
        Exponente con el que se calcula la energía.
    
    Returns
    -------
    energy_S : ndarray
        Bandas de energía a través del tiempo (formato 
        espectrograma) con dimensión (#bandas x #bins de tiempo 
        del espectrograma).     
    '''
    # Obtener el espectrograma
    t, f, S = get_spectrogram(signal_in, samplerate, 
                              N=spectrogram_params['N'], 
                              padding=spectrogram_params['padding'], 
                              repeat=spectrogram_params['repeat'], 
                              noverlap=spectrogram_params['noverlap'], 
                              window=spectrogram_params['window'], 
                              whole=False)
    
    # Definición de los intervalos
    f_intervals = np.arange(fmin, fmax, fband)

    # Definición de la lista que almacenará los datos
    energy_band = np.zeros(len(f_intervals) - 1)
    energy_S = np.zeros((len(energy_band), len(t)))

    for i in range(len(f_intervals) - 1):
        lower_lim = f_intervals[i]
        upper_lim = f_intervals[i + 1]

        # Definición de los índices de interés
        indexes = np.where((lower_lim <= f) & (f <= upper_lim))[0]

        # Definiendo el valor
        energy_S[i] = np.sum(abs(S[indexes,:]) ** power, axis=0)
    
    return energy_S
