def boundary_cycle_method(audio, samplerate, N=1024, overlap=0.5,
                          plot_spec=False):
    '''Método de detección de ciclos de respiración aplicados a pulmones.
    Basado en: Automatic detection of the respiratory cycle from recorded,
    single-channel sounds from lungs
    '''

    # Paso 2.2) Preprocesamiento

    # Aplicando transformada de Fourier con una a una sección con ventana
    # hamming se tiene la representación de un espectrograma. Representando
    # la ventana hamming
    hamm = hamming_window(N)

    # Definición de la matriz que contendrá el espectrograma
    spectrogram_list = []

    # Definición del vector de tiempos
    time = []
    t = 0

    # Iteración sobre el audio
    while audio.any():
        # Se corta la cantidad de muestras que se necesite, o bien, las que se
        # puedan cortar
        if len(audio) >= N:
            q_samples = N
            avance = int(N * (1 - overlap))
        else:
            q_samples = avance = len(audio)

        # Recorte en la cantidad de muestras
        audio_frame = audio[:q_samples]
        audio = audio[avance:]

        # Una vez obtendido el fragmento del audio, se ventanea este fragmento
        try:
            # Caso normal
            x_windowed = audio_frame * hamm
        except ValueError:
            # En el caso del final del audio, dado que puede ser de un largo
            # menor, se completa con ceros el audio hasta llegar al largo
            # deseado
            audio_frame = np.append(audio_frame, [0] * (N - len(audio_frame)))
            x_windowed = audio_frame * hamm

        # Y se aplica la transformada  de Fourier a esta ventana para obtener
        # los vectores del espectrograma
        spec_nk = np.fft.fft(x_windowed)

        # Y luego, se almacena en la matriz de espectrograma (se almacena la
        # mitad de la información dado que se repite por la fft)
        spectrogram_list.append(spec_nk[:int(len(spec_nk) / 2)])

        # Generar vector de tiempo
        time.append(t)
        t += avance / samplerate

    # Transformando el espectrograma a una matriz
    spectrogram_mat = np.zeros((len(spectrogram_list),
                                len(spectrogram_list[0])))

    for i in range(len(spectrogram_list)):
        spectrogram_mat[i][:] = abs(spectrogram_list[i])

    print("OK")

    if plot_spec:
        fig, ax = plt.subplots(figsize=(12, 6))

        color = ax.pcolor(20*np.log10(spectrogram_mat.T), cmap='viridis')
        plt.colorbar(color)

        # Índices de tiempo para la generación del x_tick
        time_indexes = np.arange(0, spectrogram_mat.shape[0], 10)
        freq_indexes = np.arange(0, spectrogram_mat.shape[1], 8)
        tiempo = [round(i * int(N * (1 - overlap)) / samplerate, 2)
                  for i in time_indexes]

        ax.xaxis.set(ticks=time_indexes)
        ax.set_xticklabels(tiempo)
        # ax.set_yticklabels(freq_indexes)

        # Labels
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Frecuencia [Hz]')
        plt.title(f'Espectrograma')

        plt.show()