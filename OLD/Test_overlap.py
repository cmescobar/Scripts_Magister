def boundary_cycle_method(audio, samplerate, N=1024, overlap=0):
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
    spectrogram = []

    # Definición del vector de tiempos
    time = []
    t = 0

    q = 0
    plt.plot(audio)

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

        # Una vez obtendido el fragmento del audio, se aplica la transformada
        # de Fourier a esta ventana para obtener los vectores del espectrograma


        # Test
        plt.plot(range(q, q+len(audio_frame)), audio_frame + 2)
        q += len(audio_frame)

        # Generar vector de tiempo
        time.append(t)
        t += avance / samplerate

    plt.show()