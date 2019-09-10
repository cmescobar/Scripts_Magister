	mean_energy_db = 20 * np.log10(mean_energy)
    energy_smoothed = smooth(mean_energy_db)

    plt.plot(mean_energy_db)
    plt.plot(energy_smoothed, 'r')
    plt.show()


	# Una vez obtenido el vector de energía media por cada frame,
    # se suavizará esta señal utilizando el algoritmo de mínimos cuadrados
    # penalizados
    x = np.linspace(0, 100, 2 ** 8)
    y = np.cos(x / 10) + (x / 50)** 2 + \
        np.asarray([rd.random()/1 for i in range(len(x))])
    
    energy_smoothed = smooth(y)

    plt.plot(y, 'r.')
    plt.plot(energy_smoothed, 'b')
    # plt.plot(time, 20*np.log10(mean_energy))
    plt.show()