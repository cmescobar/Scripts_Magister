def spectral_entropy(signal, samplerate, method='fft', 
                     nperseg_welch=None, normalize=False):
    # Transformación de la señal en un arreglo
    signal = np.array(signal)
    
    # Obtener el espectro a trabajar
    if method == 'fft':
        _, psd = periodogram(signal, samplerate)
    elif method == 'welch':
        _, psd = welch(signal, samplerate, nperseg=nperseg_welch)
    
    # Se obtiene el psd normalizado
    psd_norm = np.divide(psd, psd.sum())
    
    # Y se calcula la entropía espectral
    spect_ent = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
    
    if normalize:
        spect_ent /= np.log2(psd_norm.size)
        
    return spect_ent


def get_entropy_nmf_comps(n_comps, filename, samplerate, type='spectral', 
                          combinatorial=False, q_comb=None,
                          record=False, rec_qty=10, savetxt=False):
    
    # Se cargan los datos de las componentes
    comp_data = np.load(f'{filename}_Separation_{n_comps}.npz')
    comps = comp_data['comps']
    
    if type == 'spectral':
        # Definición de la lista de entropías para ordenar
        se_list = list()
        
        # Creación de lista iterable
        if combinatorial:
            # Se definen las posibles combinaciones
            comb_list = combinations(range(n_comps), q_comb)
            
            for i in comb_list:
                # Se hace una mezcla normalizada de las componentes
                comb_i = np.sum(np.array([comps[c] for c in i]), axis=0) / len(i)

                # Cálculo de la entropía espectral de cada componente
                se = spectral_entropy(comb_i, samplerate, normalize=True)

                # Se agrega a una lista de entropías, donde el primer elemento
                # es la entropía calculada de la combinación; y el segundo
                # elemento corresponde a los índices de esa combinación
                se_list.append((se, i))
            
        else:
            comb_list = range(n_comps)
        
            for i in comb_list:
                # Cálculo de la entropía espectral de cada componente
                se = spectral_entropy(comps[i], samplerate, normalize=True)

                # Se agrega a una lista de entropías, donde el primer elemento
                # es la entropía calculada de la combinación; y el segundo
                # elemento corresponde a los índices de esa combinación
                se_list.append((se, i))
        
        # Una vez obtenidos todas las entropías componentes con sus 
        # respectivos componentes, se ordena en función de la entropía
        se_list.sort(key=lambda x: x[0])
    
    # Opción de grabar audios de componentes
    if record:
        # Creación de la carpeta de componentes
        folder_data = f'{filename}_{n_comps}'
        
        # Preguntar si es que la carpeta que almacenará los sonidos se ha
        # creado. En caso de que no exista, se crea una carpeta
        if not os.path.isdir(folder_data):
            os.mkdir(folder_data)
        
        # Guardando
        for i, comp in enumerate(ind[:rec_qty]):
            sf.write(f'{folder_data}/{i + 1} - Comp_{comp}.wav', comps[i], samplerate)
    
    # Guardar combinaciones con entropías en un archivo txt
    if savetxt:
        # Creación de la carpeta de componentes
        folder_data = f'{filename}_{n_comps}'
        
        # Preguntar si es que la carpeta que almacenará los sonidos se ha
        # creado. En caso de que no exista, se crea una carpeta
        if not os.path.isdir(folder_data):
            os.mkdir(folder_data)
        
        # Guardando
        if combinatorial:
            fileout = f'{folder_data}/Comps_{n_comps} - Combinatorial_{q_comb}.txt'
        else:
            fileout = f'{folder_data}/Comps_{n_comps}.txt'
        
        with open(fileout, 'w', encoding='utf8') as file:
            for i in se_list:
                file.write(f'{i}\n')


def get_sound_comps_bytxt(n_comps, filename, lines_to_read, 
                          samplerate, combinatorial=False,
                          q_comb=None):
    # Creación de la carpeta de componentes
    folder_data = f'{filename}_{n_comps}'

    # Se cargan los datos de las componentes
    comp_data = np.load(f'{filename}_Separation_{n_comps}.npz')
    comps = comp_data['comps']
    
    # Definición del archivo a revisar y carpeta a crear
    if combinatorial:
        foldlook = f'{folder_data}/Comps_{n_comps} - Combinatorial_{q_comb}'
    else:
        foldlook = f'{folder_data}/Comps_{n_comps}'
        
    # Preguntar si es que la carpeta que almacenará los sonidos se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(foldlook):
        os.makedirs(foldlook)
    
    with open(f'{foldlook}.txt', 'r', encoding='utf8') as file:
        # Se obtiene todo el documento
        lines = file.readlines()
        
        # Leyendo las líneas de interés
        for i in lines_to_read:
            # Se obtiene la línea de interés
            line_i = literal_eval(lines[i].strip())
            
            # Se usan las componentes para obtener la señal
            signal_out = np.sum(np.array([comps[i] for i in line_i[1]]), axis=0)
            
            # Se define el nombre de la señal con el siguiente código:
            # [Posición en la lista de entropías] - Comps_...
            # [Componentes que lo constituyen separados por guion]
            filename_out = f'{i} - Comps_{str(line_i[1])}'
            
            # Guardando el archivo de sonido
            sf.write(f'{foldlook}/{filename_out}.wav', signal_out, samplerate)