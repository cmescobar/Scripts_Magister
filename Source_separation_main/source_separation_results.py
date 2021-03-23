# En este módulo se hace uso de las funciones disponibles en source_separation.py
# para la obtención de resultados. No se implementa nada nuevo, sino que se ordenan
# las funciones para generar resultados
import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import evaluation_metrics as evmet
from tqdm import tqdm
from ast import literal_eval
from prettytable import PrettyTable
from collections import  defaultdict
from filter_and_sampling import downsampling_signal
from source_separation import nmf_applied_all, get_components_HR_sounds


def generate_results(sep_type='on segments', ausc_zone='Anterior'):
    ''' Rutina que genera resultados de la separación de fuentes a partir de 
    un tipo de separación.
    '''
    # Parametros de separación
    N = 1024
    noverlap = int(0.5 * N)
    n_components = 5
    padding = 3 * N
    repeat = 0 # 4
    beta = 1
    sr_des = 44100 // 4
    window = 'hann'
    N_lax = 100
    N_fade = 100
    l1_ratio = 0    # 1
    alpha = 0       # 0.03
    assign_method = 'auto'
    clustering = False
    dec_criteria = 'vote'
    H_binary = True
    only_centroid = False
    
    # Definición filepath
    filepath = f'Database_manufacturing/db_HR/Source Separation/v2/{ausc_zone}/'\
                'Seed-0 - x - 0.5_Heart 0.5_Resp 0_White noise'
    
    get_components_HR_sounds(filepath, sr_des, sep_type=sep_type, assign_method=assign_method, 
                            clustering=clustering, n_components=n_components, N=N, 
                            N_lax=N_lax, N_fade=N_fade, noverlap=noverlap, 
                            padding=padding, repeat=repeat, window=window, 
                            whole=False, alpha_wiener=1, filter_out='wiener', 
                            init='random', solver='mu', beta=beta, tol=1e-4, max_iter=2000, 
                            alpha_nmf=alpha, l1_ratio=l1_ratio, random_state=0, 
                            W_0=None, H_0=None, plot_segments=True, scale='abs', 
                            ausc_zone=ausc_zone, fcut_spect_crit=200, 
                            measure_spect_crit='correlation', i_selection='max', 
                            f1_roll=20, f2_roll=150, measure_temp_crit='q_equal', 
                            H_binary=H_binary, reduce_to_H=False, dec_criteria=dec_criteria, 
                            only_centroid=only_centroid)


def generate_results_factory(sep_type='on segments', ausc_zone='Anterior'):
    ''' Rutina que genera resultados de la separación de fuentes a partir de 
    distintos tipos de separación. Hace lo mismo que "generate_results" pero
    para las combinaciones de las listas.
    
    Esta función genera los archivos de audio de la separación en sonido cardiaco
    y respiratorio para toda los archivos en la base de datos. Además, dependiendo
    del tipo de separación a usar, va generando imagenes que permiten ilustrar
    el proceso que se está realizando.
    
    '''
    # Parametros de separación
    N = [1024]          # [512, 1024]
    n_components = [2]     # [2, 5, 10, 20, 50]
    beta = [1]             # [1, 2] 
    repeat = 0      # 4
    sr_des = 44100 // 4
    window = 'hann'
    l1_ratio = 0    # 1
    alpha = 0       # 0.03
    assign_method = 'auto'
    clustering = True
    dec_criteria = 'vote'
    H_binary = True
    only_centroid = False
    
    # Definición filepath
    filepath = f'Database_manufacturing/db_HR/Source Separation/v2/{ausc_zone}/'\
                'Seed-0 - x - 0.5_Heart 0.5_Resp 0_White noise'
    
    for n_comps in n_components:
        for beta_i in beta:
            for n in N:
                
                # Parámetros
                noverlap = [int(0.90 * n)]#[int(0.5 * n), int(0.75 * n), int(0.95 * n)]
                padding = 3 * n
                N_lax = 100 if N == 1024 else 50
                N_fade = 100 if N == 1024 else 50
                
                for nov in noverlap:
                    get_components_HR_sounds(filepath, sr_des, sep_type=sep_type, 
                                            assign_method=assign_method, 
                                            clustering=clustering, n_components=n_comps, N=n, 
                                            N_lax=N_lax, N_fade=N_fade, noverlap=nov, 
                                            padding=padding, repeat=repeat, window=window, 
                                            whole=False, alpha_wiener=1, filter_out='wiener', 
                                            init='random', solver='mu', beta=beta_i, 
                                            tol=1e-4, max_iter=2000, 
                                            alpha_nmf=alpha, l1_ratio=l1_ratio, random_state=0, 
                                            W_0=None, H_0=None, plot_segments=True, scale='abs', 
                                            ausc_zone=ausc_zone, fcut_spect_crit=500, 
                                            measure_spect_crit='correlation', i_selection='max', 
                                            f1_roll=20, f2_roll=150, measure_temp_crit='q_equal', 
                                            H_binary=H_binary, reduce_to_H=False, 
                                            dec_criteria=dec_criteria, 
                                            only_centroid=only_centroid)
                    

def evaluate_result(sep_type='to all', id_rev=1, version=2, ausc_zone='Both'):
    # Parámetros
    sr_des = 44100 // 4
    
    # Definición filepath
    filepath = f'Database_manufacturing/db_HR/Source Separation/v2/{ausc_zone}/'\
                'Seed-0 - x - 0.5_Heart 0.5_Resp 0_White noise/Components'
    filepath_resp = f'Database_manufacturing/db_respiratory/Adapted v{version}/{ausc_zone}'
    filepath_heart = f'Database_manufacturing/db_heart/Manual combinations v{version}/{ausc_zone}'
    
    if sep_type == 'to all':
        filepath = f'{filepath}/Separation to all'
    elif sep_type == 'on segments':
        filepath = f'{filepath}/Separation on segments'
    elif sep_type == 'masked segments':
        filepath = f'{filepath}/Masking on segments'
    else:
        raise Exception('Opción para "sep_type" no válido.')
    
    # Búsqueda del diccionario para conocer las propiedades de la simulación
    with open(f'{filepath}/Simulation register.txt', 'r', encoding='utf8') as file:
        for line in file:
            dict_to_rev = literal_eval(line.strip())
            
            if dict_to_rev['id'] == id_rev:
                dict_sim = dict_to_rev
                break
    
    try:
        # Definición de los parámetros de la simulación
        N = dict_sim['N']
        noverlap = dict_sim['noverlap']
        window = dict_sim['window']
        clustering = dict_sim['clustering']
    except:
        raise Exception(f'Simulación con "id {id_rev}" no realizada.')
    
    # Se agrega finalmente el id de la dirección
    filepath = f'{filepath}/id {id_rev}'
    
    # Para resetear los archivos a rellenar más adelante
    open(f'{filepath}/Result Analysis - Respiration.txt','w').close()
    open(f'{filepath}/Result Analysis - Heart.txt', 'w').close()
    
    files_to_rev = [i for i in os.listdir(filepath) if i.startswith('HR')]
    
    # Definición del texto la señal
    if clustering:
        sound_type = ' clustering'
    else:
        sound_type = ' energy_lim k2'
    
    for name_i in tqdm(files_to_rev, desc=f'Desc id {id_rev}', ncols=70):
        # Obtener nombres de archivo
        _, ind, resp_name, heart_name = name_i.split(' ')
        
        # Abriendo archivos respiratorio y cardiacos ideales
        audio_resp, sr_resp = sf.read(f'{filepath_resp}/{ind} {resp_name}.wav')
        audio_heart, sr_heart = sf.read(f'{filepath_heart}/{heart_name}.wav')
        
        # Solo si es que hay que bajar puntos se baja, en caso contrario se mantiene
        if sr_des < sr_resp:
            _, resp_to = downsampling_signal(audio_resp, sr_resp, 
                                             sr_des//2-100, sr_des//2)
            _, heart_to = downsampling_signal(audio_heart, sr_heart, 
                                              sr_des//2-100, sr_des//2)
        else:
            resp_to = audio_resp
            heart_to = audio_heart
            sr_des = sr_resp
        
        # Abriendo los archivos obtenidos mediante descomposición
        if sep_type == 'to all':
            resp_comp, _ = sf.read(f'{filepath}/{name_i}/ Respiratory Sound{sound_type}.wav')
            heart_comp, _ = sf.read(f'{filepath}/{name_i}/ Heart Sound{sound_type}.wav')
        elif sep_type in ['on segments', 'masked segments']:
            resp_comp, _ = sf.read(f'{filepath}/{name_i}/Respiratory signal{sound_type}.wav')
            heart_comp, _ = sf.read(f'{filepath}/{name_i}/Heart signal{sound_type}.wav')
        
        
        # Seleccionando el largo más corto de cada uno (para hacer un punto a punto)
        minlen_resp = min(len(resp_to), len(resp_comp))
        minlen_heart = min(len(heart_to), len(heart_comp))
        minlen_to = min(len(resp_to),len(heart_to))
        
        # Recorte
        resp_to = resp_to[:minlen_resp]
        resp_comp = resp_comp[:minlen_resp]
        heart_to = heart_to[:minlen_heart]
        heart_comp = heart_comp[:minlen_heart]
        
        # Definición del factor de división por el cual se debe escalar los sonidos 
        # originales para realizar la comparación correcta debido a la normalización
        # aplicada en el momento que se crearon los datos
        max_ratio = max(abs(resp_to[:minlen_to] + heart_to[:minlen_to]))
        
        # Re escalando
        resp_to /= max_ratio
        heart_to /= max_ratio
        
        ################# RESULTADOS CUANTITATIVOS #################
        # Cálculo de MSE's
        resp_mse = evmet.MSE(resp_to, resp_comp, options='MSE', scale='abs', eps=1)
        resp_nmse = evmet.MSE(resp_to, resp_comp, options='NMSE', scale='abs', eps=1)
        resp_rmse = evmet.MSE(resp_to, resp_comp, options='RMSE', scale='abs', eps=1)
        
        heart_mse = evmet.MSE(heart_to, heart_comp, options='MSE', scale='abs', eps=1)
        heart_nmse = evmet.MSE(heart_to, heart_comp, options='NMSE', scale='abs', eps=1)
        heart_rmse = evmet.MSE(heart_to, heart_comp, options='RMSE', scale='abs', eps=1)
        
        # Cálculo de SDR
        resp_sdr = evmet.SDR(resp_to, resp_comp)
        heart_sdr = evmet.SDR(heart_to, heart_comp)
        
        # Cálculo HNRP
        hnrp = evmet.HNRP(resp_to, resp_comp)
        performance = evmet.performance_HNRP(hnrp, heart_to[:minlen_to], 
                                             (heart_to[:minlen_to] + resp_to[:minlen_to]))
        
        # Cálculo de correlación
        corr_resp = evmet.get_correlation(resp_to, resp_comp)
        corr_heart = evmet.get_correlation(heart_to, heart_comp)
        
        # Cálculo de correlación psd
        psd_corr_resp = evmet.psd_correlation(resp_to, resp_comp, sr_des, window=window, N=N,
                                              noverlap=noverlap)
        psd_corr_heart = evmet.psd_correlation(heart_to, heart_comp, sr_des, window=window, N=N,
                                               noverlap=noverlap)
        
        # Cálculo suma de errores
        error_resp = sum(abs(resp_to - resp_comp))
        error_heart = sum(abs(heart_to - heart_comp))
        
        # Creación del diccionario de información
        dict_info_resp = {'name': name_i, 'mse': resp_mse, 'nmse': resp_nmse, 
                          'rmse': resp_rmse, 'SDR': resp_sdr, 
                          'sum error': error_resp, 'correlation': corr_resp, 
                          'psd_correlation': psd_corr_resp, 
                          'HNRP': hnrp, 'p(%)': performance}
        dict_info_heart = {'name': name_i, 'mse': heart_mse, 'nmse': heart_nmse, 
                           'rmse': heart_rmse, 'SDR': heart_sdr, 
                           'sum error': error_heart, 'correlation': corr_heart, 
                           'psd_correlation': psd_corr_heart}
        
        # Registrando para cada sonido
        with open(f'{filepath}/Result Analysis - Respiration.txt', 'a', encoding='utf8') as file:
            file.write(f'{dict_info_resp}\n')
        with open(f'{filepath}/Result Analysis - Heart.txt', 'a', encoding='utf8') as file:
            file.write(f'{dict_info_heart}\n')
        
        ################# RESULTADOS GRAFICOS #################
        plt.figure(figsize=(15,5))
        f1, psd1 = evmet.get_PSD(resp_to, sr_des, window=window, N=N, noverlap=noverlap)
        f2, psd2 = evmet.get_PSD(resp_comp, sr_des, window=window, N=N, noverlap=noverlap)
        plt.plot(f1, 20*np.log10(psd1 + 1e-12), label='Original', color='C0')
        plt.plot(f2, 20*np.log10(psd2 + 1e-12), label='Obtained', color='C1')
        plt.xlabel('Frequency [Hz]')
        plt.title('Respiratory PSDs in dB')
        plt.legend(loc='upper right')
        plt.savefig(f'{filepath}/{name_i}/PSD respiratory.pdf', transparent=True)
        plt.close()
        
        plt.figure(figsize=(15,5))
        f1, psd1 = evmet.get_PSD(heart_to, sr_des, window=window, N=N, noverlap=noverlap)
        f2, psd2 = evmet.get_PSD(heart_comp, sr_des, window=window, N=N, noverlap=noverlap)
        plt.plot(f1, 20*np.log10(psd1 + 1e-12), label='Original', color='C0')
        plt.plot(f2, 20*np.log10(psd2 + 1e-12), label='Obtained', color='C1')
        plt.xlabel('Frequency [Hz]')
        plt.title('Heart PSDs in dB')
        plt.legend(loc='upper right')
        plt.savefig(f'{filepath}/{name_i}/PSD Heart.pdf', transparent=True)
        plt.close()
        
        plt.figure(figsize=(15,5))
        plt.subplot(2,1,1)
        plt.plot(resp_to, label='Original', color='C0')
        plt.plot(resp_comp, label='Obtained', color='C1')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.subplot(2,1,2)
        plt.plot(abs(resp_to - resp_comp))
        plt.xlabel('Samples')
        plt.ylabel(r'Error $|s_{in}(n) - \hat{s}_{in}|$')
        plt.suptitle('Respiratory signal: Original v/s Obtained')
        plt.savefig(f'{filepath}/{name_i}/Respiration Original & Obtained.pdf', transparent=True)
        plt.close()
        
        plt.figure(figsize=(15,5))
        plt.subplot(2,1,1)
        plt.plot(heart_to, label='Original', color='C0')
        plt.plot(heart_comp, label='Obtained', color='C1')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.subplot(2,1,2)
        plt.plot(abs(heart_to - heart_comp))
        plt.xlabel('Samples')
        plt.ylabel(r'Error $|s_{in}(n) - \hat{s}_{in}|$')
        plt.suptitle('Heart signal: Original v/s Obtained')
        plt.savefig(f'{filepath}/{name_i}/Heart Original & Obtained.pdf', transparent=True)
        plt.close()


def evaluate_results(sep_type='to all', version=2, ausc_zone='Anterior'):
    # Parámetros
    sr_des = 44100 // 4
    
    # Definición filepath
    filepath = f'Database_manufacturing/db_HR/Source Separation/v2/{ausc_zone}/'\
                'Seed-0 - x - 0.5_Heart 0.5_Resp 0_White noise/Components'
    filepath_resp = f'Database_manufacturing/db_respiratory/Adapted v{version}/{ausc_zone}/'
    filepath_heart = f'Database_manufacturing/db_heart/Manual combinations v{version}/{ausc_zone}'
    
    if sep_type == 'to all':
        filepath = f'{filepath}/Separation to all'
    elif sep_type == 'on segments':
        filepath = f'{filepath}/Separation on segments'
    elif sep_type == 'masked segments':
        filepath = f'{filepath}/Masking on segments'
    else:
        raise Exception('Opción para "sep_type" no válido.')
    
    # Obtener las carpetas de ids disponibles
    id_folders = [int(i.split(' ')[-1]) for i in os.listdir(filepath) 
                  if i.startswith('id')]
    
    # Ordenar
    id_folders.sort()
    
    for id_rev in id_folders:
        try:
            evaluate_result(sep_type=sep_type, id_rev=id_rev, version=version, 
                            ausc_zone=ausc_zone)
        except:
            print(f'Puede que la base de datos en {ausc_zone} con separación {sep_type} '
                  f'con id {id_rev} no esté completa\n')


def resume_evaluate_results(to_analyze='Respiration', sep_type='to all', version=2, 
                            ausc_zone='Anterior', order_pond=2):
    # Parámetros
    sr_des = 44100 // 4
    
    # Definición filepath
    filepath = f'Database_manufacturing/db_HR/Source Separation/v2/{ausc_zone}/'\
                'Seed-0 - x - 0.5_Heart 0.5_Resp 0_White noise/Components'
    filepath_resp = f'Database_manufacturing/db_respiratory/Adapted v{version}/{ausc_zone}/'
    filepath_heart = f'Database_manufacturing/db_heart/Manual combinations v{version}/{ausc_zone}'

    if sep_type == 'to all':
        filepath = f'{filepath}/Separation to all'
    elif sep_type == 'on segments':
        filepath = f'{filepath}/Separation on segments'
    elif sep_type == 'masked segments':
        filepath = f'{filepath}/Masking on segments'
    else:
        raise Exception('Opción para "sep_type" no válido.')
    
    # Obtener las carpetas de ids disponibles
    id_folders = [int(i.split(' ')[1]) for i in os.listdir(filepath) 
                  if i.startswith('id')]
    # Ordenando
    id_folders.sort()
    id_folders = id_folders[:72]
    
    # Definición de las listas a realizar (común)
    mse_list = list()
    nmse_list = list()
    rmse_list = list()
    sdr_list = list()
    psd_list = list()
    error_list = list()
    corr_list = list()
    
    # Listas solo para respiración
    hnrp_list = list()
    p_list = list()
    
    # Definción de las listas de listas
    mse_total = list()
    nmse_total = list()
    rmse_total = list()
    sdr_total = list()
    psd_total = list()
    error_total = list()
    corr_total = list()
    id_list = list()
    
    # Listas solo para respiración
    hnrp_total = list()
    p_total = list()
    
    # Definición de la tabla a guardar
    if to_analyze == 'Respiration':
        table = PrettyTable(['id', 'MSE', 'NMSE', 'RMSE', 'Error', 'SDR', 'Correlation',
                             'PSD correlation', 'HNRP', 'p(%)'])
    elif to_analyze == 'Heart':
        table = PrettyTable(['id', 'MSE', 'NMSE', 'RMSE', 'Error', 'SDR', 'Correlation',
                             'PSD correlation'])
    else:
        raise Exception('Error en "to_analyze"')
    
    for id_number in tqdm(id_folders, desc=f'Analyzing {sep_type}', ncols=70):        
        # Definición de la dirección a revisar
        id_dir = f'{filepath}/id {id_number}'
        
        with open(f'{id_dir}/Result Analysis - {to_analyze}.txt', 'r', encoding='utf8') as file:
            for line in file:
                # Diccionario de información
                dict_to_rev = literal_eval(line.strip())
                
                # Agregando a las listas
                mse_list.append(float(dict_to_rev['mse']))
                nmse_list.append(float(dict_to_rev['nmse']))
                rmse_list.append(float(dict_to_rev['rmse']))
                sdr_list.append(float(dict_to_rev['SDR']))
                psd_list.append(float(dict_to_rev['psd_correlation']))
                error_list.append(float(dict_to_rev['sum error']))
                corr_list.append(float(dict_to_rev['correlation']))
                
                if to_analyze == 'Respiration':
                    hnrp_list.append(float(dict_to_rev['HNRP']))
                    p_list.append(float(dict_to_rev['p(%)']))
        
        # Transformando las listas en arreglos
        mse_array = np.array(mse_list)
        nmse_array = np.array(nmse_list)
        rmse_array = np.array(rmse_list)
        sdr_array = np.array(sdr_list)
        psd_array = np.array(psd_list)
        hnrp_array = np.array(hnrp_list)
        p_array = np.array(p_list)
        error_array = np.array(error_list)
        corr_array = np.array(corr_list)
        
        # Agregar a la lista de listas
        mse_total.append(mse_array)
        nmse_total.append(nmse_array)
        rmse_total.append(rmse_array)
        sdr_total.append(sdr_array)
        psd_total.append(psd_array)
        hnrp_total.append(hnrp_array)
        p_total.append(p_array)
        error_total.append(error_array)
        corr_total.append(corr_array)
        
        # Escribiendo en la tabla
        if to_analyze == 'Respiration':
            table.add_row([id_number, 
                           "{:.4f} +- {:.4f}".format(mse_array.mean(), mse_array.std()),
                           "{:.4f} +- {:.4f}".format(nmse_array.mean(), nmse_array.std()),
                           "{:.4f} +- {:.4f}".format(rmse_array.mean(), rmse_array.std()),
                           "{:.4f} +- {:.4f}".format(error_array.mean(), error_array.std()),
                           "{:.4f} +- {:.4f}".format(sdr_array.mean(), sdr_array.std()),
                           "{:.4f} +- {:.4f}".format(corr_array.mean(), corr_array.std()),
                           "{:.4f} +- {:.4f}".format(psd_array.mean(), psd_array.std()),
                           "{:.4f} +- {:.4f}".format(hnrp_array.mean(), hnrp_array.std()),
                           "{:.4f} +- {:.4f}".format(p_array.mean(), p_array.std())])
        
        elif to_analyze == 'Heart':
            table.add_row([id_number, 
                           "{:.4f} +- {:.4f}".format(mse_array.mean(), mse_array.std()),
                           "{:.4f} +- {:.4f}".format(nmse_array.mean(), nmse_array.std()),
                           "{:.4f} +- {:.4f}".format(rmse_array.mean(), rmse_array.std()),
                           "{:.4f} +- {:.4f}".format(error_array.mean(), error_array.std()),
                           "{:.4f} +- {:.4f}".format(sdr_array.mean(), sdr_array.std()),
                           "{:.4f} +- {:.4f}".format(corr_array.mean(), corr_array.std()),
                           "{:.4f} +- {:.4f}".format(psd_array.mean(), psd_array.std())])
        
        # Reiniciar las listas
        mse_list = list()
        nmse_list = list()
        rmse_list = list()
        sdr_list = list()
        psd_list = list()
        error_list = list()
        corr_list = list()
        
        # Solo para respiración
        hnrp_list = list()
        p_list = list()
    
    # Finalmente se guarda en un archivo resumen
    with open(f'{filepath}/{to_analyze} - Analysis results.txt', 'w', encoding='utf8') as file:
        file.write(table.get_string())
    
    # Se construye una lista de promedios y desviaciones estándar para generar un .csv
    mse_total_info = [(int(id_folders[num]), i.mean(), i.std())
                      for num, i in enumerate(mse_total) if not np.isnan(i.mean())]
    nmse_total_info = [(int(id_folders[num]), i.mean(), i.std()) 
                       for num, i in enumerate(nmse_total) if not np.isnan(i.mean())]
    rmse_total_info = [(int(id_folders[num]), i.mean(), i.std()) 
                       for num, i in enumerate(rmse_total) if not np.isnan(i.mean())]
    error_total_info = [(int(id_folders[num]), i.mean(), i.std()) 
                        for num, i in enumerate(error_total) if not np.isnan(i.mean())]
    sdr_total_info = [(int(id_folders[num]), i.mean(), i.std()) 
                      for num, i in enumerate(sdr_total) if not np.isnan(i.mean())]
    corr_total_info = [(int(id_folders[num]), i.mean(), i.std()) 
                       for num, i in enumerate(corr_total) if not np.isnan(i.mean())]
    psd_total_info = [(int(id_folders[num]), i.mean(), i.std()) 
                      for num, i in enumerate(psd_total) if not np.isnan(i.mean())]
    
    # Ordenando
    mse_total_info.sort(key=lambda x: x[1])
    nmse_total_info.sort(key=lambda x: x[1])
    rmse_total_info.sort(key=lambda x: x[1])
    error_total_info.sort(key=lambda x: x[1])
    sdr_total_info.sort(key=lambda x: x[1], reverse=True)
    psd_total_info.sort(key=lambda x: x[1], reverse=True)
    corr_total_info.sort(key=lambda x: x[1], reverse=True)
    
    # Definición del diccionario a través del cuál se ordenarán posiciones
    dict_orders = defaultdict(list)
    dict_order_sum = defaultdict(int)
    
    for num, position in enumerate(mse_total_info):
        dict_orders[num].append(position[0])
        dict_order_sum[num] += position[0] ** order_pond
    '''for num, position in enumerate(nmse_total_info):
        dict_orders[num].append(position[0])
        dict_order_sum[num] += position[0] ** order_pond
    for num, position in enumerate(rmse_total_info):
        dict_orders[num].append(position[0])
        dict_order_sum[num] += position[0] ** order_pond
    for num, position in enumerate(error_total_info):
        dict_orders[num].append(position[0])
        dict_order_sum[num] += position[0] ** order_pond'''
    for num, position in enumerate(sdr_total_info):
        dict_orders[num].append(position[0])
        dict_order_sum[num] += position[0] ** order_pond
    for num, position in enumerate(psd_total_info):
        dict_orders[num].append(position[0])
        dict_order_sum[num] += position[0] ** order_pond
    '''for num, position in enumerate(corr_total_info):
        dict_orders[num].append(position[0])
        dict_order_sum[num] += position[0] ** order_pond'''
    
    
    # Finalmente se guarda en un archivo resumen
    with open(f'{filepath}/{to_analyze} - Analysis results ordered.csv', 'w', encoding='utf8') as file:
        file.write('MSE results\n')
        for line in mse_total_info:
            file.write(f'{line}\n')
        file.write('\n\n')
        
        file.write('NMSE results\n')
        for line in nmse_total_info:
            file.write(f'{line}\n')
        file.write('\n\n')
        
        file.write('RMSE results\n')
        for line in rmse_total_info:
            file.write(f'{line}\n')
        file.write('\n\n')
        
        file.write('ERROR results\n')
        for line in error_total_info:
            file.write(f'{line}\n')
        file.write('\n\n')
        
        file.write('SDR results\n')
        for line in sdr_total_info:
            file.write(f'{line}\n')
        file.write('\n\n')
        
        file.write('Correlation results\n')
        for line in corr_total_info:
            file.write(f'{line}\n')
        file.write('\n\n')
        
        file.write('PSD results\n')
        for line in psd_total_info:
            file.write(f'{line}\n')
        file.write('\n\n')
    
    # El cual se resume también a través de sus parámetros
    # Ordenando por las posiciones
    ordered_res = sorted(dict_order_sum.items(), key=lambda item: item[1])
    
    # Finalmente se guarda en un archivo resumen
    with open(f'{filepath}/{to_analyze} Analysis results ordered - [Resume].csv', 
              'w', encoding='utf8') as file:
        file.write('Resume results & Positions\n')
        for pos, _ in ordered_res:
            file.write(f'{pos},{dict_orders[pos]}\n')
        file.write('\n\n')
    
    
    # Propiedades diagramas de caja
    meanpointprops = dict(marker='.', markerfacecolor='red', markeredgecolor='red')
    
    # Creación de diagramas de caja
    plt.figure(figsize=(15,5))
    plt.boxplot(mse_total, labels=id_folders, showmeans=True, meanprops=meanpointprops)
    plt.ylabel('MSE')
    plt.title(f'{to_analyze} MSE Boxplot')
    plt.savefig(f'{filepath}/{to_analyze} mse_boxplot.pdf', transparent=True)
    plt.close()
    
    plt.figure(figsize=(15,5))
    plt.boxplot(nmse_total, labels=id_folders, showmeans=True, meanprops=meanpointprops)
    plt.ylabel('NMSE')
    plt.title(f'{to_analyze} NMSE Boxplot')
    plt.savefig(f'{filepath}/{to_analyze} nmse_boxplot.pdf', transparent=True)
    plt.close()
    
    plt.figure(figsize=(15,5))
    plt.boxplot(rmse_total, labels=id_folders, showmeans=True, meanprops=meanpointprops)
    plt.ylabel('RMSE')
    plt.title(f'{to_analyze} RMSE Boxplot')
    plt.savefig(f'{filepath}/{to_analyze} rmse_boxplot.pdf', transparent=True)
    plt.close()
    
    plt.figure(figsize=(15,5))
    plt.boxplot(error_total, labels=id_folders, showmeans=True, meanprops=meanpointprops)
    plt.ylabel('Error')
    plt.title(f'{to_analyze} Error total Boxplot')
    plt.savefig(f'{filepath}/{to_analyze} Error_boxplot.pdf', transparent=True)
    plt.close()
    
    plt.figure(figsize=(15,5))
    plt.boxplot(sdr_total, labels=id_folders, showmeans=True, meanprops=meanpointprops)
    plt.ylabel('SDR')
    plt.title(f'{to_analyze} SDR Boxplot')
    plt.savefig(f'{filepath}/{to_analyze} sdr_boxplot.pdf', transparent=True)
    plt.close()
    
    plt.figure(figsize=(15,5))
    plt.boxplot(psd_total, labels=id_folders, showmeans=True, meanprops=meanpointprops)
    plt.ylabel('PSD')
    plt.title(f'{to_analyze} PSD Correlation Boxplot')
    plt.savefig(f'{filepath}/{to_analyze} psd_boxplot.pdf', transparent=True)
    plt.close()
    
    plt.figure(figsize=(15,5))
    plt.boxplot(corr_total, labels=id_folders, showmeans=True, meanprops=meanpointprops)
    plt.ylabel('Correlation')
    plt.title(f'{to_analyze} Correlation Boxplot')
    plt.savefig(f'{filepath}/{to_analyze} corr_boxplot.pdf', transparent=True)
    plt.close()
    
    if to_analyze == 'Respiration':
        plt.figure(figsize=(15,5))
        plt.boxplot(hnrp_total, labels=id_folders, showmeans=True, meanprops=meanpointprops)
        plt.ylabel('HNRP')
        plt.title(f'{to_analyze} HNRP Boxplot')
        plt.savefig(f'{filepath}/{to_analyze} hnrp_boxplot.pdf', transparent=True)
        plt.close()
        
        plt.figure(figsize=(15,5))
        plt.boxplot(p_total, labels=id_folders, showmeans=True, meanprops=meanpointprops)
        plt.ylabel('p(%)')
        plt.title(f'{to_analyze} p(%) Boxplot')
        plt.savefig(f'{filepath}/{to_analyze} p_boxplot.pdf', transparent=True)
        plt.close()


def compare_result(N, noverlap, padding, N_lax, N_fade, n_components, beta, 
                   repeat, window, l1_ratio, alpha_nmf, clustering, 
                   to_analyze='Respiration', ausc_zone='Anterior'):    
    # Definición filepath
    filepath = f'Database_manufacturing/db_HR/Source Separation/v2/{ausc_zone}/'\
                'Seed-0 - x - 0.5_Heart 0.5_Resp 0_White noise/Components'

    # Id's que cumplen  los parámetros
    with open(f'{filepath}/Separation to all/Simulation register.txt', 
              'r', encoding='utf8') as file:
        for line in file:
            dict_to_all = literal_eval(line.strip())
            
            # Para separation on segments
            if (dict_to_all['N'] == N and 
                dict_to_all['noverlap'] == noverlap and
                dict_to_all['n_components'] == n_components and 
                dict_to_all['padding'] == padding and 
                dict_to_all['repeat'] == repeat and
                dict_to_all['window'] == window and
                dict_to_all['l1_ratio'] == l1_ratio and
                dict_to_all['alpha_nmf'] == alpha_nmf and
                dict_to_all['beta'] == beta and
                dict_to_all['clustering'] == clustering):
                id_to_all = dict_to_all['id']
                break
    
    with open(f'{filepath}/Separation on segments/Simulation register.txt', 
              'r', encoding='utf8') as file:
        for line in file:
            dict_on_seg = literal_eval(line.strip())
            
            # Para separation on segments
            if (dict_on_seg['N'] == N and 
                dict_on_seg['noverlap'] == noverlap and
                dict_on_seg['n_components'] == n_components and 
                dict_on_seg['padding'] == padding and 
                dict_on_seg['repeat'] == repeat and
                dict_on_seg['window'] == window and
                dict_on_seg['l1_ratio'] == l1_ratio and
                dict_on_seg['alpha_nmf'] == alpha_nmf and
                dict_on_seg['N_lax'] == N_lax and
                dict_on_seg['N_fade'] == N_fade and
                dict_on_seg['beta'] == beta and
                dict_on_seg['clustering'] == clustering):
                id_on_segments = dict_on_seg['id']
                break
    
    with open(f'{filepath}/Masking on segments/Simulation register.txt', 
              'r', encoding='utf8') as file:
        for line in file:
            dict_mask_seg = literal_eval(line.strip())
            
            # Para separation on segments
            if (dict_mask_seg['N'] == N and 
                dict_mask_seg['noverlap'] == noverlap and
                dict_mask_seg['n_components'] == n_components and 
                dict_mask_seg['padding'] == padding and 
                dict_mask_seg['repeat'] == repeat and
                dict_mask_seg['window'] == window and
                dict_mask_seg['l1_ratio'] == l1_ratio and
                dict_mask_seg['alpha_nmf'] == alpha_nmf and
                dict_mask_seg['N_lax'] == N_lax and
                dict_mask_seg['N_fade'] == N_fade and
                dict_mask_seg['beta'] == beta and
                dict_mask_seg['clustering'] == clustering):
                id_mask_segments = dict_mask_seg['id']
                break
    
    # Prints
    print(f'id to all: {id_to_all}')
    print(f'id on segments: {id_on_segments}')
    print(f'id mask segments: {id_mask_segments}')
    
    # Definición del nombre de las propiedades
    prop_name = f'all-{id_to_all} seg-{id_on_segments} mask{id_mask_segments}'
    
    # Definición de las direcciones con la id
    filepath_to_all = f'{filepath}/Separation to all/id {id_to_all}'
    filepath_on_segments = f'{filepath}/Separation on segments/id {id_on_segments}'
    filepath_mask_segments = f'{filepath}/Masking on segments/id {id_mask_segments}'
    
    # Definción de las listas de listas
    mse_total = list()
    nmse_total = list()
    rmse_total = list()
    sdr_total = list()
    psd_total = list()
    error_total = list()
    corr_total = list()
    
    # Listas solo para respiración
    hnrp_total = list()
    p_total = list()
    
    # Archivos a revisar
    with open(f'{filepath_to_all}/Result Analysis - {to_analyze}.txt', 
              'r', encoding='utf8') as file:
        # Definición de las listas a realizar (común)
        mse_list = list()
        nmse_list = list()
        rmse_list = list()
        sdr_list = list()
        psd_list = list()
        error_list = list()
        corr_list = list()
        
        # Listas solo para respiración
        hnrp_list = list()
        p_list = list()
        
        for line in file:
            # Diccionario de información
            dict_to_rev = literal_eval(line.strip())
            
            # Agregando a las listas
            mse_list.append(float(dict_to_rev['mse']))
            nmse_list.append(float(dict_to_rev['nmse']))
            rmse_list.append(float(dict_to_rev['rmse']))
            sdr_list.append(float(dict_to_rev['SDR']))
            psd_list.append(float(dict_to_rev['psd_correlation']))
            error_list.append(float(dict_to_rev['sum error']))
            corr_list.append(float(dict_to_rev['correlation']))
            
            if to_analyze == 'Respiration':
                hnrp_list.append(float(dict_to_rev['HNRP']))
                p_list.append(float(dict_to_rev['p(%)']))
        
        # Agregar a la lista de listas
        mse_total.append(mse_list)
        nmse_total.append(nmse_list)
        rmse_total.append(rmse_list)
        sdr_total.append(sdr_list)
        psd_total.append(psd_list)
        hnrp_total.append(hnrp_list)
        p_total.append(p_list)
        error_total.append(error_list)
        corr_total.append(corr_list)
    
    with open(f'{filepath_on_segments}/Result Analysis - {to_analyze}.txt', 
              'r', encoding='utf8') as file:
        # Definición de las listas a realizar (común)
        mse_list = list()
        nmse_list = list()
        rmse_list = list()
        sdr_list = list()
        psd_list = list()
        error_list = list()
        corr_list = list()
        
        # Listas solo para respiración
        hnrp_list = list()
        p_list = list()
                
        for line in file:
            # Diccionario de información
            dict_to_rev = literal_eval(line.strip())
            
            # Agregando a las listas
            mse_list.append(float(dict_to_rev['mse']))
            nmse_list.append(float(dict_to_rev['nmse']))
            rmse_list.append(float(dict_to_rev['rmse']))
            sdr_list.append(float(dict_to_rev['SDR']))
            psd_list.append(float(dict_to_rev['psd_correlation']))
            error_list.append(float(dict_to_rev['sum error']))
            corr_list.append(float(dict_to_rev['correlation']))
            
            if to_analyze == 'Respiration':
                hnrp_list.append(float(dict_to_rev['HNRP']))
                p_list.append(float(dict_to_rev['p(%)']))
        
        # Agregar a la lista de listas
        mse_total.append(mse_list)
        nmse_total.append(nmse_list)
        rmse_total.append(rmse_list)
        sdr_total.append(sdr_list)
        psd_total.append(psd_list)
        hnrp_total.append(hnrp_list)
        p_total.append(p_list)
        error_total.append(error_list)
        corr_total.append(corr_list)
    
    with open(f'{filepath_mask_segments}/Result Analysis - {to_analyze}.txt', 
              'r', encoding='utf8') as file:
        # Definición de las listas a realizar (común)
        mse_list = list()
        nmse_list = list()
        rmse_list = list()
        sdr_list = list()
        psd_list = list()
        error_list = list()
        corr_list = list()
        
        # Listas solo para respiración
        hnrp_list = list()
        p_list = list()
                
        for line in file:
            # Diccionario de información
            dict_to_rev = literal_eval(line.strip())
            
            # Agregando a las listas
            mse_list.append(float(dict_to_rev['mse']))
            nmse_list.append(float(dict_to_rev['nmse']))
            rmse_list.append(float(dict_to_rev['rmse']))
            sdr_list.append(float(dict_to_rev['SDR']))
            psd_list.append(float(dict_to_rev['psd_correlation']))
            error_list.append(float(dict_to_rev['sum error']))
            corr_list.append(float(dict_to_rev['correlation']))
            
            if to_analyze == 'Respiration':
                hnrp_list.append(float(dict_to_rev['HNRP']))
                p_list.append(float(dict_to_rev['p(%)']))
        
        # Agregar a la lista de listas
        mse_total.append(mse_list)
        nmse_total.append(nmse_list)
        rmse_total.append(rmse_list)
        sdr_total.append(sdr_list)
        psd_total.append(psd_list)
        hnrp_total.append(hnrp_list)
        p_total.append(p_list)
        error_total.append(error_list)
        corr_total.append(corr_list)
    
    
    # Definición de la dirección a guardar los resultados
    filepath_to_save = f'{filepath}/Results/{prop_name}'
    
    # Creación de la carpeta donde se almacenarán los resultados
    if not os.path.isdir(filepath_to_save):
        os.makedirs(filepath_to_save)
    
    # Creación de diagramas de caja
    plt.boxplot(mse_total)
    plt.xticks([1, 2, 3], ['NMF to\nall', 'NMF on\nsegments', 'NMF masked\nsegments'])
    plt.ylabel('MSE')
    plt.title(f'{to_analyze} MSE Boxplot')
    plt.savefig(f'{filepath_to_save}/{to_analyze} mse_boxplot.pdf', transparent=True)
    plt.close()
    
    plt.boxplot(nmse_total)
    plt.xticks([1, 2, 3], ['NMF to\nall', 'NMF on\nsegments', 'NMF masked\nsegments'])
    plt.ylabel('NMSE')
    plt.title(f'{to_analyze} NMSE Boxplot')
    plt.savefig(f'{filepath_to_save}/{to_analyze} nmse_boxplot.pdf', transparent=True)
    plt.close()
    
    plt.boxplot(rmse_total)
    plt.xticks([1, 2, 3], ['NMF to\nall', 'NMF on\nsegments', 'NMF masked\nsegments'])
    plt.ylabel('RMSE')
    plt.title(f'{to_analyze} RMSE Boxplot')
    plt.savefig(f'{filepath_to_save}/{to_analyze} rmse_boxplot.pdf', transparent=True)
    plt.close()
    
    plt.boxplot(error_total)
    plt.xticks([1, 2, 3], ['NMF to\nall', 'NMF on\nsegments', 'NMF masked\nsegments'])
    plt.ylabel('Error')
    plt.title(f'{to_analyze} Error total Boxplot')
    plt.savefig(f'{filepath_to_save}/{to_analyze} Error_boxplot.pdf', transparent=True)
    plt.close()
    
    plt.boxplot(sdr_total)
    plt.xticks([1, 2, 3], ['NMF to\nall', 'NMF on\nsegments', 'NMF masked\nsegments'])
    plt.ylabel('SDR')
    plt.title(f'{to_analyze} SDR Boxplot')
    plt.savefig(f'{filepath_to_save}/{to_analyze} sdr_boxplot.pdf', transparent=True)
    plt.close()
    
    plt.boxplot(psd_total)
    plt.xticks([1, 2, 3], ['NMF to\nall', 'NMF on\nsegments', 'NMF masked\nsegments'])
    plt.ylabel('PSD correlation')
    plt.title(f'{to_analyze} PSD Correlation Boxplot')
    plt.savefig(f'{filepath_to_save}/{to_analyze} psd_boxplot.pdf', transparent=True)
    plt.close()
    
    plt.boxplot(corr_total)
    plt.xticks([1, 2, 3], ['NMF to\nall', 'NMF on\nsegments', 'NMF masked\nsegments'])
    plt.ylabel('Correlation')
    plt.title(f'{to_analyze} Correlation Boxplot')
    plt.savefig(f'{filepath_to_save}/{to_analyze} corr_boxplot.pdf', transparent=True)
    plt.close()
    
    if to_analyze == 'Respiration':
        plt.boxplot(hnrp_total)
        plt.xticks([1, 2, 3], ['NMF to\nall', 'NMF on\nsegments', 'NMF masked\nsegments'])
        plt.ylabel('HNRP')
        plt.title(f'{to_analyze} HNRP Boxplot')
        plt.savefig(f'{filepath_to_save}/{to_analyze} hnrp_boxplot.pdf', transparent=True)
        plt.close()

        plt.boxplot(p_total)
        plt.xticks([1, 2, 3], ['NMF to\nall', 'NMF on\nsegments', 'NMF masked\nsegments'])
        plt.ylabel('p(%)')
        plt.title(f'{to_analyze} p(%) Boxplot')
        plt.savefig(f'{filepath_to_save}/{to_analyze} p_boxplot.pdf', transparent=True)
        plt.close()
    
    with open(f'{filepath_to_save}/{to_analyze} Resume.txt', 'w', encoding='utf8') as file:
        file.write(f'MSE all: {np.mean(mse_total[0])} +- {np.std(mse_total[0])}\n')
        file.write(f'MSE seg: {np.mean(mse_total[1])} +- {np.std(mse_total[1])}\n')
        file.write(f'MSE mask: {np.mean(mse_total[2])} +- {np.std(mse_total[2])}\n\n')
        
        file.write(f'NMSE all: {np.mean(nmse_total[0])} +- {np.std(nmse_total[0])}\n')
        file.write(f'NMSE seg: {np.mean(nmse_total[1])} +- {np.std(nmse_total[1])}\n')
        file.write(f'NMSE mask: {np.mean(nmse_total[2])} +- {np.std(nmse_total[2])}\n\n')
        
        file.write(f'RMSE all: {np.mean(rmse_total[0])} +- {np.std(rmse_total[0])}\n')
        file.write(f'RMSE seg: {np.mean(rmse_total[1])} +- {np.std(rmse_total[1])}\n')
        file.write(f'RMSE mask: {np.mean(rmse_total[2])} +- {np.std(rmse_total[2])}\n\n')
        
        file.write(f'Error all: {np.mean(error_total[0])} +- {np.std(error_total[0])}\n')
        file.write(f'Error seg: {np.mean(error_total[1])} +- {np.std(error_total[1])}\n')
        file.write(f'Error mask: {np.mean(error_total[2])} +- {np.std(error_total[2])}\n\n')
        
        file.write(f'HNRP all: {np.mean(hnrp_total[0])} +- {np.std(hnrp_total[0])}\n')
        file.write(f'HNRP seg: {np.mean(hnrp_total[1])} +- {np.std(hnrp_total[1])}\n')
        file.write(f'HNRP mask: {np.mean(hnrp_total[2])} +- {np.std(hnrp_total[2])}\n\n')
        
        file.write(f'p(%) all: {np.mean(p_total[0])} +- {np.std(p_total[0])}\n')
        file.write(f'p(%) seg: {np.mean(p_total[1])} +- {np.std(p_total[1])}\n')
        file.write(f'p(%) mask: {np.mean(p_total[2])} +- {np.std(p_total[2])}\n\n')
        
        file.write(f'Correlation all: {np.mean(corr_total[0])} +- {np.std(corr_total[0])}\n')
        file.write(f'Correlation seg: {np.mean(corr_total[1])} +- {np.std(corr_total[1])}\n')
        file.write(f'Correlation mask: {np.mean(corr_total[2])} +- {np.std(corr_total[2])}\n\n')
        
        file.write(f'PSD Correlation all: {np.mean(psd_total[0])} +- {np.std(psd_total[0])}\n')
        file.write(f'PSD Correlation seg: {np.mean(psd_total[1])} +- {np.std(psd_total[1])}\n')
        file.write(f'PSD Correlation mask: {np.mean(psd_total[2])} +- {np.std(psd_total[2])}\n\n')
        
        file.write(f'SDR all: {np.mean(sdr_total[0])} +- {np.std(sdr_total[0])}\n')
        file.write(f'SDR seg: {np.mean(sdr_total[1])} +- {np.std(sdr_total[1])}\n')
        file.write(f'SDR mask: {np.mean(sdr_total[2])} +- {np.std(sdr_total[2])}\n\n')
        
        file.write(f'Error all: {np.mean(nmse_total[0])} +- {np.std(nmse_total[0])}\n')
        file.write(f'Error seg: {np.mean(nmse_total[1])} +- {np.std(nmse_total[1])}\n')
        file.write(f'Error mask: {np.mean(nmse_total[2])} +- {np.std(nmse_total[2])}\n')
    
    with open(f'{filepath_to_save}/{to_analyze} Paper.txt', 'w', encoding='utf8') as file:
        file.write('MSE\n')
        file.write("{:.6f} \\pm {:.6f},".format(np.mean(mse_total[0]), np.std(mse_total[0])))
        file.write("{:.6f} \\pm {:.6f},".format(np.mean(mse_total[1]), np.std(mse_total[1])))
        file.write("{:.6f} \\pm {:.6f}\n\n".format(np.mean(mse_total[2]), np.std(mse_total[2])))
        
        file.write('Correlation\n')
        file.write("{:.5f} \\pm {:.5f},".format(np.mean(corr_total[0]), np.std(corr_total[0])))
        file.write("{:.5f} \\pm {:.5f},".format(np.mean(corr_total[1]), np.std(corr_total[1])))
        file.write("{:.5f} \\pm {:.5f}\n\n".format(np.mean(corr_total[2]), np.std(corr_total[2])))
        
        file.write('PSD Correlation\n')
        file.write("{:.5f} \\pm {:.5f},".format(np.mean(psd_total[0]), np.std(psd_total[0])))
        file.write("{:.5f} \\pm {:.5f},".format(np.mean(psd_total[1]), np.std(psd_total[1])))
        file.write("{:.5f} \\pm {:.5f}\n\n".format(np.mean(psd_total[2]), np.std(psd_total[2])))
        
        file.write('SDR\n')
        file.write("{:.4f} \\pm {:.4f},".format(np.mean(sdr_total[0]), np.std(sdr_total[0])))
        file.write("{:.4f} \\pm {:.4f},".format(np.mean(sdr_total[1]), np.std(sdr_total[1])))
        file.write("{:.4f} \\pm {:.4f}\n\n".format(np.mean(sdr_total[2]), np.std(sdr_total[2])))


def compare_results(to_analyze='Respiration', ausc_zone='Both'):
    # Parametros de separación
    N = [1024]      #[512, 1024]
    n_components = [2, 5, 10, 20, 50]
    beta = [1, 2]
    repeat = 0      # 4
    # sr_des = 44100 // 4
    window = 'hann'
    l1_ratio = 0    # 1
    alpha = 0       # 0.03
    # assign_method = 'auto'
    clustering = True
    # dec_criteria = 'vote'
    # H_binary = True
    # only_centroid = False

    for n_comps in n_components:
        for beta_i in beta:
            for n in N:
                noverlap = [int(0.75 * n)]      #[int(0.5 * n), int(0.75 * n), int(0.95 * n)]
                padding = 3 * n
                N_lax = 100 if N == 1024 else 50
                N_fade = 100 if N == 1024 else 50
                for nov in noverlap:
                    compare_result(N=n, noverlap=nov, padding=padding, N_lax=N_lax, 
                                   N_fade=N_fade, n_components=n_comps, beta=beta_i, 
                                   repeat=repeat, window=window, l1_ratio=l1_ratio, 
                                   alpha_nmf=alpha, clustering=clustering, 
                                   to_analyze=to_analyze, ausc_zone=ausc_zone)


def print_compare_results(sep_type='all', metric_to_print='MSE', simulations=None):
    ''' Rutina que permite imprimir los resultados registrados para cada simulación
    en la carpeta Results 
    '''
    # Carpeta objetivo
    filepath = 'Database_manufacturing/db_HR/Source Separation/v2/Both/'\
               'Seed-0 - x - 0.5_Heart 0.5_Resp 0_White noise/Components/Results'
    
    # Simulaciones en la carpeta objetivo
    if simulations is None:
        simulations = np.arange(len([i for i in os.listdir(filepath)])) + 1

    if metric_to_print == 'MSE':
        line_to_rev = 2
    elif metric_to_print == 'Correlation':
        line_to_rev = 5
    elif metric_to_print == 'PSD':
        line_to_rev = 8
    elif metric_to_print == 'SDR':
        line_to_rev = 11


    # Para cada simulación
    for sim in simulations:
        # Definición de la carpeta de simulación
        if sep_type == 'all':
            filepath_to = f'{filepath}/all-{sim} seg-{sim} mask{sim}'
            
        elif sep_type == 'to all':
            filepath_to = f'{filepath}/all-{sim}'
        
        elif sep_type == 'on segments':
            filepath_to = f'{filepath}/seg-{sim}'
        
        elif sep_type == 'masked segments':
            filepath_to = f'{filepath}/mask{sim}'
        
        
        with open(f'{filepath_to}/Respiration Paper.txt', 'r', encoding='utf8') as file:
            lines = file.readlines()
            
            print(lines[line_to_rev-1].strip())


def compare_result_specific(sep_type, N, noverlap, padding, N_lax, N_fade, 
                            n_components, beta, repeat, window, l1_ratio, 
                            alpha_nmf, clustering, to_analyze='Respiration', 
                            ausc_zone='Anterior'):    
    # Definición filepath
    filepath = f'Database_manufacturing/db_HR/Source Separation/v2/{ausc_zone}/'\
                'Seed-0 - x - 0.5_Heart 0.5_Resp 0_White noise/Components'

    # Definción de las listas de listas
    mse_total = list()
    nmse_total = list()
    rmse_total = list()
    sdr_total = list()
    psd_total = list()
    error_total = list()
    corr_total = list()
    
    # Listas solo para respiración
    hnrp_total = list()
    p_total = list()
    
    # Id's que cumplen  los parámetros
    if sep_type == 'to all':
        with open(f'{filepath}/Separation to all/Simulation register.txt', 
                'r', encoding='utf8') as file:
            for line in file:
                dict_to_all = literal_eval(line.strip())
                
                # Para separation on segments
                if (dict_to_all['N'] == N and 
                    dict_to_all['noverlap'] == noverlap and
                    dict_to_all['n_components'] == n_components and 
                    dict_to_all['padding'] == padding and 
                    dict_to_all['repeat'] == repeat and
                    dict_to_all['window'] == window and
                    dict_to_all['l1_ratio'] == l1_ratio and
                    dict_to_all['alpha_nmf'] == alpha_nmf and
                    dict_to_all['beta'] == beta and
                    dict_to_all['clustering'] == clustering):
                    id_to_all = dict_to_all['id']
                    break
        
        print(f'id to all: {id_to_all}')
        
        # Definición del nombre de las propiedades
        prop_name = f'all-{id_to_all}'
        
        # Definición de las direcciones con la id
        filepath_to_all = f'{filepath}/Separation to all/id {id_to_all}'
        
        # Archivos a revisar
        with open(f'{filepath_to_all}/Result Analysis - {to_analyze}.txt', 
                'r', encoding='utf8') as file:
            # Definición de las listas a realizar (común)
            mse_list = list()
            nmse_list = list()
            rmse_list = list()
            sdr_list = list()
            psd_list = list()
            error_list = list()
            corr_list = list()
            
            # Listas solo para respiración
            hnrp_list = list()
            p_list = list()
            
            for line in file:
                # Diccionario de información
                dict_to_rev = literal_eval(line.strip())
                
                # Agregando a las listas
                mse_list.append(float(dict_to_rev['mse']))
                nmse_list.append(float(dict_to_rev['nmse']))
                rmse_list.append(float(dict_to_rev['rmse']))
                sdr_list.append(float(dict_to_rev['SDR']))
                psd_list.append(float(dict_to_rev['psd_correlation']))
                error_list.append(float(dict_to_rev['sum error']))
                corr_list.append(float(dict_to_rev['correlation']))
                
                if to_analyze == 'Respiration':
                    hnrp_list.append(float(dict_to_rev['HNRP']))
                    p_list.append(float(dict_to_rev['p(%)']))
            
            # Agregar a la lista de listas
            mse_total.append(mse_list)
            nmse_total.append(nmse_list)
            rmse_total.append(rmse_list)
            sdr_total.append(sdr_list)
            psd_total.append(psd_list)
            hnrp_total.append(hnrp_list)
            p_total.append(p_list)
            error_total.append(error_list)
            corr_total.append(corr_list)
        
    
    elif sep_type == 'on segments':
        with open(f'{filepath}/Separation on segments/Simulation register.txt', 
                'r', encoding='utf8') as file:
            for line in file:
                dict_on_seg = literal_eval(line.strip())
                
                # Para separation on segments
                if (dict_on_seg['N'] == N and 
                    dict_on_seg['noverlap'] == noverlap and
                    dict_on_seg['n_components'] == n_components and 
                    dict_on_seg['padding'] == padding and 
                    dict_on_seg['repeat'] == repeat and
                    dict_on_seg['window'] == window and
                    dict_on_seg['l1_ratio'] == l1_ratio and
                    dict_on_seg['alpha_nmf'] == alpha_nmf and
                    dict_on_seg['N_lax'] == N_lax and
                    dict_on_seg['N_fade'] == N_fade and
                    dict_on_seg['beta'] == beta and
                    dict_on_seg['clustering'] == clustering):
                    id_on_segments = dict_on_seg['id']
                    break
        
        
        print(f'id on segments: {id_on_segments}')
        
        # Definición del nombre de las propiedades
        prop_name = f'seg-{id_on_segments}'

        # Definición de las direcciones con la id
        filepath_on_segments = f'{filepath}/Separation on segments/id {id_on_segments}'
        
            # Archivos a revisar
        with open(f'{filepath_on_segments}/Result Analysis - {to_analyze}.txt', 
                'r', encoding='utf8') as file:
            # Definición de las listas a realizar (común)
            mse_list = list()
            nmse_list = list()
            rmse_list = list()
            sdr_list = list()
            psd_list = list()
            error_list = list()
            corr_list = list()
            
            # Listas solo para respiración
            hnrp_list = list()
            p_list = list()
                    
            for line in file:
                # Diccionario de información
                dict_to_rev = literal_eval(line.strip())
                
                # Agregando a las listas
                mse_list.append(float(dict_to_rev['mse']))
                nmse_list.append(float(dict_to_rev['nmse']))
                rmse_list.append(float(dict_to_rev['rmse']))
                sdr_list.append(float(dict_to_rev['SDR']))
                psd_list.append(float(dict_to_rev['psd_correlation']))
                error_list.append(float(dict_to_rev['sum error']))
                corr_list.append(float(dict_to_rev['correlation']))
                
                if to_analyze == 'Respiration':
                    hnrp_list.append(float(dict_to_rev['HNRP']))
                    p_list.append(float(dict_to_rev['p(%)']))
            
            # Agregar a la lista de listas
            mse_total.append(mse_list)
            nmse_total.append(nmse_list)
            rmse_total.append(rmse_list)
            sdr_total.append(sdr_list)
            psd_total.append(psd_list)
            hnrp_total.append(hnrp_list)
            p_total.append(p_list)
            error_total.append(error_list)
            corr_total.append(corr_list)
    
    
    elif sep_type == 'masked segments':
        with open(f'{filepath}/Masking on segments/Simulation register.txt', 
                'r', encoding='utf8') as file:
            for line in file:
                dict_mask_seg = literal_eval(line.strip())
                
                # Para separation on segments
                if (dict_mask_seg['N'] == N and 
                    dict_mask_seg['noverlap'] == noverlap and
                    dict_mask_seg['n_components'] == n_components and 
                    dict_mask_seg['padding'] == padding and 
                    dict_mask_seg['repeat'] == repeat and
                    dict_mask_seg['window'] == window and
                    dict_mask_seg['l1_ratio'] == l1_ratio and
                    dict_mask_seg['alpha_nmf'] == alpha_nmf and
                    dict_mask_seg['N_lax'] == N_lax and
                    dict_mask_seg['N_fade'] == N_fade and
                    dict_mask_seg['beta'] == beta and
                    dict_mask_seg['clustering'] == clustering):
                    id_mask_segments = dict_mask_seg['id']
                    break
        
        print(f'id mask segments: {id_mask_segments}')

        # Definición del nombre de las propiedades
        prop_name = f'mask{id_mask_segments}'
    
        # Definición de las direcciones con la id
        filepath_mask_segments = f'{filepath}/Masking on segments/id {id_mask_segments}'
    
        # Archivos a revisar
        with open(f'{filepath_mask_segments}/Result Analysis - {to_analyze}.txt', 
                'r', encoding='utf8') as file:
            # Definición de las listas a realizar (común)
            mse_list = list()
            nmse_list = list()
            rmse_list = list()
            sdr_list = list()
            psd_list = list()
            error_list = list()
            corr_list = list()
            
            # Listas solo para respiración
            hnrp_list = list()
            p_list = list()
                    
            for line in file:
                # Diccionario de información
                dict_to_rev = literal_eval(line.strip())
                
                # Agregando a las listas
                mse_list.append(float(dict_to_rev['mse']))
                nmse_list.append(float(dict_to_rev['nmse']))
                rmse_list.append(float(dict_to_rev['rmse']))
                sdr_list.append(float(dict_to_rev['SDR']))
                psd_list.append(float(dict_to_rev['psd_correlation']))
                error_list.append(float(dict_to_rev['sum error']))
                corr_list.append(float(dict_to_rev['correlation']))
                
                if to_analyze == 'Respiration':
                    hnrp_list.append(float(dict_to_rev['HNRP']))
                    p_list.append(float(dict_to_rev['p(%)']))
            
            # Agregar a la lista de listas
            mse_total.append(mse_list)
            nmse_total.append(nmse_list)
            rmse_total.append(rmse_list)
            sdr_total.append(sdr_list)
            psd_total.append(psd_list)
            hnrp_total.append(hnrp_list)
            p_total.append(p_list)
            error_total.append(error_list)
            corr_total.append(corr_list)
    
    
    # Definición de la dirección a guardar los resultados
    filepath_to_save = f'{filepath}/Results/{prop_name}'
    
    # Creación de la carpeta donde se almacenarán los resultados
    if not os.path.isdir(filepath_to_save):
        os.makedirs(filepath_to_save)
    
        
    with open(f'{filepath_to_save}/{to_analyze} Paper.txt', 'w', encoding='utf8') as file:
        file.write('MSE\n')
        file.write("{:.6f} \\pm {:.6f}\n\n".format(np.mean(mse_total[0]), np.std(mse_total[0])))
        
        file.write('Correlation\n')
        file.write("{:.5f} \\pm {:.5f}\n\n".format(np.mean(corr_total[0]), np.std(corr_total[0])))
        
        file.write('PSD Correlation\n')
        file.write("{:.5f} \\pm {:.5f}\n\n".format(np.mean(psd_total[0]), np.std(psd_total[0])))
        
        file.write('SDR\n')
        file.write("{:.4f} \\pm {:.4f}\n\n".format(np.mean(sdr_total[0]), np.std(sdr_total[0])))


def compare_results_specific(sep_type, to_analyze='Respiration', ausc_zone='Both'):
    # Parametros de separación
    N = [1024]      #[512, 1024]
    n_components = [2]
    beta = [1,2]
    repeat = 0      # 4
    # sr_des = 44100 // 4
    window = 'hann'
    l1_ratio = 0    # 1
    alpha = 0       # 0.03
    # assign_method = 'auto'
    clustering = True
    # dec_criteria = 'vote'
    # H_binary = True
    # only_centroid = False

    for n_comps in n_components:
        for beta_i in beta:
            for n in N:
                noverlap = [int(0.90 * n)]      #[int(0.5 * n), int(0.75 * n), int(0.95 * n)]
                padding = 3 * n
                N_lax = 100 if N == 1024 else 50
                N_fade = 100 if N == 1024 else 50
                for nov in noverlap:
                    compare_result_specific(sep_type=sep_type, N=n, noverlap=nov, padding=padding, 
                                            N_lax=N_lax, N_fade=N_fade, n_components=n_comps, 
                                            beta=beta_i, repeat=repeat, window=window, 
                                            l1_ratio=l1_ratio, alpha_nmf=alpha, 
                                            clustering=clustering, to_analyze=to_analyze, 
                                            ausc_zone=ausc_zone)


def testing_module_1():
    # Parametros de separación
    N = 1024 * 2
    noverlap = N // 2
    n_components = 2
    padding = 3 * N
    repeat = 4
    sr_des = 44100 // 4
    window = 'hann'
    assign_method = 'manual'
    
    # Definición filepath
    filepath = 'Database_manufacturing/db_HR/Source Separation/v2/Anterior/'\
                'Seed-0 - x - 0.5_Heart 0.5_Resp 0_White noise'
                
    dir_file = f'{filepath}/HR 1 148_1b1_Al_sc_Meditron Seed[10315]_S1[63]_S2[62].wav'
    filepath_to_save = f'{filepath}/Components/Separation to all'

    filepath_to_save_id = f'{filepath_to_save}/id 1'

    # Preguntar si es que la carpeta que almacenará las imágenes se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(filepath_to_save_id):
        os.makedirs(filepath_to_save_id)
        
    nmf_applied_all(dir_file, filepath_to_save_id, sr_des, assign_method, clustering=False, 
                    n_components=n_components, N=N, noverlap=noverlap, padding=padding, 
                    repeat=repeat, window=window, whole=False, alpha_wiener=1, 
                    filter_out='wiener', init='random', solver='cd', beta=2, 
                    tol=1e-4, max_iter=1000, alpha_nmf=0, l1_ratio=0, 
                    random_state=0, W_0=None, H_0=None, scale='abs', version=2, 
                    ausc_zone='Anterior')




# Módulo de testeo
if __name__ == '__main__':
    # Parámetros
    sep_type = 'to all'                # 'to all', 'on segments', 'masked segments'
    ausc_zone = 'Both'                      # 'Both', 'Anterior', 'Posterior'
    to_analyze = 'Respiration'              # 'Respiration', 'Heart'
    metric_to_print = 'SDR'         # MSE, Correlation, PSD, SDR
    
    # Definición de la función de interés
    func_to = 'generate_results'        # 'generate_results', 'generate_results_factory',
                                        # 'evaluate_results', 'resume_evaluate_results',
                                        # 'compare_results', 'print_compare_results',
                                        # 'evaluate_result', 'compare_result_specific'
    sep_type_print = 'on segments'
    id_rev = 14
    simulations = [14, 15]

    
    #################            RUTINA            #################
    
    
    if func_to == 'generate_results':
        print(f'Generating {sep_type}...')
        generate_results(ausc_zone=ausc_zone, sep_type=sep_type)
    
    elif func_to == 'generate_results_factory':
        print(f'Generating {sep_type}...')
        generate_results_factory(ausc_zone=ausc_zone, sep_type=sep_type)
        
    elif func_to == 'evaluate_result':
        print(f'Evaluating {sep_type}...')
        evaluate_result(sep_type=sep_type, id_rev=id_rev, version=2, ausc_zone=ausc_zone)
    
    elif func_to == 'evaluate_results':
        print(f'Evaluating {sep_type}...')
        evaluate_results(sep_type=sep_type,  version=2, ausc_zone=ausc_zone)
    
    elif func_to == 'resume_evaluate_results':
        print(f'Resuming {sep_type}...')
        resume_evaluate_results(to_analyze=to_analyze, sep_type=sep_type, version=2, 
                                ausc_zone=ausc_zone, order_pond=1/2)

    elif func_to == 'compare_results':
        print(f'Comparing {to_analyze} results...')
        compare_results(to_analyze=to_analyze, ausc_zone=ausc_zone)
        
    elif func_to == 'compare_results_specific':
        print(f'Comparing {to_analyze} results...')
        compare_results_specific(sep_type=sep_type, to_analyze=to_analyze, 
                                 ausc_zone=ausc_zone)
        
    elif func_to == 'print_compare_results':
        print_compare_results(sep_type=sep_type_print, 
                              metric_to_print=metric_to_print,
                              simulations=simulations)
    