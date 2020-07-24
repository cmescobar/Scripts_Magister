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
from filter_and_sampling import downsampling_signal
from source_separation import nmf_applied_all, get_components_HR_sounds


def generate_results(ausc_zone='Anterior', sep_type='on segments'):
    # Parametros de separación
    N = 1024
    noverlap = int(0.9 * N)
    n_components = 2
    padding = 3 * N
    repeat = 0
    sr_des = 44100 // 4
    window = 'hann'
    
    # Definición filepath
    filepath = f'Database_manufacturing/db_HR/Source Separation/v2/{ausc_zone}/'\
                'Seed-0 - x - 0.5_Heart 0.5_Resp 0_White noise'

    get_components_HR_sounds(filepath, sr_des, sep_type=sep_type, assign_method='manual', 
                             clustering=False, n_components=n_components, N=N, N_lax=200, N_fade=200, 
                             noverlap=noverlap, padding=padding, repeat=repeat, window=window, 
                             whole=False, alpha_wiener=1, filter_out='wiener', 
                             init='random', solver='mu', beta=2, tol=1e-4, max_iter=1000, 
                             alpha_nmf=0, l1_ratio=0, random_state=0, 
                             W_0=None, H_0=None, plot_segments=True, scale='abs', 
                             ausc_zone=ausc_zone, fcut_spect_crit=200, 
                             measure_spect_crit='correlation', i_selection='max', f1_roll=20, 
                             f2_roll=150, measure_temp_crit='q_equal', H_binary=True, 
                             reduce_to_H=False, dec_criteria='or')


def evaluate_result(sep_type='to all', id_rev=1, version=2, ausc_zone='Anterior'):
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
    except:
        raise Exception(f'Simulación con "id {id_rev}" no realizada.')
    
    # Se agrega finalmente el id de la dirección
    filepath = f'{filepath}/id {id_rev}'
    
    # Para resetear los archivos a rellenar más adelante
    open(f'{filepath}/Result Analysis - Respiration.txt','w').close()
    open(f'{filepath}/Result Analysis - Heart.txt', 'w').close()
    
    files_to_rev = [i for i in os.listdir(filepath) if i.startswith('HR')]
    
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
            resp_comp, _ = sf.read(f'{filepath}/{name_i}/ Respiratory Sound.wav')
            heart_comp, _ = sf.read(f'{filepath}/{name_i}/ Heart Sound.wav')
        elif sep_type == 'on segments':
            resp_comp, _ = sf.read(f'{filepath}/{name_i}/Respiratory signal.wav')
            heart_comp, _ = sf.read(f'{filepath}/{name_i}/Heart signal.wav')
        
        # Seleccionando el largo más corto de cada uno (para hacer un punto a punto)
        minlen_resp = min(len(resp_to), len(resp_comp))
        minlen_heart = min(len(heart_to), len(heart_comp))
        
        # Recorte
        resp_to = resp_to[:minlen_resp]
        resp_comp = resp_comp[:minlen_resp]
        heart_to = heart_to[:minlen_heart]
        heart_comp = heart_comp[:minlen_heart]
        
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
        performance = evmet.performance_HNRP(hnrp, 0.5)
        
        # Cálculo de correlación psd
        correlation_resp = evmet.psd_correlation(resp_to, resp_comp, sr_des, window=window, N=N,
                                                noverlap=noverlap)
        correlation_heart = evmet.psd_correlation(heart_to, heart_comp, sr_des, window=window, N=N,
                                                  noverlap=noverlap)
        
        # Cálculo suma de errores
        error_resp = sum(abs(resp_to - resp_comp))
        error_heart = sum(abs(heart_to - heart_comp))
        
        # Creación del diccionario de información
        dict_info_resp = {'name': name_i, 'mse': resp_mse, 'nmse': resp_nmse, 
                          'rmse': resp_rmse, 'SDR': resp_sdr, 
                          'sum error': error_resp, 'psd_correlation': correlation_resp, 
                          'HNRP': hnrp, 'p(%)': performance}
        dict_info_heart = {'name': name_i, 'mse': heart_mse, 'nmse': heart_nmse, 
                           'rmse': heart_rmse, 'SDR': heart_sdr, 
                           'sum error': error_heart, 'psd_correlation': correlation_heart}
        
        # Registrando para cada sonido
        with open(f'{filepath}/Result Analysis - Respiration.txt', 'a', encoding='utf8') as file:
            file.write(f'{dict_info_resp}\n')
        with open(f'{filepath}/Result Analysis - Heart.txt', 'a', encoding='utf8') as file:
            file.write(f'{dict_info_heart}\n')
        
        ################# RESULTADOS GRAFICOS #################
        plt.figure(figsize=(17,7))
        f1, psd1 = evmet.get_PSD(resp_to, sr_des, window=window, N=N, noverlap=noverlap)
        f2, psd2 = evmet.get_PSD(resp_comp, sr_des, window=window, N=N, noverlap=noverlap)
        plt.plot(f1, 20*np.log10(psd1 + 1e-12), label='Original', color='C0')
        plt.plot(f2, 20*np.log10(psd2 + 1e-12), label='Obtained', color='C1')
        plt.xlabel('Frequency [Hz]')
        plt.title('Respiratory PSDs in dB')
        plt.legend(loc='upper right')
        plt.savefig(f'{filepath}/{name_i}/PSD respiratory.png')
        plt.close()
        
        plt.figure(figsize=(17,7))
        f1, psd1 = evmet.get_PSD(heart_to, sr_des, window=window, N=N, noverlap=noverlap)
        f2, psd2 = evmet.get_PSD(heart_comp, sr_des, window=window, N=N, noverlap=noverlap)
        plt.plot(f1, 20*np.log10(psd1 + 1e-12), label='Original', color='C0')
        plt.plot(f2, 20*np.log10(psd2 + 1e-12), label='Obtained', color='C1')
        plt.xlabel('Frequency [Hz]')
        plt.title('Heart PSDs in dB')
        plt.legend(loc='upper right')
        plt.savefig(f'{filepath}/{name_i}/PSD Heart.png')
        plt.close()
        
        plt.figure(figsize=(17,7))
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
        plt.savefig(f'{filepath}/{name_i}/Respiration Original & Obtained.png')
        plt.close()
        
        plt.figure(figsize=(17,7))
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
        plt.savefig(f'{filepath}/{name_i}/Heart Original & Obtained.png')
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
    else:
        raise Exception('Opción para "sep_type" no válido.')
    
    # Obtener las carpetas de ids disponibles
    id_folders = [int(i.split(' ')[-1]) for i in os.listdir(filepath) 
                  if i.startswith('id')]
    
    # Ordenar
    id_folders.sort()
    
    for id_rev in id_folders:
        evaluate_result(sep_type=sep_type, id_rev=id_rev, version=version, 
                        ausc_zone=ausc_zone)


def resume_evaluate_results(to_analyze='Respiration', sep_type='to all', version=2, 
                            ausc_zone='Anterior'):
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
    else:
        raise Exception('Opción para "sep_type" no válido.')
    
    # Obtener las carpetas de ids disponibles
    id_folders = [f'{filepath}/{i}' for i in os.listdir(filepath) 
                  if i.startswith('id')]
    
    # Definición de las listas a realizar (común)
    mse_list = list()
    nmse_list = list()
    rmse_list = list()
    sdr_list = list()
    psd_list = list()
    error_list = list()
    
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
    
    # Listas solo para respiración
    hnrp_total = list()
    p_total = list()
    
    # Definición de la tabla a guardar
    if to_analyze == 'Respiration':
        table = PrettyTable(['id', 'MSE', 'NMSE', 'RMSE', 'Error', 'SDR', 'PSD correlation',
                             'HNRP', 'p(%)'])
    elif to_analyze == 'Heart':
        table = PrettyTable(['id', 'MSE', 'NMSE', 'RMSE', 'Error', 'SDR', 'PSD correlation'])
    else:
        raise Exception('Error en "to_analyze"')
    
    for id_rev in tqdm(id_folders, desc=f'Analyzing {sep_type}', ncols=70):
        # Número del ID
        id_number = id_rev.split(' ')[-1]
        
        with open(f'{id_rev}/Result Analysis - {to_analyze}.txt', 'r', encoding='utf8') as file:
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
        
        # Transformando las listas en arreglos
        mse_array = np.array(mse_list)
        nmse_array = np.array(nmse_list)
        rmse_array = np.array(rmse_list)
        sdr_array = np.array(sdr_list)
        psd_array = np.array(psd_list)
        hnrp_array = np.array(hnrp_list)
        p_array = np.array(p_list)
        error_array = np.array(error_list)
        
        # Escribiendo en la tabla
        if to_analyze == 'Respiration':
            table.add_row([id_number, 
                           "{:.4f} +- {:.4f}".format(mse_array.mean(), mse_array.std()),
                           "{:.4f} +- {:.4f}".format(nmse_array.mean(), nmse_array.std()),
                           "{:.4f} +- {:.4f}".format(rmse_array.mean(), rmse_array.std()),
                           "{:.4f} +- {:.4f}".format(error_array.mean(), error_array.std()),
                           "{:.4f} +- {:.4f}".format(sdr_array.mean(), sdr_array.std()),
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
                           "{:.4f} +- {:.4f}".format(psd_array.mean(), psd_array.std())])
    
    # Finalmente se guarda en un archivo resumen
    with open(f'{filepath}/Analysis results.txt', 'w', encoding='utf8') as file:
        file.write(table.get_string())
    
    # Creación de diagramas de caja
    plt.boxplot(mse_total)
    plt.ylabel('MSE')
    plt.title(f'{to_analyze} MSE Boxplot')
    plt.savefig(f'{filepath}/{to_analyze} mse_boxplot.png')
    plt.close()
    
    plt.boxplot(nmse_total)
    plt.ylabel('NMSE')
    plt.title(f'{to_analyze} NMSE Boxplot')
    plt.savefig(f'{filepath}/{to_analyze} nmse_boxplot.png')
    plt.close()
    
    plt.boxplot(rmse_total)
    plt.ylabel('RMSE')
    plt.title(f'{to_analyze} RMSE Boxplot')
    plt.savefig(f'{filepath}/{to_analyze} rmse_boxplot.png')
    plt.close()
    
    plt.boxplot(error_total)
    plt.ylabel('RMSE')
    plt.title(f'{to_analyze} Error total Boxplot')
    plt.savefig(f'{filepath}/{to_analyze} Error_boxplot.png')
    plt.close()
    
    plt.boxplot(sdr_total)
    plt.ylabel('SDR')
    plt.title(f'{to_analyze} SDR Boxplot')
    plt.savefig(f'{filepath}/{to_analyze} sdr_boxplot.png')
    plt.close()
    
    plt.boxplot(psd_total)
    plt.ylabel('PSD')
    plt.title(f'{to_analyze} PSD Boxplot')
    plt.savefig(f'{filepath}/{to_analyze} psd_boxplot.png')
    plt.close()
    
    if to_analyze == 'Respiration':
        plt.boxplot(hnrp_total)
        plt.ylabel('HNRP')
        plt.title(f'{to_analyze} HNRP Boxplot')
        plt.savefig(f'{filepath}/{to_analyze} hnrp_boxplot.png')
        plt.close()
        
        plt.boxplot(p_total)
        plt.ylabel('p(%)')
        plt.title(f'{to_analyze} p(%) Boxplot')
        plt.savefig(f'{filepath}/{to_analyze} p_boxplot.png')
        plt.close()


def compare_results(to_analyze='Respiration', ausc_zone='Anterior'):
    # Parametros de separación
    N = 1024
    noverlap = int(0.9 * N)
    n_components = 2
    padding = 3 * N
    repeat = 0
    window = 'hann'
    
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
                dict_to_all['window'] == window):
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
                dict_on_seg['window'] == window):
                id_on_segments = dict_on_seg['id']
    
    print(f'id to all: {id_to_all}')
    print(f'id on segments: {id_on_segments}')
    
    # Definición de las direcciones con la id
    filepath_to_all = f'{filepath}/Separation to all/id {id_to_all}'
    filepath_on_segments = f'{filepath}/Separation on segments/id {id_on_segments}'
    
    # Definción de las listas de listas
    mse_total = list()
    nmse_total = list()
    rmse_total = list()
    sdr_total = list()
    psd_total = list()
    error_total = list()
    
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
    
    with open(f'{filepath_on_segments}/Result Analysis - {to_analyze}.txt', 
              'r', encoding='utf8') as file:
        # Definición de las listas a realizar (común)
        mse_list = list()
        nmse_list = list()
        rmse_list = list()
        sdr_list = list()
        psd_list = list()
        error_list = list()
        
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
    
    # Creación de diagramas de caja
    plt.boxplot(mse_total)
    plt.xticks([1, 2], ['NMF\nto all', 'NMF\non segments'])
    plt.ylabel('MSE')
    plt.title(f'{to_analyze} MSE Boxplot')
    plt.savefig(f'{filepath}/{to_analyze} mse_boxplot.png')
    plt.close()
    
    plt.boxplot(nmse_total)
    plt.xticks([1, 2], ['NMF\nto all', 'NMF\non segments'])
    plt.ylabel('NMSE')
    plt.title(f'{to_analyze} NMSE Boxplot')
    plt.savefig(f'{filepath}/{to_analyze} nmse_boxplot.png')
    plt.close()
    
    plt.boxplot(rmse_total)
    plt.xticks([1, 2], ['NMF\nto all', 'NMF\non segments'])
    plt.ylabel('RMSE')
    plt.title(f'{to_analyze} RMSE Boxplot')
    plt.savefig(f'{filepath}/{to_analyze} rmse_boxplot.png')
    plt.close()
    
    plt.boxplot(error_total)
    plt.xticks([1, 2], ['NMF\nto all', 'NMF\non segments'])
    plt.ylabel('Error')
    plt.title(f'{to_analyze} Error total Boxplot')
    plt.savefig(f'{filepath}/{to_analyze} Error_boxplot.png')
    plt.close()
    
    plt.boxplot(sdr_total)
    plt.xticks([1, 2], ['NMF\nto all', 'NMF\non segments'])
    plt.ylabel('SDR')
    plt.title(f'{to_analyze} SDR Boxplot')
    plt.savefig(f'{filepath}/{to_analyze} sdr_boxplot.png')
    plt.close()
    
    plt.boxplot(psd_total)
    plt.xticks([1, 2], ['NMF\nto all', 'NMF\non segments'])
    plt.ylabel('PSD')
    plt.title(f'{to_analyze} PSD Boxplot')
    plt.savefig(f'{filepath}/{to_analyze} psd_boxplot.png')
    plt.close()
    
    if to_analyze == 'Respiration':
        plt.boxplot(hnrp_total)
        plt.xticks([1, 2], ['NMF\nto all', 'NMF\non segments'])
        plt.ylabel('HNRP')
        plt.title(f'{to_analyze} HNRP Boxplot')
        plt.savefig(f'{filepath}/{to_analyze} hnrp_boxplot.png')
        plt.close()
        
        plt.boxplot(p_total)
        plt.xticks([1, 2], ['NMF\nto all', 'NMF\non segments'])
        plt.ylabel('p(%)')
        plt.title(f'{to_analyze} p(%) Boxplot')
        plt.savefig(f'{filepath}/{to_analyze} p_boxplot.png')
        plt.close()
    

def testing_module_1():
    # Parametros de separación
    N = 1024 * 2
    noverlap = N // 2
    n_components = 2
    padding = 3 * N
    repeat = 4
    sr_des = 44100 // 4
    window = 'hann'
    
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
        
    nmf_applied_all(dir_file, filepath_to_save_id, sr_des, clustering=False, n_components=n_components, 
                    N=N, noverlap=noverlap, padding=padding, repeat=repeat, window=window, 
                    whole=False, alpha_wiener=1, filter_out='wiener', init='random', 
                    solver='cd', beta=2, tol=1e-4, max_iter=1000, alpha_nmf=0, l1_ratio=0, 
                    random_state=0, W_0=None, H_0=None, scale='abs', version=2, ausc_zone='Anterior')



generate_results(ausc_zone='Both', sep_type='on segments')
# evaluate_results(sep_type='on segments', version=2, ausc_zone='Both')
# compare_results(to_analyze='Heart', ausc_zone='Both')



"""
generate_results(ausc_zone='Anterior', sep_type='on segments')
generate_results(ausc_zone='Anterior', sep_type='to all')

resume_evaluate_results(to_analyze='Heart', sep_type='to all', version=2, 
                        ausc_zone='Anterior')
resume_evaluate_results(to_analyze='Respiration', sep_type='to all', version=2, 
                        ausc_zone='Posterior')
resume_evaluate_results(to_analyze='Respiration', sep_type='on segments', version=2, 
                        ausc_zone='Anterior')

evaluate_results(sep_type='to all', version=2, ausc_zone='Anterior')
evaluate_results(sep_type='to all', version=2, ausc_zone='Posterior')
evaluate_results(sep_type='on segments', version=2, ausc_zone='Anterior')

evaluate_result(sep_type='to all', id_rev=1, version=2, ausc_zone='Anterior')
testing_module_1()
"""