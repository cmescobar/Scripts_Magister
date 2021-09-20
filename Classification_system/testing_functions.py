import os, gc, pickle
import numpy as np
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt
from ast import literal_eval
from tqdm import tqdm
from datetime import datetime
from scipy.io  import wavfile
from sklearn.metrics import confusion_matrix, accuracy_score
from heart_sound_segmentation.evaluation_functions import eval_sound_model_db
from respiratory_sound_classification.respiratory_sound_management import \
    get_ML_data_oncycles, get_db_system, train_test_filebased, get_ML_data, get_label_filename,\
    get_db_spectrograms, get_db_MFCC, get_db_energy
from pybalu.feature_transformation import normalize



def test_hss():
    # Función auxiliar para desplegar los archivos a seleccionar
    def _file_selection(filenames):
        print('Seleccione el archivo que desea descomponer:')
        for num, name in enumerate(filenames):
            print(f'[{num + 1}] {name}')
            
        # Definición de la selección
        selection = int(input('Selección: '))
        
        # Se retorna
        try:
            return filenames[selection-1].strip('.wav')
        except:
            raise Exception('No ha seleccionado un archivo válido.')
    
    
    ################        Parámetros      ################

    # Carpeta de ubicación de la base de datos
    db_folder = 'cardiorespiratory_database'
    # Síntoma del paciente (Healthy, Pneumonia)
    symptom = 'Healthy'
    # Posición de auscultación (toracic, trachea, all)
    ausc_pos = 'toracic'        
    # Prioridad de los sonidos a revisar (1, 2, 3)
    priority = 1
    
    # Parámetros de descomposición
    model_name = 'segnet_based_12_10'
    # Parámetros del filtro pasa bajos a la salida de la red
    lowpass_params = {'freq_pass': 140, 'freq_stop': 150}


    ################        Rutina      ################
    
    # Definición de la carpeta a revisar
    filepath = f'{db_folder}/{symptom}/{ausc_pos}/Priority_{priority}'
    
    # Definición de la dirección del modelo a utilizar
    model_to = f'heart_sound_segmentation/models/{model_name}.h5'
    
    # Definición del archivo a revisar
    filenames = [i for i in os.listdir(filepath) if i.endswith('.wav')]
    
    # Obtención del archivo
    filename = f'{filepath}/{_file_selection(filenames)}'
    
    # Salida de la red
    y_hat, y_out3, y_out4 = \
                    eval_sound_model_db(filename, model_to, 
                                     lowpass_params=lowpass_params,
                                     plot_outputs=True)
    
    return y_hat, y_out3, y_out4


def classificaton_results(db_process, class_params, objective_label, objective_set):
    # Parámetros
    # db_process = 'preprocessed'
    # class_params = {'classifier': 'knn', 'param': '5'}      # knn, svm, mlp
    # objective_label = 'wheeze'                            # wheeze, crackle
    # objective_set = 'train'
    
    # Parámetros de caracaterísticas (no modificar)
    N = 1024
    noverlap = int(0.9 * N)
    spec_params = {'N': N, 'noverlap': noverlap, 'window': 'hann', 
                   'padding': 0, 'repeat': 0}
    mfcc_params = {'n_mfcc': 50, 'n_filters': 50, 'spec_params': spec_params,
                   'freq_lim': 2000, 'norm_filters': True, 'power': 2}
    lfcc_params = {'n_mfcc': 50, 'n_filters': 50, 'spec_params': spec_params,
                   'freq_lim': 2000, 'norm_filters': True, 'power': 2}
    energy_params = {'spec_params': spec_params, 'fmin': 0, 'fmax': 1000, 
                     'fband': 20}
    
    # Definición de la carpeta con la base de datos
    db_folder = f'{db_process}_signals'
    db_original = 'C:/Users/Chris/Desktop/Scripts_Magister/Respiratory_Sound_Database/audio_and_txt_files'
    
    
    
    ###############     RUTINA     ###############
        
    # Obtención del conjunto de entrenamiento y testeo
    train_test_folder = '/'.join(db_original.split('/')[:-1])
    dir_div_file = f'{train_test_folder}/ICBHI_challenge_train_test.txt'
    train_list, test_list = train_test_filebased(dir_div_file, db_root=db_original,
                                                 db_folder=db_folder)
    
    
    
    # Definición del nombre del modelo en base a los parámetros
    model_name = f'respiratory_sound_classification/{db_folder}/'\
                    f'{class_params["classifier"]}_{class_params["param"]}_'\
                    f'{objective_label}_{db_process}'
    
    
    # Control de versiones
    if os.path.isfile(f'{model_name}_{objective_set}.txt'):
        print(f'Archivo con el nombre {model_name}_{objective_set} ya existe...')
        return
    
    
    
    # Cargar el modelo
    if class_params["classifier"] in ['knn', 'svm']:
        model = pickle.load(open(f'{model_name}.pickle', 'rb'))
    elif class_params["classifier"] == 'mlp':
        model = tf.keras.models.load_model(f'{model_name}.h5')
    
    
    if objective_set == 'train':
        # Obteniendo los datos de entrenamiento
        X_data, Y_wheeze, Y_crackl = \
                get_ML_data(train_list, spec_params=spec_params, mfcc_params=mfcc_params, 
                            lfcc_params=lfcc_params, energy_params=energy_params)

    elif objective_set == 'test':
        # Obteniendo los datos de testeo
        X_data, Y_wheeze, Y_crackl = \
                get_ML_data(test_list, spec_params=spec_params, mfcc_params=mfcc_params, 
                            lfcc_params=lfcc_params, energy_params=energy_params)
    
    else:
        raise Exception('Error en la selección del conjunto objective_set')
    
    
    # Obtener las características para transformar las variables
    with open(f'respiratory_sound_classification/{objective_label}_features_params_{db_process}.txt', 
                'r', encoding='utf8') as file:
        params_dict = literal_eval(file.readline().strip())
    
    # Definición de las características
    s_clean = np.array(params_dict['s_clean'])
    a_norm = np.array(params_dict['a_norm'])
    b_norm = np.array(params_dict['b_norm'])
    s_sfs = np.array(params_dict['s_sfs'])
    
    # Transformando las características del conjunto de testeo en 
    # base a los parámetros
    print('Transformando')
    X_data = X_data[:, s_clean]         # 1) Clean
    X_data = a_norm * X_data + b_norm   # 2) Normalización
    X_data = X_data[:, s_sfs]           # 3) Selección
    
    print(f'Prediciendo con {model_name}...')
    print(X_data.shape)
    # Evaluando
    Y_pred = model.predict(X_data)
    
    # Si es la red neuronal, modificar el Y_pred
    if class_params["classifier"] == 'mlp':
        Y_pred = np.where(Y_pred < 0.5, 0, 1)[:, 0]
    
    
    # Comparando con las etiquetas
    if objective_label == 'wheeze':
        conf_mat = confusion_matrix(Y_wheeze, Y_pred)
        
    elif objective_label == 'crackle':
        conf_mat = confusion_matrix(Y_crackl, Y_pred)
    
    
    print(conf_mat)
    
    # Registrando la matriz
    with open(f'{model_name}_{objective_set}.txt', 'w', encoding='utf8') as file:
        file.write(f'{conf_mat.tolist()}')


def get_results(db_process, class_params, objective_label, objective_set,
                print_info='conf_matrix', print_detail=False):
    # Definición de la carpeta con la base de datos
    db_folder = f'{db_process}_signals'
    
    # Definición del nombre del modelo en base a los parámetros
    model_name = f'respiratory_sound_classification/{db_folder}/'\
                    f'{class_params["classifier"]}_{class_params["param"]}_'\
                    f'{objective_label}_{db_process}'
    
    if objective_set == 'train':        
        # Abriendo el archivo
        with open(f'{model_name}_{objective_set}.txt', 'r', encoding='utf8') as file:
            conf_mat = literal_eval(file.readline().strip())
    
    elif objective_set == 'test':
        # Abriendo el archivo
        with open(f'{model_name}_results.txt', 'r', encoding='utf8') as file:
            conf_mat = literal_eval(file.readline().strip())

    
    # Se printea
    if print_info == 'conf_matrix':
        if print_detail:
            print(f'{model_name}: {conf_mat}')
        else:
            print(f'{conf_mat}')
    
    elif print_info == 'values':
        # Obtener los elementos de cada componente
        tp = conf_mat[1][1]
        tn = conf_mat[0][0]
        fp = conf_mat[0][1]
        fn = conf_mat[1][0]
        
        # Calculando los valores de interés
        accuracy  = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall    = tp / (tp + fn)
        
        if print_detail:
            print(f'{model_name}: {accuracy},{precision},{recall}')
        else:
            print(f'{accuracy},{precision},{recall}')
        

def classificaton_results_CNN(db_process, objective_label, objective_set):
    # Definición de la carpeta de interés
    root = 'respiratory_sound_classification'
    
    # Definición de la carpeta con la base de datos
    db_folder = f'{db_process}_signals'
    db_original = 'C:/Users/Chris/Desktop/Scripts_Magister/Respiratory_Sound_Database/audio_and_txt_files'
    
    
    
    ###############     RUTINA     ###############
    
    # Obtención del conjunto de entrenamiento y testeo
    train_test_folder = '/'.join(db_original.split('/')[:-1])
    dir_div_file = f'{train_test_folder}/ICBHI_challenge_train_test.txt'
    train_list, test_list = train_test_filebased(dir_div_file, db_root=db_original,
                                                 db_folder=db_folder)
    
    # Definición de los nombres a revisar
    if objective_set == 'train':
        filenames = train_list
    elif objective_set == 'test':
        filenames = test_list
    
    
    # Definición del modelo a usar
    model_name = f'{root}/{db_folder}/definitive_segnet_based_{objective_label}_2/'\
                 f'definitive_segnet_based_2_{objective_label}'
    # Obtener el modelo
    model = tf.keras.models.load_model(f'{model_name}.h5')
    
    # Definición de la matriz de confusión base
    base_conf_mat = np.zeros((2,2))
        
    # Para cada archivo
    for filename in filenames:
        print(filename)     
        # Obtención del archivo de audio .wav
        try:
            samplerate, audio = wavfile.read(f'{filename}.wav')
        except:
            audio, samplerate = sf.read(f'{filename}.wav')
    
        # Normalizando el audio
        audio = audio / max(abs(audio))
        
        # Dejando solo el nombre
        filename_clean = '_'.join(filename.split('/')[-1].split('_')[:-1])
        
        # Obteniendo las etiquetas en específico
        Y_wheeze, Y_crackl = \
                get_label_filename(filename_clean, samplerate=samplerate, 
                                   length_desired=len(audio))
                
        # Acondicionar la entrada
        audio_to = np.expand_dims(np.expand_dims(audio, -1), 0)
        
        # Usando el modelo para predecir
        Y_pred = model.predict(audio_to)
        # print('audio shape: ', audio.shape)
        # print('Y_pred shape: ', Y_pred.shape)
        
        # Haciendo binario
        Y_pred = np.where(Y_pred[0, :, 0] >= 0.5, 1, 0)
        # print('Y_pred shape new: ',Y_pred.shape)
        
        # plt.plot(Y_wheeze)
        # plt.plot(Y_pred)
        # plt.show()
        
        # Paddeando en caso de que la salida sea más corta
        if len(Y_pred) < len(audio):
            Y_pred = np.concatenate((Y_pred, [0] * abs(len(audio) - len(Y_pred)) ))
        
        # Calculando la matriz de confusión
        if objective_label == 'wheeze':
            conf_mat = confusion_matrix(Y_wheeze, Y_pred, labels=[0,1])
        elif objective_label == 'crackle':
            conf_mat = confusion_matrix(Y_crackl, Y_pred, labels=[0,1])
        
        # Agregando a la matriz de interés
        base_conf_mat += conf_mat
        
        
    # Registrando la matriz
    with open(f'{model_name}_{objective_set}.txt', 'w', encoding='utf8') as file:
        file.write(f'{base_conf_mat.astype(int).tolist()}')
        

def get_results_CNN(db_process, objective_label, objective_set,
                    print_info='conf_matrix', print_detail=False):
    # Definición de la carpeta con la base de datos
    db_folder = f'{db_process}_signals'
    
    # Definición de la carpeta de interés
    root = 'respiratory_sound_classification'
    
    # Definición del nombre del modelo en base a los parámetros
    model_name = f'{root}/{db_folder}/definitive_segnet_based_{objective_label}_2/'\
                 f'definitive_segnet_based_2_{objective_label}'

    # Abriendo el archivo
    with open(f'{model_name}_{objective_set}.txt', 'r', encoding='utf8') as file:
        conf_mat = literal_eval(file.readline().strip())
        
    
    # Se printea
    if print_info == 'conf_matrix':
        if print_detail:
            print(f'{model_name}: {conf_mat}')
        else:
            print(f'{conf_mat}')
    
    elif print_info == 'values':
        # Obtener los elementos de cada componente
        tp = conf_mat[1][1]
        tn = conf_mat[0][0]
        fp = conf_mat[0][1]
        fn = conf_mat[1][0]
        
        # Calculando los valores de interés
        accuracy  = (tp + tn) / (tp + tn + fp + fn)
        try:
            precision = tp / (tp + fp)
        except:
            precision = '-'
        
        try:
            recall    = tp / (tp + fn)
        except:
            recall    = '-'
        
        if print_detail:
            print(f'{model_name}: {accuracy},{precision},{recall}')
        else:
            print(f'{accuracy},{precision},{recall}')
 
 
def get_number_labels(db_process):
    # Parámetros de caracaterísticas (no modificar)
    N = 1024
    noverlap = int(0.9 * N)
    spec_params = {'N': N, 'noverlap': noverlap, 'window': 'hann', 
                   'padding': 0, 'repeat': 0}
    mfcc_params = {'n_mfcc': 50, 'n_filters': 50, 'spec_params': spec_params,
                   'freq_lim': 2000, 'norm_filters': True, 'power': 2}
    lfcc_params = {'n_mfcc': 50, 'n_filters': 50, 'spec_params': spec_params,
                   'freq_lim': 2000, 'norm_filters': True, 'power': 2}
    energy_params = {'spec_params': spec_params, 'fmin': 0, 'fmax': 1000, 
                     'fband': 20}
    
    # Definición de la carpeta con la base de datos
    db_folder = f'{db_process}_signals'
    db_original = 'C:/Users/Chris/Desktop/Scripts_Magister/Respiratory_Sound_Database/audio_and_txt_files'
    
    
    
    ###############     RUTINA     ###############
        
    # Obtención del conjunto de entrenamiento y testeo
    train_test_folder = '/'.join(db_original.split('/')[:-1])
    dir_div_file = f'{train_test_folder}/ICBHI_challenge_train_test.txt'
    train_list, test_list = train_test_filebased(dir_div_file, db_root=db_original,
                                                 db_folder=db_folder)
        
    
    # Obteniendo los datos de entrenamiento
    _, Y_wheeze_train, Y_crackl_train = \
            get_ML_data(train_list, spec_params=spec_params, mfcc_params=mfcc_params, 
                        lfcc_params=lfcc_params, energy_params=energy_params)

    # Obteniendo los datos de testeo
    _, Y_wheeze_test, Y_crackl_test = \
            get_ML_data(test_list, spec_params=spec_params, mfcc_params=mfcc_params, 
                        lfcc_params=lfcc_params, energy_params=energy_params)
    
    # Obteniendo las cantidades
    q_wheeze_train = len(np.where(Y_wheeze_train == 1)[0])
    q_crackl_train = len(np.where(Y_crackl_train == 1)[0])
    q_wheeze_test = len(np.where(Y_wheeze_test == 1)[0])
    q_crackl_test = len(np.where(Y_crackl_test == 1)[0])
    
    print('Total:', Y_wheeze_train.shape[0] + Y_wheeze_test.shape[0])
    print('Wheeze train:', q_wheeze_train)
    print('Crackl train:', q_crackl_train)
    print('Wheeze test:', q_wheeze_test)
    print('Crackl test:', q_crackl_test)
    

def get_number_labels_CNN(db_process):
    # Definición de la carpeta de interés
    root = 'respiratory_sound_classification'
    
    # Definición de la carpeta con la base de datos
    db_folder = f'{db_process}_signals'
    db_original = 'C:/Users/Chris/Desktop/Scripts_Magister/Respiratory_Sound_Database/audio_and_txt_files'
    
    
    
    ###############     RUTINA     ###############
    
    # Obtención del conjunto de entrenamiento y testeo
    train_test_folder = '/'.join(db_original.split('/')[:-1])
    dir_div_file = f'{train_test_folder}/ICBHI_challenge_train_test.txt'
    train_list, test_list = train_test_filebased(dir_div_file, db_root=db_original,
                                                 db_folder=db_folder)
    
    # Definición de contadores de etiquetas
    q_total_train = 0
    q_total_test = 0
    q_wheeze_train = 0
    q_crackl_train = 0
    q_wheeze_test = 0
    q_crackl_test = 0
    
    # Para cada archivo
    for filename in train_list:        
        # Obtención del archivo de audio .wav
        try:
            samplerate, audio = wavfile.read(f'{filename}.wav')
        except:
            audio, samplerate = sf.read(f'{filename}.wav')
    
        # Normalizando el audio
        audio = audio / max(abs(audio))
        
        # Dejando solo el nombre
        filename_clean = '_'.join(filename.split('/')[-1].split('_')[:-1])
        
        # Obteniendo las etiquetas en específico
        Y_wheeze_train, Y_crackl_train = \
                get_label_filename(filename_clean, samplerate=samplerate, 
                                   length_desired=len(audio))
                
        # Añadiendo a los contadores
        q_total_train += len(audio)
        q_wheeze_train += len(np.where(Y_wheeze_train == 1)[0])
        q_crackl_train += len(np.where(Y_crackl_train == 1)[0])
        
    
    # Para cada archivo
    for filename in test_list:        
        # Obtención del archivo de audio .wav
        try:
            samplerate, audio = wavfile.read(f'{filename}.wav')
        except:
            audio, samplerate = sf.read(f'{filename}.wav')
    
        # Normalizando el audio
        audio = audio / max(abs(audio))
        
        # Dejando solo el nombre
        filename_clean = '_'.join(filename.split('/')[-1].split('_')[:-1])
        
        # Obteniendo las etiquetas en específico
        Y_wheeze_test, Y_crackl_test = \
                get_label_filename(filename_clean, samplerate=samplerate, 
                                   length_desired=len(audio))
                
        # Añadiendo a los contadores
        q_total_test += len(audio)
        q_wheeze_test += len(np.where(Y_wheeze_test == 1)[0])
        q_crackl_test += len(np.where(Y_crackl_test == 1)[0])
        
    
    print('Total train:', q_total_train)
    print('Total test:', q_total_test)
    print('Wheeze train:', q_wheeze_train)
    print('Crackl train:', q_crackl_train)
    print('Wheeze test:', q_wheeze_test)
    print('Crackl test:', q_crackl_test)
    
    print('Wheeze prop:',  100 * (q_wheeze_train + q_wheeze_test) / (q_total_train + q_total_test))
    print('Crackle prop:', 100 * (q_crackl_train + q_crackl_test) / (q_total_train + q_total_test))


def plot_features_MLP(db_process, objective_label):
    # Parámetros
    # db_process = 'preprocessed'
    # class_params = {'classifier': 'knn', 'param': '5'}      # knn, svm, mlp
    # objective_label = 'wheeze'                            # wheeze, crackle
    # objective_set = 'train'
    
    # Parámetros de caracaterísticas (no modificar)
    N = 1024
    noverlap = int(0.9 * N)
    spec_params = {'N': N, 'noverlap': noverlap, 'window': 'hann', 
                   'padding': 0, 'repeat': 0}
    mfcc_params = {'n_mfcc': 50, 'n_filters': 50, 'spec_params': spec_params,
                   'freq_lim': 2000, 'norm_filters': True, 'power': 2}
    lfcc_params = {'n_mfcc': 50, 'n_filters': 50, 'spec_params': spec_params,
                   'freq_lim': 2000, 'norm_filters': True, 'power': 2}
    energy_params = {'spec_params': spec_params, 'fmin': 0, 'fmax': 1000, 
                     'fband': 20}
    
    # Definición de la carpeta con la base de datos
    db_folder = f'{db_process}_signals'
    db_original = 'C:/Users/Chris/Desktop/Scripts_Magister/Respiratory_Sound_Database/audio_and_txt_files'
    
    
    
    ###############     RUTINA     ###############
        
    # Obtención del conjunto de entrenamiento y testeo
    train_test_folder = '/'.join(db_original.split('/')[:-1])
    dir_div_file = f'{train_test_folder}/ICBHI_challenge_train_test.txt'
    train_list, test_list = train_test_filebased(dir_div_file, db_root=db_original,
                                                 db_folder=db_folder)
    
    # Obteniendo los datos de entrenamiento
    X_data_train, Y_wheeze, Y_crackl = \
            get_ML_data(train_list, spec_params=spec_params, mfcc_params=mfcc_params, 
                        lfcc_params=lfcc_params, energy_params=energy_params)

    # # Obteniendo los datos de testeo
    # X_data_test, Y_wheeze, Y_crackl = \
    #         get_ML_data(test_list, spec_params=spec_params, mfcc_params=mfcc_params, 
    #                     lfcc_params=lfcc_params, energy_params=energy_params)

    
    
    # Obtener las características para transformar las variables
    with open(f'respiratory_sound_classification/{objective_label}_features_params_{db_process}.txt', 
                'r', encoding='utf8') as file:
        params_dict = literal_eval(file.readline().strip())
    
    # Definición de las características
    s_clean = np.array(params_dict['s_clean'])
    a_norm = np.array(params_dict['a_norm'])
    b_norm = np.array(params_dict['b_norm'])
    s_sfs = np.array(params_dict['s_sfs'])
    
    # Transformando las características del conjunto de testeo en 
    # base a los parámetros
    print('Transformando')
    X_data_train = X_data_train[:, s_clean]         # 1) Clean
    X_data_train = a_norm * X_data_train + b_norm   # 2) Normalización
    X_data_train = X_data_train[:, s_sfs]           # 3) Selección
    
    # X_data_test = X_data_test[:, s_clean]         # 1) Clean
    # X_data_test = a_norm * X_data_test + b_norm   # 2) Normalización
    # X_data_test = X_data_test[:, s_sfs]           # 3) Selección
    
    print(X_data_train.shape)
    # print(X_data_test.shape)
    
    wheeze_data = np.zeros((0, X_data_train.shape[1]))
    crackl_data = np.zeros((0, X_data_train.shape[1]))
    
    for i in tqdm(range(X_data_train.shape[0]), ncols=120, desc='Plotting'):
        if Y_wheeze[i] == 1:
            wheeze_data = np.concatenate((wheeze_data, [X_data_train[i]]), axis=0)
        
        if Y_crackl[i] == 1:
            crackl_data = np.concatenate((crackl_data, [X_data_train[i]]), axis=0)

    
    print('Wheeze_data: ',  wheeze_data.shape)
    print('Crackle_data: ', crackl_data.shape)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.pcolormesh(wheeze_data)
    plt.colorbar()
    
    plt.subplot(1,2,2)
    plt.pcolormesh(crackl_data)
    plt.colorbar()
    
    plt.show()
    
    plt.figure()
    plt.pcolormesh(np.concatenate((wheeze_data, [[100] * 60] , crackl_data), axis=0))
    plt.colorbar()
    
    plt.show()


def plot_features_CNN(db_process, objective_label):
    # Parámetros
    # db_process = 'preprocessed'
    # class_params = {'classifier': 'knn', 'param': '5'}      # knn, svm, mlp
    # objective_label = 'wheeze'                            # wheeze, crackle
    # objective_set = 'train'
    
    # Parámetros de caracaterísticas (no modificar)
    N = 1024
    noverlap = int(0.9 * N)
    spec_params = {'N': N, 'noverlap': noverlap, 'window': 'hann', 
                   'padding': 0, 'repeat': 0}
    mfcc_params = {'n_mfcc': 50, 'n_filters': 50, 'spec_params': spec_params,
                   'freq_lim': 2000, 'norm_filters': True, 'power': 2}
    lfcc_params = {'n_mfcc': 50, 'n_filters': 50, 'spec_params': spec_params,
                   'freq_lim': 2000, 'norm_filters': True, 'power': 2}
    energy_params = {'spec_params': spec_params, 'fmin': 0, 'fmax': 1000, 
                     'fband': 20}
    
    # Definición de la carpeta con la base de datos
    db_folder = f'{db_process}_signals'
    db_original = 'C:/Users/Chris/Desktop/Scripts_Magister/Respiratory_Sound_Database/audio_and_txt_files'
    
    
    
    ###############     RUTINA     ###############
        
    # Obtención del conjunto de entrenamiento y testeo
    train_test_folder = '/'.join(db_original.split('/')[:-1])
    dir_div_file = f'{train_test_folder}/ICBHI_challenge_train_test.txt'
    train_list, test_list = train_test_filebased(dir_div_file, db_root=db_original,
                                                 db_folder=db_folder)
    
    # Obteniendo los datos de entrenamiento
    X_data_train, Y_wheeze, Y_crackl = \
            get_ML_data(train_list, spec_params=spec_params, mfcc_params=mfcc_params, 
                        lfcc_params=lfcc_params, energy_params=energy_params)

    # # Obteniendo los datos de testeo
    # X_data_test, Y_wheeze, Y_crackl = \
    #         get_ML_data(test_list, spec_params=spec_params, mfcc_params=mfcc_params, 
    #                     lfcc_params=lfcc_params, energy_params=energy_params)

    
    
    # Obtener las características para transformar las variables
    with open(f'respiratory_sound_classification/{objective_label}_features_params_{db_process}.txt', 
                'r', encoding='utf8') as file:
        params_dict = literal_eval(file.readline().strip())
    
    # Definición de las características
    s_clean = np.array(params_dict['s_clean'])
    a_norm = np.array(params_dict['a_norm'])
    b_norm = np.array(params_dict['b_norm'])
    s_sfs = np.array(params_dict['s_sfs'])
    
    # Transformando las características del conjunto de testeo en 
    # base a los parámetros
    print('Transformando')
    X_data_train = X_data_train[:, s_clean]         # 1) Clean
    X_data_train = a_norm * X_data_train + b_norm   # 2) Normalización
    X_data_train = X_data_train[:, s_sfs]           # 3) Selección
    
    # X_data_test = X_data_test[:, s_clean]         # 1) Clean
    # X_data_test = a_norm * X_data_test + b_norm   # 2) Normalización
    # X_data_test = X_data_test[:, s_sfs]           # 3) Selección
    
    print(X_data_train.shape)
    # print(X_data_test.shape)
    
    wheeze_data = np.zeros((0, X_data_train.shape[1]))
    crackl_data = np.zeros((0, X_data_train.shape[1]))
    
    for i in tqdm(range(X_data_train.shape[0]), ncols=120, desc='Plotting'):
        if Y_wheeze[i] == 1:
            wheeze_data = np.concatenate((wheeze_data, [X_data_train[i]]), axis=0)
        
        if Y_crackl[i] == 1:
            crackl_data = np.concatenate((crackl_data, [X_data_train[i]]), axis=0)

    
    print('Wheeze_data: ',  wheeze_data.shape)
    print('Crackle_data: ', crackl_data.shape)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.pcolormesh(wheeze_data)
    plt.colorbar()
    
    plt.subplot(1,2,2)
    plt.pcolormesh(crackl_data)
    plt.colorbar()
    
    plt.show()
    
    plt.figure()
    plt.pcolormesh(np.concatenate((wheeze_data, [[100] * 60] , crackl_data), axis=0))
    plt.colorbar()
    
    plt.show()


def plot_features_points_MLP(db_process, objective_label, class_params):
    # Parámetros
    # db_process = 'preprocessed'
    # class_params = {'classifier': 'knn', 'param': '5'}      # knn, svm, mlp
    # objective_label = 'wheeze'                            # wheeze, crackle
    # objective_set = 'train'
    
    # Parámetros de caracaterísticas (no modificar)
    N = 1024
    noverlap = int(0.9 * N)
    spec_params = {'N': N, 'noverlap': noverlap, 'window': 'hann', 
                   'padding': 0, 'repeat': 0}
    mfcc_params = {'n_mfcc': 50, 'n_filters': 50, 'spec_params': spec_params,
                   'freq_lim': 2000, 'norm_filters': True, 'power': 2}
    lfcc_params = {'n_mfcc': 50, 'n_filters': 50, 'spec_params': spec_params,
                   'freq_lim': 2000, 'norm_filters': True, 'power': 2}
    energy_params = {'spec_params': spec_params, 'fmin': 0, 'fmax': 1000, 
                     'fband': 20}
    
    # Definición de la carpeta con la base de datos
    db_folder = f'{db_process}_signals'
    db_original = 'C:/Users/Chris/Desktop/Scripts_Magister/Respiratory_Sound_Database/audio_and_txt_files'
    
    
    # Definición del nombre del modelo en base a los parámetros
    model_name = f'respiratory_sound_classification/{db_folder}/'\
                    f'{class_params["classifier"]}_{class_params["param"]}_'\
                    f'{objective_label}_{db_process}'
    
    
    ###############     RUTINA     ###############
        
    # Obtención del conjunto de entrenamiento y testeo
    train_test_folder = '/'.join(db_original.split('/')[:-1])
    dir_div_file = f'{train_test_folder}/ICBHI_challenge_train_test.txt'
    train_list, test_list = train_test_filebased(dir_div_file, db_root=db_original,
                                                 db_folder=db_folder)
    
    # Obteniendo los datos de entrenamiento
    X_data_train, Y_wheeze, Y_crackl = \
            get_ML_data(train_list, spec_params=spec_params, mfcc_params=mfcc_params, 
                        lfcc_params=lfcc_params, energy_params=energy_params)

    # # Obteniendo los datos de testeo
    # X_data_test, Y_wheeze, Y_crackl = \
    #         get_ML_data(test_list, spec_params=spec_params, mfcc_params=mfcc_params, 
    #                     lfcc_params=lfcc_params, energy_params=energy_params)

    
    
    # Obtener las características para transformar las variables
    with open(f'respiratory_sound_classification/{objective_label}_features_params_{db_process}.txt', 
                'r', encoding='utf8') as file:
        params_dict = literal_eval(file.readline().strip())
    
    # Definición de las características
    s_clean = np.array(params_dict['s_clean'])
    a_norm = np.array(params_dict['a_norm'])
    b_norm = np.array(params_dict['b_norm'])
    s_sfs = np.array(params_dict['s_sfs'])
    
    # Transformando las características del conjunto de testeo en 
    # base a los parámetros
    print('Transformando')
    X_data_train = X_data_train[:, s_clean]         # 1) Clean
    X_data_train = a_norm * X_data_train + b_norm   # 2) Normalización
    X_data_train = X_data_train[:, s_sfs]           # 3) Selección
    
    # X_data_test = X_data_test[:, s_clean]         # 1) Clean
    # X_data_test = a_norm * X_data_test + b_norm   # 2) Normalización
    # X_data_test = X_data_test[:, s_sfs]           # 3) Selección
    
    print(X_data_train.shape)
    # print(X_data_test.shape)
    
    for i in tqdm(range(X_data_train.shape[0]), ncols=70, desc='Plotting'):
        if Y_wheeze[i] == 0:
            plt.plot(X_data_train[i,0], X_data_train[i,1], color='blue', ls='', marker='o')
        elif Y_wheeze[i] == 1:
            plt.plot(X_data_train[i,0], X_data_train[i,1], color='red', ls='', marker='x')
    
    plt.show()


def classificaton_results_oncycles(db_process, class_params, objective_label, objective_set):
    # Parámetros
    # db_process = 'preprocessed'
    # class_params = {'classifier': 'knn', 'param': '5'}      # knn, svm, mlp
    # objective_label = 'wheeze'                            # wheeze, crackle
    # objective_set = 'train'
    
    # Parámetros de caracaterísticas (no modificar)
    N = 1024
    noverlap = int(0.9 * N)
    spec_params = {'N': N, 'noverlap': noverlap, 'window': 'hann', 
                   'padding': 0, 'repeat': 0}
    mfcc_params = {'n_mfcc': 50, 'n_filters': 50, 'spec_params': spec_params,
                   'freq_lim': 2000, 'norm_filters': True, 'power': 2}
    lfcc_params = {'n_mfcc': 50, 'n_filters': 50, 'spec_params': spec_params,
                   'freq_lim': 2000, 'norm_filters': True, 'power': 2}
    energy_params = {'spec_params': spec_params, 'fmin': 0, 'fmax': 1000, 
                     'fband': 20}
    
    # Definición de la carpeta con la base de datos
    db_folder = f'{db_process}_signals'
    db_original = 'C:/Users/Chris/Desktop/Scripts_Magister/Respiratory_Sound_Database/audio_and_txt_files'
    
    
    
    ###############     RUTINA     ###############
        
    # Obtención del conjunto de entrenamiento y testeo
    train_test_folder = '/'.join(db_original.split('/')[:-1])
    dir_div_file = f'{train_test_folder}/ICBHI_challenge_train_test.txt'
    train_list, test_list = train_test_filebased(dir_div_file, db_root=db_original,
                                                 db_folder=db_folder)
    
    
    
    # Definición del nombre del modelo en base a los parámetros
    model_name = f'respiratory_sound_classification/{db_folder}/oncycle/'\
                    f'{class_params["classifier"]}_{class_params["param"]}_'\
                    f'{objective_label}_{db_process}'
    
    
    # Control de versiones
    if os.path.isfile(f'{model_name}_{objective_set}.txt'):
        print(f'Archivo con el nombre {model_name}_{objective_set} ya existe...')
        return
    
    
    
    # Cargar el modelo
    if class_params["classifier"] in ['knn', 'svm']:
        model = pickle.load(open(f'{model_name}.pickle', 'rb'))
    elif class_params["classifier"] == 'mlp':
        model = tf.keras.models.load_model(f'{model_name}.h5')
    
    
    if objective_set == 'train':
        # Obteniendo los datos de entrenamiento
        X_data, Y_wheeze, Y_crackl = \
                get_ML_data_oncycles(train_list, mfcc_params=mfcc_params, 
                                     lfcc_params=lfcc_params, 
                                     energy_params=energy_params)

    elif objective_set == 'test':
        # Obteniendo los datos de testeo
        X_data, Y_wheeze, Y_crackl = \
                get_ML_data_oncycles(test_list, mfcc_params=mfcc_params, 
                                     lfcc_params=lfcc_params, 
                                     energy_params=energy_params)
    
    else:
        raise Exception('Error en la selección del conjunto objective_set')
    
    
    # Obtener las características para transformar las variables
    with open(f'respiratory_sound_classification/{objective_label}_features_params_{db_process}_oncycle.txt', 
               'r', encoding='utf8') as file:
        params_dict = literal_eval(file.readline().strip())
    
    # Definición de las características
    s_clean = np.array(params_dict['s_clean'])
    a_norm = np.array(params_dict['a_norm'])
    b_norm = np.array(params_dict['b_norm'])
    s_sfs = np.array(params_dict['s_sfs'])
    
    # Transformando las características del conjunto de testeo en 
    # base a los parámetros
    print('Transformando')
    X_data = X_data[:, s_clean]         # 1) Clean
    X_data = a_norm * X_data + b_norm   # 2) Normalización
    X_data = X_data[:, s_sfs]           # 3) Selección
    
    print(f'Prediciendo con {model_name}...')
    print(X_data.shape)
    # Evaluando
    Y_pred = model.predict(X_data)
    
    # Si es la red neuronal, modificar el Y_pred
    if class_params["classifier"] == 'mlp':
        Y_pred = np.where(Y_pred < 0.5, 0, 1)[:, 0]
    
    
    # Comparando con las etiquetas
    if objective_label == 'wheeze':
        conf_mat = confusion_matrix(Y_wheeze, Y_pred)
        
    elif objective_label == 'crackle':
        conf_mat = confusion_matrix(Y_crackl, Y_pred)
    
    print(conf_mat)
    
    # Registrando la matriz
    with open(f'{model_name}_{objective_set}.txt', 'w', encoding='utf8') as file:
        file.write(f'{conf_mat.tolist()}')


def get_results_oncycle(db_process, class_params, objective_label, objective_set,
                        print_info='conf_matrix', print_detail=False):
    # Definición de la carpeta con la base de datos
    db_folder = f'{db_process}_signals'
    
    # Definición del nombre del modelo en base a los parámetros
    model_name = f'respiratory_sound_classification/{db_folder}/oncycle/'\
                    f'{class_params["classifier"]}_{class_params["param"]}_'\
                    f'{objective_label}_{db_process}'
    
    if objective_set == 'train':        
        # Abriendo el archivo
        with open(f'{model_name}_{objective_set}.txt', 'r', encoding='utf8') as file:
            conf_mat = literal_eval(file.readline().strip())
    
    elif objective_set == 'test':
        # Abriendo el archivo
        with open(f'{model_name}_results.txt', 'r', encoding='utf8') as file:
            conf_mat = literal_eval(file.readline().strip())

    
    # Se printea
    if print_info == 'conf_matrix':
        if print_detail:
            print(f'{model_name}: {conf_mat}')
        else:
            print(f'{conf_mat}')
    
    elif print_info == 'values':
        # Obtener los elementos de cada componente
        tp = conf_mat[1][1]
        tn = conf_mat[0][0]
        fp = conf_mat[0][1]
        fn = conf_mat[1][0]
        
        # Calculando los valores de interés
        accuracy  = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall    = tp / (tp + fn)
        
        if print_detail:
            print(f'{model_name}: {accuracy},{precision},{recall}')
        else:
            print(f'{accuracy},{precision},{recall}')


def plot_segments(db_process, feature='spectrogram', normalize_bool=True):
    # Parámetros de caracaterísticas (no modificar)
    N = 1024
    noverlap = int(0.5 * N)
    spec_params = {'N': N, 'noverlap': noverlap, 'window': 'hann', 
                   'padding': 0, 'repeat': 0}
    mfcc_params = {'n_mfcc': 50, 'n_filters': 50, 'spec_params': spec_params,
                   'freq_lim': 2000, 'norm_filters': True, 'power': 2}
    lfcc_params = {'n_mfcc': 50, 'n_filters': 50, 'spec_params': spec_params,
                   'freq_lim': 2000, 'norm_filters': True, 'power': 2}
    energy_params = {'spec_params': spec_params, 'fmin': 0, 'fmax': 1001, 
                     'fband': 20}
       
    # Definición de la carpeta con la base de datos
    db_folder = f'{db_process}_signals'
    db_original = 'C:/Users/Chris/Desktop/Scripts_Magister/Respiratory_Sound_Database/audio_and_txt_files'
    
    ###############     RUTINA     ###############
        
    # Obtención del conjunto de entrenamiento y testeo
    train_test_folder = '/'.join(db_original.split('/')[:-1])
    dir_div_file = f'{train_test_folder}/ICBHI_challenge_train_test.txt'
    train_list, test_list = train_test_filebased(dir_div_file, db_root=db_original,
                                                 db_folder=db_folder)
    
    if feature == 'spectrogram':
        S_silenc_train, S_wheeze_train, S_crackl_train = \
                        get_db_spectrograms(train_list, spec_params)
        
        S_silenc_test, S_wheeze_test, S_crackl_test = \
                        get_db_spectrograms(test_list, spec_params)
    
    elif feature == 'MFCC':
        S_silenc_train, S_wheeze_train, S_crackl_train = \
                        get_db_MFCC(train_list, spec_params)
        
        S_silenc_test, S_wheeze_test, S_crackl_test = \
                        get_db_MFCC(test_list, spec_params)
                        
    elif feature == 'energy':
        S_silenc_train, S_wheeze_train, S_crackl_train = \
                        get_db_energy(train_list, spec_params, 
                                      energy_params)
        
        S_silenc_test, S_wheeze_test, S_crackl_test = \
                        get_db_energy(test_list, spec_params, 
                                      energy_params)
                        
    elif feature == 'system':
        S_silenc_train, S_wheeze_train, S_crackl_train = \
                        get_db_system(train_list, spec_params, 
                                      mfcc_params=mfcc_params,
                                      lfcc_params=lfcc_params, 
                                      energy_params=energy_params)
        
        S_silenc_test, S_wheeze_test, S_crackl_test = \
                        get_db_system(test_list, spec_params, 
                                      mfcc_params=mfcc_params,
                                      lfcc_params=lfcc_params, 
                                      energy_params=energy_params)
    
    # Pasando a array
    S_silenc_train = np.hstack(S_silenc_train)
    S_wheeze_train = np.hstack(S_wheeze_train)
    S_crackl_train = np.hstack(S_crackl_train)
    
    S_silenc_test = np.hstack(S_silenc_test)
    S_wheeze_test = np.hstack(S_wheeze_test)
    S_crackl_test = np.hstack(S_crackl_test)
    
    # Normalizando
    if normalize_bool:
        S_silenc_train, _, _ = normalize(S_silenc_train.T)
        S_wheeze_train, _, _ = normalize(S_wheeze_train.T)
        S_crackl_train, _, _ = normalize(S_crackl_train.T)
        
        S_silenc_test, _, _ = normalize(S_silenc_test.T)
        S_wheeze_test, _, _ = normalize(S_wheeze_test.T)
        S_crackl_test, _, _ = normalize(S_crackl_test.T)
        
        S_silenc_train = S_silenc_train.T
        S_wheeze_train = S_wheeze_train.T
        S_crackl_train = S_crackl_train.T
        S_silenc_test = S_silenc_test.T
        S_wheeze_test = S_wheeze_test.T
        S_crackl_test = S_crackl_test.T
       
    
    # plt.figure(figsize=(12,5))
    # plt.subplot(1,3,1)
    # plt.pcolormesh(20 * np.log10(abs(S_silenc) + 1e-10), cmap='jet', vmin=-200, vmax=0)
    # plt.colorbar()
    
    # plt.subplot(1,3,2)
    # plt.pcolormesh(20 * np.log10(abs(S_wheeze) + 1e-10), cmap='jet', vmin=-200, vmax=0)
    # plt.colorbar()
    
    # plt.subplot(1,3,3)
    # plt.pcolormesh(20 * np.log10(abs(S_crackl) + 1e-10), cmap='jet', vmin=-200, vmax=0)
    # plt.colorbar()
    
    # plt.show()

    plt.figure(figsize=(12,5))
    plt.subplot(2,3,1)
    plt.pcolormesh(abs(S_silenc_train), cmap='jet', vmin=0, vmax=15)
    plt.colorbar()
    plt.title('Train silenc')
    
    plt.subplot(2,3,2)
    plt.pcolormesh(abs(S_wheeze_train), cmap='jet', vmin=0, vmax=15)
    plt.colorbar()
    plt.title('Train wheeze')
    
    plt.subplot(2,3,3)
    plt.pcolormesh(abs(S_crackl_train), cmap='jet', vmin=0, vmax=15)
    plt.colorbar()
    plt.title('Train crackl')
    
    
    plt.subplot(2,3,4)
    plt.pcolormesh(abs(S_silenc_test), cmap='jet', vmin=0, vmax=15)
    plt.colorbar()
    plt.title('Test silenc')
    
    plt.subplot(2,3,5)
    plt.pcolormesh(abs(S_wheeze_test), cmap='jet', vmin=0, vmax=15)
    plt.colorbar()
    plt.title('Test wheeze')
    
    plt.subplot(2,3,6)
    plt.pcolormesh(abs(S_crackl_test), cmap='jet', vmin=0, vmax=15)
    plt.colorbar()
    plt.title('Test crackl')
    
    plt.show()




# Módulo de testeo
if __name__ == '__main__':
    func_to = 'plot_segments'
    
    
    if func_to == 'classificaton_results':
        # Parámetros
        db_process_list = ['preprocessed', 'unpreprocessed']
        # class_params = {'classifier': 'knn', 'param': '5'}      # knn, svm, mlp
        # class_params_list = [{'classifier': 'knn', 'param': '3'},
        #                      {'classifier': 'svm', 'param': 'rbf'},
        #                      {'classifier': 'svm', 'param': 'poly'}]
        class_params_list = [{'classifier': 'mlp', 'param': True}]
        objective_label_list = ['wheeze', 'crackle']            # wheeze, crackle
        objective_set_list = ['train']
        
        # Para cada iteración
        for class_params_i in class_params_list:
            for process_i in db_process_list:
                for objective_label_i in objective_label_list:
                    for objective_set_i in objective_set_list:
                        # Tiempo de inicio
                        init_time = datetime.now()
                        classificaton_results_CNN(db_process=process_i, 
                                                  class_params=class_params_i,
                                                  objective_label=objective_label_i,
                                                  objective_set=objective_set_i)
                        print(f'Duracion: {datetime.now() - init_time} hrs')
                        
                        # Eliminando las variables registradas que no se referencian en memoria
                        print("Recolectando registros de memoria sin uso...")
                        n = gc.collect()
                        print("Número de objetos inalcanzables recolectados por el GC:", n)
                        print("Basura incoleccionable:", gc.garbage)

    
    elif func_to == 'classificaton_results_CNN':
        # Parámetros
        db_process_list = ['unpreprocessed']
        objective_label_list = ['wheeze']            # wheeze, crackle
        objective_set_list = ['test']

        # Para cada iteración
        for process_i in db_process_list:
            for objective_label_i in objective_label_list:
                for objective_set_i in objective_set_list:
                    # Tiempo de inicio
                    init_time = datetime.now()
                    classificaton_results_CNN(db_process=process_i,
                                              objective_label=objective_label_i,
                                              objective_set=objective_set_i)
                    print(f'Duracion: {datetime.now() - init_time} hrs')
                    
                    # Eliminando las variables registradas que no se referencian en memoria
                    print("Recolectando registros de memoria sin uso...")
                    n = gc.collect()
                    print("Número de objetos inalcanzables recolectados por el GC:", n)
                    print("Basura incoleccionable:", gc.garbage)
    
    
    elif func_to == 'get_results':
        db_process_list = ['preprocessed', 'unpreprocessed']
        class_params_list = [{'classifier': 'knn', 'param': '3'},
                             {'classifier': 'knn', 'param': '5'},
                             {'classifier': 'svm', 'param': 'rbf'},
                             {'classifier': 'svm', 'param': 'poly'},
                             {'classifier': 'mlp', 'param': True}]
        objective_label_list = ['wheeze']           # wheeze, crackle
        objective_set_list = ['train'] #, 'test']
        print_info = 'values'
        print_detail = True
        
        # Para cada iteración
        for objective_label_i in objective_label_list:
            for process_i in db_process_list:
                for objective_set_i in objective_set_list:
                    for class_params_i in class_params_list:
                        get_results(db_process=process_i, 
                                    class_params=class_params_i,
                                    objective_label=objective_label_i,
                                    objective_set=objective_set_i,
                                    print_info=print_info,
                                    print_detail=print_detail)

    
    elif func_to == 'get_results_CNN':
        db_process_list = ['unpreprocessed']
        objective_label_list = ['wheeze']           # wheeze, crackle
        objective_set_list = ['test'] #, 'test']
        print_info = 'values'
        print_detail = True
        
        # Para cada iteración
        for objective_label_i in objective_label_list:
            for process_i in db_process_list:
                for objective_set_i in objective_set_list:
                    get_results_CNN(db_process=process_i, 
                                    objective_label=objective_label_i,
                                    objective_set=objective_set_i,
                                    print_info=print_info,
                                    print_detail=print_detail)

    
    elif func_to == 'get_number_labels':
        get_number_labels('preprocessed')
        
    
    elif func_to == 'get_number_labels_CNN':
        get_number_labels_CNN('preprocessed')
    
    
    elif func_to == 'plot_features_MLP':
        db_process_list = ['preprocessed', 'unpreprocessed']
        # class_params = {'classifier': 'knn', 'param': '5'}      # knn, svm, mlp
        # class_params_list = [{'classifier': 'knn', 'param': '3'},
        #                      {'classifier': 'svm', 'param': 'rbf'},
        #                      {'classifier': 'svm', 'param': 'poly'}]
        class_params_list = [{'classifier': 'mlp', 'param': True}]
        objective_label_list = ['wheeze', 'crackle']            # wheeze, crackle
        
        # Tiempo de inicio
        init_time = datetime.now()
        plot_features_MLP(db_process='preprocessed', objective_label='wheeze')
        print(f'Duracion: {datetime.now() - init_time} hrs')
    
    
    elif func_to == 'classificaton_results_oncycles':
        # Parámetros
        db_process_list = ['preprocessed', 'unpreprocessed']
        # class_params = {'classifier': 'knn', 'param': '5'}      # knn, svm, mlp
        # class_params_list = [{'classifier': 'knn', 'param': '3'},
        #                      {'classifier': 'knn', 'param': '5'},
        #                      {'classifier': 'svm', 'param': 'rbf'},
        #                      {'classifier': 'svm', 'param': 'poly'}]
        class_params_list = [{'classifier': 'mlp', 'param': True}]
        objective_label_list = ['wheeze', 'crackle']            # wheeze, crackle
        objective_set_list = ['train']
        
        # Para cada iteración
        for class_params_i in class_params_list:
            for process_i in db_process_list:
                for objective_label_i in objective_label_list:
                    for objective_set_i in objective_set_list:
                        # Tiempo de inicio
                        init_time = datetime.now()
                        classificaton_results_oncycles(db_process=process_i, 
                                                       class_params=class_params_i,
                                                       objective_label=objective_label_i,
                                                       objective_set=objective_set_i)
                        print(f'Duracion: {datetime.now() - init_time} hrs')
                        
                        # Eliminando las variables registradas que no se referencian en memoria
                        print("Recolectando registros de memoria sin uso...")
                        n = gc.collect()
                        print("Número de objetos inalcanzables recolectados por el GC:", n)
                        print("Basura incoleccionable:", gc.garbage)
    
    
    elif func_to == 'get_results_oncycle':
        db_process_list = ['preprocessed', 'unpreprocessed']
        # class_params_list = [{'classifier': 'knn', 'param': '3'},
        #                      {'classifier': 'knn', 'param': '5'},
        #                      {'classifier': 'svm', 'param': 'rbf'},
        #                      {'classifier': 'svm', 'param': 'poly'},]
                            #  {'classifier': 'mlp', 'param': True}]
        class_params_list = [{'classifier': 'mlp', 'param': True}]
        objective_label_list = ['wheeze']           # wheeze, crackle
        objective_set_list = ['test'] #, 'test']
        print_info = 'values'
        print_detail = False
        
        # Para cada iteración
        for objective_label_i in objective_label_list:
            for process_i in db_process_list:
                for objective_set_i in objective_set_list:
                    for class_params_i in class_params_list:
                        get_results_oncycle(db_process=process_i, 
                                            class_params=class_params_i,
                                            objective_label=objective_label_i,
                                            objective_set=objective_set_i,
                                            print_info=print_info,
                                            print_detail=print_detail)

    
    elif func_to == 'plot_segments':
        plot_segments(db_process='unpreprocessed', feature='system')    
    
    