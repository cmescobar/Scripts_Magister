import numpy as np
import tensorflow as tf
from ast import literal_eval
from paper_DNN_models import segnet_based_3_6, segnet_based_3_14


def get_architecture_results_OLD():
    '''Función utilizada para obtener los resultados de la comparación de 
    arquitecturas.
    '''
    
    ###########         Parámetros         ###########
    # Definición del experimento a revisar
    # experiments = ['cnn_dnn_1_1', 'cnn_dnn_1_2', 'cnn_dnn_1_3', 'cnn_dnn_1_4']
    # experiments = ['segnet_based_1_3', 'segnet_based_1_4', 'segnet_based_1_1',
    #                'segnet_based_1_2']
    experiments = ['segnet_based_2_1', 'segnet_based_2_2', 'segnet_based_2_3',
                   'segnet_based_2_4', 'segnet_based_2_5', 'segnet_based_2_6',
                   'segnet_based_2_7', 'segnet_based_2_8', 'segnet_based_2_9',
                   'segnet_based_2_10', 'segnet_based_2_11', 'segnet_based_2_12']

    # Definición del parámetro de interés a obtener
    interest_info = 'f1-score'

    # Tipo de conjunto de datos a revisar
    set_type = 'test'
    ###########         Parámetros         ###########


    ###########         Rutina         ###########

    # Definición de un contenedor de la información
    register_list = list()

    # Definición del prefijo a revisar
    if set_type == 'train' or set_type == 'test':
        suf = ''
    elif set_type == 'validation':
        suf = 'val_'
    else:
        print('set_type no válido.')
        exit() 


    for exp in experiments:
        with open(f'Paper_models/{exp}/{exp}.txt', 'r', encoding='utf8') as file:
            for line in file:
                try:
                    # Obtener la información de la línea
                    info = literal_eval(line.strip())
                    
                    # Recuperar la información de interés
                    # print(info[interest_info][0] * 100)
                    
                    # Guardando la información
                    if interest_info == 'f1-score':
                        register_list.append((info[f'{suf}precision'][0] * 100, 
                                            info[f'{suf}recall'][0]    * 100))
                    else:
                        register_list.append(info[f'{suf}{interest_info}'][0] * 100)
                    
                except:
                    # Eliminar el encabezado
                    info_test = literal_eval(line.strip().replace('Testing_info: ', ''))
                

        # Parámetros de interés a incluir para el cálculo del valor
        values = register_list[-4:]

        if interest_info == 'f1_score':
            values_to = [2 * (i[0] * i[1]) / (i[0] + i[1]) for i in values]
        else:
            values_to = values
        
        if set_type in ['train', 'validation']:
            print(f"{round(np.mean(values_to), 2)} $\pm$ {round(np.std(values_to), 2)}" )
        elif set_type == 'test':
            if interest_info == 'f1-score':
                print(2 * (info_test['recall'] * info_test['precision']) /
                    (info_test['recall'] + info_test['precision'])  * 100)
            else:
                print(info_test[f'{suf}{interest_info}'] * 100)

        # print(values_to)


def get_architecture_results():
    '''Función utilizada para obtener los resultados de la comparación de 
    arquitecturas.
    '''
    
    ###########         Parámetros         ###########
    # Definición del experimento a revisar
    # experiments = ['cnn_dnn_1_1', 'cnn_dnn_1_2', 'cnn_dnn_1_3', 'cnn_dnn_1_4']
    # experiments = ['segnet_based_1_3', 'segnet_based_1_4', 'segnet_based_1_1',
    #                'segnet_based_1_2']
    # experiments = ['segnet_based_1_3_all', 'segnet_based_1_4_all', 
    #                'segnet_based_1_1_all', 'segnet_based_1_2_all']
    # experiments = ['segnet_based_2_1', 'segnet_based_2_2', 'segnet_based_2_3',
    #                'segnet_based_2_4', 'segnet_based_2_5', 'segnet_based_2_6',
    #                'segnet_based_2_7', 'segnet_based_2_8', 'segnet_based_2_9',
    #                'segnet_based_2_10', 'segnet_based_2_11', 'segnet_based_2_12',
    #                'segnet_based_2_13', 'segnet_based_2_14', 'segnet_based_2_15',
    #                'segnet_based_2_16', 'segnet_based_2_17', 'segnet_based_2_18',
    #                'segnet_based_2_19', 'segnet_based_2_20', 'segnet_based_2_21']
    # experiments = ['segnet_based_3_1', 'segnet_based_3_2', 'segnet_based_3_3',
    #                'segnet_based_3_4', 'segnet_based_3_5', 'segnet_based_3_6',
    #                'segnet_based_3_7', 'segnet_based_3_8', 'segnet_based_3_9',
    #                'segnet_based_3_10', 'segnet_based_3_11', 'segnet_based_3_12',
    #                'segnet_based_3_13', 'segnet_based_3_14']
    # experiments = ['segnet_based_4_1', 'segnet_based_4_2', 'segnet_based_4_3',
    #                'segnet_based_4_4']
    # experiments = ['segnet_based_6_1', 'segnet_based_6_2', 'segnet_based_6_3',
    #                'segnet_based_6_4', 'segnet_based_6_5', 'segnet_based_6_6',
    #                'segnet_based_6_7', 'segnet_based_6_8', 'segnet_based_6_9',
    #                'segnet_based_6_10']
    # experiments = ['segnet_based_7_1', 'segnet_based_7_2', 'segnet_based_7_3',
    #                'segnet_based_7_4', 'segnet_based_7_5', 'segnet_based_7_6',
    #                'segnet_based_7_7', 'segnet_based_7_8', 'segnet_based_7_9',
    #                'segnet_based_7_10']
    # experiments = ['segnet_based_8_1', 'segnet_based_8_2', 'segnet_based_8_3',
    #                'segnet_based_8_4', 'segnet_based_8_5', 'segnet_based_8_6',
    #                'segnet_based_8_7', 'segnet_based_8_8', 'segnet_based_8_9',
    #                'segnet_based_8_10', 'segnet_based_8_11', 'segnet_based_8_12',
    #                'segnet_based_8_13']
    # experiments = ['segnet_based_9_1']
    # experiments = ['segnet_based_10_1', 'segnet_based_10_2']
    experiments = ['segnet_based_11_1', 'segnet_based_11_2']
    
    
    for exp in experiments:
        with open(f'Paper_models/{exp}/{exp}.txt', 'r', encoding='utf8') as file:
            lines = file.readlines()[-6:]

            # Defininición de las listas de interés para el entrenamiento
            accuracy_train_list = list()
            recall_train_list = list()
            precision_train_list = list()
            
            # Resultados del entrenamiento
            for line in lines[:4]:
                # Transformando a diccionario
                dict_to_rev = literal_eval(line.strip())
                
                # Agregando a las listas
                accuracy_train_list.extend(dict_to_rev['accuracy'])
                recall_train_list.extend(dict_to_rev['recall'])
                precision_train_list.extend(dict_to_rev['precision'])
                
                # print(dict_to_rev)
            
            # Valores de entrenamiento
            accuracy_train = np.mean(accuracy_train_list) * 100
            recall_train = np.mean(recall_train_list) * 100
            precision_train = np.mean(precision_train_list) * 100
            
            # Defininición de las listas de interés para la validación
            val_dict = literal_eval(lines[-2].strip().replace('Validation_info_epoch_20: ', ''))
            
            accuracy_val = val_dict['accuracy'] * 100
            recall_val = val_dict['recall'] * 100
            precision_val = val_dict['precision'] * 100
            
            
            # Obteniendo la información de interés para el testeo
            test_dict = literal_eval(lines[-1].strip().replace('Testing_info: ', ''))

            accuracy_test = test_dict['accuracy'] * 100
            recall_test = test_dict['recall'] * 100
            precision_test = test_dict['precision'] * 100
            
        print(f'{accuracy_train}, {recall_train}, {precision_train},, '
              f'{accuracy_val}, {recall_val}, {precision_val},, '
              f'{accuracy_test}, {recall_test}, {precision_test}')


def get_crossval_results():
    '''Función utilizada para obtener la validación cruzada de la red final.
    '''
    
    ###########         Parámetros         ###########
    # Definición del experimento a revisar
    experiments = ['segnet_based_12_10', 'segnet_based_12_2', 'segnet_based_12_3',
                   'segnet_based_12_4', 'segnet_based_12_5', 'segnet_based_12_6',
                   'segnet_based_12_7', 'segnet_based_12_8', 'segnet_based_12_9',
                   'segnet_based_12_10']

    
    
    for exp in experiments:
        with open(f'Paper_models/{exp}/{exp}.txt', 'r', encoding='utf8') as file:
            lines = file.readlines()[-19:]
            
            # Defininición de las listas de interés para el entrenamiento
            accuracy_train_list = list()
            recall_train_list = list()
            precision_train_list = list()
            
            # Resultados del entrenamiento
            for line in lines[:-1]:
                # Transformando a diccionario
                dict_to_rev = literal_eval(line.strip())
                
                # Agregando a las listas
                accuracy_train_list.extend(dict_to_rev['accuracy'])
                recall_train_list.extend(dict_to_rev['recall'])
                precision_train_list.extend(dict_to_rev['precision'])
                
                # print(dict_to_rev)
            
            # Valores de entrenamiento
            accuracy_train = np.mean(accuracy_train_list) * 100
            recall_train = np.mean(recall_train_list) * 100
            precision_train = np.mean(precision_train_list) * 100
                       
            
            # Obteniendo la información de interés para el testeo
            test_dict = literal_eval(lines[-1].strip().replace('Testing_info: ', ''))

            accuracy_test = test_dict['accuracy'] * 100
            recall_test = test_dict['recall'] * 100
            precision_test = test_dict['precision'] * 100
            
        print(f'{accuracy_train}, {recall_train}, {precision_train},, '
              f'{accuracy_test}, {recall_test}, {precision_test}')



def get_qparams_network():
    '''Rutina para contar la cantidad de parámetros entrenables de una red'''
    # models = ['cnn_dnn_1_1', 'cnn_dnn_1_2', 'cnn_dnn_1_3', 'cnn_dnn_1_4']
    # models = ['segnet_based_1_3_all', 'segnet_based_1_4_all', 'segnet_based_1_1_all',
    #           'segnet_based_1_2_all']
    # models = ['segnet_based_3_1', 'segnet_based_3_2', 'segnet_based_3_3',
    #             'segnet_based_3_4', 'segnet_based_3_5', 'segnet_based_3_6',
    #             'segnet_based_3_7', 'segnet_based_3_8', 'segnet_based_3_9',
    #             'segnet_based_3_10', 'segnet_based_3_11', 'segnet_based_3_12',
    #             'segnet_based_3_13', 'segnet_based_3_14']
    # models = ['segnet_based_3_6', 'segnet_based_3_14']
    # models = ['segnet_based_6_1', 'segnet_based_6_2', 'segnet_based_6_3',
    #           'segnet_based_6_4', 'segnet_based_6_5', 'segnet_based_6_6',
    #           'segnet_based_6_7', 'segnet_based_6_8', 'segnet_based_6_9',
    #           'segnet_based_6_10']
    # models = ['segnet_based_7_1', 'segnet_based_7_2', 'segnet_based_7_3',
    #           'segnet_based_7_4', 'segnet_based_7_5', 'segnet_based_7_6',
    #           'segnet_based_7_7', 'segnet_based_7_8', 'segnet_based_7_9',
    #           'segnet_based_7_10']
    models = ['segnet_based_8_1', 'segnet_based_8_2', 'segnet_based_8_3',
              'segnet_based_8_4', 'segnet_based_8_5', 'segnet_based_8_6',
              'segnet_based_8_7', 'segnet_based_8_8', 'segnet_based_8_9'
              'segnet_based_8_10', 'segnet_based_8_11', 'segnet_based_8_12',
              'segnet_based_8_13']
    
    for model_name in models:
        try:
            # Cargar el modelo
            model = tf.keras.models.load_model(f'Paper_models/{model_name}/{model_name}.h5')
            
        except:
            if model_name == 'segnet_based_3_6':
                model = segnet_based_3_6((1024,3), padding_value=2, name='Testeo')
            
            elif model_name == 'segnet_based_3_14':
                model = segnet_based_3_14((1024,3), padding_value=2, name='Testeo')
            
        
        # Contar la cantidad de variables entrenables
        trainable_count = sum([tf.keras.backend.count_params(i) 
                            for i in model.trainable_weights])
        non_trainable_count = sum([tf.keras.backend.count_params(i) 
                                for i in model.non_trainable_weights])
        
        # print(model.summary())
        print(trainable_count)
            
            

if __name__ == '__main__':
    # get_architecture_results()
    get_crossval_results()
    # get_qparams_network()
    pass
