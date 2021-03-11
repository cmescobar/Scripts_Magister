import os
from heart_sound_segmentation.evaluation_functions import eval_sound_model_db


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
