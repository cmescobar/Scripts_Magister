import os
import soundfile as sf
from ast import literal_eval

DIAGNOSIS_CSV = 'Respiratory_Sound_Database/patient_diagnosis.csv'
AUDIO_FILES = 'Respiratory_Sound_Database/audio_and_txt_files'


def get_patient_by_symptom(symptom):
    '''Esta función permite retornar una lista de pacientes que tiene la
    etiqueta de la enfermedad respiratoria pedida en la variable
    "symptom".
    - "symptom": (string) Enfermedad a buscar

    Retorna una lista (out) con los ID de los pacientes que estén con ese
    síntoma.'''

    # Definición de la variable de salida
    out = []

    with open(DIAGNOSIS_CSV, 'r', encoding='utf8') as file:
        for line in file:
            id, patient_symptom = line.strip('\n').split(',')
            if patient_symptom == symptom:
                out.append(id)
    return out


def get_dir_audio_by_id(id_list):
    '''Esta función permite retornar una lista de la dirección de los audios de
     los pacientes que se encuentran en la lista de identificadores "id_list"
    - "id_list": (list) Lista de id's de los pacientes a buscar

    Retorna una lista (out) con las direcciones de los audios de esta lista
    de id's.'''

    # Definición del vector de salida
    out = []

    files = os.listdir(AUDIO_FILES)
    for i in files:
        if i.endswith('.wav') and i[:3] in id_list:
            out.append(AUDIO_FILES + '/' + i)

    return out


def get_dir_audiotxt_by_id(id_list):
    '''Esta función permite retornar una lista de la dirección de los audios de
     los pacientes que se encuentran en la lista de identificadores "id_list"
    - "id_list": (list) Lista de id's de los pacientes a buscar

    Retorna una lista (out) con las direcciones de los txt de los audios de esta
    lista de id's.'''

    # Definición del vector de salida
    out = []

    files = os.listdir(AUDIO_FILES)
    for i in files:
        if i.endswith('.txt') and i[:3] in id_list:
            out.append(AUDIO_FILES + '/' + i)

    return out


def get_dir_audiotxt_by_symptom(symptom, wav=True):
    '''Esta función permite retornar una lista de pacientes que tiene la
    etiqueta de la enfermedad respiratoria pedida en la variable
    "symptom".
    - "symptom": (string) Enfermedad a buscar

    Retorna una lista (out) con direcciones de los audios de los pacientes que
    estén con ese síntoma. '''

    # Lista las id de pacientes por síntoma
    id_list = get_patient_by_symptom(symptom)

    if wav:
        return get_dir_audio_by_id(id_list)
    else:
        return get_dir_audiotxt_by_id(id_list)


def get_audio_folder_by_symptom(symptom, sep_type='all'):
    # Corroborar la opción de separación
    if sep_type not in ['all', 'tracheal', 'toracic']:
        print('The option is not valid. Please try again.')
        return

    # Obtener las id de los pacientes
    symptom_list = get_patient_by_symptom(symptom)

    # Luego obtener las direcciones de los archivos de audio para cada caso
    symptom_dirs = get_dir_audio_by_id(symptom_list)

    # Definición de la carpeta dónde se guardará la información
    folder_data = f'Interest_Audios/{symptom}/{sep_type}'
    
    # Preguntar si es que la carpeta que almacenará los sonidos se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(folder_data):
        os.mkdir(folder_data)

    # Lectura de cada uno de los archivos de audios para la reescritura
    # normalizada del archivo de audio
    for i in symptom_dirs:
        if sep_type == 'tracheal'and 'Tc' not in i:
            continue
        elif sep_type == 'toracic' and 'Tc' in i:
            continue

        # Lectura del audio
        audio, samplerate = sf.read(i)

        # Normalizando al archivo de audio
        audio /= max(abs(audio))

        # Definición de la carpeta donde se guardará
        folder_to_save = f"{folder_data}/{i.split('/')[-1]}"
        
        # Escribiendo
        print(f"Re-writing file sound {i.split('/')[-1]}...")
        sf.write(folder_to_save, audio, samplerate, 'PCM_24')
        print("Re-writing complete!\n")


def get_segmentation_points_by_filename(symptom, filename):
    # Dirección de los puntos originales
    filepath = f"Results/{symptom}/{symptom}_resp_original_points.txt"
    # Abriendo el archivo
    with open(filepath, 'r', encoding='utf8') as file:
        for i in file:
            # Obteniendo el nombre
            name_in_file = i.split(';')[0]
            if name_in_file == filename:
                # Se obtiene la lista en texto
                list_txt = i.split(';')[1].strip()
                # Transformando a lista
                return literal_eval(list_txt)
            
# Test module
'''symptom = "Pneumonia"
get_audio_folder_by_symptom(symptom, sep_type='all')
get_audio_folder_by_symptom(symptom, sep_type='tracheal')
get_audio_folder_by_symptom(symptom, sep_type='toracic')
get_segmentation_points_by_filename("Healthy")'''