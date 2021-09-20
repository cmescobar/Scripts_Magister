import numpy as np
import tensorflow as tf
from pybalu.feature_selection import sfs
from pybalu.feature_transformation import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, linear_model, svm
from sklearn.metrics import confusion_matrix, accuracy_score
from respiratory_sound_classification.features import pybalu_clean, pybalu_sfs



def ML_classification_system(X_train, Y_train, X_test, Y_test, 
                             clean_params=None, sel_params=None, 
                             class_params=None):
    '''Diseño del sistema de clasificación basado en Machine 
    Learning.
    
    Parameters
    ----------
    X_train : ndarray
        Datos de entrenamiento.
    Y_train : ndarray
        Etiquetas de los datos de entrenamiento.
    X_test : ndarray
        Datos de testeo.
    Y_test : ndarray
        Etiquetas de los datos de testeo.
    clean_params: dict or None, optional
        Parámetros del proceso de limpieza de características. 
        Si es None se utilizan características por defecto: 
        'tol': 13-5, 'show': True. Por defecto es None.
    sel_params: dict or None, optional
        Parámetros del proceso de selección de características. 
        Si es None se utilizan características por defecto: 
        'n_features': 10, 'show': True. Por defecto es None.
    class_params: dict or None, optional
        Parámetros del proceso de clasificación. Si es None se 
        utilizan características por defecto: 
        'classifier': 'knn', 'k_neigh': 10. Por defecto es None. 
        En caso de usar 'svm', es posible modificar el 'kernel'.
        
    Returns
    -------
    classifier : class
        Clasificador entrenado.
    X_test : ndarray
        Matriz de testeo modificada (en caso de que X_test no 
        sea None).
    params_out : dict
        Parámetros obtenidos a partir del entrenamiento del
        sistema sobre los datos. Se entrega información de las
        características del clean ('s_clean'), normalización
        ('a_norm' y 'b_norm'), y de la selección de 
        características ('s_sfs').
    Y_pred : ndarray or None
        Predicción realizada por el sistema (en caso de que
        Y_test no sea None). Si no se entrega Y_test, la salida
        será None.
    '''
    # Parámetros por defecto
    if clean_params is None:
        clean_params = {'tol': 1e-5, 'show': True}
    
    if sel_params is None:
        sel_params = {'n_features': 10, 'show': True}
    
    if class_params is None:
        class_params = {'classifier': 'knn', 'k_neigh': 10}
        
    Y_pred = None
    
    
    #### Pipeline de la etapa de clasificación ####
    
    ## 1) Limpieza de las características
    s_clean = pybalu_clean(X_train, tol=clean_params['tol'], 
                           show=clean_params['show'])
    
    # Aplicando la limpieza
    X_train = X_train[:, s_clean]
    
    
    ## 2) Normalización de los datos
    X_train, a_norm, b_norm = normalize(X_train)
    
    
    ## 3) Selección de características
    s_sfs = sfs(X_train, Y_train, show=sel_params['show'],
                n_features=sel_params['n_features'])
    
    # Aplicando la selección
    X_train = X_train[:, s_sfs]
    
    
    ## 4) Proceso de clasificación   
    if class_params['classifier'] == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=\
                                          class_params['k_neigh'])
        
    elif class_params['classifier'] == 'svm':
        classifier = svm.SVC(kernel=class_params['kernel'])
    
    else:
        raise Exception('Opción de clasificador no definida '
                        'correctamente.')
    
    # Ajustando el clasificador
    classifier.fit(X_train, Y_train)
    
    
    # Aplicando todo el proceso a los datos de testeo
    if X_test is not None:
        X_test = X_test[:, s_clean]         # 1) Clean
        X_test = a_norm * X_test + b_norm   # 2) Normalización
        X_test = X_test[:, s_sfs]           # 3) Selección
        
        # Aplicando el clasificador
        if Y_test is not None:
            Y_pred = classifier.predict(X_test)

    
    # Definición del diccionario de parámetros
    params_out = {'a_norm': a_norm, 'b_norm': b_norm, 's_clean': s_clean,
                  's_sfs': s_sfs}
        
    return classifier, X_test, params_out, Y_pred


def NN_MLP_classification_system(X_train, Y_train, X_test, Y_test, 
                                 clean_params=None, sel_params=None, 
                                 mlp_params=None):
    '''Diseño del sistema de clasificación basado en Redes Neuronales
    Multicapas.
    
    Parameters
    ----------
    X_train : ndarray
        Datos de entrenamiento.
    Y_train : ndarray
        Etiquetas de los datos de entrenamiento.
    X_test : ndarray
        Datos de testeo.
    Y_test : ndarray
        Etiquetas de los datos de testeo.
    clean_params: dict or None, optional
        Parámetros del proceso de limpieza de características. 
        Si es None se utilizan características por defecto: 
        {'tol': 13-5, 'show': True}. Por defecto es None.
    sel_params: dict or None, optional
        Parámetros del proceso de selección de características. 
        Si es None se utilizan características por defecto: 
        {'n_features': 10, 'show': True}. Por defecto es None.
    mlp_params : dict or None, optional
        Parámetros del preoceso de clasificación con MLP. Si es
        None se utilizan las características por defecto:
        {'optimizer': 'Adam', 'loss': 'binary_crossentropy',
         'batch_size': None, 'epochs': 100, 'verbose': 1, 
         'metrics': ['accuracy', tf.keras.metrics.Recall(), 
                      tf.keras.metrics.Precision()],
         'out_layer': 'sigmoid', 'preprocessing': True}
        
    Returns
    -------
    classifier : class
        Clasificador entrenado.
    X_test : ndarray
        Matriz de testeo modificada (en caso de que X_test no 
        sea None).
    params_out : dict
        Parámetros obtenidos a partir del entrenamiento del
        sistema sobre los datos. Se entrega información de las
        características del clean ('s_clean'), normalización
        ('a_norm' y 'b_norm'), y de la selección de 
        características ('s_sfs').
    Y_pred : ndarray or None
        Predicción realizada por el sistema (en caso de que
        Y_test no sea None). Si no se entrega Y_test, la salida
        será None.
    '''
    # Parámetros por defecto
    if clean_params is None:
        clean_params = {'tol': 1e-5, 'show': True}
    
    if sel_params is None:
        sel_params = {'n_features': 10, 'show': True}
        
    if mlp_params is None:
        mlp_params = {'optimizer': 'Adam', 'loss': 'binary_crossentropy',
                      'batch_size': None, 'epochs': 100, 'verbose': 1, 
                      'metrics': ['accuracy', tf.keras.metrics.Recall(), 
                                  tf.keras.metrics.Precision()],
                      'out_layer': 'sigmoid', 'preprocessing': True}
    
    Y_pred = None
    
    
    #### Pipeline de la etapa de clasificación ####
    
    # Rutina de preprocesamiento
    if mlp_params['preprocessing']:
        ## 1) Limpieza de las características
        s_clean = pybalu_clean(X_train, tol=clean_params['tol'], 
                               show=clean_params['show'])

        # Aplicando la limpieza
        X_train = X_train[:, s_clean]


        ## 2) Normalización de los datos
        X_train, a_norm, b_norm = normalize(X_train)


        ## 3) Selección de características
        s_sfs = sfs(X_train, Y_train, show=sel_params['show'],
                    n_features=sel_params['n_features'])

        # Aplicando la selección
        X_train = X_train[:, s_sfs]

    
    
    ## 4) Proceso de clasificación
    
    # Definición del modelo
    model = MLP_network(input_shape=(X_train.shape[1],),
                        out_layer=mlp_params['out_layer'])
    
    # Compilando modelos
    model.compile(optimizer=mlp_params['optimizer'], 
                  loss=mlp_params['loss'],
                  metrics=mlp_params['metrics'])
    
    
    # Definición de los vectores
    if mlp_params['out_layer'] == 'softmax':
        # One-Hot
        Y_train_to = \
            np.array([Y_train, np.ones(len(Y_train)) - Y_train]).T
    
    elif mlp_params['out_layer'] == 'sigmoid':
        # Normal
        Y_train_to = Y_train
    
    
    # Ajustando el Modelo
    history = model.fit(x=X_train, y=Y_train_to, 
                        batch_size=mlp_params['batch_size'],
                        epochs=mlp_params['epochs'],
                        verbose=mlp_params['verbose'])
    
    
    # Aplicando todo el proceso a los datos de testeo
    if X_test is not None:
        # Si se realizó el preprocesamiento, se actualiza
        if mlp_params['preprocessing']:
            X_test = X_test[:, s_clean]         # 1) Clean
            X_test = X_test * a_norm + b_norm   # 2) Normalización
            X_test = X_test[:, s_sfs]           # 3) Selección
        
        # Aplicando el clasificador
        if Y_test is not None:
            Y_pred = model.predict(X_test)

    
    # Definición del diccionario de parámetros
    if mlp_params['preprocessing']: 
        params_out = {'a_norm': a_norm, 'b_norm': b_norm, 's_clean': s_clean,
                      's_sfs': s_sfs, 'history': history}
    else:
        params_out = {'history': history}
    
    
    return model, X_test, params_out, Y_pred


def MLP_network_OLD(input_shape, out_layer='sigmoid'):
    '''Función que define una red de perceptrones multicapas para 
    clasificar.
    
    Parameters
    ----------
    input_shape : list or ndarray
        Dimensión de la información de entrada.
    out_layer : {'sigmoid', 'softmax'}, optional
        Función a usar en la capa de salida de la red. Por defecto
        es 'sigmoid'.
    
    Returns
    -------
    model: tensorflow.keras.Model
        Modelo del sistema.
    '''
    
    def _layer(input_layer, units, kernel_initializer, 
               bias_initializer, name):
        '''Función auxiliar que modela las capas Dense + batchnorm +
        Activation ReLU'''
        # Aplicando la concatenación de capas
        x_dense = tf.keras.layers.Dense(units=units, 
                                        bias_initializer=bias_initializer,
                                        kernel_initializer=kernel_initializer,
                                        name=f'Dense_{name}')(input_layer)
        x_dense = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_dense)
        x_dense = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_dense)

        return x_dense
    
    
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')
    
    
    # Definición de la red misma
    x_layer = _layer(x_in, units=500, kernel_initializer='he_normal', 
                     bias_initializer='he_normal', name='Layer_1')
    x_layer = _layer(x_layer, units=200, kernel_initializer='he_normal', 
                     bias_initializer='he_normal', name='Layer_2')
    x_layer = _layer(x_layer, units=100, kernel_initializer='he_normal', 
                     bias_initializer='he_normal', name='Layer_3')
    x_layer = _layer(x_layer, units=80, kernel_initializer='he_normal', 
                     bias_initializer='he_normal', name='Layer_4')
    x_layer = _layer(x_layer, units=30, kernel_initializer='he_normal', 
                     bias_initializer='he_normal', name='Layer_5')
    x_layer = _layer(x_layer, units=10, kernel_initializer='he_normal', 
                     bias_initializer='he_normal', name='Layer_6')
    x_layer = _layer(x_layer, units=5, kernel_initializer='he_normal', 
                     bias_initializer='he_normal', name='Layer_7')
    
    # Definición de la salida
    if out_layer == 'softmax':
        x_out = tf.keras.layers.Dense(2, activation='softmax', 
                                      kernel_initializer='he_normal', 
                                      bias_initializer='he_normal',
                                      name='softmax_out')(x_layer)
    elif out_layer == 'sigmoid':
        x_out = tf.keras.layers.Dense(1, activation='sigmoid', 
                                      kernel_initializer='he_normal', 
                                      bias_initializer='he_normal',
                                      name='sigmoid_out')(x_layer)
    else:
        raise Exception(f'Opción de parámetro "out_layer"={out_layer} '
                        f'no válido.')
    
    # Definir el modelo
    model = tf.keras.Model(inputs=x_in, outputs=x_out, name='Red_MLP')
    
    return model


def MLP_network(input_shape, out_layer='sigmoid'):
    '''Función que define una red de perceptrones multicapas para 
    clasificar.
    
    Parameters
    ----------
    input_shape : list or ndarray
        Dimensión de la información de entrada.
    out_layer : {'sigmoid', 'softmax'}, optional
        Función a usar en la capa de salida de la red. Por defecto
        es 'sigmoid'.
    
    Returns
    -------
    model: tensorflow.keras.Model
        Modelo del sistema.
    '''
    
    def _layer(input_layer, units, kernel_initializer, 
               bias_initializer, name):
        '''Función auxiliar que modela las capas Dense + batchnorm +
        Activation ReLU'''
        # Aplicando la concatenación de capas
        x_dense = tf.keras.layers.Dense(units=units, 
                                        bias_initializer=bias_initializer,
                                        kernel_initializer=kernel_initializer,
                                        name=f'Dense_{name}')(input_layer)
        x_dense = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_dense)
        x_dense = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_dense)

        return x_dense
    
    
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')
    
    
    # Definición de la red misma
    x_layer = _layer(x_in, units=40, kernel_initializer='he_normal', 
                     bias_initializer='he_normal', name='Layer_1')
    x_layer = _layer(x_layer, units=30, kernel_initializer='he_normal', 
                     bias_initializer='he_normal', name='Layer_2')
    
    # Definición de la salida
    if out_layer == 'softmax':
        x_out = tf.keras.layers.Dense(2, activation='softmax', 
                                      kernel_initializer='he_normal', 
                                      bias_initializer='he_normal',
                                      name='softmax_out')(x_layer)
    elif out_layer == 'sigmoid':
        x_out = tf.keras.layers.Dense(1, activation='sigmoid', 
                                      kernel_initializer='he_normal', 
                                      bias_initializer='he_normal',
                                      name='sigmoid_out')(x_layer)
    else:
        raise Exception(f'Opción de parámetro "out_layer"={out_layer} '
                        f'no válido.')
    
    # Definir el modelo
    model = tf.keras.Model(inputs=x_in, outputs=x_out, name='Red_MLP')
    
    return model


def CNN_network_OLD(input_shape, padding_value, out_layer='sigmoid'):
    '''Función que define una red CNN para extraer características y 
    clasificar.
    
    Parameters
    ----------
    padding_value : float
        Valor utilizado para hacer padding en la señal.
    out_layer : {'sigmoid', 'softmax'}, optional
        Función a usar en la capa de salida de la red. Por defecto
        es 'sigmoid'.
    
    Returns
    -------
    model: tensorflow.keras.Model
        Modelo del sistema.
    '''
    def _conv_bn_act_layer(input_layer, filters, kernel_size, padding,
                           kernel_initializer, bias_initializer, name):
        '''Función auxiliar que modela las capas azules conv + batchnorm +
        Activation ReLU para realizar el ENCODING.'''
        # Aplicando la concatenación de capas
        x_conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer=bias_initializer,
                                        padding=padding, 
                                        name=f'Conv_{name}')(input_layer)
        x_conv = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_conv)
        x_conv = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_conv)

        return x_conv
    
    
    def _cnn_layers(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar "n_layers_conv" capas CNN seguida de 
        una capa de Maxpooling.  
        '''
        # Definición de la salida de este bloque
        x_enc = input_layer
        
        # Aplicando "n_layers_conv" capas convolucionales de codificación
        for i in range(n_layers_conv):
            x_enc = _conv_bn_act_layer(x_enc, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'],
                                       bias_initializer=layer_params['bias_initializer'],
                                       name=f"{layer_params['name']}_{i}")

        # Finalmente la capa de MaxPooling
        x_enc = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, 
                                             padding='valid',
                                             name=f"MaxPool_Conv_{layer_params['name']}")(x_enc)
        return x_enc
    
    
    def _mlp_layers(input_layer, units, kernel_initializer, 
               bias_initializer, name):
        '''Función auxiliar que modela las capas Dense + batchnorm +
        Activation ReLU'''
        # Aplicando la concatenación de capas
        x_dense = tf.keras.layers.Dense(units=units, 
                                        bias_initializer=bias_initializer,
                                        kernel_initializer=kernel_initializer,
                                        name=f'Dense_{name}')(input_layer)
        x_dense = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_dense)
        x_dense = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_dense)

        return x_dense
    
    
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')
    
    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value)(x_in)
                                             
    # Definición de la CNN
    layer_params_1 = {'filters': 50, 'kernel_size': 100, 'padding': 'same',
                      'kernel_initializer': 'he_normal',
                      'bias_initializer': 'he_normal', 'name': 'cnn_1'}
    x_layer = _cnn_layers(x_masked, n_layers_conv=2, layer_params=layer_params_1)
                                             
    layer_params_2 = {'filters': 30, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal',
                      'bias_initializer': 'he_normal', 'name': 'cnn_2'}
    x_layer = _cnn_layers(x_layer, n_layers_conv=2, layer_params=layer_params_2)
                                             
    layer_params_3 = {'filters': 10, 'kernel_size': 25, 'padding': 'same',
                      'kernel_initializer': 'he_normal',
                      'bias_initializer': 'he_normal', 'name': 'cnn_3'}
    x_layer = _cnn_layers(x_layer, n_layers_conv=3, layer_params=layer_params_3)
                                             
    layer_params_4 = {'filters': 7, 'kernel_size': 13, 'padding': 'same',
                      'kernel_initializer': 'he_normal',
                      'bias_initializer': 'he_normal', 'name': 'cnn_4'}
    x_layer = _cnn_layers(x_layer, n_layers_conv=3, layer_params=layer_params_4)
                      
    
    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_layer = tf.keras.layers.Flatten()(x_layer)                                     
    
    
    # Definición de la red misma
    x_layer = _mlp_layers(x_layer, units=500, kernel_initializer='he_normal', 
                          bias_initializer='he_normal', name='Layer_1')
    x_layer = _mlp_layers(x_layer, units=200, kernel_initializer='he_normal', 
                          bias_initializer='he_normal', name='Layer_2')
    x_layer = _mlp_layers(x_layer, units=100, kernel_initializer='he_normal', 
                          bias_initializer='he_normal', name='Layer_3')
    x_layer = _mlp_layers(x_layer, units=80, kernel_initializer='he_normal', 
                          bias_initializer='he_normal', name='Layer_4')
    x_layer = _mlp_layers(x_layer, units=30, kernel_initializer='he_normal', 
                          bias_initializer='he_normal', name='Layer_5')
    x_layer = _mlp_layers(x_layer, units=10, kernel_initializer='he_normal', 
                          bias_initializer='he_normal', name='Layer_6')
    x_layer = _mlp_layers(x_layer, units=5, kernel_initializer='he_normal', 
                          bias_initializer='he_normal', name='Layer_7')
    
    # Definición de la salida
    if out_layer == 'softmax':
        x_out = tf.keras.layers.Dense(2, activation='softmax', 
                                      kernel_initializer='he_normal', 
                                      bias_initializer='he_normal',
                                      name='softmax_out')(x_layer)
    elif out_layer == 'sigmoid':
        x_out = tf.keras.layers.Dense(1, activation='sigmoid', 
                                      kernel_initializer='he_normal', 
                                      bias_initializer='he_normal',
                                      name='sigmoid_out')(x_layer)
    else:
        raise Exception(f'Opción de parámetro "out_layer"={out_layer} '
                        f'no válido.')
    
    # Definir el modelo
    model = tf.keras.Model(inputs=x_in, outputs=x_out, name='Red_CNN')
    
    return model

                                             
def CNN_network(input_shape, padding_value, out_layer='sigmoid'):
    '''Función que define una red CNN para extraer características y 
    clasificar.
    
    Parameters
    ----------
    padding_value : float
        Valor utilizado para hacer padding en la señal.
    out_layer : {'sigmoid', 'softmax'}, optional
        Función a usar en la capa de salida de la red. Por defecto
        es 'sigmoid'.
    
    Returns
    -------
    model: tensorflow.keras.Model
        Modelo del sistema.
    '''
    def _conv_bn_act_layer(input_layer, filters, kernel_size, padding,
                           kernel_initializer, bias_initializer, name):
        '''Función auxiliar que modela las capas azules conv + batchnorm +
        Activation ReLU para realizar el ENCODING.'''
        # Aplicando la concatenación de capas
        x_conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer=bias_initializer,
                                        padding=padding, 
                                        name=f'Conv_{name}')(input_layer)
        x_conv = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_conv)
        x_conv = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_conv)

        return x_conv
    
    
    def _cnn_layers(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar "n_layers_conv" capas CNN seguida de 
        una capa de Maxpooling.  
        '''
        # Definición de la salida de este bloque
        x_enc = input_layer
        
        # Aplicando "n_layers_conv" capas convolucionales de codificación
        for i in range(n_layers_conv):
            x_enc = _conv_bn_act_layer(x_enc, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'],
                                       bias_initializer=layer_params['bias_initializer'],
                                       name=f"{layer_params['name']}_{i}")

        # Finalmente la capa de MaxPooling
        x_enc = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, 
                                             padding='valid',
                                             name=f"MaxPool_Conv_{layer_params['name']}")(x_enc)
        return x_enc
    
    
    def _mlp_layers(input_layer, units, kernel_initializer, 
               bias_initializer, name):
        '''Función auxiliar que modela las capas Dense + batchnorm +
        Activation ReLU'''
        # Aplicando la concatenación de capas
        x_dense = tf.keras.layers.Dense(units=units, 
                                        bias_initializer=bias_initializer,
                                        kernel_initializer=kernel_initializer,
                                        name=f'Dense_{name}')(input_layer)
        x_dense = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_dense)
        x_dense = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_dense)

        return x_dense
    
    
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')
                                             
    # Definición de la CNN
    layer_params_1 = {'filters': 15, 'kernel_size': 100, 'padding': 'same',
                      'kernel_initializer': 'he_normal',
                      'bias_initializer': 'he_normal', 'name': 'cnn_1'}
    x_layer = _cnn_layers(x_in, n_layers_conv=2, layer_params=layer_params_1)
                                             
    layer_params_2 = {'filters': 15, 'kernel_size': 100, 'padding': 'same',
                      'kernel_initializer': 'he_normal',
                      'bias_initializer': 'he_normal', 'name': 'cnn_2'}
    x_layer = _cnn_layers(x_layer, n_layers_conv=2, layer_params=layer_params_2)
                                             
    layer_params_3 = {'filters': 15, 'kernel_size': 100, 'padding': 'same',
                      'kernel_initializer': 'he_normal',
                      'bias_initializer': 'he_normal', 'name': 'cnn_3'}
    x_layer = _cnn_layers(x_layer, n_layers_conv=2, layer_params=layer_params_3)
                                             
    layer_params_4 = {'filters': 15, 'kernel_size': 100, 'padding': 'same',
                      'kernel_initializer': 'he_normal',
                      'bias_initializer': 'he_normal', 'name': 'cnn_4'}
    x_layer = _cnn_layers(x_layer, n_layers_conv=2, layer_params=layer_params_4)
                      
    layer_params_5 = {'filters': 15, 'kernel_size': 100, 'padding': 'same',
                      'kernel_initializer': 'he_normal',
                      'bias_initializer': 'he_normal', 'name': 'cnn_5'}
    x_layer = _cnn_layers(x_layer, n_layers_conv=2, layer_params=layer_params_5)
                                             
    layer_params_6 = {'filters': 15, 'kernel_size': 100, 'padding': 'same',
                      'kernel_initializer': 'he_normal',
                      'bias_initializer': 'he_normal', 'name': 'cnn_6'}
    x_layer = _cnn_layers(x_layer, n_layers_conv=2, layer_params=layer_params_6)
    
    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_layer = tf.keras.layers.Flatten()(x_layer)                                     
    
    
    # Definición de la red misma
    x_layer = _mlp_layers(x_layer, units=40, kernel_initializer='he_normal', 
                          bias_initializer='he_normal', name='Layer_1')
    
    # Definición de la salida
    if out_layer == 'softmax':
        x_out = tf.keras.layers.Dense(2, activation='softmax', 
                                      kernel_initializer='he_normal', 
                                      bias_initializer='he_normal',
                                      name='softmax_out')(x_layer)
    elif out_layer == 'sigmoid':
        x_out = tf.keras.layers.Dense(1, activation='sigmoid', 
                                      kernel_initializer='he_normal', 
                                      bias_initializer='he_normal',
                                      name='sigmoid_out')(x_layer)
    else:
        raise Exception(f'Opción de parámetro "out_layer"={out_layer} '
                        f'no válido.')
    
    # Definir el modelo
    model = tf.keras.Model(inputs=x_in, outputs=x_out, name='Red_CNN')
    
    return model

                                             
def CNN_network_2(input_shape, padding_value, out_layer='sigmoid'):
    '''Función que define una red CNN para extraer características y 
    clasificar.
    
    Parameters
    ----------
    padding_value : float
        Valor utilizado para hacer padding en la señal.
    out_layer : {'sigmoid', 'softmax'}, optional
        Función a usar en la capa de salida de la red. Por defecto
        es 'sigmoid'.
    
    Returns
    -------
    model: tensorflow.keras.Model
        Modelo del sistema.
    '''
    def _conv_bn_act_layer(input_layer, filters, kernel_size, padding,
                           kernel_initializer, bias_initializer, name):
        '''Función auxiliar que modela las capas azules conv + batchnorm +
        Activation ReLU para realizar el ENCODING.'''
        # Aplicando la concatenación de capas
        x_conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer=bias_initializer,
                                        padding=padding, 
                                        name=f'Conv_{name}')(input_layer)
        x_conv = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_conv)
        x_conv = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_conv)

        return x_conv
    
    
    def _cnn_layers(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar "n_layers_conv" capas CNN seguida de 
        una capa de Maxpooling.  
        '''
        # Definición de la salida de este bloque
        x_enc = input_layer
        
        # Aplicando "n_layers_conv" capas convolucionales de codificación
        for i in range(n_layers_conv):
            x_enc = _conv_bn_act_layer(x_enc, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'],
                                       bias_initializer=layer_params['bias_initializer'],
                                       name=f"{layer_params['name']}_{i}")

        # Finalmente la capa de MaxPooling
        x_enc = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, 
                                             padding='valid',
                                             name=f"MaxPool_Conv_{layer_params['name']}")(x_enc)
        return x_enc
    
    
    def _mlp_layers(input_layer, units, kernel_initializer, 
               bias_initializer, name):
        '''Función auxiliar que modela las capas Dense + batchnorm +
        Activation ReLU'''
        # Aplicando la concatenación de capas
        x_dense = tf.keras.layers.Dense(units=units, 
                                        bias_initializer=bias_initializer,
                                        kernel_initializer=kernel_initializer,
                                        name=f'Dense_{name}')(input_layer)
        x_dense = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_dense)
        x_dense = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_dense)

        return x_dense
    
    
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')
    
    # Definición de la CNN
    layer_params_1 = {'filters': 100, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal',
                      'bias_initializer': 'he_normal', 'name': 'cnn_1'}
    x_layer = _cnn_layers(x_in, n_layers_conv=2, layer_params=layer_params_1)
                                             
    layer_params_2 = {'filters': 100, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal',
                      'bias_initializer': 'he_normal', 'name': 'cnn_2'}
    x_layer = _cnn_layers(x_layer, n_layers_conv=2, layer_params=layer_params_2)
                                             
    layer_params_3 = {'filters': 100, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal',
                      'bias_initializer': 'he_normal', 'name': 'cnn_3'}
    x_layer = _cnn_layers(x_layer, n_layers_conv=2, layer_params=layer_params_3)
                                             
    layer_params_4 = {'filters': 100, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal',
                      'bias_initializer': 'he_normal', 'name': 'cnn_4'}
    x_layer = _cnn_layers(x_layer, n_layers_conv=2, layer_params=layer_params_4)
                      
    layer_params_5 = {'filters': 100, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal',
                      'bias_initializer': 'he_normal', 'name': 'cnn_5'}
    x_layer = _cnn_layers(x_layer, n_layers_conv=2, layer_params=layer_params_5)
                                             
    layer_params_6 = {'filters': 100, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal',
                      'bias_initializer': 'he_normal', 'name': 'cnn_6'}
    x_layer = _cnn_layers(x_layer, n_layers_conv=2, layer_params=layer_params_6)
    
    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_layer = tf.keras.layers.GlobalMaxPool1D()(x_layer)                                     
    
    
    # Definición de la red misma
    x_layer = _mlp_layers(x_layer, units=40, kernel_initializer='he_normal', 
                          bias_initializer='he_normal', name='Layer_1')
    
    # Definición de la salida
    if out_layer == 'softmax':
        x_out = tf.keras.layers.Dense(2, activation='softmax', 
                                      kernel_initializer='he_normal', 
                                      bias_initializer='he_normal',
                                      name='softmax_out')(x_layer)
    elif out_layer == 'sigmoid':
        x_out = tf.keras.layers.Dense(1, activation='sigmoid', 
                                      kernel_initializer='he_normal', 
                                      bias_initializer='he_normal',
                                      name='sigmoid_out')(x_layer)
    else:
        raise Exception(f'Opción de parámetro "out_layer"={out_layer} '
                        f'no válido.')
    
    # Definir el modelo
    model = tf.keras.Model(inputs=x_in, outputs=x_out, name='Red_CNN')
    
    return model

                                       
def CNN_network_2D(input_shape, padding_value, out_layer='softmax'):
    '''Función que define una red CNN para extraer características y 
    clasificar.
    
    Parameters
    ----------
    padding_value : float
        Valor utilizado para hacer padding en la señal.
    out_layer : {'sigmoid', 'softmax'}, optional
        Función a usar en la capa de salida de la red. Por defecto
        es 'softmax'.
    
    Returns
    -------
    model: tensorflow.keras.Model
        Modelo del sistema.
    '''
    def _conv_bn_act_layer(input_layer, filters, kernel_size, padding,
                           kernel_initializer, bias_initializer, name):
        '''Función auxiliar que modela las capas azules conv + batchnorm +
        Activation ReLU para realizar el ENCODING.'''
        # Aplicando la concatenación de capas
        x_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, 
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer=bias_initializer,
                                        padding=padding, 
                                        name=f'Conv_{name}')(input_layer)
        x_conv = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_conv)
        x_conv = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_conv)

        return x_conv
    
    
    def _cnn_layers(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar "n_layers_conv" capas CNN seguida de 
        una capa de Maxpooling.  
        '''
        # Definición de la salida de este bloque
        x_enc = input_layer
        
        # Aplicando "n_layers_conv" capas convolucionales de codificación
        for i in range(n_layers_conv):
            x_enc = _conv_bn_act_layer(x_enc, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'],
                                       bias_initializer=layer_params['bias_initializer'],
                                       name=f"{layer_params['name']}_{i}")

        # Finalmente la capa de MaxPooling
        x_enc = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding='valid',
                                             name=f"MaxPool_Conv_{layer_params['name']}")(x_enc)
        return x_enc
    
    
    def _mlp_layers(input_layer, units, kernel_initializer, 
               bias_initializer, name):
        '''Función auxiliar que modela las capas Dense + batchnorm +
        Activation ReLU'''
        # Aplicando la concatenación de capas
        x_dense = tf.keras.layers.Dense(units=units, 
                                        bias_initializer=bias_initializer,
                                        kernel_initializer=kernel_initializer,
                                        name=f'Dense_{name}')(input_layer)
        x_dense = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_dense)
        x_dense = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_dense)

        return x_dense
    
    
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')
    
    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value)(x_in)
                                             
    # Definición de la CNN
    layer_params_1 = {'filters': 15, 'kernel_size': (5,5), 'padding': 'same',
                      'kernel_initializer': 'he_normal',
                      'bias_initializer': 'he_normal', 'name': 'cnn_1'}
    x_layer = _cnn_layers(x_masked, n_layers_conv=2, layer_params=layer_params_1)
                                             
    layer_params_2 = {'filters': 15, 'kernel_size': (4,4), 'padding': 'same',
                      'kernel_initializer': 'he_normal',
                      'bias_initializer': 'he_normal', 'name': 'cnn_2'}
    x_layer = _cnn_layers(x_layer, n_layers_conv=2, layer_params=layer_params_2)
                                             
    layer_params_3 = {'filters': 15, 'kernel_size': (3,3), 'padding': 'same',
                      'kernel_initializer': 'he_normal',
                      'bias_initializer': 'he_normal', 'name': 'cnn_3'}
    x_layer = _cnn_layers(x_layer, n_layers_conv=2, layer_params=layer_params_3)
    
                                             
    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_layer = tf.keras.layers.Flatten()(x_layer)                                     
    
    
    # Definición de la red misma
    x_layer = _mlp_layers(x_layer, units=40, kernel_initializer='he_normal', 
                          bias_initializer='he_normal', name='Layer_1')
    
    # Definición de la salida
    if out_layer == 'softmax':
        x_out = tf.keras.layers.Dense(2, activation='softmax', 
                                      kernel_initializer='he_normal', 
                                      bias_initializer='he_normal',
                                      name='softmax_out')(x_layer)
    elif out_layer == 'sigmoid':
        x_out = tf.keras.layers.Dense(1, activation='sigmoid', 
                                      kernel_initializer='he_normal', 
                                      bias_initializer='he_normal',
                                      name='sigmoid_out')(x_layer)
    else:
        raise Exception(f'Opción de parámetro "out_layer"={out_layer} '
                        f'no válido.')
    
    # Definir el modelo
    model = tf.keras.Model(inputs=x_in, outputs=x_out, name='Red_CNN')
    
    return model

                                             
def train_test_definition(X_data, Y_data, index_test, patient_groups,
                          patient_register, kfold=10):
    '''Función que permite retornar los conjuntos de entrenamiento
    y testeo en base a la división de la base de datos realizada 
    previamente para hacer una validación cruzada.
    
    Parameters
    ----------
    X_data : ndarray
        Matriz de características.
    Y_data : ndarray
        Etiquetas de la matriz de características.
    index_test : int
        Índice del grupo de testeo en la validación cruzada.
    patient_gropus : dict
        Diccionario que contiene los pacientes que corresponden
        a cada grupo de la validación cruzada.
    patient_register : dict
        Diccionario que contiene las entradas de cada paciente
        en la matriz de características.
    kfold : int, optional
        k de la validación cruzada que se realiza. Por defecto 
        es 10.
    
    Returns
    -------
    X_train : ndarray
        Datos de entrenamiento.
    Y_train : ndarray
        Etiquetas de los datos de entrenamiento.
    X_test : ndarray
        Datos de testeo.
    Y_test : ndarray
        Etiquetas de los datos de testeo.
    '''
    # Definición de los pacientes de testeo 
    test_patients = patient_groups[index_test]
    
    # Y entrenamiento
    train_patients = list()
    for i in range(1, kfold + 1):
        if i != index_test:
            train_patients.extend(patient_groups[i])
            
    # Definición de las entradas de entrenamiento y testeo
    train_indexes = list()
    test_indexes = list()
    
    for i in train_patients:
        train_indexes.extend(patient_register[str(i)])
    
    for i in test_patients:
        test_indexes.extend(patient_register[str(i)])

    # Aplicando los indices sobre los datos
    X_train = X_data[train_indexes]
    Y_train = Y_data[train_indexes]
    X_test  = X_data[test_indexes]
    Y_test  = Y_data[test_indexes]
    
    return X_train, Y_train, X_test, Y_test
    
    
def crossval_results(X_data, Y_data, patient_groups, patient_register, 
                     experiment_type='ML', clean_params=None, 
                     sel_params=None, class_params=None, mlp_params=None,
                     kfold=10):
    '''Función que permite calcular el desempeño del clasificador
    mediante una validación cruzada de los datos.
    
    Parameters
    ----------
    X_data : ndarray
        Matriz de características.
    Y_data : ndarray
        Etiquetas de la matriz de características.
    experiment_type : {'ML', 'NN-MLP' 'CNN'}, optional
        Tipo de sistema a estudiar. 'ML' corresponde a un diseño
        estilo Machine-Learning (Rec. de Patrones). 'NN-MLP'
        corresponde a un diseño que utiliza como salida un 
        clasificador de perceptrones multicapas. 'CNN' es un
        diseño que utiliza una CNN con arquitectura clásica
        (AlexNet o VGG-16) para clasificar cada segmento.
        Por defecto es 'ML'.
    clean_params: dict or None, optional
        Parámetros del proceso de limpieza de características. 
        Si es None se utilizan características por defecto: 
        'tol': 13-5, 'show': True. Por defecto es None.
    sel_params: dict or None, optional
        Parámetros del proceso de selección de características. 
        Si es None se utilizan características por defecto: 
        'n_features': 10, 'show': True. Por defecto es None.
    class_params: dict or None, optional
        Parámetros del proceso de clasificación. Si es None se 
        utilizan características por defecto: 
        'classifier': 'knn', 'k_neigh': 10. Por defecto es None. 
        En caso de usar 'svm', es posible modificar el 'kernel'.
    mlp_params : dict or None, optional
        Parámetros del preoceso de clasificación con MLP. Si es
        None se utilizan las características por defecto:
        {'optimizer': 'Adam', 'loss': 'binary_crossentropy',
         'batch_size': None, 'epochs': 100, 'verbose': 1, 
         'metrics': ['accuracy', tf.keras.metrics.Recall(), 
                      tf.keras.metrics.Precision()],
         'out_layer': 'sigmoid', 'preprocessing': True}
    kfold : int, optional
        k de las repeticiones de la validación cruzada k-fold.
        Por defecto es 10.
    
    Returns
    -------
    confmat_list: list
        Lista de las matrices de confusión para cada iteración.
    accuracy_list : list
        Lista de las accuracys para cada iteración.
    '''
    # Definición de la lista de matrices de confusión
    confmat_list = list()
    
    # Iteraciónes del k-fold cross validation
    for index in range(1, kfold + 1):
        # Definición de la base de datos
        X_train, Y_train, X_test, Y_test = \
            train_test_definition(X_data, Y_data, index_test=index, 
                                  patient_groups=patient_groups,
                                  patient_register=patient_register,
                                  kfold=kfold)

        # Aplicando el clasificador
        if experiment_type == 'ML':
            _, X_test, _, Y_pred = \
                    ML_classification_system(X_train, Y_train, X_test, Y_test, 
                                             clean_params=clean_params, 
                                             sel_params=sel_params, 
                                             class_params=class_params)
            
        elif experiment_type == 'NN-MLP':
            _, X_test, _, Y_pred = \
                NN_MLP_classification_system(X_train, Y_train, X_test, Y_test, 
                                             clean_params=clean_params, 
                                             sel_params=sel_params, 
                                             mlp_params=mlp_params)
            
            # Modificar el Y_pred
            Y_pred = np.where(Y_pred < 0.5, 0, 1)[:, 0]

        elif experiment_type == 'CNN':
            pass
    
        else:
            raise Exception('Opción no válida para "experiment_type".')
    
        # Obteniendo la matriz de confusión
        conf_mat = confusion_matrix(Y_test, Y_pred)
        
        # Agregando a la lista
        confmat_list.append(conf_mat)
    
    # Cálculo de los resultados finales
    accuracy_list = list()
    
    for cmat in confmat_list:
        accuracy_i = np.sum(np.diag(cmat)) / np.sum(cmat)
        accuracy_list.append(accuracy_i)
        
    print(f'Accuracy {kfold}-fold CV: {np.mean(accuracy_list)} +- '
          f'{np.std(accuracy_list)}')
    
    return confmat_list, accuracy_list
                                             
                                             
def events_eval_results(X_train, Y_train, X_test, Y_test, experiment_type='ML', 
                        clean_params=None, sel_params=None, class_params=None, 
                        mlp_params=None):
    '''Función que permite calcular el desempeño del clasificador
    mediante una validación cruzada de los datos.
    
    Parameters
    ----------
    X_train : ndarray
        Datos de entrenamiento.
    Y_train : ndarray
        Etiquetas de los datos de entrenamiento.
    X_test : ndarray
        Datos de testeo.
    Y_test : ndarray
        Etiquetas de los datos de testeo.
    experiment_type : {'ML', 'NN-MLP' 'CNN'}, optional
        Tipo de sistema a estudiar. 'ML' corresponde a un diseño
        estilo Machine-Learning (Rec. de Patrones). 'NN-MLP'
        corresponde a un diseño que utiliza como salida un 
        clasificador de perceptrones multicapas. 'CNN' es un
        diseño que utiliza una CNN con arquitectura clásica
        (AlexNet o VGG-16) para clasificar cada segmento.
        Por defecto es 'ML'.
    clean_params: dict or None, optional
        Parámetros del proceso de limpieza de características. 
        Si es None se utilizan características por defecto: 
        'tol': 13-5, 'show': True. Por defecto es None.
    sel_params: dict or None, optional
        Parámetros del proceso de selección de características. 
        Si es None se utilizan características por defecto: 
        'n_features': 10, 'show': True. Por defecto es None.
    class_params: dict or None, optional
        Parámetros del proceso de clasificación. Si es None se 
        utilizan características por defecto: 
        'classifier': 'knn', 'k_neigh': 10. Por defecto es None. 
        En caso de usar 'svm', es posible modificar el 'kernel'.
    mlp_params : dict or None, optional
        Parámetros del preoceso de clasificación con MLP. Si es
        None se utilizan las características por defecto:
        {'optimizer': 'Adam', 'loss': 'binary_crossentropy',
         'batch_size': None, 'epochs': 100, 'verbose': 1, 
         'metrics': ['accuracy', tf.keras.metrics.Recall(), 
                      tf.keras.metrics.Precision()],
         'out_layer': 'sigmoid', 'preprocessing': True}.
    
    Returns
    -------
    confmat_list: list
        Lista de las matrices de confusión para cada iteración.
    accuracy_list : list
        Lista de las accuracys para cada iteración.
    '''    
    # Aplicando el clasificador
    if experiment_type == 'ML':
        classifier, X_test, params_out, Y_pred = \
                ML_classification_system(X_train, Y_train, X_test, Y_test, 
                                         clean_params=clean_params, 
                                         sel_params=sel_params, 
                                         class_params=class_params)

    elif experiment_type == 'NN-MLP':
        classifier, X_test, params_out, Y_pred = \
            NN_MLP_classification_system(X_train, Y_train, X_test, Y_test, 
                                         clean_params=clean_params, 
                                         sel_params=sel_params, 
                                         mlp_params=mlp_params)

        # Modificar el Y_pred
        Y_pred = np.where(Y_pred < 0.5, 0, 1)[:, 0]

    elif experiment_type == 'CNN':
        pass

    else:
        raise Exception('Opción no válida para "experiment_type".')

    # Obteniendo la matriz de confusión
    conf_mat = confusion_matrix(Y_test, Y_pred)
    accuracy = np.sum(np.diag(conf_mat)) / np.sum(conf_mat)
        
    print(f'Accuracy: {accuracy}')
    
    return conf_mat, accuracy, classifier, params_out


def oncycle_eval_results(X_train, Y_train, X_test, Y_test, experiment_type='ML', 
                         clean_params=None, sel_params=None, class_params=None, 
                         mlp_params=None):
    '''Función que permite calcular el desempeño del clasificador
    mediante una validación cruzada de los datos.
    
    Parameters
    ----------
    X_train : ndarray
        Datos de entrenamiento.
    Y_train : ndarray
        Etiquetas de los datos de entrenamiento.
    X_test : ndarray
        Datos de testeo.
    Y_test : ndarray
        Etiquetas de los datos de testeo.
    experiment_type : {'ML', 'NN-MLP' 'CNN'}, optional
        Tipo de sistema a estudiar. 'ML' corresponde a un diseño
        estilo Machine-Learning (Rec. de Patrones). 'NN-MLP'
        corresponde a un diseño que utiliza como salida un 
        clasificador de perceptrones multicapas. 'CNN' es un
        diseño que utiliza una CNN con arquitectura clásica
        (AlexNet o VGG-16) para clasificar cada segmento.
        Por defecto es 'ML'.
    clean_params: dict or None, optional
        Parámetros del proceso de limpieza de características. 
        Si es None se utilizan características por defecto: 
        'tol': 13-5, 'show': True. Por defecto es None.
    sel_params: dict or None, optional
        Parámetros del proceso de selección de características. 
        Si es None se utilizan características por defecto: 
        'n_features': 10, 'show': True. Por defecto es None.
    class_params: dict or None, optional
        Parámetros del proceso de clasificación. Si es None se 
        utilizan características por defecto: 
        'classifier': 'knn', 'k_neigh': 10. Por defecto es None. 
        En caso de usar 'svm', es posible modificar el 'kernel'.
    mlp_params : dict or None, optional
        Parámetros del preoceso de clasificación con MLP. Si es
        None se utilizan las características por defecto:
        {'optimizer': 'Adam', 'loss': 'binary_crossentropy',
         'batch_size': None, 'epochs': 100, 'verbose': 1, 
         'metrics': ['accuracy', tf.keras.metrics.Recall(), 
                      tf.keras.metrics.Precision()],
         'out_layer': 'sigmoid', 'preprocessing': True}.
    
    Returns
    -------
    confmat_list: list
        Lista de las matrices de confusión para cada iteración.
    accuracy_list : list
        Lista de las accuracys para cada iteración.
    '''    
    # Aplicando el clasificador
    if experiment_type == 'ML':
        classifier, X_test, params_out, Y_pred = \
                ML_classification_system(X_train, Y_train, X_test, Y_test, 
                                         clean_params=clean_params, 
                                         sel_params=sel_params, 
                                         class_params=class_params)

    elif experiment_type == 'NN-MLP':
        classifier, X_test, params_out, Y_pred = \
            NN_MLP_classification_system(X_train, Y_train, X_test, Y_test, 
                                         clean_params=clean_params, 
                                         sel_params=sel_params, 
                                         mlp_params=mlp_params)

        # Modificar el Y_pred
        Y_pred = np.where(Y_pred < 0.5, 0, 1)[:, 0]

    else:
        raise Exception('Opción no válida para "experiment_type".')

    # Obteniendo la matriz de confusión
    conf_mat = confusion_matrix(Y_test, Y_pred)
    accuracy = np.sum(np.diag(conf_mat)) / np.sum(conf_mat)
        
    print(f'Accuracy: {accuracy}')
    
    return conf_mat, accuracy, classifier, params_out


def conditioning_features_train(X_train, Y_train, clean_params=None, 
                                sel_params=None):
    '''Función que permite obtener los parámetros que se utilizan
    para limpiar, normalizar y seleccionar características en base
    a los datos de entrenamiento.
    
    Parameters
    ----------
    X_train : ndarray
        Datos de entrenamiento.
    Y_train : ndarray
        Etiquetas de los datos de entrenamiento.
    clean_params: dict or None, optional
        Parámetros del proceso de limpieza de características. 
        Si es None se utilizan características por defecto: 
        {'tol': 13-5, 'show': True}. Por defecto es None.
    sel_params: dict or None, optional
        Parámetros del proceso de selección de características. 
        Si es None se utilizan características por defecto: 
        {'n_features': 10, 'show': True}. Por defecto es None.
    
    Returns
    -------
    params_out : dict
        Diccionario que contiene los parámetros de interés para la
        limpieza, normalización y selección de las características.
    '''
    # Parámetros por defecto
    if clean_params is None:
        clean_params = {'tol': 1e-5, 'show': True}
    
    if sel_params is None:
        sel_params = {'n_features': 10, 'show': True}

        
    #### Pipeline de la etapa de clasificación ####
    
    ## 1) Limpieza de las características
    s_clean = pybalu_clean(X_train, tol=clean_params['tol'], 
                           show=clean_params['show'])
    
    # Aplicando la limpieza
    X_train = X_train[:, s_clean]
    
    
    ## 2) Normalización de los datos
    X_train, a_norm, b_norm = normalize(X_train)
    
    
    ## 3) Selección de características
    s_sfs = sfs(X_train, Y_train, show=sel_params['show'],
                n_features=sel_params['n_features'])
    
    # Aplicando la selección
    X_train = X_train[:, s_sfs]
    
    # Definición del diccionario de parámetros
    params_out = {'a_norm': a_norm.tolist(), 'b_norm': b_norm.tolist(), 
                  's_clean': s_clean.tolist(), 's_sfs': s_sfs.tolist()}

    return params_out


def conditioning_features_train_opt(X_train, Y_train, clean_params=None, 
                                    sel_params=None):
    '''Función que permite obtener los parámetros que se utilizan
    para limpiar, normalizar y seleccionar características en base
    a los datos de entrenamiento.
    
    Parameters
    ----------
    X_train : ndarray
        Datos de entrenamiento.
    Y_train : ndarray
        Etiquetas de los datos de entrenamiento.
    clean_params: dict or None, optional
        Parámetros del proceso de limpieza de características. 
        Si es None se utilizan características por defecto: 
        {'tol': 13-5, 'show': True}. Por defecto es None.
    sel_params: dict or None, optional
        Parámetros del proceso de selección de características. 
        Si es None se utilizan características por defecto: 
        {'n_features': 10, 'show': True}. Por defecto es None.
    
    Returns
    -------
    params_out : dict
        Diccionario que contiene los parámetros de interés para la
        limpieza, normalización y selección de las características.
    '''
    # Parámetros por defecto
    if clean_params is None:
        clean_params = {'tol': 1e-5, 'show': True}
    
    if sel_params is None:
        sel_params = {'n_features': 10, 'show': True}

        
    #### Pipeline de la etapa de clasificación ####
    
    ## 1) Limpieza de las características
    s_clean = pybalu_clean(X_train, tol=clean_params['tol'], 
                           show=clean_params['show'])
    
    # Aplicando la limpieza
    X_train = X_train[:, s_clean]
    
    
    ## 2) Normalización de los datos
    X_train, a_norm, b_norm = normalize(X_train)
    
    
    ## 3) Selección de características
    s_sfs, j_values = pybalu_sfs(X_train, Y_train, show=sel_params['show'],
                                 n_features=sel_params['n_features']+1)
    
    # Aplicando la selección
    X_train = X_train[:, s_sfs]
    
    # Definición del diccionario de parámetros
    params_out = {'a_norm': a_norm.tolist(), 'b_norm': b_norm.tolist(), 
                  's_clean': s_clean.tolist(), 's_sfs': s_sfs.tolist(),
                  'j_values': j_values}

    return params_out




# Módulo de testeo
if __name__ == '__main__':
    # Función a aplicar
    func_to = 'conditioning_features_train'
    
    if func_to == 'conditioning_features_train':
        from respiratory_sound_classification.respiratory_sound_management import \
            get_model_data_idxs, get_training_weights, train_test_filebased, get_ML_data
        
        
        db_folder = 'unpreprocessed_signals'
        db_original = 'C:/Users/Chris/Desktop/Scripts_Magister/'\
                      'Respiratory_Sound_Database/audio_and_txt_files'
        
        # Parámetros de los espectrogramas generales
        N = 1024
        noverlap = int(0.9 * N)
        spec_params = {'N': N, 'noverlap': noverlap, 'window': 'hann', 
                    'padding': 0, 'repeat': 0}

        # Parámetros MFCC
        mfcc_params = {'n_mfcc': 50, 'n_filters': 50, 'spec_params': spec_params,
                    'freq_lim': 2000, 'norm_filters': True, 'power': 2}
        lfcc_params = {'n_mfcc': 50, 'n_filters': 50, 'spec_params': spec_params,
                    'freq_lim': 2000, 'norm_filters': True, 'power': 2}
        energy_params = {'spec_params': spec_params, 'fmin': 0, 'fmax': 1000, 
                        'fband': 20}

        # Parámetros generales
        clean_params = {'tol': 1e-5, 'show': True}
        sel_params = {'n_features': 60, 'show': True}
        
        # Definición de la carpeta a buscar los archivos
        train_test_folder = '/'.join(db_original.split('/')[:-1])
        file_traintest = f'{train_test_folder}/ICBHI_challenge_train_test.txt'

        
        # Definición de las listas de entrenamiento y testeo
        train_list, test_list = \
                train_test_filebased(file_traintest, db_original, 
                                     db_folder)

        # Obteniendo los datos de entrenamiento
        X_train, Y_wheeze_tr, Y_crackl_tr = \
                get_ML_data(train_list, spec_params=spec_params, 
                            mfcc_params=mfcc_params, 
                            lfcc_params=lfcc_params, 
                            energy_params=energy_params)
                

        # Obtención de los parámetros para el sistema crackle
        params_crackl = \
            conditioning_features_train(X_train, Y_crackl_tr, 
                                        clean_params=clean_params, 
                                        sel_params=sel_params)
            
        # Obtención de los parámetros para el sistema wheeze
        params_wheeze = \
            conditioning_features_train(X_train, Y_wheeze_tr, 
                                        clean_params=clean_params, 
                                        sel_params=sel_params)
            
        # Registrando
        with open('crackle_features_params.txt', 'w', encoding='utf8') as file:
            file.write(f'{params_crackl}')
            
        with open('wheeze_features_params.txt', 'w', encoding='utf8') as file:
            file.write(f'{params_wheeze}')
