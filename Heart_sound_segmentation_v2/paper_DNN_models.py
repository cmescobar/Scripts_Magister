import tensorflow as tf


#######     1) Comparación de arquitecturas     ####### 

def cnn_dnn_1_1(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    En este caso se hace uso de una entrada común para todas las envolventes.
    
    Envolventes usadas:
    - Filtros homomórficos
    - Envolventes de Hilbert
    - DWT
    - Energy envolve
    
    Salida de 3 etiquetas:
    - S1
    - S2
    - None
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''
    # Definición de la entrada 
    x_in = tf.keras.Input(shape=input_shape, dtype='float32',
                          name='Input_channel')
    
    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value,
                                       input_shape=(input_shape[0], 1),
                                       name='Masking')(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     kernel_initializer='he_normal', 
                                     name='Conv1')(x_masked)
    x_conv1 = tf.keras.layers.BatchNormalization(name='Batchnorm_Conv1')(x_conv1)
    x_conv1 = tf.keras.layers.Activation('relu', name='Act_Conv1')(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     kernel_initializer='he_normal', 
                                     name='Conv2')(x_conv1)
    x_conv2 = tf.keras.layers.BatchNormalization(name='Batchnorm_Conv2')(x_conv2)
    x_conv2 = tf.keras.layers.Activation('relu', name='Act_Conv2')(x_conv2)

    ### Tercer conv ###
    x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     kernel_initializer='he_normal', 
                                     name='Conv3')(x_conv2)
    x_conv3 = tf.keras.layers.BatchNormalization(name='Batchnorm_Conv3')(x_conv3)
    x_conv3 = tf.keras.layers.Activation('relu', name='Act_Conv3')(x_conv3)

    ### Cuarto conv ###
    x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     kernel_initializer='he_normal', 
                                     name='Conv4')(x_conv3)
    x_conv4 = tf.keras.layers.BatchNormalization(name='Batchnorm_Conv4')(x_conv4)
    x_conv4 = tf.keras.layers.Activation('relu', name='Act_Conv4')(x_conv4)
    
    
    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv4)
    
    
    # Definición de las capas fully connected
    ### Fully Connected 1 ###
    x_full1 = tf.keras.layers.Dense(units=50, activation='relu',
                                    kernel_initializer='he_normal',
                                    name='FC1')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

    ### Fully Connected 2 ###
    x_full2 = tf.keras.layers.Dense(units=50, activation='relu',
                                    kernel_initializer='he_normal',
                                    name='FC2')(x_full1)
    x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

    ### Fully Connected 3 ###
    x_full3 = tf.keras.layers.Dense(units=50, activation='relu',
                                    kernel_initializer='he_normal',
                                    name='FC3')(x_full2)
    x_full3 = tf.keras.layers.BatchNormalization()(x_full3)
    
    
    # Definición de la última capa (activación)
    x_out = tf.keras.layers.Dense(units=3, activation='softmax', 
                                  kernel_initializer='he_normal',
                                  name='Heart_out')(x_full3)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in, x_out)
    
    return model


def cnn_dnn_1_2(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    En este caso se hace uso de una entrada distinta para cada envolvente.
    
    Envolventes usadas:
    - Filtros homomórficos
    - Envolventes de Hilbert
    - DWT
    - Energy envolve
    
    Salida de 3 etiquetas:
    - S1
    - S2
    - None
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''    
    # Definición de una lista auxiliar de las entradas y de capas FC de salida
    x_in_list = list()
    full_list = list()
    
    # Aplicando una separación para cada canal
    for i in range(input_shape[1]):        
        # Definición de la entrada
        x_in = tf.keras.Input(shape=(input_shape[0], 1), dtype='float32',
                              name=f'Input_ch{i+1}')
        
        # Agregando a la lista
        x_in_list.append(x_in)

        # Definición de la capa de máscara
        x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                           input_shape=(input_shape[0], 1),
                                           name=f'Masking_ch{i+1}')(x_in)
    
        # Definición de las capas convolucionales
        ### Primer conv ###
        x_conv1 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                         kernel_initializer='he_normal', 
                                         name=f'Conv1_ch{i+1}')(x_masked)
        x_conv1 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv1_ch{i+1}')(x_conv1)
        x_conv1 = tf.keras.layers.Activation('relu', name=f'Act_Conv1_ch{i+1}')(x_conv1)

        ### Segundo conv ###
        x_conv2 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                         kernel_initializer='he_normal', 
                                         name=f'Conv2_ch{i+1}')(x_conv1)
        x_conv2 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv2_ch{i+1}')(x_conv2)
        x_conv2 = tf.keras.layers.Activation('relu', name=f'Act_Conv2_ch{i+1}')(x_conv2)

        ### Tercer conv ###
        x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                         kernel_initializer='he_normal', 
                                         name=f'Conv3_ch{i+1}')(x_conv2)
        x_conv3 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv3_ch{i+1}')(x_conv3)
        x_conv3 = tf.keras.layers.Activation('relu', name=f'Act_Conv3_ch{i+1}')(x_conv3)

        ### Cuarto conv ###
        x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                         kernel_initializer='he_normal', 
                                         name=f'Conv4_ch{i+1}')(x_conv3)
        x_conv4 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv4_ch{i+1}')(x_conv4)
        x_conv4 = tf.keras.layers.Activation('relu', name=f'Act_Conv4_ch{i+1}')(x_conv4)
        
        
        # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
        x_flat = tf.keras.layers.Flatten()(x_conv4)
        
        
        # Definición de las capas fully connected
        ### Fully Connected 1 ###
        x_full1 = tf.keras.layers.Dense(units=50, activation='relu',
                                        kernel_initializer='he_normal',
                                        name=f'FC1_ch{i+1}')(x_flat)
        x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

        ### Fully Connected 2 ###
        x_full2 = tf.keras.layers.Dense(units=50, activation='relu',
                                        kernel_initializer='he_normal',
                                        name=f'FC2_ch{i+1}')(x_full1)
        x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

        ### Fully Connected 3 ###
        x_full3 = tf.keras.layers.Dense(units=50, activation='relu',
                                        kernel_initializer='he_normal',
                                        name=f'FC3_ch{i+1}')(x_full2)
        x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

                
        # Agregando a las listas
        full_list.append(x_full3)
    
    # Definición de una capa de aplanamiento
    fully_concat = tf.keras.layers.concatenate(full_list, name='fully_concat')
    
    
    # Definición de la última capa (activación)
    x_out = tf.keras.layers.Dense(units=3, activation='softmax', 
                                  kernel_initializer='he_normal',
                                  name='Heart_out')(fully_concat)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in_list, x_out)
    
    return model


def cnn_dnn_1_3(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    En este caso se hace uso de una entrada común para todas las envolventes. 
    Es lo mismo que cnn_dnn_1_1, pero utilizando capas de pooling luego de la 2 y la 4.
    
    Envolventes usadas:
    - Filtros homomórficos
    - Envolventes de Hilbert
    - DWT
    - Energy envolve
    
    Salida de 3 etiquetas:
    - S1
    - S2
    - None
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''
    # Definición de la entrada 
    x_in = tf.keras.Input(shape=input_shape, dtype='float32',
                          name='Input_channel')
    
    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value,
                                       input_shape=(input_shape[0], 1),
                                       name='Masking')(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     kernel_initializer='he_normal', 
                                     name='Conv1')(x_masked)
    x_conv1 = tf.keras.layers.BatchNormalization(name='Batchnorm_Conv1')(x_conv1)
    x_conv1 = tf.keras.layers.Activation('relu', name='Act_Conv1')(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     kernel_initializer='he_normal', 
                                     name='Conv2')(x_conv1)
    x_conv2 = tf.keras.layers.BatchNormalization(name='Batchnorm_Conv2')(x_conv2)
    x_conv2 = tf.keras.layers.Activation('relu', name='Act_Conv2')(x_conv2)
    x_conv2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, name='MaxPool_1')(x_conv2)

    ### Tercer conv ###
    x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     kernel_initializer='he_normal', 
                                     name='Conv3')(x_conv2)
    x_conv3 = tf.keras.layers.BatchNormalization(name='Batchnorm_Conv3')(x_conv3)
    x_conv3 = tf.keras.layers.Activation('relu', name='Act_Conv3')(x_conv3)

    ### Cuarto conv ###
    x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     kernel_initializer='he_normal', 
                                     name='Conv4')(x_conv3)
    x_conv4 = tf.keras.layers.BatchNormalization(name='Batchnorm_Conv4')(x_conv4)
    x_conv4 = tf.keras.layers.Activation('relu', name='Act_Conv4')(x_conv4)
    x_conv4 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, name='MaxPool_2')(x_conv4)
    
    
    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv4)
    
    
    # Definición de las capas fully connected
    ### Fully Connected 1 ###
    x_full1 = tf.keras.layers.Dense(units=50, activation='relu',
                                    kernel_initializer='he_normal',
                                    name='FC1')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

    ### Fully Connected 2 ###
    x_full2 = tf.keras.layers.Dense(units=50, activation='relu',
                                    kernel_initializer='he_normal',
                                    name='FC2')(x_full1)
    x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

    ### Fully Connected 3 ###
    x_full3 = tf.keras.layers.Dense(units=50, activation='relu',
                                    kernel_initializer='he_normal',
                                    name='FC3')(x_full2)
    x_full3 = tf.keras.layers.BatchNormalization()(x_full3)
    
    
    # Definición de la última capa (activación)
    x_out = tf.keras.layers.Dense(units=3, activation='softmax', 
                                  kernel_initializer='he_normal',
                                  name='Heart_out')(x_full3)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in, x_out)
    
    return model


def cnn_dnn_1_4(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    En este caso se hace uso de una entrada distinta para cada envolvente.
    Es lo mismo que cnn_dnn_1_2, pero utilizando capas de pooling luego de la 2 y 
    la 4 para cada canal.
    
    Envolventes usadas:
    - Filtros homomórficos
    - Envolventes de Hilbert
    - DWT
    - Energy envolve
    
    Salida de 3 etiquetas:
    - S1
    - S2
    - None
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''    
    # Definición de una lista auxiliar de las entradas y de capas FC de salida
    x_in_list = list()
    full_list = list()
    
    # Aplicando una separación para cada canal
    for i in range(input_shape[1]):        
        # Definición de la entrada
        x_in = tf.keras.Input(shape=(input_shape[0], 1), dtype='float32',
                              name=f'Input_ch{i+1}')
        
        # Agregando a la lista
        x_in_list.append(x_in)

        # Definición de la capa de máscara
        x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                           input_shape=(input_shape[0], 1),
                                           name=f'Masking_ch{i+1}')(x_in)
    
        # Definición de las capas convolucionales
        ### Primer conv ###
        x_conv1 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                         kernel_initializer='he_normal', 
                                         name=f'Conv1_ch{i+1}')(x_masked)
        x_conv1 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv1_ch{i+1}')(x_conv1)
        x_conv1 = tf.keras.layers.Activation('relu', name=f'Act_Conv1_ch{i+1}')(x_conv1)

        ### Segundo conv ###
        x_conv2 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                         kernel_initializer='he_normal', 
                                         name=f'Conv2_ch{i+1}')(x_conv1)
        x_conv2 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv2_ch{i+1}')(x_conv2)
        x_conv2 = tf.keras.layers.Activation('relu', name=f'Act_Conv2_ch{i+1}')(x_conv2)
        x_conv2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, name=f'MaxPool_1_ch{i+1}')(x_conv2)

        ### Tercer conv ###
        x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                         kernel_initializer='he_normal', 
                                         name=f'Conv3_ch{i+1}')(x_conv2)
        x_conv3 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv3_ch{i+1}')(x_conv3)
        x_conv3 = tf.keras.layers.Activation('relu', name=f'Act_Conv3_ch{i+1}')(x_conv3)

        ### Cuarto conv ###
        x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                         kernel_initializer='he_normal', 
                                         name=f'Conv4_ch{i+1}')(x_conv3)
        x_conv4 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv4_ch{i+1}')(x_conv4)
        x_conv4 = tf.keras.layers.Activation('relu', name=f'Act_Conv4_ch{i+1}')(x_conv4)
        x_conv4 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, name=f'MaxPool_2_ch{i+1}')(x_conv4)
        
        
        # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
        x_flat = tf.keras.layers.Flatten()(x_conv4)
        
        
        # Definición de las capas fully connected
        ### Fully Connected 1 ###
        x_full1 = tf.keras.layers.Dense(units=50, activation='relu',
                                        kernel_initializer='he_normal',
                                        name=f'FC1_ch{i+1}')(x_flat)
        x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

        ### Fully Connected 2 ###
        x_full2 = tf.keras.layers.Dense(units=50, activation='relu',
                                        kernel_initializer='he_normal',
                                        name=f'FC2_ch{i+1}')(x_full1)
        x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

        ### Fully Connected 3 ###
        x_full3 = tf.keras.layers.Dense(units=50, activation='relu',
                                        kernel_initializer='he_normal',
                                        name=f'FC3_ch{i+1}')(x_full2)
        x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

                
        # Agregando a las listas
        full_list.append(x_full3)
    
    # Definición de una capa de aplanamiento
    fully_concat = tf.keras.layers.concatenate(full_list, name='fully_concat')
    
    
    # Definición de la última capa (activación)
    x_out = tf.keras.layers.Dense(units=3, activation='softmax', 
                                  kernel_initializer='he_normal',
                                  name='Heart_out')(fully_concat)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in_list, x_out)
    
    return model


def segnet_based_1_1(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Se utiliza el mismo canal para todas las envolventes.
    
    Envolventes usadas:
    - Filtros homomórficos
    - Envolventes de Hilbert
    - DWT
    - Energy envolve
    
    Salida de 3 etiquetas:
    - S1
    - S2
    - None
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    def _conv_bn_act_layer(input_layer, filters, kernel_size, padding,
                          kernel_initializer, name):
        '''Función auxiliar que modela las capas azules conv + batchnorm +
        Activation ReLU para realizar el ENCODING.'''
        # Aplicando la concatenación de capas
        x_conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                        kernel_initializer=kernel_initializer,
                                        padding=padding, 
                                        name=f'Conv_{name}')(input_layer)
        x_conv = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_conv)
        x_conv = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_conv)

        return x_conv
    
    
    def _encoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar "n_layers_conv" capas CNN seguida de 
        una capa de Maxpooling, tal como se puede ver en la figura 2 de [1].  
        '''
        # Definición de la salida de este bloque
        x_enc = input_layer
        
        # Aplicando "n_layers_conv" capas convolucionales de codificación
        for i in range(n_layers_conv):
            x_enc = _conv_bn_act_layer(x_enc, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'], 
                                       name=f"{layer_params['name']}_{i}")

        # Finalmente la capa de MaxPooling
        x_enc = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, 
                                             padding='valid',
                                             name=f"MaxPool_Conv_{layer_params['name']}")(x_enc)
        return x_enc
    
    
    def _decoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar una capa de upsampling seguido de 
        "n_layers_conv" capas CNN, tal como se puede ver en la figura 2 de [1].  
        '''
        # Capa de upsampling
        x_dec = tf.keras.layers.UpSampling1D(size=2, name=f"Upsampling_"\
                                                          f"{layer_params['name']}")(input_layer)
        
        # Aplicando "n_layers_conv" capas convolucionales de decodificación
        for i in range(n_layers_conv):
            x_dec = _conv_bn_act_layer(x_dec, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'], 
                                       name=f"{layer_params['name']}_{i}")

        return x_dec
    
    
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value)(x_in)

    ############        Definición de las capas convolucionales        ############
    
    ### Encoding ###
    
    # Primera capa de encoding
    layer_params_1 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'enc1'}
    x_enc1 = _encoding_layer(x_masked, n_layers_conv=2, layer_params=layer_params_1)
    
    # Segunda capa de encoding
    layer_params_2 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'enc2'}
    x_enc2 = _encoding_layer(x_enc1, n_layers_conv=2, layer_params=layer_params_2)
    
    # Tercera capa de encoding
    layer_params_3 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'enc3'}
    x_enc3 = _encoding_layer(x_enc2, n_layers_conv=3, layer_params=layer_params_3)
    
    # Cuarta capa de encoding
    layer_params_4 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'enc4'}
    x_enc4 = _encoding_layer(x_enc3, n_layers_conv=3, layer_params=layer_params_4)
    
    
    ### Decoding ###
    
    # Cuarta capa de salida del decoding
    layer_params_4 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'dec4'}
    x_dec4 = _decoding_layer(x_enc4, n_layers_conv=3, layer_params=layer_params_4)
    
    # Tercera capa de salida del decoding
    layer_params_3 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'dec3'}
    x_dec3 = _decoding_layer(x_dec4, n_layers_conv=3, layer_params=layer_params_3)
    
    # Segunda capa de salida del decoding
    layer_params_2 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'dec2'}
    x_dec2 = _decoding_layer(x_dec3, n_layers_conv=2, layer_params=layer_params_2)
    
    # Primera capa de salida del decoding
    layer_params_1 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'dec1'}
    x_dec1 = _decoding_layer(x_dec2, n_layers_conv=2, layer_params=layer_params_1)
                                       
    
    # Aplicando reshape
    # x_reshaped = tf.keras.layers.Reshape((input_shape[0], input_shape[1] * 2))(x_dec1)
    
    # Definición de la capa de salida
    x_out = tf.keras.layers.Dense(3, activation='softmax', kernel_initializer='he_normal',
                                  name='softmax_out')(x_dec1)
    
    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=x_out, name=name)
    
    return model


def segnet_based_1_2(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Se utiliza un canal distinto para cada envolvente
    
    Envolventes usadas:
    - Filtros homomórficos
    - Envolventes de Hilbert
    - DWT
    - Energy envolve
    
    Salida de 3 etiquetas:
    - S1
    - S2
    - None
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    def _conv_bn_act_layer(input_layer, filters, kernel_size, padding,
                          kernel_initializer, name):
        '''Función auxiliar que modela las capas azules conv + batchnorm +
        Activation ReLU para realizar el ENCODING.'''
        # Aplicando la concatenación de capas
        x_conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                        kernel_initializer=kernel_initializer,
                                        padding=padding, 
                                        name=f'Conv_{name}')(input_layer)
        x_conv = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_conv)
        x_conv = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_conv)

        return x_conv
    
    
    def _encoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar "n_layers_conv" capas CNN seguida de 
        una capa de Maxpooling, tal como se puede ver en la figura 2 de [1].  
        '''
        # Definición de la salida de este bloque
        x_enc = input_layer
        
        # Aplicando "n_layers_conv" capas convolucionales de codificación
        for i in range(n_layers_conv):
            x_enc = _conv_bn_act_layer(x_enc, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'], 
                                       name=f"{layer_params['name']}_{i}")

        # Finalmente la capa de MaxPooling
        x_enc = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, 
                                             padding='valid',
                                             name=f"MaxPool_Conv_{layer_params['name']}")(x_enc)
        return x_enc
    
    
    def _decoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar una capa de upsampling seguido de 
        "n_layers_conv" capas CNN, tal como se puede ver en la figura 2 de [1].  
        '''
        # Capa de upsampling
        x_dec = tf.keras.layers.UpSampling1D(size=2, name=f"Upsampling_"\
                                                          f"{layer_params['name']}")(input_layer)
        
        # Aplicando "n_layers_conv" capas convolucionales de decodificación
        for i in range(n_layers_conv):
            x_dec = _conv_bn_act_layer(x_dec, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'], 
                                       name=f"{layer_params['name']}_{i}")

        return x_dec
    
    # Definición de una lista auxiliar de entradas y salidas
    x_in_list = list()
    list_decs = list()
    
    for i in range(input_shape[1]):
        # Definición de la entrada
        x_in = tf.keras.Input(shape=(input_shape[0], 1), dtype='float32',
                              name=f'Input_ch{i+1}')

        # Agregando a la lista
        x_in_list.append(x_in)
        
        # Definición de la capa de máscara
        x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                           name=f'Masking_ch{i+1}')(x_in)


        ############        Definición de las capas convolucionales        ############
        
        ### Encoding ###
        
        # Primera capa de encoding
        layer_params_1 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'enc1_ch{i+1}'}
        x_enc1 = _encoding_layer(x_masked, n_layers_conv=2, layer_params=layer_params_1)
        
        # Segunda capa de encoding
        layer_params_2 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'enc2_ch{i+1}'}
        x_enc2 = _encoding_layer(x_enc1, n_layers_conv=2, layer_params=layer_params_2)
        
        # Tercera capa de encoding
        layer_params_3 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'enc3_ch{i+1}'}
        x_enc3 = _encoding_layer(x_enc2, n_layers_conv=3, layer_params=layer_params_3)
        
        # Cuarta capa de encoding
        layer_params_4 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'enc4_ch{i+1}'}
        x_enc4 = _encoding_layer(x_enc3, n_layers_conv=3, layer_params=layer_params_4)
        
        
        ### Decoding ###
        
        # Cuarta capa de salida del decoding
        layer_params_4 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'dec4_ch{i+1}'}
        x_dec4 = _decoding_layer(x_enc4, n_layers_conv=3, layer_params=layer_params_4)
        
        # Tercera capa de salida del decoding
        layer_params_3 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'dec3_ch{i+1}'}
        x_dec3 = _decoding_layer(x_dec4, n_layers_conv=3, layer_params=layer_params_3)
        
        # Segunda capa de salida del decoding
        layer_params_2 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'dec2_ch{i+1}'}
        x_dec2 = _decoding_layer(x_dec3, n_layers_conv=2, layer_params=layer_params_2)
        
        # Primera capa de salida del decoding
        layer_params_1 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'dec1_ch{i+1}'}
        x_dec1 = _decoding_layer(x_dec2, n_layers_conv=2, layer_params=layer_params_1)
                                        
        # Agregando a la lista de salida
        list_decs.append(x_dec1)
    
    # Concatenando la lista de envolventes para generar una matriz
    x_conc = tf.keras.layers.concatenate(list_decs, axis=-1, name='Concatenate_layer')
    
    # Definición de la capa de salida
    x_out = tf.keras.layers.Dense(3, activation='softmax', kernel_initializer='he_normal',
                                  name='softmax_out')(x_conc)
    
    # Definición del modelo
    model = tf.keras.Model(inputs=x_in_list, outputs=x_out, name=name)
    
    return model


def segnet_based_1_3(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Se utiliza el mismo canal para todas las envolventes.
    Es lo mismo que segnet_based_1_1, pero se utiliza para segmentos de 
    largo 128 con step 16.
    
    Envolventes usadas:
    - Señal original
    - Filtros homomórficos
    - Envolventes de Hilbert
    - DWT
    - Energy envolve
    
    Salida de 3 etiquetas:
    - S1
    - S2
    - None
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    return segnet_based_1_1(input_shape, padding_value, name=name)


def segnet_based_1_4(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Se utiliza el mismo canal para todas las envolventes.
    Es lo mismo que segnet_based_1_2, pero se utiliza para segmentos de 
    largo 128 con step 16.
    
    Envolventes usadas:
    - Señal original
    - Filtros homomórficos
    - Envolventes de Hilbert
    - DWT
    - Energy envolve
    
    Salida de 3 etiquetas:
    - S1
    - S2
    - None
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    return segnet_based_1_2(input_shape, padding_value, name=name)



# Comentarios
# -----------
# A partir de los resultados, es posible notar que la arquitectura basada
# en Segnet supera considerablemente a una arquitectura CNN + Fully connected,
# por lo que se decide escoger Segnet para los siguientes experimentos.
# Se puede apreciar que existe una pequeña mejora entre usar un canal
# único para todas las envolentes y usar un canal para cada envolvente.
# Sin embargo, la mejora que ofrece cada canal es solo sobre el entrenamiento,
# pero en testeo se cae. 

# Por lo tanto, se escoge para seguir el segnet_based_1_1 ya que utiliza
# una cantidad mucho menor de parámetros logrando el mismo desempeño, e 
# incluso mejor en los resultados del testeo.


#######     2) Análisis de envolventes     ####### 

def segnet_based_2_1(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Se utiliza el mismo canal para todas las envolventes.
    
    Envolventes usadas:
    - Señal original
    - Filtros homomórficos
    - Envolventes de Hilbert
    - DWT
    - Energy envolve
    
    Salida de 3 etiquetas:
    - S1
    - S2
    - None
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    return segnet_based_1_1(input_shape, padding_value, name=name)


def segnet_based_2_2(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Se utiliza el mismo canal para todas las envolventes.
    
    Envolventes usadas:
    - Señal original
    - Filtros homomórficos
    - Envolventes de Hilbert
    - DWT
    - Energy envolve
    - Spectral tracking
    
    Salida de 3 etiquetas:
    - S1
    - S2
    - None
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    return segnet_based_1_1(input_shape, padding_value, name=name)


def segnet_based_2_3(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Se utiliza el mismo canal para todas las envolventes.
    
    Envolventes usadas:
    - Señal original
    - Filtros homomórficos
    - Envolventes de Hilbert
    - DWT
    - Energy envolve
    - Spectral tracking
    - Variance Fractal Dimension
    
    Salida de 3 etiquetas:
    - S1
    - S2
    - None
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    return segnet_based_1_1(input_shape, padding_value, name=name)


def segnet_based_2_4(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Se utiliza el mismo canal para todas las envolventes.
    
    Envolventes usadas:
    - Señal original
    - Filtros homomórficos
    - Envolventes de Hilbert
    - DWT
    - Energy envolve
    - Spectral tracking
    - Variance Fractal Dimension
    - Multiscale Wavelet Product
    
    Salida de 3 etiquetas:
    - S1
    - S2
    - None
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    return segnet_based_1_1(input_shape, padding_value, name=name)


# Versión con todas las envolventes
def segnet_based_2_5(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Se utiliza el mismo canal para todas las envolventes.
    
    Envolventes usadas:
    - Señal original
    - Filtros homomórficos
    - Envolventes de Hilbert
    - DWT
    - Energy envolve
    - Spectral tracking
    - Variance Fractal Dimension
    - Multiscale Wavelet Product
    - Envolventes de Hilbert modificadas
    
    Salida de 3 etiquetas:
    - S1
    - S2
    - None
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    return segnet_based_1_1(input_shape, padding_value, name=name)


def segnet_based_2_6(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Se utiliza el mismo canal para todas las envolventes.
    
    Envolventes usadas:
    - Señal original
    - Envolventes de Hilbert
    - DWT
    - Energy envolve
    - Spectral tracking
    - Variance Fractal Dimension
    - Multiscale Wavelet Product
    - Envolventes de Hilbert modificadas
    
    Salida de 3 etiquetas:
    - S1
    - S2
    - None
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    return segnet_based_1_1(input_shape, padding_value, name=name)


def segnet_based_2_7(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Se utiliza el mismo canal para todas las envolventes.
    
    Envolventes usadas:
    - Envolventes de Hilbert
    - DWT
    - Energy envolve
    - Spectral tracking
    - Variance Fractal Dimension
    - Multiscale Wavelet Product
    - Envolventes de Hilbert modificadas
    
    Salida de 3 etiquetas:
    - S1
    - S2
    - None
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    return segnet_based_1_1(input_shape, padding_value, name=name)


def segnet_based_2_8(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Se utiliza el mismo canal para todas las envolventes.
    
    Envolventes usadas:
    - Envolventes de Hilbert
    - DWT
    - Energy envolve
    - Variance Fractal Dimension
    - Multiscale Wavelet Product
    - Envolventes de Hilbert modificadas
    
    Salida de 3 etiquetas:
    - S1
    - S2
    - None
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    return segnet_based_1_1(input_shape, padding_value, name=name)


def segnet_based_2_9(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Se utiliza el mismo canal para todas las envolventes.
    
    Envolventes usadas:
    - Envolvente de Hilbert normal
    - DWT
    - Energy envolve
    - Variance Fractal Dimension
    - Multiscale Wavelet Product
    
    Salida de 3 etiquetas:
    - S1
    - S2
    - None
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    return segnet_based_1_1(input_shape, padding_value, name=name)


def segnet_based_2_10(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Se utiliza el mismo canal para todas las envolventes.
    
    Envolventes usadas:
    - Envolvente de Hilbert (solo clásico)
    - DWT
    - Energy envolve
    - Multiscale Wavelet Product
    
    Salida de 3 etiquetas:
    - S1
    - S2
    - None
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    return segnet_based_1_1(input_shape, padding_value, name=name)


def segnet_based_2_11(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Se utiliza el mismo canal para todas las envolventes.
    
    Envolventes usadas:
    - Envolvente de Hilbert modificadas
    - DWT
    - Energy envolve
    - Multiscale Wavelet Product
    
    Salida de 3 etiquetas:
    - S1
    - S2
    - None
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    return segnet_based_1_1(input_shape, padding_value, name=name)


def segnet_based_2_12(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Se utiliza el mismo canal para todas las envolventes.
    
    Envolventes usadas:
    - Filtros homomórficos
    - Envolventes de Hilbert
    - DWT
    - Multiscale Wavelet Product
    - Envolventes de Hilbert modificadas
    
    Salida de 3 etiquetas:
    - S1
    - S2
    - None
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    return segnet_based_1_1(input_shape, padding_value, name=name)





# Módulo de testeo
if __name__ == '__main__':
    # Crear gráfico de la red neuronal 
    model = cnn_dnn_1_1((128,3), padding_value=2, name='Testeo')
    print(model.summary())
    tf.keras.utils.plot_model(model, to_file='Testeo.png', show_shapes=True, 
                              expand_nested=True)
