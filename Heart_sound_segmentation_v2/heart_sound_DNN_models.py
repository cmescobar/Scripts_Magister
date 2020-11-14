import tensorflow as tf


def model_2_1(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    
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
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=200, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv1')(x_masked)
    x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=200, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv2')(x_conv1_norm)
    x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

    ### Tercer conv ###
    x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv3')(x_conv2_norm)
    x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

    ### Cuarto conv ###
    x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv4')(x_conv3_norm)
    x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)

    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
    
    
    # Definición de las capas fully connected
    x_full1 = tf.keras.layers.Dense(units=30, activation='relu',
                                    kernel_initializer='glorot_uniform')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

    x_full2 = tf.keras.layers.Dense(units=80, activation='relu',
                                    kernel_initializer='glorot_uniform')(x_full1)
    x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

    x_full3 = tf.keras.layers.Dense(units=70, activation='relu',
                                    kernel_initializer='glorot_uniform')(x_full2)
    x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

    # Definición de la última capa (activación)
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(x_full3)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(x_full3)


    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=[x_out_s1, x_out_s2], name=name)
    
    return model


def model_2_2(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    
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
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv1')(x_masked)
    x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv2')(x_conv1_norm)
    x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

    ### Tercer conv ###
    x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv3')(x_conv2_norm)
    x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

    ### Cuarto conv ###
    x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv4')(x_conv3_norm)
    x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)

    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
    
    
    # Definición de las capas fully connected
    x_full1 = tf.keras.layers.Dense(units=30, activation='relu')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

    x_full2 = tf.keras.layers.Dense(units=80, activation='relu')(x_full1)
    x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

    x_full3 = tf.keras.layers.Dense(units=70, activation='relu')(x_full2)
    x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

    # Definición de la última capa (activación)
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(x_full3)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(x_full3)


    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=[x_out_s1, x_out_s2], name=name)
    
    return model


def model_2_3(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Es una variación del model_2 en el que se quita la mitad de las CNN.
    
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
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv1')(x_masked)
    x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv2')(x_conv1_norm)
    x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)


    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv2_norm)
    
    
    # Definición de las capas fully connected
    x_full1 = tf.keras.layers.Dense(units=30, activation='relu')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

    x_full2 = tf.keras.layers.Dense(units=80, activation='relu')(x_full1)
    x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

    x_full3 = tf.keras.layers.Dense(units=70, activation='relu')(x_full2)
    x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

    # Definición de la última capa (activación)
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(x_full3)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(x_full3)


    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=[x_out_s1, x_out_s2], name=name)
    
    return model


def model_2_4(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Es una variación del model_2 en el que se agrega el doble de las CNN. 
    
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
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv1')(x_masked)
    x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv2')(x_conv1_norm)
    x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

    ### Tercer conv ###
    x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv3')(x_conv2_norm)
    x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

    ### Cuarto conv ###
    x_conv4 = tf.keras.layers.Conv1D(filters=30, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv4')(x_conv3_norm)
    x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)
    
    ### Quinto conv ###
    x_conv5 = tf.keras.layers.Conv1D(filters=50, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv5')(x_conv4_norm)
    x_conv5_norm = tf.keras.layers.BatchNormalization()(x_conv5)
    
    ### Sexto conv ###
    x_conv6 = tf.keras.layers.Conv1D(filters=40, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv6')(x_conv5_norm)
    x_conv6_norm = tf.keras.layers.BatchNormalization()(x_conv6)
    
    ### Séptimo conv ###
    x_conv7 = tf.keras.layers.Conv1D(filters=30, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv7')(x_conv6_norm)
    x_conv7_norm = tf.keras.layers.BatchNormalization()(x_conv7)

    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv7_norm)
    
    
    # Definición de las capas fully connected
    x_full1 = tf.keras.layers.Dense(units=30, activation='relu')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

    x_full2 = tf.keras.layers.Dense(units=80, activation='relu')(x_full1)
    x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

    x_full3 = tf.keras.layers.Dense(units=70, activation='relu')(x_full2)
    x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

    # Definición de la última capa (activación)
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(x_full3)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(x_full3)


    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=[x_out_s1, x_out_s2], name=name)
    
    return model


def model_2_5(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Es una variación del model_2 en el que se deja la mitad de las FC.
    
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
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv1')(x_masked)
    x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv2')(x_conv1_norm)
    x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

    ### Tercer conv ###
    x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv3')(x_conv2_norm)
    x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

    ### Cuarto conv ###
    x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv4')(x_conv3_norm)
    x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)

    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
    
    
    # Definición de las capas fully connected
    x_full1 = tf.keras.layers.Dense(units=30, activation='relu')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

    # Definición de la última capa (activación)
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(x_full1)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(x_full1)


    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=[x_out_s1, x_out_s2], name=name)
    
    return model


def model_2_6(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    En esta versión se usa el doble de capas FC.
    
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
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv1')(x_masked)
    x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv2')(x_conv1_norm)
    x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

    ### Tercer conv ###
    x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv3')(x_conv2_norm)
    x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

    ### Cuarto conv ###
    x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv4')(x_conv3_norm)
    x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)

    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
    
    # Definición de las capas fully connected
    x_full1 = tf.keras.layers.Dense(units=30, activation='relu')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

    x_full2 = tf.keras.layers.Dense(units=80, activation='relu')(x_full1)
    x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

    x_full3 = tf.keras.layers.Dense(units=70, activation='relu')(x_full2)
    x_full3 = tf.keras.layers.BatchNormalization()(x_full3)
    
    x_full4 = tf.keras.layers.Dense(units=70, activation='relu')(x_full3)
    x_full4 = tf.keras.layers.BatchNormalization()(x_full4)
    
    x_full5 = tf.keras.layers.Dense(units=70, activation='relu')(x_full4)
    x_full5 = tf.keras.layers.BatchNormalization()(x_full5)
    
    x_full6 = tf.keras.layers.Dense(units=70, activation='relu')(x_full5)
    x_full6 = tf.keras.layers.BatchNormalization()(x_full6)

    # Definición de la última capa (activación)
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(x_full6)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(x_full6)


    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=[x_out_s1, x_out_s2], name=name)
    
    return model


def model_2_7(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    En esta variación de usa la mitad de las capas convolucionales y la mitad de
    las capas FC.
    
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
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv1')(x_masked)
    x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv2')(x_conv1_norm)
    x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)


    
    
    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv2_norm)
    
    
    
    # Definición de las capas fully connected
    x_full1 = tf.keras.layers.Dense(units=30, activation='relu')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

    # Definición de la última capa (activación)
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(x_full1)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(x_full1)

    
    
    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=[x_out_s1, x_out_s2], name=name)
    
    return model


def model_2_8(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    En esta versión se usa 10 de capas FC.
    
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
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='he_normal', 
                                     name='Conv1')(x_masked)
    x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='he_normal', 
                                     name='Conv2')(x_conv1_norm)
    x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

    ### Tercer conv ###
    x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='he_normal', 
                                     name='Conv3')(x_conv2_norm)
    x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

    ### Cuarto conv ###
    x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='he_normal', 
                                     name='Conv4')(x_conv3_norm)
    x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)

    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
    
    # Definición de las capas fully connected
    x_full1 = tf.keras.layers.Dense(units=30, activation='relu', 
                                    kernel_initializer='he_normal')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

    x_full2 = tf.keras.layers.Dense(units=80, activation='relu', 
                                    kernel_initializer='he_normal')(x_full1)
    x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

    x_full3 = tf.keras.layers.Dense(units=70, activation='relu', 
                                    kernel_initializer='he_normal')(x_full2)
    x_full3 = tf.keras.layers.BatchNormalization()(x_full3)
    
    x_full4 = tf.keras.layers.Dense(units=70, activation='relu', 
                                    kernel_initializer='he_normal')(x_full3)
    x_full4 = tf.keras.layers.BatchNormalization()(x_full4)
    
    x_full5 = tf.keras.layers.Dense(units=70, activation='relu', 
                                    kernel_initializer='he_normal')(x_full4)
    x_full5 = tf.keras.layers.BatchNormalization()(x_full5)
    
    x_full6 = tf.keras.layers.Dense(units=70, activation='relu', 
                                    kernel_initializer='he_normal')(x_full5)
    x_full6 = tf.keras.layers.BatchNormalization()(x_full6)
    
    x_full7 = tf.keras.layers.Dense(units=70, activation='relu', 
                                    kernel_initializer='he_normal')(x_full6)
    x_full7 = tf.keras.layers.BatchNormalization()(x_full7)
    
    x_full8 = tf.keras.layers.Dense(units=70, activation='relu', 
                                    kernel_initializer='he_normal')(x_full7)
    x_full8 = tf.keras.layers.BatchNormalization()(x_full8)
    
    x_full9 = tf.keras.layers.Dense(units=70, activation='relu', 
                                    kernel_initializer='he_normal')(x_full8)
    x_full9 = tf.keras.layers.BatchNormalization()(x_full9)
    
    x_full10 = tf.keras.layers.Dense(units=70, activation='relu', 
                                    kernel_initializer='he_normal')(x_full9)
    x_full10 = tf.keras.layers.BatchNormalization()(x_full10)

    # Definición de la última capa (activación)
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='he_normal',
                                     name='S1_out')(x_full10)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='he_normal',
                                     name='S2_out')(x_full10)


    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=[x_out_s1, x_out_s2], name=name)
    
    return model


def model_3(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    En este caso a diferencia del modelo 2, se hace que la salida sea solo un canal.
    Por lo tanto, la salida tendrá información tanto de S1 como de S2.
    
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
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape,
                                       name='Masking')(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=200, padding='same', 
                                     activation='relu', 
                                     kernel_initializer='glorot_uniform', 
                                     name='Conv1')(x_masked)
    x_conv1_norm = tf.keras.layers.BatchNormalization(name='BatchConv1')(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=200, padding='same', 
                                     activation='relu', 
                                     kernel_initializer='glorot_uniform', 
                                     name='Conv2')(x_conv1_norm)
    x_conv2_norm = tf.keras.layers.BatchNormalization(name='BatchConv2')(x_conv2)

    ### Tercer conv ###
    x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', 
                                     kernel_initializer='glorot_uniform', 
                                     name='Conv3')(x_conv2_norm)
    x_conv3_norm = tf.keras.layers.BatchNormalization(name='BatchConv3')(x_conv3)

    ### Cuarto conv ###
    x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', 
                                     kernel_initializer='glorot_uniform', 
                                     name='Conv4')(x_conv3_norm)
    x_conv4_norm = tf.keras.layers.BatchNormalization(name='BatchConv4')(x_conv4)

    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
    
    
    # Definición de las capas fully connected
    x_full1 = tf.keras.layers.Dense(units=30, activation='relu', 
                                    name='FC1')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization(name='BatchFC1')(x_full1)
    
    
    x_full2 = tf.keras.layers.Dense(units=80, activation='relu', 
                                    name='FC2')(x_full1)
    x_full2 = tf.keras.layers.BatchNormalization(name='BatchFC2')(x_full2)
    
    
    x_full3 = tf.keras.layers.Dense(units=70, activation='relu', 
                                    name='FC3')(x_full2)
    x_full3 = tf.keras.layers.BatchNormalization(name='BatchFC3')(x_full3)

    # Definición de la última capa (activación)
    x_out = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                  kernel_initializer='glorot_uniform',
                                  name='Heart_out')(x_full3)


    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=x_out, name=name)
    
    return model


def model_4_1(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 2_1 pero aplicado a un mayor número de envolventes.
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    
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
    return model_2_1(input_shape, padding_value, name=name)


def model_4_2(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 2_1 pero aplicado a un mayor número de envolventes.
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    
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
    return model_2_1(input_shape, padding_value, name=name)


def model_4_3(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 2_1 pero aplicado a un mayor número de envolventes.
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    
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
    return model_2_1(input_shape, padding_value, name=name)


def model_4_4(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 2_1 pero aplicado a un mayor número de envolventes.
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    - Variance Fractal Dimension
    
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
    return model_2_1(input_shape, padding_value, name=name)


def model_5_1(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 2_1 pero aplicado a un mayor número de envolventes.
    Además cada envolvente posee una red neuronal para cada envolvente.
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    - Variance Fractal Dimension
    
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
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')
    
    # Definición de la lista de salida s1 y s2
    s1_list = list()
    s2_list = list()
    
    # Aplicando una separación para cada canal
    for i in range(input_shape[1]):
        # Se obtiene la i-ésima envolvente de la matriz de entrada
        channel_i = tf.keras.layers.Lambda(lambda x : 
            tf.keras.backend.expand_dims(x[:,:,i], axis=-1), name=f'Ch_{i+1}')(x_in)
    
        # Se hace el path del modelo
        model_i = model_2_1((input_shape[0], 1), padding_value, name=f'model_2_1_ch{i+1}')
        
        # Conectando el canal a la entrada del modelo (salida es una lista)
        s1_out, s2_out = model_i(channel_i)
        
        # Agregando a la lista de parámetros
        s1_list.append(s1_out)
        s2_list.append(s2_out)
    
    # Definición de una capa de aplanamiento
    s1_concat = tf.keras.layers.concatenate(s1_list, name='s1_concat')
    s2_concat = tf.keras.layers.concatenate(s2_list, name='s2_concat')
    
    # Definición de las 2 capas de salida
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='he_normal',
                                     name='S1_out')(s1_concat)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='he_normal',
                                     name='S2_out')(s2_concat)
        
    # Creación del modelo final
    model = tf.keras.Model(x_in, [x_out_s1, x_out_s2])
    
    return model


def model_5_1_1(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 2_1 pero aplicado a un mayor número de envolventes.
    Además cada envolvente posee una red neuronal para cada envolvente. 
    
    La diferencia con el model_5_1 es que en esta ocasión se realizan las conexiones 
    manuales
    
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    - Variance Fractal Dimension
    
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
    # Definición de una lista auxiliar de capas FC de salida
    s1_list = list()
    s2_list = list()
    
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)
    
    # Aplicando una separación para cada canal
    for i in range(input_shape[1]):
        # Se obtiene la i-ésima envolvente de la matriz de entrada
        channel_i = tf.keras.layers.Lambda(lambda x : 
            tf.keras.backend.expand_dims(x[:,:,i], axis=-1), name=f'Ch_{i+1}')(x_masked)
    
        # Definición de las capas convolucionales
        ### Primer conv ###
        x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='he_normal', 
                                        name=f'Conv1_ch{i+1}')(channel_i)
        x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

        ### Segundo conv ###
        x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='he_normal', 
                                        name=f'Conv2_ch{i+1}')(x_conv1_norm)
        x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

        ### Tercer conv ###
        x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='he_normal', 
                                        name=f'Conv3_ch{i+1}')(x_conv2_norm)
        x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

        ### Cuarto conv ###
        x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='he_normal', 
                                        name=f'Conv4_ch{i+1}')(x_conv3_norm)
        x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)
        
        
        # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
        x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
        
        
        # Definición de las capas fully connected
        ### Fully Connected 1 ###
        x_full1 = tf.keras.layers.Dense(units=30, activation='relu',
                                        kernel_initializer='he_normal',
                                        name=f'FC1_ch{i+1}')(x_flat)
        x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

        ### Fully Connected 2 ###
        x_full2 = tf.keras.layers.Dense(units=80, activation='relu',
                                        kernel_initializer='he_normal',
                                        name=f'FC2_ch{i+1}')(x_full1)
        x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

        ### Fully Connected 3 ###
        x_full3 = tf.keras.layers.Dense(units=70, activation='relu',
                                        kernel_initializer='he_normal',
                                        name=f'FC3_ch{i+1}')(x_full2)
        x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

        # Definición de las 2 capas de salida
        s1_lay = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                       kernel_initializer='he_normal',
                                       name=f'S1_out_ch{i+1}')(x_full3)
        s2_lay = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                       kernel_initializer='he_normal',
                                       name=f'S2_out_ch{i+1}')(x_full3)
        
        # Agregando a las listas
        s1_list.append(s1_lay)
        s2_list.append(s2_lay)
    
    # Definición de una capa de aplanamiento
    s1_concat = tf.keras.layers.concatenate(s1_list, name='s1_concat')
    s2_concat = tf.keras.layers.concatenate(s2_list, name='s2_concat')
    
    # Definición de las 2 capas de salida
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='he_normal',
                                     name='S1_out')(s1_concat)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='he_normal',
                                     name='S2_out')(s2_concat)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in, [x_out_s1, x_out_s2])
    
    return model


def model_5_2_1(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 5_1_1 pero con un mayor número de FC a la salida.
    
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    - Variance Fractal Dimension
    
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
    # Definición de una lista auxiliar de capas FC de salida
    full_list = list()
    
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)
    
    # Aplicando una separación para cada canal
    for i in range(input_shape[1]):
        # Se obtiene la i-ésima envolvente de la matriz de entrada
        channel_i = tf.keras.layers.Lambda(lambda x : 
            tf.keras.backend.expand_dims(x[:,:,i], axis=-1), name=f'Ch_{i+1}')(x_masked)
    
        # Definición de las capas convolucionales
        ### Primer conv ###
        x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv1_ch{i+1}')(channel_i)
        x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

        ### Segundo conv ###
        x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv2_ch{i+1}')(x_conv1_norm)
        x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

        ### Tercer conv ###
        x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv3_ch{i+1}')(x_conv2_norm)
        x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

        ### Cuarto conv ###
        x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv4_ch{i+1}')(x_conv3_norm)
        x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)
        
        
        # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
        x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
        
        
        # Definición de las capas fully connected
        ### Fully Connected 1 ###
        x_full1 = tf.keras.layers.Dense(units=30, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC1_ch{i+1}')(x_flat)
        x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

        ### Fully Connected 2 ###
        x_full2 = tf.keras.layers.Dense(units=80, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC2_ch{i+1}')(x_full1)
        x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

        ### Fully Connected 3 ###
        x_full3 = tf.keras.layers.Dense(units=70, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC3_ch{i+1}')(x_full2)
        x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

                
        # Agregando a las listas
        full_list.append(x_full3)
    
    # Definición de una capa de aplanamiento
    fully_concat = tf.keras.layers.concatenate(full_list, name='fully_concat')
    
    ### Fully Connected 1 out ###
    x_full1_out = tf.keras.layers.Dense(units=30, activation='relu',
                                   kernel_initializer='glorot_uniform',
                                   name=f'FC1_out')(fully_concat)
    x_full1_out = tf.keras.layers.BatchNormalization()(x_full1_out)
    
    
    ### Fully Connected 2 out ###
    x_full2_out = tf.keras.layers.Dense(units=30, activation='relu',
                                   kernel_initializer='glorot_uniform',
                                   name=f'FC2_out')(x_full1_out)
    x_full2_out = tf.keras.layers.BatchNormalization()(x_full2_out)
    
    
    ### Fully Connected 3 out ###
    x_full3_out = tf.keras.layers.Dense(units=30, activation='relu',
                                   kernel_initializer='glorot_uniform',
                                   name=f'FC3_out')(x_full2_out)
    x_full3_out = tf.keras.layers.BatchNormalization()(x_full3_out)
    
    
    # Definición de las 2 capas de salida
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(x_full3_out)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(x_full3_out)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in, [x_out_s1, x_out_s2])
    
    return model


def model_5_2_2(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 5_2_1, pero con menos envolventes.
    
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    
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
    # Definición de una lista auxiliar de capas FC de salida
    full_list = list()
    
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)
    
    # Aplicando una separación para cada canal
    for i in range(input_shape[1]):
        # Se obtiene la i-ésima envolvente de la matriz de entrada
        channel_i = tf.keras.layers.Lambda(lambda x : 
            tf.keras.backend.expand_dims(x[:,:,i], axis=-1), name=f'Ch_{i+1}')(x_masked)
    
        # Definición de las capas convolucionales
        ### Primer conv ###
        x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv1_ch{i+1}')(channel_i)
        x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

        ### Segundo conv ###
        x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv2_ch{i+1}')(x_conv1_norm)
        x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

        ### Tercer conv ###
        x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv3_ch{i+1}')(x_conv2_norm)
        x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

        ### Cuarto conv ###
        x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv4_ch{i+1}')(x_conv3_norm)
        x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)
        
        
        # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
        x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
        
        
        # Definición de las capas fully connected
        ### Fully Connected 1 ###
        x_full1 = tf.keras.layers.Dense(units=30, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC1_ch{i+1}')(x_flat)
        x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

        ### Fully Connected 2 ###
        x_full2 = tf.keras.layers.Dense(units=80, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC2_ch{i+1}')(x_full1)
        x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

        ### Fully Connected 3 ###
        x_full3 = tf.keras.layers.Dense(units=70, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC3_ch{i+1}')(x_full2)
        x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

                
        # Agregando a las listas
        full_list.append(x_full3)
    
    # Definición de una capa de aplanamiento
    fully_concat = tf.keras.layers.concatenate(full_list, name='fully_concat')
    
    ### Fully Connected 1 out ###
    x_full1_out = tf.keras.layers.Dense(units=30, activation='relu',
                                   kernel_initializer='glorot_uniform',
                                   name=f'FC1_out')(fully_concat)
    x_full1_out = tf.keras.layers.BatchNormalization()(x_full1_out)
    
    
    ### Fully Connected 2 out ###
    x_full2_out = tf.keras.layers.Dense(units=30, activation='relu',
                                   kernel_initializer='glorot_uniform',
                                   name=f'FC2_out')(x_full1_out)
    x_full2_out = tf.keras.layers.BatchNormalization()(x_full2_out)
    
    
    ### Fully Connected 3 out ###
    x_full3_out = tf.keras.layers.Dense(units=30, activation='relu',
                                   kernel_initializer='glorot_uniform',
                                   name=f'FC3_out')(x_full2_out)
    x_full3_out = tf.keras.layers.BatchNormalization()(x_full3_out)
    
    
    # Definición de las 2 capas de salida
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(x_full3_out)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(x_full3_out)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in, [x_out_s1, x_out_s2])
    
    return model


def model_5_2_3(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 5_2_1, pero sin la capa lambda de dispersión. En este
    caso se hace uso de una entrada distinta para cada envolvente
    
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    - Variance Fractal Dimension
    
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
        x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv1_ch{i+1}')(x_masked)
        x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

        ### Segundo conv ###
        x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv2_ch{i+1}')(x_conv1_norm)
        x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

        ### Tercer conv ###
        x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv3_ch{i+1}')(x_conv2_norm)
        x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

        ### Cuarto conv ###
        x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv4_ch{i+1}')(x_conv3_norm)
        x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)
        
        
        # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
        x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
        
        
        # Definición de las capas fully connected
        ### Fully Connected 1 ###
        x_full1 = tf.keras.layers.Dense(units=30, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC1_ch{i+1}')(x_flat)
        x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

        ### Fully Connected 2 ###
        x_full2 = tf.keras.layers.Dense(units=80, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC2_ch{i+1}')(x_full1)
        x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

        ### Fully Connected 3 ###
        x_full3 = tf.keras.layers.Dense(units=70, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC3_ch{i+1}')(x_full2)
        x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

                
        # Agregando a las listas
        full_list.append(x_full3)
    
    # Definición de una capa de aplanamiento
    fully_concat = tf.keras.layers.concatenate(full_list, name='fully_concat')
    
    ### Fully Connected 1 out ###
    x_full1_out = tf.keras.layers.Dense(units=30, activation='relu',
                                   kernel_initializer='glorot_uniform',
                                   name=f'FC1_out')(fully_concat)
    x_full1_out = tf.keras.layers.BatchNormalization()(x_full1_out)
    
    
    ### Fully Connected 2 out ###
    x_full2_out = tf.keras.layers.Dense(units=30, activation='relu',
                                   kernel_initializer='glorot_uniform',
                                   name=f'FC2_out')(x_full1_out)
    x_full2_out = tf.keras.layers.BatchNormalization()(x_full2_out)
    
    
    ### Fully Connected 3 out ###
    x_full3_out = tf.keras.layers.Dense(units=30, activation='relu',
                                   kernel_initializer='glorot_uniform',
                                   name=f'FC3_out')(x_full2_out)
    x_full3_out = tf.keras.layers.BatchNormalization()(x_full3_out)
    
    
    # Definición de las 2 capas de salida
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(x_full3_out)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(x_full3_out)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in_list, [x_out_s1, x_out_s2])
    
    return model


def model_5_2_4(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 5_2_1, pero sin la capa lambda de dispersión. En este
    caso se hace uso de una entrada distinta para cada envolvente, y además se aplica
    una etapa sigmoide (como en el model_2_1) extra en cada canal.
    
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    - Variance Fractal Dimension
    
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
    s1_list = list()
    s2_list = list()
    
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
        x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv1_ch{i+1}')(x_masked)
        x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

        ### Segundo conv ###
        x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv2_ch{i+1}')(x_conv1_norm)
        x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

        ### Tercer conv ###
        x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv3_ch{i+1}')(x_conv2_norm)
        x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

        ### Cuarto conv ###
        x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv4_ch{i+1}')(x_conv3_norm)
        x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)
        
        
        # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
        x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
        
        
        # Definición de las capas fully connected
        ### Fully Connected 1 ###
        x_full1 = tf.keras.layers.Dense(units=30, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC1_ch{i+1}')(x_flat)
        x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

        ### Fully Connected 2 ###
        x_full2 = tf.keras.layers.Dense(units=80, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC2_ch{i+1}')(x_full1)
        x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

        ### Fully Connected 3 ###
        x_full3 = tf.keras.layers.Dense(units=70, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC3_ch{i+1}')(x_full2)
        x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

        # Definición de las 2 capas de salida
        s1_lay = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                       kernel_initializer='glorot_uniform',
                                       name=f'S1_out_ch{i+1}')(x_full3)
        s2_lay = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                       kernel_initializer='glorot_uniform',
                                       name=f'S2_out_ch{i+1}')(x_full3)
                
        # Agregando a las listas
        s1_list.append(s1_lay)
        s2_list.append(s2_lay)
    
    # Definición de una capa de aplanamiento
    s1_concat = tf.keras.layers.concatenate(s1_list, name='s1_concat')
    s2_concat = tf.keras.layers.concatenate(s2_list, name='s2_concat')
    
    
    # Definición de las 2 capas de salida
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(s1_concat)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(s2_concat)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in_list, [x_out_s1, x_out_s2])
    
    return model


# model = model_5_2_4((128,3), padding_value=2, name='Testeo')
# print(model.summary())
# tf.keras.utils.plot_model(model, to_file='Testeo.png', show_shapes=True, expand_nested=True)
