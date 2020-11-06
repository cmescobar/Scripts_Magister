import tensorflow as tf


def model_2(input_shape, padding_value, name=None):
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

