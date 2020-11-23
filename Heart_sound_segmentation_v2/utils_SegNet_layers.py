# Código auxiliar utilizado para las capas de Maxpooling y Upsampling de 
# la red SegNet. Modificado para señales 1D pero basado en el código 
# disponible en: https://github.com/ykamikawa/tf-keras-SegNet

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class MaxPoolingWithArgmax1D(Layer):
    def __init__(self, pool_size=2, strides=2, padding="same", **kwargs):
        super(MaxPoolingWithArgmax1D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        # Obtener variables
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        
        # Acondicionando el string del padding
        padding = padding.upper()
        
        # Definición de los argumtento del maxpool con argmax
        ksize = [1, pool_size]
        strides = [1, strides]
        
        # Aplicando maxpool
        output, argmax = tf.nn.max_pool_with_argmax(inputs, ksize=ksize, 
                                                    strides=strides, 
                                                    padding=padding)

        # Casteando
        argmax = K.cast(argmax, K.floatx())
        
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx] if dim is not None else None
            for idx, dim in enumerate(input_shape)
        ]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with tf.variable_scope(self.name):
            mask = K.cast(mask, "int32")
            input_shape = tf.shape(updates, out_type="int32")
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1] * self.size[0],
                    input_shape[2] * self.size[1],
                    input_shape[3],
                )
            self.output_shape1 = output_shape

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype="int32")
            batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
            batch_range = K.reshape(
                K.tf.range(output_shape[0], dtype="int32"), shape=batch_shape
            )
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = K.tf.range(output_shape[3], dtype="int32")
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = K.tf.size(updates)
            indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.size[0],
            mask_shape[2] * self.size[1],
            mask_shape[3],
        )
        



# Módulo de testeo
import numpy as np
if __name__ == '__main__':
    np.random.seed(0)
    test_np = np.expand_dims([np.random.randint(0,9,20)], -1)
    print(test_np.shape)
    test = tf.constant(test_np)
    print(test)
    
    # Definición de los argumtento del maxpool con argmax
    pool_size = 2
    strides = 2
    
    ksize = [1, pool_size]
    strides = [1, strides]
    
    # Aplicando maxpool
    output, argmax = tf.nn.max_pool_with_argmax(test, ksize=2, 
                                                strides=2, 
                                                padding='VALID',
                                                data_format='NWC')