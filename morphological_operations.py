import numpy as np


def erosion_dilation_operation(signal_in, Q, op_type='erosion',
                               g_type='zeros',g_def=None):
    '''Definición de la operación de erosión/dilatación basada en 
    transformaciones morfológicas. 
    
    Disponible en: 
    - https://en.wikipedia.org/wiki/Erosion_(morphology)
    - https://en.wikipedia.org/wiki/Dilation_(morphology)
    
    Ref. anexa: https://opencv-python-tutroals.readthedocs.io/en/latest/
    py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    
    Parámetros
    - signal_in: Señal a erosionar/dilatar
    - Q: Tamaño del kernel g(n) (si es par, se le suma 1)
    - g_type: Tipo de kernel a usar.
        - [zeros]: Vector cero de largo Q (o Q+1)
        - [user_defined]: Vector de largo Q (o Q+1) a definir por el usuario
        
    ** Nota **: Se usa Q+1 para que el queden ventanas de largo impar con un
    único punto central.
    '''
    # Propiedad de Q (si es par, se le suma 1)
    if Q % 2 == 0:
        Q += 1
        
    # Definición del radio de la ventana (window ratio)
    ratio = Q // 2
    
    # Definición de N
    N = len(signal_in)
    
    # Definición del kernel g(n)
    if g_type == 'user_defined':
        g_n = g_def[:Q+1]
    elif g_type == 'zeros':
        g_n = np.zeros(Q+1)
        
    # Definición de la función a aplicar
    if op_type == 'erosion':
        func = lambda x: min(x)
    elif op_type == 'dilation':
        func = lambda x: max(x)
    
    # Creación del vector de erosión
    signal_out = np.zeros(N)

    # Se hace aplica erosión dependiendo del segmento. Esto se hace ya que las
    # ventanas que están en los bordes de la señal no alcanzan necesariamente a
    # juntar los Q puntos. Por ende, es necesario ajustar los tamaños de las
    # ventanas en los bordes
    for n in range(N):
        if 0 <= n <= Q//2 - 1:
            signal_out[n] = func(signal_in[:n+ratio+1])
        elif Q//2 <= n <= N - Q//2 - 1:
            signal_out[n] = func(signal_in[n-ratio:n+ratio+1])
        elif N - Q//2 <= n <= N - 1:
            signal_out[n] = func(signal_in[n-ratio:])
    
    # Se entrega finalmente el vector erosionado
    return signal_out


def closing_operation(signal_in, Q=30, g_type='zeros',g_def=None,
                      normalized=True):
    '''Operación morfológica que permite obtener la envolvente de la señal.
    Corresponde a dilatar y luego erosionar la señal.
    Referencias: 
    - Qingshu Liu, et.al. An automatic segmentation method for heart sounds.
      2018. Biomedical Engineering.
    - https://homepages.inf.ed.ac.uk/rbf/HIPR2/close.htm'''
    
    # En primer lugar se dilata la señal
    dilated_signal = erosion_dilation_operation(signal_in, Q=Q, 
                                                op_type='dilation',
                                                g_type=g_type, g_def=g_def)
    # Y luego de erosiona
    closed_signal =  erosion_dilation_operation(dilated_signal, Q=Q, 
                                                op_type='erosion',
                                                g_type=g_type, g_def=g_def)
    
    # Normalización
    if normalized:
        return closed_signal / max(abs(closed_signal))
    else:
        return closed_signal

    
def opening_operation(signal_in, Q=50, g_type='zeros',g_def=None,
                      normalized=True):
    '''Operación morfológica que permite obtener la envolvente de la señal.
    Corresponde a dilatar y luego erosionar la señal.
    Referencias: 
    - Qingshu Liu, et.al. An automatic segmentation method for heart sounds.
      2018. Biomedical Engineering.
    - https://homepages.inf.ed.ac.uk/rbf/HIPR2/close.htm'''
    
    # En primer lugar se dilata la señal
    eroded_signal = erosion_dilation_operation(signal_in, Q=Q, 
                                               op_type='erosion',
                                               g_type=g_type, g_def=g_def)
    # Y luego de erosiona
    opened_signal = erosion_dilation_operation(eroded_signal, Q=Q, 
                                               op_type='dilation',
                                               g_type=g_type, g_def=g_def)
    
    # Normalización
    if normalized:
        return opened_signal / max(abs(opened_signal))
    else:
        return opened_signal
