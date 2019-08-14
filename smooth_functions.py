import numpy as np
from scipy.fftpack import dct, idct
from scipy.optimize import brent


def smooth(signal):
    '''Este algoritmo permite realizar una suavización de la señal mediante
    el uso de regresión por mínimos cuadrados penalizados, el cual busca
    minimizar una función de la forma:
    F(y_est) = RSS + S*P(y_est)

    Donde P(y_est) = ||D*y_est||^2  y D es una matriz posible de
    diagonalizar. A raíz de esto, la solución que se obtiene es de la forma:
    (In + sD^T D) * y_est = y

    Con lo cual es posible obtener una expresión de y_est en función de 'y',
    y que esta expresión minimice la función de penalización F(y_est)
    '''
    # Definición de la matriz diagonal
    lambda_list = np.asarray([-2 + 2 * np.cos(np.pi * i / len(signal))
                              for i in range(len(signal))])

    # Búsqueda del 's' mínimo
    p_min = brent(gcv_to_score, args=(signal, lambda_list))
    s_min = get_s(p_min)

    # Obtención del gamma_vect óptimo
    gamma_vect = 1 / (1 + s_min * lambda_list ** 2)

    return idct(gamma_vect * dct(signal, norm='ortho'), norm='ortho')


def robust_smooth(signal, iters=20, tol_limit=1e-5):
    '''Algoritmo de suavización similar a la función smooth, pero utilizando
    pesos Wi '''
    # Definición de la matriz diagonal
    lambda_list = np.asarray([-2 + 2 * np.cos(np.pi * i / len(signal))
                              for i in range(len(signal))])

    # Definición de la matriz de pesos Wi
    wi = np.asarray([1] * len(signal))

    # Definición del vector de estados anteriores
    y_before = signal

    # Realizando la iteración 5 veces (tomando en cuenta la recomendación del
    # paper)
    for _ in range(iters):
        tolerance = float('inf')
        while tolerance > tol_limit:
            # Definición del dct_y utilizado para la iteración
            dct_y = dct(wi * (signal - y_before) + y_before, norm='ortho')
            # Búsqueda del 's' mínimo
            p_min = brent(gcv_to_score, args=(signal, lambda_list, wi,
                          dct_y, True)) 
            s_min = get_s(p_min)
            
            # Obtención del gamma_vect óptimo
            gamma_vect = 1 / (1 + s_min * lambda_list ** 2)

            # Definición del nuevo estado
            y_est = idct(gamma_vect * dct_y, norm='ortho')
            #print(y_est)
            #plt.plot(y_est)
            #plt.show()
            # Se redefine también la tolerancia como
            tolerance = np.linalg.norm(y_before - y_est) / \
                np.linalg.norm(y_est)

            # Y finalmente, se redefine el nuevo estado 'anterior'
            y_before = y_est

        # Una vez que se obtiene el punto de óptimo a buscar, se procede a
        # redefinir la matriz de pesos utilizando la función bisquare
        wi = bisquare_weight_argument(signal - y_est, s_min)

    return y_est


def bisquare_weight_argument(r, s):
    '''Función que permite modificar los pesos para el modelo de suavizamiento
    de la señal'''
    
    # Definición del valor h
    h = np.sqrt(1 + np.sqrt(1 + 16 * s)) / np.sqrt(2 + 32 * s)
    # Definición del MAD
    mad = np.median(abs(r - np.median(r)))
    # Definición del vector u_i
    u_vect = abs(r / (1.4826 * mad * np.sqrt(1 - h)))
    return np.asarray([(1 - (u_i / 4.685) ** 2) ** 2
            if (abs(u_i / 4.685) < 1) else 0 for u_i in u_vect])


def gcv_to_score(p, y, lambda_list, wi_list=np.asarray([]),
    dct_y=np.asarray([]), robust=False):
    '''La validación cruzada general corresponde a la expresión a optimizar en
     el algoritmo de suavizamiento mostrado utilizado en la función smooth y 
     robust_smooth. Los parámetros de esta función corresponden a:
    
    - p: Valor a optimizar (representando a 's', el verdadero valor a optimizar
      según el algoritmo del paper).
    
    - y: Señal de entrada.
    
    - lambda_list: Matriz diagonal 'A' representada en forma de lista.'''
    
    # Definición de n
    n = len(y)
    # Definición de s
    s = get_s(p)
    # Definición del vector gamma
    gamma_vect = 1 / (1 + s * lambda_list ** 2)
    # Calculando los RSS para cada uno de los casos
    if robust:
        # Calculando el valor del vector y_est
        y_est = idct(gamma_vect * dct(y, norm='ortho'), norm='ortho')
        # Es posible definir el valor de la RSS considerando penalizaciones
        rss = np.linalg.norm(np.sqrt(wi_list) * (y - y_est)) ** 2
    else:
        # Calculando los parámetros necesarios para obtener la expresión de la
        # validación cruzada. En primer lugar, obteniendo la RSS (suma residual
        # cuadrada)
        rss = np.linalg.norm((gamma_vect - 1) * dct(y, norm='ortho')) ** 2
        
    # Y la traza de la matriz A
    trace_h = sum(gamma_vect)
    # Es posible obtener una expresión para la validación cruzada generalizada
    return n * rss / ((n - trace_h) ** 2)


def get_s(p):
    return 10 ** p
