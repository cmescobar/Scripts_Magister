from testing_functions import test_hss
from process_functions import preprocessing_audio


if __name__ == '__main__':
    # Definición de la función a revisar
    test_func = 'preprocessing_audio'
    
    if test_func == 'test_hss':
        test_hss()
    
    
    elif test_func == 'preprocessing_audio':
        # Parámetros de la función
        lowpass_params = {'freq_pass': 140, 'freq_stop': 150}
        model_name = 'definitive_segnet_based'
        
        # Parámetros base de datos
        symptom = 'Healthy'
        priority = 1
        ausc_pos = 'toracic'
        nmf_method = 'to_all'
        
        # Definición de los parámetros NMF
        nmf_parameters = {'n_components': 10, 'N': 1024, 'N_lax': 100, 
                          'N_fade': 100, 'noverlap': 768, 'repeat': 0, 
                          'padding': 0, 'window': 'hamming', 'init': 'random',
                          'solver': 'mu', 'beta': 1, 'tol': 1e-4, 
                          'max_iter': 200, 'alpha_nmf': 0, 'l1_ratio': 0, 
                          'random_state': 0, 'dec_criteria': 'vote'}
        
        # Aplicando la rutina
        preprocessing_audio(model_name, lowpass_params, symptom=symptom,
                            nmf_parameters=nmf_parameters,  
                            ausc_pos=ausc_pos, priority=priority,
                            nmf_method=nmf_method,
                            plot_segmentation=False,
                            plot_separation=True)
        
        