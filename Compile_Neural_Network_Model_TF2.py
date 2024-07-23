#%% All Required Imports

#-------------------------Keras' General Imports------------------------------#

# import tensorflow as tf

# import keras
# from tensorflow import keras

#%% Compile Neural Network Model

def Compile_Neural_Network_Model(Input_Neural_Network_Model,
                                 Selected_Optimizer,
                                 Selected_Loss,
                                 Selected_Metrics):
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    print("\nCompiling Neural Network Model...\n")
    
#-----------------------------------------------------------------------------#
    
    Input_Neural_Network_Model.compile(optimizer=Selected_Optimizer,
                                       loss=Selected_Loss,
                                       metrics=Selected_Metrics)
    
#-----------------------------------------------------------------------------#
    
    print("\nNeural Network Model was compiled successfully!\n")