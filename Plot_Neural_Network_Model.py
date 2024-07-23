#%% All Required Imports

#-------------------------Keras' General Imports------------------------------#

import tensorflow as tf
# import keras
# from tensorflow import keras

#%% Plot Neural Network Model

def Plot_Created_Model(Created_Neural_Network_Model):
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    print("\nPloting the Neural Network Model...\n")
    
#-----------------------------------------------------------------------------#
    
    tf.keras.utils.plot_model(Created_Neural_Network_Model,
                              to_file="Model_Architecture.png",
                              show_shapes=True,
                              show_layer_names=True,
                              rankdir="TB",
                              expand_nested=False,
                              dpi=96)
    
# =============================================================================
#     tf.keras.utils.plot_model(Created_Neural_Network_Model,
#                               to_file="Model_Architecture.png",
#                               show_shapes=True,
#                               show_layer_names=True,
#                               rankdir="LR",
#                               expand_nested=False,
#                               dpi=96)
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    print("\nThe Neural Network Model was ploted successfully!\n")