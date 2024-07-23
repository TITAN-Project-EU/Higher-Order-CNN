#%% All Required Imports

#-------------------------Keras' General Imports------------------------------#

import tensorflow as tf

# import keras
# from tensorflow import keras

#%% Function Definition

def PLCC_TF(y_true, y_pred):
    
#%% Define PLCC-Metric via Tensorflow functions
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    # print("\nDefining PLCC-Metric via Tensorflow functions...\n")
    
#-----------------------------------------------------------------------------#
    
    Mean_Value_Y_True=tf.math.reduce_mean(y_true)
    
    Mean_Value_Y_Pred=tf.math.reduce_mean(y_pred)
    
#-----------------------------------------------------------------------------#
    
    Difference_Y_True=y_true-Mean_Value_Y_True
    
    Difference_Y_Pred=y_pred-Mean_Value_Y_Pred
    
#-----------------------------------------------------------------------------#
    
    PLCC_Numerator=tf.math.reduce_mean(tf.multiply(Difference_Y_True,
                                                   Difference_Y_Pred))
    
#-----------------------------------------------------------------------------#
    
    PLCC_Denominator=tf.math.reduce_std(Difference_Y_True)*tf.math.reduce_std(Difference_Y_Pred)
    
#-----------------------------------------------------------------------------#
    
    # print("\nPLCC-Metric via Tensorflow functions was defined successfully!\n")
    
#%% Create Customized PLCC-Metric via Tensorflow Metrics
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    # print("\nCreating Customized PLCC-Metric via functions...\n")
    
#-----------------------------------------------------------------------------#
    
    PLCC_Metric_Tensorflow=PLCC_Numerator/PLCC_Denominator
    
#-----------------------------------------------------------------------------#
    
    # print("\nCustomized PLCC-Metric via Tensorflow functions was created successfully!\n")
    
#-----------------------------------------------------------------------------#
    
#%% Return the required values
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    # print('\nReturning the required values...\n')
    
#-----------------------------------------------------------------------------#
    
    return PLCC_Metric_Tensorflow
    
#-----------------------------------------------------------------------------#
    
    # print('\nThe required values were returned successfully!\n')