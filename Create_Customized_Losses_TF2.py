#%% All Required Imports

#-------------------------Keras' General Imports------------------------------#

# import tensorflow as tf

# import keras
# from tensorflow import keras

#---------------------Keras Probabilistic Losses' Imports---------------------#

# from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras.losses import CategoricalCrossentropy

# from tensorflow.keras.losses import SparseCategoricalCrossentropy

#----------------------Keras Regression Losses' Imports-----------------------#

from tensorflow.keras.losses import MeanSquaredError

# from tensorflow.keras.losses import MeanAbsoluteError

#%% Function Definition

def Create_Customized_Losses(Supervised_Learning_Problem):
    
#%% Customize Losses Options
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    print("\nCustomizing Losses Options...\n")
    
#-----------------------------------------------------------------------------#
    
#----------------------------Probabilistic Losses-----------------------------#
    
    # Binary_Cross_Entropy_Loss=BinaryCrossentropy(name="Binary_Cross_Entropy")
    
    Categorical_Cross_Entropy_Loss=CategoricalCrossentropy(name="Categorical_Cross_Entropy")
    
    # Sparse_Categorical_Cross_Entropy_Loss=SparseCategoricalCrossentropy(name="Sparse_Categorical_Cross_Entropy")
    
#-----------------------------Regression Losses-------------------------------#
    
    MSE_Loss=MeanSquaredError(name="MSE")
    
    # MAE_Loss=MeanAbsoluteError(name="MAE")
    
#-----------------------------------------------------------------------------#
    
    print("\nLosses Options were customized successfully!\n")
    
#%% Create Customized Losses
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    print("\nCreating Customized Losses...\n")
    
#-----------------------------------------------------------------------------#
    
    if Supervised_Learning_Problem=="Classification":
        # Customized_Loss=Binary_Cross_Entropy_Loss
        Customized_Loss=Categorical_Cross_Entropy_Loss
        # Customized_Loss=Sparse_Categorical_Cross_Entropy_Loss
    elif Supervised_Learning_Problem=="Regression":
        Customized_Loss=MSE_Loss
        # Customized_Loss=MAE_Loss
    else:
        pass
    
#-----------------------------------------------------------------------------#
    
    print("\nCustomized Losses were created successfully!\n")
    
#%% Return the required values
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    print('\nReturning the required values...\n')
    
#-----------------------------------------------------------------------------#
    
    return Customized_Loss
    
#-----------------------------------------------------------------------------#
    
    print('\nThe required values were returned successfully!\n')