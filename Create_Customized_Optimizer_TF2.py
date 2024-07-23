#%% All Required Imports

#---------------------------Keras' General Imports----------------------------#

import tensorflow as tf

# import keras
# from tensorflow import keras

#-----------------------Keras Optimizers' Imports-----------------------------#

# =============================================================================
# from keras.optimizers import SGD
# from keras.optimizers import Adagrad
# from keras.optimizers import Adam,Adamax,Nadam
# from keras.optimizers import Adadelta
# from keras.optimizers import RMSprop
# =============================================================================

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import Adam,Adamax,Nadam
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import RMSprop

#%% Function Definition

def Create_Customized_Optimizer(Optimizer_Name,Optimizers_Import_Source):
    
#%% Customize Optimizer Options
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    print("\nCustomizing Optimizer Options...\n")
    
#-----------------------------------------------------------------------------#
    
    if Optimizer_Name=="SGD":
        Initial_Learning_Rate=0.01  # Initial Learning Rate to be used for training
        # Initial_Learning_Rate=0.001  # Initial Learning Rate to be used for training
        # Initial_Learning_Rate=0.0001  # Initial Learning Rate to be used for training
        
        Momentum=0  # Contribution of the Parameter Update Step of the Previous Iteration to the Current Iteration of SGD
        # Momentum=0.9    # Contribution of the Parameter Update Step of the Previous Iteration to the Current Iteration of SGD
        
        Nesterov_Momentum=False # Whether to apply Nesterov Momentum
        # Nesterov_Momentum=True # Whether to apply Nesterov Momentum
    elif Optimizer_Name=="Adagrad":
        Initial_Learning_Rate=0.01  # Initial Learning Rate to be used for training
        # Initial_Learning_Rate=0.001 # Initial Learning Rate to be used for training
        # Initial_Learning_Rate=0.0001 # Initial Learning Rate to be used for training
        
        Initial_Accumulator_Value=0.1   # Starting value for the Accumulators
        
        Epsilon_Value=1e-7  # Denominator Offset
    elif Optimizer_Name=="Adam":
        # Initial_Learning_Rate=0.001 # Initial Learning Rate to be used for training
        # Initial_Learning_Rate=0.01  # Initial Learning Rate to be used for training
        Initial_Learning_Rate=0.0001  # Initial Learning Rate to be used for training
        
        Beta_1=0.9  # Exponential Decay Rate for the 1st Moment Estimates
        Beta_2=0.999    # Exponential Decay Rate for the 2nd Moment Estimates
        
        Epsilon_Value=1e-7  # Denominator Offset
        
        AMSGrad=False   # Whether to apply AMSGrad Variant of this Algorithm from the paper "On the Convergence of Adam and Beyond"
        # AMSGrad=True   # Whether to apply AMSGrad Variant of this Algorithm from the paper "On the Convergence of Adam and Beyond"
    elif Optimizer_Name=="Adamax":
        Initial_Learning_Rate=0.001 # Initial Learning Rate to be used for training
        # Initial_Learning_Rate=0.01  # Initial Learning Rate to be used for training
        # Initial_Learning_Rate=0.0001  # Initial Learning Rate to be used for training
        
        Beta_1=0.9  # Exponential Decay Rate for the 1st Moment Estimates
        Beta_2=0.999    # Exponential Decay Rate for the 2nd Moment Estimates
        
        Epsilon_Value=1e-7  # Denominator Offset
    elif Optimizer_Name=="Nadam":
        Initial_Learning_Rate=0.001 # Initial Learning Rate to be used for training
        # Initial_Learning_Rate=0.01  # Initial Learning Rate to be used for training
        # Initial_Learning_Rate=0.0001  # Initial Learning Rate to be used for training
        
        Beta_1=0.9  # Exponential Decay Rate for the 1st Moment Estimates
        Beta_2=0.999    # Exponential Decay Rate for the Exponentially Weighted Infinity Norm
        
        Epsilon_Value=1e-7  # Denominator Offset
    elif Optimizer_Name=="Adadelta":
        Initial_Learning_Rate=1 # Initial Learning Rate to be used for training
        # Initial_Learning_Rate=0.001 # Initial Learning Rate to be used for training
        # Initial_Learning_Rate=0.01 # Initial Learning Rate to be used for training
        
        Rho_Value=0.95  # Decay Rate
        
        Epsilon_Value=1e-6  # Denominator Offset
    elif Optimizer_Name=="RMSprop":
        Initial_Learning_Rate=0.001 # Initial Learning Rate to be used for training
        # Initial_Learning_Rate=0.01  # Initial Learning Rate to be used for training
        # Initial_Learning_Rate=0.0001  # Initial Learning Rate to be used for training
        
        Rho_Value=0.9  # Discounting Factor for the History/Coming Gradient
        
        Momentum=0  # Contribution of the Parameter Update Step of the Previous Iteration to the Current Iteration of RMSprop
        # Momentum=0.9    # Contribution of the Parameter Update Step of the Previous Iteration to the Current Iteration of RMSprop
        
        Epsilon_Value=1e-7  # Denominator Offset
        
        # Centered_Normalization=True    # If True,Gradients are normalized by the Estimated Variance of the Gradient
        Centered_Normalization=False    # If False,Gradients are normalized by the Uncentered Second Moment
        # Setting this to True may help with Training,but is slightly more expensive in terms of Computation and Memory
    else:
        pass
        
#-----------------------------------------------------------------------------#
    
    print("\nOptimizer Options were customized successfully!\n")
    
#%% Create Customized Optimizer
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    print("\nCreating Customized Optimizer...\n")
    
#-----------------------------------------------------------------------------#
    
    if Optimizer_Name=="SGD":
        if Optimizers_Import_Source=="Keras":
            Customized_Optimizer=SGD(learning_rate=Initial_Learning_Rate,
                                     momentum=Momentum,
                                     nesterov=Nesterov_Momentum)
        elif Optimizers_Import_Source=="Tensorflow":
            Customized_Optimizer=tf.optimizers.SGD(learning_rate=Initial_Learning_Rate,
                                                   momentum=Momentum,
                                                   nesterov=Nesterov_Momentum)
        else:
            pass
    elif Optimizer_Name=="Adagrad":
        if Optimizers_Import_Source=="Keras":
            Customized_Optimizer=Adagrad(learning_rate=Initial_Learning_Rate,
                                         initial_accumulator_value=Initial_Accumulator_Value,
                                         epsilon=Epsilon_Value)
        elif Optimizers_Import_Source=="Tensorflow":
            Customized_Optimizer=tf.optimizers.Adagrad(learning_rate=Initial_Learning_Rate,
                                                       initial_accumulator_value=Initial_Accumulator_Value,
                                                       epsilon=Epsilon_Value)
        else:
            pass
    elif Optimizer_Name=="Adam":
        if Optimizers_Import_Source=="Keras":
            Customized_Optimizer=Adam(learning_rate=Initial_Learning_Rate,
                                      beta_1=Beta_1,beta_2=Beta_2,
                                      epsilon=Epsilon_Value,
                                      amsgrad=AMSGrad)
        elif Optimizers_Import_Source=="Tensorflow":
            Customized_Optimizer=tf.optimizers.Adam(learning_rate=Initial_Learning_Rate,
                                                    beta_1=Beta_1,beta_2=Beta_2,
                                                    epsilon=Epsilon_Value,
                                                    amsgrad=AMSGrad)
        else:
            pass
    elif Optimizer_Name=="Adamax":
        if Optimizers_Import_Source=="Keras":
            Customized_Optimizer=Adamax(learning_rate=Initial_Learning_Rate,
                                        beta_1=Beta_1,beta_2=Beta_2,
                                        epsilon=Epsilon_Value)
        elif Optimizers_Import_Source=="Tensorflow":
            Customized_Optimizer=tf.optimizers.Adamax(learning_rate=Initial_Learning_Rate,
                                                      beta_1=Beta_1,beta_2=Beta_2,
                                                      epsilon=Epsilon_Value)
            pass
        else:
            pass
    elif Optimizer_Name=="Nadam":
        if Optimizers_Import_Source=="Keras":
            Customized_Optimizer=Nadam(learning_rate=Initial_Learning_Rate,
                                       beta_1=Beta_1,beta_2=Beta_2,
                                       epsilon=Epsilon_Value)
        elif Optimizers_Import_Source=="Tensorflow":
            Customized_Optimizer=tf.optimizers.Nadam(learning_rate=Initial_Learning_Rate,
                                                     beta_1=Beta_1,beta_2=Beta_2,
                                                     epsilon=Epsilon_Value)
            pass
        else:
            pass
    elif Optimizer_Name=="Adadelta":
        if Optimizers_Import_Source=="Keras":
            Customized_Optimizer=Adadelta(learning_rate=Initial_Learning_Rate,
                                          rho=Rho_Value,
                                          epsilon=Epsilon_Value)
        elif Optimizers_Import_Source=="Tensorflow":
            Customized_Optimizer=tf.optimizers.Adadelta(learning_rate=Initial_Learning_Rate,
                                                        rho=Rho_Value,
                                                        epsilon=Epsilon_Value)
        else:
            pass
    elif Optimizer_Name=="RMSprop":
        if Optimizers_Import_Source=="Keras":
            Customized_Optimizer=RMSprop(learning_rate=Initial_Learning_Rate,
                                         rho=Rho_Value,
                                         momentum=Momentum,
                                         epsilon=Epsilon_Value,
                                         centered=Centered_Normalization)
        elif Optimizers_Import_Source=="Tensorflow":
            Customized_Optimizer=tf.optimizers.RMSprop(learning_rate=Initial_Learning_Rate,
                                                       rho=Rho_Value,
                                                       momentum=Momentum,
                                                       epsilon=Epsilon_Value,
                                                       centered=Centered_Normalization)
        else:
            pass
    else:
        pass
    
#-----------------------------------------------------------------------------#
    
    print("\nCustomized Optimizer was created successfully!\n")
    
#%% Return the required values
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    print('\nReturning the required values...\n')
    
#-----------------------------------------------------------------------------#
    
    return Customized_Optimizer
    
#-----------------------------------------------------------------------------#
    
    print('\nThe required values were returned successfully!\n')