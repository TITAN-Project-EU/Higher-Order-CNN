from __future__ import print_function

#%% Enable Tensorflow Eager Execution

# =============================================================================
# import tensorflow as tf
# 
# tf.enable_eager_execution()
# 
# # =============================================================================
# # import tensorflow.contrib.eager as tfe
# # tfe.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)
# # =============================================================================
# 
# # tf.compat.v1.disable_eager_execution()
# =============================================================================

#%%------------------Keras Results' Reproducibility Imports-------------------#

# Seed-Value
# Apparently you may use different Seed-Values at each stage

#Seed_Value=0
Seed_Value=1337

# 1. Set the "PYTHONHASHSEED" Environment Variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(Seed_Value)

# 2. Set the "Python" built-in Pseudo-Random Generator at a fixed value
import random
random.seed(Seed_Value)

# 3. Set the "Numpy" Pseudo-Random Generator at a fixed value
import numpy as np
np.random.seed(Seed_Value)

# 4. Set the "Tensorflow" Pseudo-Random Generator at a fixed value
import tensorflow as tf
# tf.set_random_seed(Seed_Value)
tf.random.set_seed(Seed_Value)

# =============================================================================
# # 5. Configure a new Global "Tensorflow" Session
# # # from keras import backend as K
# # from tensorflow.keras import backend as K
# # session_conf=tf.ConfigProto(intra_op_parallelism_threads=1,
# #                             inter_op_parallelism_threads=1)
# # sess=tf.Session(graph=tf.get_default_graph(),config=session_conf)
# 
# session_conf=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
#                                       inter_op_parallelism_threads=1)
# sess=tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)
# =============================================================================

#%%---------------------------All Required Imports----------------------------#

#---------------------------Keras' General Imports----------------------------#

# =============================================================================
# # import keras
# from tensorflow import keras
# =============================================================================

#-----------------------Keras Models' Creation Imports------------------------#

#--------------------------ETC Model-4D-Convolution---------------------------#

# =============================================================================
# from Create_ETC_Model_4D_Convolution_TF import Create_ETC_Model_4D_Functional_API
# =============================================================================

#--------------------------ETC Model-4D-Correlation---------------------------#

from Create_ETC_Model_4D_Correlation_TF import Create_ETC_Model_4D_Functional_API

#-------------------Keras Optimizers' Customization Imports-------------------#

from Create_Customized_Optimizer_TF2 import Create_Customized_Optimizer

#---------------------Keras Losses' Customization Imports---------------------#

from Create_Customized_Losses_TF2 import Create_Customized_Losses

#---------------------Keras Metrics' Customization Imports--------------------#

from Create_Customized_Metrics_Regression_TF2 import Create_Customized_Metrics_Regression

#----------------------Keras Models' Compilation Imports----------------------#

from Compile_Neural_Network_Model_TF2 import Compile_Neural_Network_Model

#-------------------------Keras Models' Plot Imports--------------------------#

from Plot_Neural_Network_Model import Plot_Created_Model

#----------------Tensor Decomposition Operations' Imports---------------------#

import tensorly as tl
tl.set_backend('tensorflow')

#%%----------------------Define Training Process Parameters-------------------#

# =============================================================================
# print("\014")
# print("\033[H\033[J")
# =============================================================================

#-----------------------------------------------------------------------------#

print("\nDefining Training Process Parameters...\n")

#-------------------------Number of Starting Neurons--------------------------#

Number_of_Starting_Neurons=4
# Number_of_Starting_Neurons=8
# Number_of_Starting_Neurons=16
# Number_of_Starting_Neurons=32
# Number_of_Starting_Neurons=64
# Number_of_Starting_Neurons=128
# Number_of_Starting_Neurons=256

#-------------------------Number of Different Classes-------------------------#

num_classes=1

#-----------------------------------------------------------------------------#

print("\nTraining Process Parameters were defined successfully!\n")

#%%---------------------Define Neural Network Parameters----------------------#

# =============================================================================
# print("\014")
# print("\033[H\033[J")
# =============================================================================

#-----------------------------------------------------------------------------#

print("\nDefining Neural Network Parameters...\n")

#--------------------------------ETC Model-4D---------------------------------#

# input_shape=(32,32,5,4,1) # Multi-Spectral Time-Series Images
input_shape=(32,32,6,4,1) # Multi-Spectral Time-Series Images
# input_shape=(32,32,12,4,1) # Multi-Spectral Time-Series Images
# input_shape=(32,32,18,4,1) # Multi-Spectral Time-Series Images

#-----------------------------------------------------------------------------#

print("\nNeural Network Parameters were defined successfully!\n")

#%% Create the Neural Network Model

# =============================================================================
# print("\014")
# print("\033[H\033[J")
# =============================================================================

#-----------------------------------------------------------------------------#

print("\nCreating the Neural Network Model...\n")

#--------------------------------ETC Model-4D---------------------------------#

Neural_Network_Model=Create_ETC_Model_4D_Functional_API(input_shape,
                                                        Number_of_Starting_Neurons,
                                                        num_classes)

#-----------------------------------------------------------------------------#

print("\nThe Neural Network Model was created successfully!\n")

#%%----------Customize Optimizer for the built Neural Network Model-----------#

# =============================================================================
# print("\014")
# print("\033[H\033[J")
# =============================================================================

#-----------------------------------------------------------------------------#

print("\nCustomizing Optimizer for the built Neural Network Model...\n")

#-----------------------------------------------------------------------------#

# Selected_Optimizer_Name="SGD"

# Selected_Optimizer_Name="Adagrad"

Selected_Optimizer_Name="Adam"
# Selected_Optimizer_Name="Adamax"
# Selected_Optimizer_Name="Nadam"

# Selected_Optimizer_Name="Adadelta"

# Selected_Optimizer_Name="RMSprop"

#-----------------------------------------------------------------------------#

# Selected_Optimizers_Import_Source="Keras"
Selected_Optimizers_Import_Source="Tensorflow"

#-----------------------------------------------------------------------------#

Selected_Optimizer_Customized=Create_Customized_Optimizer(Selected_Optimizer_Name,
                                                          Selected_Optimizers_Import_Source)

#-----------------------------------------------------------------------------#

print("\nOptimizer was successfully customized for the built Neural Network Model!\n")

#%%------------Customize Loss for the built Neural Network Model--------------#

# =============================================================================
# print("\014")
# print("\033[H\033[J")
# =============================================================================

#-----------------------------------------------------------------------------#

print("\nCustomizing Loss for the built Neural Network Model...\n")

#-----------------------------------------------------------------------------#

# Machine_Learning_Problem="Classification"
Machine_Learning_Problem="Regression"

#-----------------------------------------------------------------------------#

Selected_Loss_Customized=Create_Customized_Losses(Machine_Learning_Problem)

#-----------------------------------------------------------------------------#

print("\nLoss was successfully customized for the built Neural Network Model!\n")

#%%-----------Customize Metrics for the built Neural Network Model------------#

# =============================================================================
# print("\014")
# print("\033[H\033[J")
# =============================================================================

#-----------------------------------------------------------------------------#

print("\nCustomizing Metrics for the built Neural Network Model...\n")

#-----------------------------------------------------------------------------#

# Number_of_Metrics=1
# Number_of_Metrics=2

# Number_of_Metrics=6

Number_of_Metrics=7

#-----------------------------------------------------------------------------#

Selected_Metrics_Customized=Create_Customized_Metrics_Regression(Number_of_Metrics)

#-----------------------------------------------------------------------------#

print("\nMetrics were successfully customized for the built Neural Network Model!\n")

#%%------------------Compile the built Neural Network Model-------------------#

# =============================================================================
# print("\014")
# print("\033[H\033[J")
# =============================================================================

#-----------------------------------------------------------------------------#

print("\nCompiling the built Neural Network Model...\n")

#-----------------------------------------------------------------------------#

Compile_Neural_Network_Model(Neural_Network_Model,
                             Selected_Optimizer_Customized,
                             Selected_Loss_Customized,
                             Selected_Metrics_Customized)

#-----------------------------------------------------------------------------#

print("\nThe built Neural Network Model was compiled successfully!\n")

#%% Preview the built Neural Network Model

# =============================================================================
# print("\014")
# print("\033[H\033[J")
# =============================================================================

#-----------------------------------------------------------------------------#

print("\nPreviewing the built Neural Network Model...\n")

#-----------------------------------------------------------------------------#

Neural_Network_Model.summary()

#-----------------------------------------------------------------------------#

print("\nThe built Neural Network Model was previewed successfully!\n")

#%%---------------------Plot the Trained Neural Network Model-----------------#

# =============================================================================
# print("\014")
# print("\033[H\033[J")
# =============================================================================

#-----------------------------------------------------------------------------#

print("\nPlotting the Trained Neural Network Model...\n")

#-----------------------------------------------------------------------------#

Plot_Created_Model(Neural_Network_Model)

#-----------------------------------------------------------------------------#

print("\nThe Trained Neural Network Model was plotted successfully!\n")