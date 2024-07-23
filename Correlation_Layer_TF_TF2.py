#%% All Required Imports

#----------------Arithmetic and Ploting Operations' Imports-------------------#

import numpy as np

#---------------------------Keras' General Imports----------------------------#

import tensorflow as tf
# import keras
from tensorflow import keras

#----------------Tensor Decomposition Operations' Imports---------------------#

import tensorly as tl
# tl.set_backend('tensorflow')

#-----------------Correlation Features Transformation Imports-----------------#

# from Non_Linear_Transformation import Non_Linear_Transformation

#%% Layer Definition

@tf.keras.utils.register_keras_serializable()
class Correlation_Layer_TF(tf.keras.layers.Layer):
    
#%% Input-Independent Initialization(Layer Constructor Function)
    
    def __init__(self,
                 Tensor_Order,
                 Correlation_Subsection,
                 Filter_Size,
                 Number_of_Filters,
                 Correlation_Strides,
                 Weights_Initializer,
                 Bias_Initializer,
                 **kwargs):
        
#-----------------------------------------------------------------------------#
        
        super(Correlation_Layer_TF,self).__init__(**kwargs)
        
#-----------------------------------------------------------------------------#
        
        self.Tensor_Order=Tensor_Order    # Set Tensor Order
        self.Correlation_Subsection=Correlation_Subsection    # Set Correlation Subsection
        self.Filter_Size=Filter_Size  # Set Filter Size
        self.Number_of_Filters=Number_of_Filters  # Set Number of Filters
        self.Correlation_Strides=Correlation_Strides  # Set Correlation Strides
        self.Weights_Initializer=keras.initializers.get(Weights_Initializer)  # Set Weights Initializer
        self.Bias_Initializer=keras.initializers.get(Bias_Initializer)  # Set Bias Initializer
        
#-----------------------------------------------------------------------------#
        
        # super(Correlation_Layer_TF,self).__init__(**kwargs)
    
#%% Rest of the Initialization-Given the Shapes of the Input Tensors
    
    def build(self,input_shape):
        
#-----------------------------------------------------------------------------#
        
        if tl.get_backend()=='tensorflow':
            Weights_Size=tl.concatenate([self.Filter_Size,
                                         [(input_shape.dims)[-1].value],
                                         [self.Number_of_Filters]],
                                        axis=0)
            Bias_Size=tl.concatenate([[1 for i in range(self.Tensor_Order)],
                                      [self.Number_of_Filters]],
                                     axis=0)
        else:
            Weights_Size=tl.concatenate((self.Filter_Size,
                                         (input_shape.dims)[-1].value,
                                         self.Number_of_Filters),
                                        axis=None)
            Bias_Size=tl.concatenate((tl.ones((1,self.Tensor_Order)),
                                      self.Number_of_Filters),
                                     axis=None)
        
#-----------------------------------------------------------------------------#
        
        if tl.get_backend()=='tensorflow':
            self.Weights=self.add_weight(
                shape=tl.to_numpy(Weights_Size),
                initializer=self.Weights_Initializer,
                name="Kernel",
                trainable=True,
                dtype='float32')
            self.Bias=self.add_weight(
                shape=tl.to_numpy(Bias_Size),
                initializer=self.Bias_Initializer,
                name="Bias",
                trainable=True,
                dtype='float32')
        else:
            self.Weights=self.add_weight(
                shape=Weights_Size.astype(int),
                initializer=self.Weights_Initializer,
                name="Kernel",
                trainable=True,
                dtype='float32')
            self.Bias=self.add_weight(
                shape=Bias_Size.astype(int),
                initializer=self.Bias_Initializer,
                name="Bias",
                trainable=True,
                dtype='float32')
        
#-----------------------------------------------------------------------------#
        
        # super(Correlation_Layer_TF,self).build(input_shape)  # Be sure to call this at the end
    
#%% Forward Computation(Layer Forward Function for Prediction)
    
    def call(self,inputs):
        
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
        
        if self.Tensor_Order<=3:
            if self.Correlation_Subsection==2:
                Z=tf.nn.convolution(input=inputs,
                                    filters=self.Weights,
                                    strides=self.Correlation_Strides,
                                    padding="SAME")
            elif self.Correlation_Subsection==3:
                Z=tf.nn.convolution(input=inputs,
                                    filters=self.Weights,
                                    strides=self.Correlation_Strides,
                                    padding="VALID")
            else:
                # print('\nNot acceptable Correlation Subsection...\n')
                pass
            
            if tl.get_backend()=='tensorflow':
                Z=tf.math.add(Z,self.Bias)
            else:
                Z=np.add(Z,self.Bias)
        
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
        
        if self.Tensor_Order==4:
            
#-------------------------Obtain the Size of Tensor A-------------------------#
            
# =============================================================================
#             print("\014")
#             print("\033[H\033[J")
# =============================================================================
            
            # print('\nObtaining the Size of Tensor A...\n')
            
            (Number_of_Input_Images,
             Input_Height,Input_Width,
             Input_Depth,Input_Frames,
             Number_of_Input_Channels)=tuple(inputs.get_shape().as_list())
            
            # print('\nThe Size of Tensor A was obtained successfully!\n')
            
#-------------------------Obtain the Size of Tensor B-------------------------#
            
# =============================================================================
#             print("\014")
#             print("\033[H\033[J")
# =============================================================================
            
            # print('\nObtaining the Size of Tensor B...\n')
            
            (Filter_Height,Filter_Width,
             Filter_Depth,Filter_Frames)=self.Filter_Size
            
            # print('\nThe Size of Tensor B was obtained successfully!\n')
            
#--------------Compute Tensor-based Correlation-Loop-Computation--------------#
            
# =============================================================================
#             print("\014")
#             print("\033[H\033[J")
# =============================================================================
            
            # print('\nComputing the Correlation between Tensors A and B-Loop-Computation...\n')
            
#-----------------------------------------------------------------------------#
            
            if self.Correlation_Subsection==2:
                Padding_Mode="SAME"
            elif self.Correlation_Subsection==3:
                Padding_Mode="VALID"
            else:
                # print('\nNot acceptable Correlation Subsection...\n')
                pass
            
#-----------------------------------------------------------------------------#
            
            Stack_Axis=4
            
#-----------------------------------------------------------------------------#
            
            Frame_Input_Corr_3D=tf.unstack(inputs,
                                           axis=Stack_Axis)
            
            Frame_Kernel_Corr_3D=tf.unstack(self.Weights,
                                            axis=Stack_Axis-1)
            
#-----------------------------------------------------------------------------#
            
            if Filter_Frames % 2==1:
                # Uneven Filter Size: Same size to Left and Right
                Filter_Left=int(Filter_Frames/2)
                Filter_Right=int(Filter_Frames/2)
            else:
                # Even Filter Size: One more to Right
                Filter_Left=int(Filter_Frames/2)-1
                Filter_Right=int(Filter_Frames/2)
            
#-----------------------------------------------------------------------------#
            
            # The Start Index is important for Strides and Dilation
            # The Strides start with the First Element that works and is VALID
            Start_Index=0
            if Padding_Mode=="VALID":
                for i in range(Input_Frames):
                    if len(range(max(i-Filter_Left,0),
                                 min(i+Filter_Right+1,Input_Frames),
                                 1))==Filter_Frames:
                        # We found the First Index that doesn't need Padding
                        break
                Start_Index=i
            
#-----------------------------------------------------------------------------#
            
            # Loop over all t_j in t
            Frame_Results=[]
            for i in range(Start_Index,Input_Frames,self.Correlation_Strides[-1]):
                Frame_Output_Corr_3D=[]
                
                if Padding_Mode=="VALID":
                    
                    # Get Indices t_s
                    Indices_t_s=range(max(i-Filter_Left,0),
                                      min(i+Filter_Right+1,Input_Frames),
                                      1)
                    
                    # Check if Padding="VALID"
                    if len(Indices_t_s)==Filter_Frames:
                        
                        # Sum over all Remaining Index_t_i in Indices_t_s
                        for j,Index_t_i in enumerate(Indices_t_s):
                            Frame_Output_Corr_3D.append(tf.nn.convolution(input=Frame_Input_Corr_3D[Index_t_i],
                                                                          filters=Frame_Kernel_Corr_3D[j],
                                                                          strides=(1,)+self.Correlation_Strides[:-1]+(1,),
                                                                          padding=Padding_Mode))
                        Frame_Output_Corr_3D_Sum=tf.add_n(Frame_Output_Corr_3D)
                        Frame_Results.append(Frame_Output_Corr_3D_Sum)
                elif Padding_Mode=="SAME":
                    
                    # Get Indices t_s
                    Indices_t_s=range(i-Filter_Left,
                                      (i+1)+Filter_Right,
                                      1)
                    
                    for Kernel_j,j in enumerate(Indices_t_s):
                        # We can just leave out the Invalid t Coordinates
                        # since they will be padded with 0's and therfore
                        # don't contribute to the Sum
                        if 0<=j<Input_Frames:
                            Frame_Output_Corr_3D.append(tf.nn.convolution(input=Frame_Input_Corr_3D[j],
                                                                          filters=Frame_Kernel_Corr_3D[Kernel_j],
                                                                          strides=(1,)+self.Correlation_Strides[:-1]+(1,),
                                                                          padding=Padding_Mode))
                    Frame_Output_Corr_3D_Sum=tf.add_n(Frame_Output_Corr_3D)
                    Frame_Results.append(Frame_Output_Corr_3D_Sum)
                else:
                    # print('\nNot acceptable Correlation Subsection...\n')
                    pass
            
#-----------------------------------------------------------------------------#
            
            Z=tf.stack(Frame_Results,
                       axis=Stack_Axis)
            
#-----------------------------------------------------------------------------#
            
            # print('\nThe Correlation between Tensors A and B-Loop Computation was computed successfully!\n')
            
#-----------------------------------------------------------------------------#
            
            if tl.get_backend()=='tensorflow':
                Z=tf.math.add(Z,self.Bias)
            else:
                Z=np.add(Z,self.Bias)
        
#-------------------------Return the required values--------------------------#
        
# =============================================================================
#         print("\014")
#         print("\033[H\033[J")
# =============================================================================
        
        # print('\nReturning the required values...\n')
        
        return Z
    
#%% Implement get_config to Enable Serialization-This is *Optional*
    
    def get_config(self):
        
#-----------------------------------------------------------------------------#
        
        config=super(Correlation_Layer_TF,self).get_config()
        
#-----------------------------------------------------------------------------#
        
        config.update({"Tensor_Order":self.Tensor_Order,
                       "Correlation_Subsection":self.Correlation_Subsection,
                       "Filter_Size":self.Filter_Size,
                       "Number_of_Filters":self.Number_of_Filters,
                       "Correlation_Strides":self.Correlation_Strides,
                       "Weights_Initializer":keras.initializers.serialize(self.Weights_Initializer),
                       "Bias_Initializer":keras.initializers.serialize(self.Bias_Initializer)})
        
#-----------------------------------------------------------------------------#
        
        return config