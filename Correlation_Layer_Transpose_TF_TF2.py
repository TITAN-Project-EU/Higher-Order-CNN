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

#--------------------Correlation-Transpose-TF Layer Imports-------------------#

from Tranpose_Convolution_Output_Shape import Compute_Transpose_Convolution_Output_Length

#%% Layer Definition

@tf.keras.utils.register_keras_serializable()
class Correlation_Layer_Transpose_TF(tf.keras.layers.Layer):
    
#%% Input-Independent Initialization(Layer Constructor Function)
    
    def __init__(self,
                 Tensor_Order,
                 Correlation_Transpose_Subsection,
                 Filter_Size,
                 Number_of_Filters,
                 Correlation_Transpose_Strides,
                 Weights_Initializer,
                 Bias_Initializer,
                 **kwargs):
        
#-----------------------------------------------------------------------------#
        
        super(Correlation_Layer_Transpose_TF,self).__init__(**kwargs)
        
#-----------------------------------------------------------------------------#
        
        self.Tensor_Order=Tensor_Order    # Set Tensor Order
        self.Correlation_Transpose_Subsection=Correlation_Transpose_Subsection    # Set Correlation Subsection
        self.Filter_Size=Filter_Size  # Set Filter Size
        self.Number_of_Filters=Number_of_Filters  # Set Number of Filters
        self.Correlation_Transpose_Strides=Correlation_Transpose_Strides  # Set Correlation Strides
        self.Weights_Initializer=keras.initializers.get(Weights_Initializer)  # Set Weights Initializer
        self.Bias_Initializer=keras.initializers.get(Bias_Initializer)  # Set Bias Initializer
        
#-----------------------------------------------------------------------------#
        
        # super(Correlation_Layer_Transpose_TF,self).__init__(**kwargs)
    
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
        
        # super(Correlation_Layer_Transpose_TF,self).build(input_shape)  # Be sure to call this at the end
    
#%% Forward Computation(Layer Forward Function for Prediction)
    
    def call(self,inputs):
        
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
        
        if self.Tensor_Order<=3:
            if self.Correlation_Transpose_Subsection==2:
                Z=tf.nn.conv_transpose(input=inputs,
                                       filters=self.Weights,
                                       strides=self.Correlation_Transpose_Strides,
                                       padding="SAME")
            elif self.Correlation_Transpose_Subsection==3:
                Z=tf.nn.conv_transpose(input=inputs,
                                       filters=self.Weights,
                                       strides=self.Correlation_Transpose_Strides,
                                       padding="VALID")
            else:
                # print('\nNot acceptable Correlation-Transpose Subsection...\n')
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
            
#---------Compute Tensor-based Correlation-Transpose-Loop-Computation---------#
            
# =============================================================================
#             print("\014")
#             print("\033[H\033[J")
# =============================================================================
            
            # print('\nComputing the Correlation-Transpose between Tensors A and B-Loop-Computation...\n')
            
#-----------------------------------------------------------------------------#
            
            if self.Correlation_Transpose_Subsection==2:
                Padding_Mode="SAME"
                Padding_Output_Shape="same"
            elif self.Correlation_Transpose_Subsection==3:
                Padding_Mode="VALID"
                Padding_Output_Shape="valid"
            else:
                Padding_Output_Shape="full"
                # print('\nNot acceptable Correlation-Transpose Subsection...\n')
                # pass
            
#-----------------------------------------------------------------------------#
            
            Input_Shape=inputs.get_shape().as_list()
            
#-----------------------------------------------------------------------------#
            
            Input_Shape_Dynamic=tf.shape(inputs)
            Batch_Size_Dynamic=Input_Shape_Dynamic[0]
            
#-----------------------------------------------------------------------------#
            
            Output_Shape=[i for i in inputs.get_shape().as_list()]
            for Index,Value in enumerate(Input_Shape[1:-1]):
                Output_Shape[Index+1]=Compute_Transpose_Convolution_Output_Length(dim_size=Input_Shape[Index+1],
                                                                                  stride_size=self.Correlation_Transpose_Strides[Index],
                                                                                  kernel_size=self.Filter_Size[Index],
                                                                                  padding=Padding_Output_Shape,
                                                                                  output_padding=None,
                                                                                  dilation=1)
            Output_Shape=tuple(Output_Shape)
            # Output_Shape=tf.convert_to_tensor(Output_Shape)
            
#-----------------------------------------------------------------------------#
            
            Stack_Axis=4
            
#-----------------------------------------------------------------------------#
            
            Frame_Input_Corr_Transpose_3D=tf.unstack(inputs,
                                                     axis=Stack_Axis)
            
            Frame_Kernel_Corr_Transpose_3D=tf.unstack(self.Weights,
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
            for i in range(Start_Index,Input_Frames):
                Frame_Output_Corr_Transpose_3D=[]
                
                if Padding_Mode=="VALID":
                    
                    # Get Indices t_s
                    Indices_t_s=range(max(i-Filter_Left,0),
                                      min(i+Filter_Right+1,Input_Frames),
                                      1)
                    
                    # Check if Padding="VALID"
                    if len(Indices_t_s)==Filter_Frames:
                        
                        # Sum over all Remaining Index_t_i in Indices_t_s
                        for j,Index_t_i in enumerate(Indices_t_s):
# =============================================================================
#                             Frame_Output_Corr_Transpose_3D.append(tf.nn.conv_transpose(input=Frame_Input_Corr_Transpose_3D[Index_t_i],
#                                                                                        filters=Frame_Kernel_Corr_Transpose_3D[j],
#                                                                                        output_shape=Output_Shape[:-2]+(Output_Shape[-1],),
#                                                                                        strides=(1,)+self.Correlation_Transpose_Strides[:-1]+(1,),
#                                                                                        padding=Padding_Mode))
# =============================================================================
# =============================================================================
#                             Frame_Output_Corr_Transpose_3D.append(tf.nn.conv_transpose(input=Frame_Input_Corr_Transpose_3D[Index_t_i],
#                                                                                        filters=Frame_Kernel_Corr_Transpose_3D[j],
#                                                                                        output_shape=(-1,)+Output_Shape[1:-2]+(Output_Shape[-1],),
#                                                                                        strides=(1,)+self.Correlation_Transpose_Strides[:-1]+(1,),
#                                                                                        padding=Padding_Mode))
# =============================================================================
# =============================================================================
#                             Frame_Output_Corr_Transpose_3D.append(tf.nn.conv_transpose(input=Frame_Input_Corr_Transpose_3D[Index_t_i],
#                                                                                        filters=Frame_Kernel_Corr_Transpose_3D[j],
#                                                                                        output_shape=(1,)+Output_Shape[1:-2]+(Output_Shape[-1],),
#                                                                                        strides=(1,)+self.Correlation_Transpose_Strides[:-1]+(1,),
#                                                                                        padding=Padding_Mode))
# =============================================================================
                            Frame_Output_Corr_Transpose_3D.append(tf.nn.conv_transpose(input=Frame_Input_Corr_Transpose_3D[Index_t_i],
                                                                                       filters=Frame_Kernel_Corr_Transpose_3D[j],
                                                                                       output_shape=tf.stack([Batch_Size_Dynamic,
                                                                                                              Output_Shape[1],Output_Shape[2],
                                                                                                              Output_Shape[3],Output_Shape[-1]]),
                                                                                       strides=(1,)+self.Correlation_Transpose_Strides[:-1]+(1,),
                                                                                       padding=Padding_Mode))
                        Frame_Output_Corr_Transpose_3D_Sum=tf.add_n(Frame_Output_Corr_Transpose_3D)
                        
                        # Frame_Results.append(Frame_Output_Corr_Transpose_3D_Sum)
                        
                        Frame_Results.extend([Frame_Output_Corr_Transpose_3D_Sum
                                              for k in range(Compute_Transpose_Convolution_Output_Length(dim_size=1,
                                                                                                         stride_size=self.Correlation_Transpose_Strides[-1],
                                                                                                         kernel_size=self.Filter_Size[-1],
                                                                                                         padding=Padding_Output_Shape,
                                                                                                         output_padding=None,
                                                                                                         dilation=1))])
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
# =============================================================================
#                             Frame_Output_Corr_Transpose_3D.append(tf.nn.conv_transpose(input=Frame_Input_Corr_Transpose_3D[j],
#                                                                                        filters=Frame_Kernel_Corr_Transpose_3D[Kernel_j],
#                                                                                        output_shape=Output_Shape[:-2]+(Output_Shape[-1],),
#                                                                                        strides=(1,)+self.Correlation_Transpose_Strides[:-1]+(1,),
#                                                                                        padding=Padding_Mode))
# =============================================================================
# =============================================================================
#                             Frame_Output_Corr_Transpose_3D.append(tf.nn.conv_transpose(input=Frame_Input_Corr_Transpose_3D[j],
#                                                                                        filters=Frame_Kernel_Corr_Transpose_3D[Kernel_j],
#                                                                                        output_shape=(-1,)+Output_Shape[1:-2]+(Output_Shape[-1],),
#                                                                                        strides=(1,)+self.Correlation_Transpose_Strides[:-1]+(1,),
#                                                                                        padding=Padding_Mode))
# =============================================================================
# =============================================================================
#                             Frame_Output_Corr_Transpose_3D.append(tf.nn.conv_transpose(input=Frame_Input_Corr_Transpose_3D[j],
#                                                                                        filters=Frame_Kernel_Corr_Transpose_3D[Kernel_j],
#                                                                                        output_shape=(1,)+Output_Shape[1:-2]+(Output_Shape[-1],),
#                                                                                        strides=(1,)+self.Correlation_Transpose_Strides[:-1]+(1,),
#                                                                                        padding=Padding_Mode))
# =============================================================================
                            Frame_Output_Corr_Transpose_3D.append(tf.nn.conv_transpose(input=Frame_Input_Corr_Transpose_3D[j],
                                                                                       filters=Frame_Kernel_Corr_Transpose_3D[Kernel_j],
                                                                                       output_shape=tf.stack([Batch_Size_Dynamic,
                                                                                                              Output_Shape[1],Output_Shape[2],
                                                                                                              Output_Shape[3],Output_Shape[-1]]),
                                                                                       strides=(1,)+self.Correlation_Transpose_Strides[:-1]+(1,),
                                                                                       padding=Padding_Mode))
                    Frame_Output_Corr_Transpose_3D_Sum=tf.add_n(Frame_Output_Corr_Transpose_3D)
                    
                    # Frame_Results.append(Frame_Output_Corr_Transpose_3D_Sum)
                    
                    Frame_Results.extend([Frame_Output_Corr_Transpose_3D_Sum
                                          for k in range(Compute_Transpose_Convolution_Output_Length(dim_size=1,
                                                                                                     stride_size=self.Correlation_Transpose_Strides[-1],
                                                                                                     kernel_size=self.Filter_Size[-1],
                                                                                                     padding=Padding_Output_Shape,
                                                                                                     output_padding=None,
                                                                                                     dilation=1))])
                else:
                    # print('\nNot acceptable Correlation-Transpose Subsection...\n')
                    pass
            
#-----------------------------------------------------------------------------#
            
            if Padding_Mode=="VALID":
                if len(Frame_Results)<Output_Shape[-2]:
                    # print("Further Padding Actions must take place...")
                    Frame_Results.extend([Frame_Output_Corr_Transpose_3D_Sum for l in range(Output_Shape[-2]-len(Frame_Results))])
                elif len(Frame_Results)>Output_Shape[-2]:
                    # print("Further Cropping Actions must take place...")
                    Frame_Results=Frame_Results[:Output_Shape[-2]-len(Frame_Results)]
                else:
                    pass
                    # print("No Further Actions must take place...")
            
#-----------------------------------------------------------------------------#
            
            Z=tf.stack(Frame_Results,
                       axis=Stack_Axis)
            
#-----------------------------------------------------------------------------#
            
            # print('\nThe Correlation-Transpose between Tensors A and B-Loop Computation was computed successfully!\n')
            
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
        
        config=super(Correlation_Layer_Transpose_TF,self).get_config()
        
#-----------------------------------------------------------------------------#
        
        config.update({"Tensor_Order":self.Tensor_Order,
                       "Correlation_Transpose_Subsection":self.Correlation_Transpose_Subsection,
                       "Filter_Size":self.Filter_Size,
                       "Number_of_Filters":self.Number_of_Filters,
                       "Correlation_Transpose_Strides":self.Correlation_Transpose_Strides,
                       "Weights_Initializer":keras.initializers.serialize(self.Weights_Initializer),
                       "Bias_Initializer":keras.initializers.serialize(self.Bias_Initializer)})
        
#-----------------------------------------------------------------------------#
        
        return config