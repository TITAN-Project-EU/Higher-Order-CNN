#%% All Required Imports

#----------------Arithmetic and Ploting Operations' Imports-------------------#

# =============================================================================
# import numpy as np
# =============================================================================

#---------------------------Keras' General Imports----------------------------#

import tensorflow as tf
# import keras
# from tensorflow import keras

#----------------Tensor Decomposition Operations' Imports---------------------#

# =============================================================================
# import tensorly as tl
# # tl.set_backend('tensorflow')
# =============================================================================

#-----------------Convolution Features Transformation Imports-----------------#

# from Non_Linear_Transformation import Non_Linear_Transformation

#%% Layer Definition

@tf.keras.utils.register_keras_serializable()
class Max_Pooling_Layer_TF(tf.keras.layers.Layer):
    
#%% Input-Independent Initialization(Layer Constructor Function)
    
    def __init__(self,
                 Tensor_Order,
                 Max_Pooling_Subsection,
                 Max_Pooling_Window_Size,
                 Max_Pooling_Strides,
                 **kwargs):
        
#-----------------------------------------------------------------------------#
        
        super(Max_Pooling_Layer_TF,self).__init__(**kwargs)
        
#-----------------------------------------------------------------------------#
        
        self.Tensor_Order=Tensor_Order    # Set Tensor Order
        self.Max_Pooling_Subsection=Max_Pooling_Subsection    # Set Max-Pooling Subsection
        self.Max_Pooling_Window_Size=Max_Pooling_Window_Size  # Set Max-Pooling Window Size
        self.Max_Pooling_Strides=Max_Pooling_Strides  # Set Max-Pooling Strides
        
#-----------------------------------------------------------------------------#
        
        # super(Max_Pooling_Layer_TF,self).__init__(**kwargs)
    
#%% Forward Computation(Layer Forward Function for Prediction)
    
    def call(self,inputs):
        
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
        
        if self.Tensor_Order<=3:
            if self.Max_Pooling_Subsection==2:
                Z=tf.nn.max_pool(input=inputs,
                                 ksize=self.Max_Pooling_Window_Size,
                                 strides=self.Max_Pooling_Strides,
                                 padding="SAME")
            elif self.Max_Pooling_Subsection==3:
                Z=tf.nn.max_pool(input=inputs,
                                 ksize=self.Max_Pooling_Window_Size,
                                 strides=self.Max_Pooling_Strides,
                                 padding="VALID")
            else:
                # print('\nNot acceptable Max-Pooling Subsection...\n')
                pass
        
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
            
#--------------------Obtain the Size of Max-Pooling Window--------------------#
            
# =============================================================================
#             print("\014")
#             print("\033[H\033[J")
# =============================================================================
            
            # print('\nObtaining the Size of Max-Pooling Window...\n')
            
            (Max_Pooling_Window_Height,Max_Pooling_Window_Width,
             Max_Pooling_Window_Depth,Max_Pooling_Window_Frames)=self.Max_Pooling_Window_Size
            
            # print('\nThe Size of Max-Pooling Window was obtained successfully!\n')
            
#---------------Compute Max-Pooling Operation-Loop-Computation----------------#
            
# =============================================================================
#             print("\014")
#             print("\033[H\033[J")
# =============================================================================
            
            # print('\nComputing the Max-Pooling Operation on Tensor A-Loop-Computation...\n')
            
#-----------------------------------------------------------------------------#
            
            if self.Max_Pooling_Subsection==2:
                Padding_Mode="SAME"
            elif self.Max_Pooling_Subsection==3:
                Padding_Mode="VALID"
            else:
                # print('\nNot acceptable Max-Pooling Subsection...\n')
                pass
            
#-----------------------------------------------------------------------------#
            
            Stack_Axis=4
            
#-----------------------------------------------------------------------------#
            
            Frame_Input_Max_Pooling_3D=tf.unstack(inputs,
                                                  axis=Stack_Axis)
            
#-----------------------------------------------------------------------------#
            
            if Max_Pooling_Window_Frames % 2==1:
                # Uneven Filter Size: Same size to Left and Right
                Filter_Left=int(Max_Pooling_Window_Frames/2)
                Filter_Right=int(Max_Pooling_Window_Frames/2)
            else:
                # Even Filter Size: One more to Right
                Filter_Left=int(Max_Pooling_Window_Frames/2)-1
                Filter_Right=int(Max_Pooling_Window_Frames/2)
            
#-----------------------------------------------------------------------------#
            
            # The Start Index is important for Strides and Dilation
            # The Strides start with the First Element that works and is VALID
            Start_Index=0
            if Padding_Mode=="VALID":
                for i in range(Input_Frames):
                    if len(range(max(i-Filter_Left,0),
                                 min(i+Filter_Right+1,Input_Frames)))==Max_Pooling_Window_Frames:
                        # We found the First Index that doesn't need Padding
                        break
                Start_Index=i
            
#-----------------------------------------------------------------------------#
            
            # Loop over all z_j in t
            Frame_Results=[]
            for i in range(Start_Index,Input_Frames,self.Max_Pooling_Strides[-1]):
                
                if Padding_Mode=="VALID":
                    
                    # Get Indices z_s
                    Indices_t_s=range(max(i-Filter_Left,0),
                                      min(i+Filter_Right+1,Input_Frames))
                    
                    # Check if Padding="VALID"
                    if len(Indices_t_s)==Max_Pooling_Window_Frames:
                        
                        Frame_Output_MP_3D=[]
                        
                        # Sum over all Remaining Index_z_i in Indices_t_s
                        for j,Index_z_i in enumerate(Indices_t_s):
                            Frame_Output_MP_3D.append(tf.nn.max_pool(input=Frame_Input_Max_Pooling_3D[Index_z_i],
                                                                     ksize=(1,)+self.Max_Pooling_Window_Size[:-1]+(1,),
                                                                     strides=(1,)+self.Max_Pooling_Strides[:-1]+(1,),
                                                                     padding=Padding_Mode))
                        # Frame_Output_MP_3D_Sum=tf.add_n(Frame_Output_MP_3D)
                        if len(Frame_Output_MP_3D)>1:
                            Frame_Output_MP_3D_Max=tf.keras.layers.maximum(Frame_Output_MP_3D)
                        else:
                            Frame_Output_MP_3D.append(Frame_Output_MP_3D[0])
                            Frame_Output_MP_3D_Max=tf.keras.layers.maximum(Frame_Output_MP_3D)
                        
                        Frame_Results.append(Frame_Output_MP_3D_Max)
                elif Padding_Mode=="SAME":
                    
                    # Get Indices z_s
                    Indices_t_s=range(i-Filter_Left,
                                      (i+1)+Filter_Right)
                    
                    Frame_Output_MP_3D=[]
                    
                    for Kernel_j,j in enumerate(Indices_t_s):
                        # We can just leave out the Invalid t Coordinates
                        # since they will be padded with 0's and therfore
                        # don't contribute to the Sum
                        if 0<=j<Input_Frames:
                            Frame_Output_MP_3D.append(tf.nn.max_pool(input=Frame_Input_Max_Pooling_3D[j],
                                                                     ksize=(1,)+self.Max_Pooling_Window_Size[:-1]+(1,),
                                                                     strides=(1,)+self.Max_Pooling_Strides[:-1]+(1,),
                                                                     padding=Padding_Mode))
                    # Frame_Output_MP_3D_Sum=tf.add_n(Frame_Output_MP_3D)
                    if len(Frame_Output_MP_3D)>1:
                        Frame_Output_MP_3D_Max=tf.keras.layers.maximum(Frame_Output_MP_3D)
                    else:
                        Frame_Output_MP_3D.append(Frame_Output_MP_3D[0])
                        Frame_Output_MP_3D_Max=tf.keras.layers.maximum(Frame_Output_MP_3D)
                    
                    Frame_Results.append(Frame_Output_MP_3D_Max)
                else:
                    # print('\nNot acceptable Max-Pooling Subsection...\n')
                    pass
            
#-----------------------------------------------------------------------------#
            
            Z=tf.stack(Frame_Results,
                       axis=Stack_Axis)
            
#-----------------------------------------------------------------------------#
            
            # print('\nThe Max-Pooling Operation on Tensor A-Loop Computation was computed successfully!\n')
        
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
        
        config=super(Max_Pooling_Layer_TF,self).get_config()
        
#-----------------------------------------------------------------------------#
        
        config.update({"Tensor_Order":self.Tensor_Order,
                       "Max_Pooling_Subsection":self.Max_Pooling_Subsection,
                       "Max_Pooling_Window_Size":self.Max_Pooling_Window_Size,
                       "Max_Pooling_Strides":self.Max_Pooling_Strides})
        
#-----------------------------------------------------------------------------#
        
        return config