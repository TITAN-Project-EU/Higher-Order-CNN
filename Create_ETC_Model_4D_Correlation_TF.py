#%% All Required Imports

#-------------------------Keras Models' Imports-------------------------------#

from tensorflow.keras.models import Model

#------------------------Keras Core Layers' Imports---------------------------#

from tensorflow.keras.layers import Input

#---------------------Keras Convolution Layers' Imports-----------------------#

from tensorflow.keras.layers import Conv1D

#--------------------Keras Normalization Layers' Imports----------------------#

from tensorflow.keras.layers import BatchNormalization

#--------------------Keras Regularization Layers' Imports---------------------#

from tensorflow.keras.layers import SpatialDropout1D

#----------------------Keras Reshaping Layers' Imports------------------------#

from tensorflow.keras.layers import Reshape

#-----------------------Keras Merging Layers' Imports-------------------------#

from tensorflow.keras.layers import Add

#----------------------Keras Activation Layers' Imports-----------------------#

from tensorflow.keras.layers import Activation

#-------------------------Correlation-TF Layer Imports------------------------#

from Correlation_Layer_TF_TF2 import Correlation_Layer_TF

#--------------------Correlation-Transpose TF Layer Imports-------------------#

from Correlation_Layer_Transpose_TF_TF2 import Correlation_Layer_Transpose_TF

#-------------------------Max-Pooling-TF Layer Imports------------------------#

from Max_Pooling_Layer_TF_TF2 import Max_Pooling_Layer_TF

#---------------------Tensorflow Add-Ons Layers' Imports----------------------#

from tensorflow_addons.layers import WeightNormalization

#%% Function Definition

def Create_ETC_Model_4D_Parameter_Generation(Input_Size,
                                             Start_Neurons,
                                             Number_of_Classes):
    
#%% Define Image Input Layer Parameters
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    print("\nDefining Image Input Layer Parameters...\n")
    
#-----------------------------------------------------------------------------#
    
#------------------------------4D-Input Shape---------------------------------#
    
# =============================================================================
#     Input_Shape=(32,32,5,4,1)   # Multi-Modal Time-Series Images
#     # Input_Shape=(32,32,6,4,1)   # Multi-Modal Time-Series Images
#     # Input_Shape=(32,32,12,4,1)   # Multi-Modal Time-Series Images
#     # Input_Shape=(32,32,18,4,1)   # Multi-Modal Time-Series Images
# =============================================================================
    
    Input_Shape=Input_Size
    
#-----------------------------------------------------------------------------#
    
    print("\nNeural Network Parameters were defined successfully!\n")
    
#%% Define 4-D Convolutional Layer Parameters
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    print("\nDefining 4-D Convolutional Layer Parameters...\n")
    
#-----------------------------------------------------------------------------#
    
#------------------------------4D-Filter Size---------------------------------#
    
    Filter_Size_Convolution_Encoder=(2,2,2,2)   # Height,Width,Depth and Cube of the Filters
    # Filter_Size_Convolution_Encoder=(3,3,3,3)   # Height,Width,Depth and Cube of the Filters
    # Filter_Size_Convolution_Encoder=(4,4,4,4)   # Height,Width,Depth and Cube of the Filters
    
    Filter_Size_Convolution_Last=(1,1,1,1)   # Height,Width,Depth and Cube of the Filters
    
#------------------------------Filters per Layer------------------------------#
    
    Number_of_Filters_Convolution_Encoder_1st_Stack=1*Start_Neurons # Number of Filters
    Number_of_Filters_Convolution_Encoder_2nd_Stack=2*Start_Neurons # Number of Filters
    Number_of_Filters_Convolution_Encoder_3rd_Stack=2*Start_Neurons # Number of Filters
    
    Number_of_Filters_Convolution_Decoder_Reshape=1*Start_Neurons # Number of Filters
    
    Number_of_Filters_Convolution_Last=Number_of_Classes # Number of Filters
    
#-----------4D-Padding Mode to apply to Input Borders of Convolution----------#
    
    Padding_Mode_Convolution_Encoder="same"   # Convolution Subsection
    
# =============================================================================
#     Padding_Mode_Convolution="valid"   # Convolution Subsection
# =============================================================================
    
    if Padding_Mode_Convolution_Encoder=="same":
        # print('\nConvolution Subsection is: SAME\n')
        Convolution_Shape_Encoder=2
    elif Padding_Mode_Convolution_Encoder=="valid":
        # print('\nConvolution Subsection is: VALID\n')
        Convolution_Shape_Encoder=3
    else:
        # print('\nConvolution Subsection is: FULL\n')
        Convolution_Shape_Encoder=1
    
    Padding_Mode_Convolution_Last="same"   # Convolution Subsection
    
# =============================================================================
#     Padding_Mode_Convolution_Last="valid"   # Convolution Subsection
# =============================================================================
    
    if Padding_Mode_Convolution_Last=="same":
        # print('\nConvolution Subsection is: SAME\n')
        Convolution_Shape_Last=2
    elif Padding_Mode_Convolution_Last=="valid":
        # print('\nConvolution Subsection is: VALID\n')
        Convolution_Shape_Last=3
    else:
        # print('\nConvolution Subsection is: FULL\n')
        Convolution_Shape_Last=1
    
#----------------4D-Strides of Convolution along its Input Size---------------#
    
    Strides_Convolution_Encoder=(2,2,1,1)   # Strides of the Convolution along the Height,Width,Depth and Cube
    
    Strides_Convolution_Decoder_Reshape=(1,1,1,1)   # Strides of the Convolution along the Height,Width,Depth and Cube
    
    Strides_Convolution_Last=(1,1,Input_Size[-3],1)   # Strides of the Convolution along the Height,Width,Depth and Cube
    
#-----------------------------------------------------------------------------#
    
    print("\n4-D Convolutional Layer Parameters were defined successfully!\n")
    
#%% Define 4-D Max-Pooling Layer Parameters
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    print("\nDefining 4-D Max-Pooling Layer Parameters...\n")
    
#-----------------------------------------------------------------------------#
    
#---------------------------4D-Max-Pooling Size-------------------------------#
    
    Pool_Size=(2,2,1,1) # Dimensions of Pooling Regions
    
#-----------4D-Padding Mode to apply to Input Borders of Max-Pooling----------#
    
    Padding_Mode_Max_Pooling="same"   # Max-Pooling Subsection
    
# =============================================================================
#     Padding_Mode_Max_Pooling="valid"   # Max-Pooling Subsection
# =============================================================================
    
    if Padding_Mode_Max_Pooling=="same":
        # print('\nMax-Pooling Subsection is: SAME\n')
        Max_Pooling_Shape=2
    elif Padding_Mode_Max_Pooling=="valid":
        # print('\nMax-Pooling Subsection is: VALID\n')
        Max_Pooling_Shape=3
    else:
        # print('\nMax-Pooling Subsection is: FULL\n')
        Max_Pooling_Shape=1
    
#----------------4D-Strides of Max-Pooling along its Input Size---------------#
    
    Strides_Max_Pooling=(2,2,1,1)   # Strides of the Max-Pooling along the Height,Width,Depth and Cube
    
#-----------------------------------------------------------------------------#
    
    print("\n4-D Max-Pooling Layer Parameters were defined successfully!\n")
    
#%% Define 1-D Convolutional Layer Parameters
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    print("\nDefining 1-D Convolutional Layer Parameters...\n")
    
#-----------------------------------------------------------------------------#
    
#------------------------------1D-Filter Size---------------------------------#
    
    Filter_Size_Convolution_TCN=2   # Height of the Filters
    # Filter_Size_Convolution_TCN=3   # Height of the Filters
    # Filter_Size_Convolution_TCN=4   # Height of the Filters
    
    Filter_Size_Convolution_TCN_Extra=1   # Height of the Filters
    
#------------------------------Filters per Layer------------------------------#
    
    Number_of_Filters_Convolution_TCN_1st_Stack=2*Start_Neurons # Number of Filters
    
    # Number_of_Filters_Convolution_TCN_2nd_Stack=49 # Number of Filters
    # Number_of_Filters_Convolution_TCN_2nd_Stack=2*Start_Neurons # Number of Filters
    # Number_of_Filters_Convolution_TCN_2nd_Stack=1*Start_Neurons # Number of Filters
    Number_of_Filters_Convolution_TCN_2nd_Stack=64 # Number of Filters
    
    # Number_of_Filters_Convolution_TCN_3rd_Stack=49 # Number of Filters
    # Number_of_Filters_Convolution_TCN_3rd_Stack=2*Start_Neurons # Number of Filters
    Number_of_Filters_Convolution_TCN_3rd_Stack=2*Start_Neurons # Number of Filters
    
#-----------1D-Padding Mode to apply to Input Borders of Convolution----------#
    
# =============================================================================
#     Padding_Mode_Convolution_TCN="same"   # Convolution Subsection
# =============================================================================
    
# =============================================================================
#     Padding_Mode_Convolution_TCN="valid"   # Convolution Subsection
# =============================================================================
    
    Padding_Mode_Convolution_TCN="causal"   # Convolution Subsection
    
    Padding_Mode_Convolution_TCN_Extra="same"   # Convolution Subsection
    
#-----------------------1D-Dilation Rate of Convolution-----------------------#
    
    Dilation_Rate_TCN_1st_Stack=1   # Dilation Rate to use for Dilated Convolution
    Dilation_Rate_TCN_2nd_Stack=2   # Dilation Rate to use for Dilated Convolution
    Dilation_Rate_TCN_3rd_Stack=4   # Dilation Rate to use for Dilated Convolution
    
#-----------------------------------------------------------------------------#
    
    print("\n1-D Convolutional Layer Parameters were defined successfully!\n")
    
#%% Define 1D Spatial-Dropout Layer Parameters
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    print("\nDefining 1D Spatial-Dropout Layer Parameters...\n")
    
#-----------------------------------------------------------------------------#
    
#------------------------------Dropout Rate-----------------------------------#
    
    Rate=0.3 # Fraction of the Input Units to Drop
    # Rate=0.4 # Fraction of the Input Units to Drop
    # Rate=0.5 # Fraction of the Input Units to Drop
    
#-----------------------------------------------------------------------------#
    
    print("\n1D Spatial-Dropout Layer Parameters were defined successfully!\n")
    
#%% Define 4-D Convolutional-Transpose Layer Parameters
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    print("\nDefining 4-D Convolutional-Transpose Layer Parameters...\n")
    
#-----------------------------------------------------------------------------#
    
#------------------------------4D-Filter Size---------------------------------#
    
    Filter_Size_Convolution_Transpose_Decoder=(2,2,2,2)   # Height,Width,Depth and Cube of the Filters
    # Filter_Size_Convolution_Transpose_Decoder=(3,3,3,3)   # Height,Width,Depth and Cube of the Filters
    # Filter_Size_Convolution_Transpose_Decoder=(4,4,4,4)   # Height,Width,Depth and Cube of the Filters
    
    # Filter_Size_Convolution_Transpose=(2,2,1,1)   # Height,Width,Depth and Cube of the Filters
    
#------------------------------Filters per Layer------------------------------#
    
# =============================================================================
#     Number_of_Filters_Convolution_Transpose_Decoder_1st_Stack=2*Start_Neurons # Number of Filters
#     Number_of_Filters_Convolution_Transpose_Decoder_2nd_Stack=2*Start_Neurons # Number of Filters 
#     Number_of_Filters_Convolution_Transpose_Decoder_3rd_Stack=1*Start_Neurons # Number of Filters
# =============================================================================
    
    Number_of_Filters_Convolution_Transpose_Decoder_1st_Stack=Number_of_Filters_Convolution_Decoder_Reshape # Number of Filters
    Number_of_Filters_Convolution_Transpose_Decoder_2nd_Stack=Number_of_Filters_Convolution_Decoder_Reshape # Number of Filters 
    Number_of_Filters_Convolution_Transpose_Decoder_3rd_Stack=Number_of_Filters_Convolution_Decoder_Reshape # Number of Filters
    
#------4D-Padding Mode to apply to Input Borders of Convolution-Transpose-----#
    
    Padding_Mode_Convolution_Transpose_Decoder="same"   # Convolution-Transpose Subsection
    
# =============================================================================
#     Padding_Mode_Convolution_Transpose_Decoder="valid"   # Convolution-Transpose Subsection
# =============================================================================
    
    if Padding_Mode_Convolution_Transpose_Decoder=="same":
        # print('\nConvolution-Transpose Subsection is: SAME\n')
        Convolution_Transpose_Shape_Decoder=2
    elif Padding_Mode_Convolution_Transpose_Decoder=="valid":
        # print('\nConvolution-Transpose Subsection is: VALID\n')
        Convolution_Transpose_Shape_Decoder=3
    else:
        # print('\nConvolution-Transpose Subsection is: FULL\n')
        Convolution_Transpose_Shape_Decoder=1
    
#-----------4D-Strides of Convolution-Transpose along its Input Size----------#
    
    # Strides_Convolution_Transpose_Decoder=(2,2,2,2)   # Strides of the Convolution-Transpose along the Height,Width,Depth and Cube
    Strides_Convolution_Transpose_Decoder=(2,2,1,1)   # Strides of the Convolution-Transpose along the Height,Width,Depth and Cube
    
#-----------------------------------------------------------------------------#
    
    print("\n4-D Convolutional-Transpose Layer Parameters were defined successfully!\n")
    
#%% Return the required values
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    print('\nReturning the required values...\n')
    
#-----------------------------------------------------------------------------#
    
    return [Input_Shape,
            Filter_Size_Convolution_Encoder,Filter_Size_Convolution_Last,
            Number_of_Filters_Convolution_Encoder_1st_Stack,
            Number_of_Filters_Convolution_Encoder_2nd_Stack,
            Number_of_Filters_Convolution_Encoder_3rd_Stack,
            Number_of_Filters_Convolution_Decoder_Reshape,
            Number_of_Filters_Convolution_Last,
            Convolution_Shape_Encoder,Convolution_Shape_Last,
            Strides_Convolution_Encoder,
            Strides_Convolution_Decoder_Reshape,
            Strides_Convolution_Last,
            Pool_Size,Max_Pooling_Shape,Strides_Max_Pooling,
            Filter_Size_Convolution_TCN,Filter_Size_Convolution_TCN_Extra,
            Number_of_Filters_Convolution_TCN_1st_Stack,
            Number_of_Filters_Convolution_TCN_2nd_Stack,
            Number_of_Filters_Convolution_TCN_3rd_Stack,
            Padding_Mode_Convolution_TCN,Padding_Mode_Convolution_TCN_Extra,
            Dilation_Rate_TCN_1st_Stack,
            Dilation_Rate_TCN_2nd_Stack,
            Dilation_Rate_TCN_3rd_Stack,
            Rate,
            Filter_Size_Convolution_Transpose_Decoder,
            Number_of_Filters_Convolution_Transpose_Decoder_1st_Stack,
            Number_of_Filters_Convolution_Transpose_Decoder_2nd_Stack,
            Number_of_Filters_Convolution_Transpose_Decoder_3rd_Stack,
            Convolution_Transpose_Shape_Decoder,
            Strides_Convolution_Transpose_Decoder]
    
#-----------------------------------------------------------------------------#
    
    print('\nThe required values were returned successfully!\n')
    
#%% Build Neural Network Model with Functional API

# =============================================================================
# print("\014")
# print("\033[H\033[J")
# =============================================================================

def Create_ETC_Model_4D_Functional_API(Input_Size,Start_Neurons,Number_of_Classes):
    
    from Create_ETC_Model_4D_Correlation_TF import Create_ETC_Model_4D_Parameter_Generation
    
    [Input_Shape,
     Filter_Size_Convolution_Encoder,Filter_Size_Convolution_Last,
     Number_of_Filters_Convolution_Encoder_1st_Stack,
     Number_of_Filters_Convolution_Encoder_2nd_Stack,
     Number_of_Filters_Convolution_Encoder_3rd_Stack,
     Number_of_Filters_Convolution_Decoder_Reshape,
     Number_of_Filters_Convolution_Last,
     Convolution_Shape_Encoder,Convolution_Shape_Last,
     Strides_Convolution_Encoder,
     Strides_Convolution_Decoder_Reshape,
     Strides_Convolution_Last,
     Pool_Size,Max_Pooling_Shape,Strides_Max_Pooling,
     Filter_Size_Convolution_TCN,Filter_Size_Convolution_TCN_Extra,
     Number_of_Filters_Convolution_TCN_1st_Stack,
     Number_of_Filters_Convolution_TCN_2nd_Stack,
     Number_of_Filters_Convolution_TCN_3rd_Stack,
     Padding_Mode_Convolution_TCN,Padding_Mode_Convolution_TCN_Extra,
     Dilation_Rate_TCN_1st_Stack,
     Dilation_Rate_TCN_2nd_Stack,
     Dilation_Rate_TCN_3rd_Stack,
     Rate,
     Filter_Size_Convolution_Transpose_Decoder,
     Number_of_Filters_Convolution_Transpose_Decoder_1st_Stack,
     Number_of_Filters_Convolution_Transpose_Decoder_2nd_Stack,
     Number_of_Filters_Convolution_Transpose_Decoder_3rd_Stack,
     Convolution_Transpose_Shape_Decoder,
     Strides_Convolution_Transpose_Decoder]=Create_ETC_Model_4D_Parameter_Generation(Input_Size,
                                                                                     Start_Neurons,
                                                                                     Number_of_Classes)
    
#-----------------------------------------------------------------------------#
    
    print("\nBuilding Neural Network Model with Functional API...\n")
    
#-----------------------------------Input Layer-------------------------------#
    
    Neural_Network_Input=Input(shape=Input_Shape,
                               name="Input")
    
#-------------------------Encoder-1st Stack of Layers-------------------------#
    
    Encoder_1st_Stack=Correlation_Layer_TF(Tensor_Order=4,
                                           Correlation_Subsection=Convolution_Shape_Encoder,
                                           Filter_Size=Filter_Size_Convolution_Encoder,
                                           Number_of_Filters=Number_of_Filters_Convolution_Encoder_1st_Stack,
                                           Correlation_Strides=Strides_Convolution_Encoder,
                                           Weights_Initializer="glorot_uniform",
                                           Bias_Initializer="zeros",
                                           name="Conv_4D_1")(Neural_Network_Input)
    
    Encoder_1st_Stack=Activation('relu',
                                 name="ReLU_1")(Encoder_1st_Stack)
    
    Encoder_1st_Stack=Max_Pooling_Layer_TF(Tensor_Order=4,
                                           Max_Pooling_Subsection=Max_Pooling_Shape,
                                           Max_Pooling_Window_Size=Pool_Size,
                                           Max_Pooling_Strides=Strides_Max_Pooling,
                                           name="Max_Pool_4D_1")(Encoder_1st_Stack)
    
#-------------------------Encoder-2nd Stack of Layers-------------------------#
    
    Encoder_2nd_Stack=Correlation_Layer_TF(Tensor_Order=4,
                                           Correlation_Subsection=Convolution_Shape_Encoder,
                                           Filter_Size=Filter_Size_Convolution_Encoder,
                                           Number_of_Filters=Number_of_Filters_Convolution_Encoder_2nd_Stack,
                                           Correlation_Strides=Strides_Convolution_Encoder,
                                           Weights_Initializer="glorot_uniform",
                                           Bias_Initializer="zeros",
                                           name="Conv_4D_2")(Encoder_1st_Stack)
    
    Encoder_2nd_Stack=Activation('relu',
                                 name="ReLU_2")(Encoder_2nd_Stack)
    
    Encoder_2nd_Stack=Max_Pooling_Layer_TF(Tensor_Order=4,
                                           Max_Pooling_Subsection=Max_Pooling_Shape,
                                           Max_Pooling_Window_Size=Pool_Size,
                                           Max_Pooling_Strides=Strides_Max_Pooling,
                                           name="Max_Pool_4D_2")(Encoder_2nd_Stack)
    
#-------------------------Encoder-3rd Stack of Layers-------------------------#
    
    Encoder_3rd_Stack=Correlation_Layer_TF(Tensor_Order=4,
                                           Correlation_Subsection=Convolution_Shape_Encoder,
                                           Filter_Size=Filter_Size_Convolution_Encoder,
                                           Number_of_Filters=Number_of_Filters_Convolution_Encoder_3rd_Stack,
                                           Correlation_Strides=Strides_Convolution_Encoder,
                                           Weights_Initializer="glorot_uniform",
                                           Bias_Initializer="zeros",
                                           name="Conv_4D_3")(Encoder_2nd_Stack)
    
    Encoder_3rd_Stack=Activation('relu',
                                 name="ReLU_3")(Encoder_3rd_Stack)
    
#----------------------------Reshape Layer-Encoder----------------------------#
    
    Encoder_Reshape_Layer=Reshape(target_shape=(Encoder_3rd_Stack.shape[-2]*Encoder_3rd_Stack.shape[-3],
                                                Encoder_3rd_Stack.shape[-1]),
                                  name="Reshape_Encoder")(Encoder_3rd_Stack)
    
#---------------------------TCN-1st Stack of Layers---------------------------#
    
    TCN_1st_Stack=WeightNormalization(
        Conv1D(filters=Number_of_Filters_Convolution_TCN_1st_Stack,
               kernel_size=Filter_Size_Convolution_TCN,
               padding=Padding_Mode_Convolution_TCN,
               dilation_rate=Dilation_Rate_TCN_1st_Stack,
               name="Conv_1D_1"))(Encoder_Reshape_Layer)
    
    TCN_1st_Stack=Activation('relu',
                             name="ReLU_4")(TCN_1st_Stack)
    
    TCN_1st_Stack=SpatialDropout1D(rate=Rate,
                                   name="Dropout_1")(TCN_1st_Stack)
    
    TCN_1st_Stack=WeightNormalization(
        Conv1D(filters=Number_of_Filters_Convolution_TCN_1st_Stack,
               kernel_size=Filter_Size_Convolution_TCN,
               padding=Padding_Mode_Convolution_TCN,
               dilation_rate=Dilation_Rate_TCN_1st_Stack,
               name="Conv_1D_2"))(TCN_1st_Stack)
    
    TCN_1st_Stack=Activation('relu',
                             name="ReLU_5")(TCN_1st_Stack)
    
    TCN_1st_Stack=SpatialDropout1D(rate=Rate,
                                   name="Dropout_2")(TCN_1st_Stack)
    
    TCN_1st_Stack=Add(name="Add_1")([Encoder_Reshape_Layer,
                                     TCN_1st_Stack])
    
    TCN_1st_Stack=Activation('relu',
                             name="ReLU_6")(TCN_1st_Stack)
    
#---------------------------TCN-2nd Stack of Layers---------------------------#
    
    TCN_2nd_Stack=WeightNormalization(
        Conv1D(filters=Number_of_Filters_Convolution_TCN_2nd_Stack,
               kernel_size=Filter_Size_Convolution_TCN,
               padding=Padding_Mode_Convolution_TCN,
               dilation_rate=Dilation_Rate_TCN_2nd_Stack,
               name="Conv_1D_3"))(TCN_1st_Stack)
    
    TCN_2nd_Stack=Activation('relu',
                             name="ReLU_7")(TCN_2nd_Stack)
    
    TCN_2nd_Stack=SpatialDropout1D(rate=Rate,
                                   name="Dropout_3")(TCN_2nd_Stack)
    
    TCN_2nd_Stack=WeightNormalization(
        Conv1D(filters=Number_of_Filters_Convolution_TCN_2nd_Stack,
               kernel_size=Filter_Size_Convolution_TCN,
               padding=Padding_Mode_Convolution_TCN,
               dilation_rate=Dilation_Rate_TCN_2nd_Stack,
               name="Conv_1D_4"))(TCN_2nd_Stack)
    
    TCN_2nd_Stack=Activation('relu',
                             name="ReLU_8")(TCN_2nd_Stack)
    
    TCN_2nd_Stack=SpatialDropout1D(rate=Rate,
                                   name="Dropout_4")(TCN_2nd_Stack)
    
    TCN_1st_Stack_Extra=Conv1D(filters=Number_of_Filters_Convolution_TCN_2nd_Stack,
                               kernel_size=Filter_Size_Convolution_TCN_Extra,
                               padding=Padding_Mode_Convolution_TCN_Extra,
                               name="Conv_1D_5")(TCN_1st_Stack)
    TCN_2nd_Stack=Add(name="Add_2")([TCN_1st_Stack_Extra,
                                     TCN_2nd_Stack])
    
    TCN_2nd_Stack=Activation('relu',
                             name="ReLU_9")(TCN_2nd_Stack)
    
#---------------------------TCN-3rd Stack of Layers---------------------------#
    
    TCN_3rd_Stack=WeightNormalization(
        Conv1D(filters=Number_of_Filters_Convolution_TCN_2nd_Stack,
               kernel_size=Filter_Size_Convolution_TCN,
               padding=Padding_Mode_Convolution_TCN,
               dilation_rate=Dilation_Rate_TCN_3rd_Stack,
               name="Conv_1D_6"))(TCN_2nd_Stack)
    
    TCN_3rd_Stack=Activation('relu',
                             name="ReLU_10")(TCN_3rd_Stack)
    
    TCN_3rd_Stack=SpatialDropout1D(rate=Rate,
                                   name="Dropout_5")(TCN_3rd_Stack)
    
    TCN_3rd_Stack=WeightNormalization(
        Conv1D(filters=Number_of_Filters_Convolution_TCN_2nd_Stack,
               kernel_size=Filter_Size_Convolution_TCN,
               padding=Padding_Mode_Convolution_TCN,
               dilation_rate=Dilation_Rate_TCN_3rd_Stack,
               name="Conv_1D_7"))(TCN_3rd_Stack)
    
    TCN_3rd_Stack=Activation('relu',
                             name="ReLU_11")(TCN_3rd_Stack)
    
    TCN_3rd_Stack=SpatialDropout1D(rate=Rate,
                                   name="Dropout_6")(TCN_3rd_Stack)
    
    TCN_3rd_Stack=Add(name="Add_3")([TCN_2nd_Stack,
                                     TCN_3rd_Stack])
    
    TCN_3rd_Stack=Activation('relu',
                             name="ReLU_12")(TCN_3rd_Stack)
    
#----------------------------Reshape Layer-Decoder----------------------------#
    
# =============================================================================
#     Decoder_Reshape_Layer=Reshape(target_shape=(7,7,
#                                                 Encoder_3rd_Stack.shape[-3],
#                                                 Encoder_3rd_Stack.shape[-2],
#                                                 1),
#                                   name="Reshape_Decoder")(TCN_3rd_Stack)
# =============================================================================
    
    Decoder_Reshape_Layer=Reshape(target_shape=(8,8,
                                                Encoder_3rd_Stack.shape[-3],
                                                Encoder_3rd_Stack.shape[-2],
                                                1),
                                  name="Reshape_Decoder")(TCN_3rd_Stack)
    
    Decoder_Reshape_Layer=Correlation_Layer_TF(Tensor_Order=4,
                                               Correlation_Subsection=Convolution_Shape_Encoder,
                                               Filter_Size=Filter_Size_Convolution_Encoder,
                                               Number_of_Filters=Number_of_Filters_Convolution_Decoder_Reshape,
                                               Correlation_Strides=Strides_Convolution_Decoder_Reshape,
                                               Weights_Initializer="glorot_uniform",
                                               Bias_Initializer="zeros",
                                               name="Conv_4D_Reshape_Decoder")(Decoder_Reshape_Layer)
    
    Decoder_Reshape_Layer=Activation('relu',
                                     name="ReLU_Reshape_Decoder")(Decoder_Reshape_Layer)
    
#-------------------------Decoder-1st Stack of Layers-------------------------#
    
    Decoder_1st_Stack=Correlation_Layer_Transpose_TF(Tensor_Order=4,
                                                     Correlation_Transpose_Subsection=Convolution_Transpose_Shape_Decoder,
                                                     Filter_Size=Filter_Size_Convolution_Transpose_Decoder,
                                                     Number_of_Filters=Number_of_Filters_Convolution_Transpose_Decoder_1st_Stack,
                                                     Correlation_Transpose_Strides=Strides_Convolution_Transpose_Decoder,
                                                     Weights_Initializer="glorot_uniform",
                                                     Bias_Initializer="zeros",
                                                     name="Conv_4DT_1")(Decoder_Reshape_Layer)
    
    Decoder_1st_Stack=Activation('relu',
                                 name="ReLU_13")(Decoder_1st_Stack)
    
    Decoder_1st_Stack=BatchNormalization(name="BN_1")(Decoder_1st_Stack)
    
#-------------------------Decoder-2nd Stack of Layers-------------------------#
    
    Decoder_2nd_Stack=Correlation_Layer_Transpose_TF(Tensor_Order=4,
                                                     Correlation_Transpose_Subsection=Convolution_Transpose_Shape_Decoder,
                                                     Filter_Size=Filter_Size_Convolution_Transpose_Decoder,
                                                     Number_of_Filters=Number_of_Filters_Convolution_Transpose_Decoder_2nd_Stack,
                                                     Correlation_Transpose_Strides=Strides_Convolution_Transpose_Decoder,
                                                     Weights_Initializer="glorot_uniform",
                                                     Bias_Initializer="zeros",
                                                     name="Conv_4DT_2")(Decoder_1st_Stack)
    
    Decoder_2nd_Stack=Activation('relu',
                                 name="ReLU_14")(Decoder_2nd_Stack)
    
    Decoder_2nd_Stack=BatchNormalization(name="BN_2")(Decoder_2nd_Stack)
    
#-------------------------Decoder-3rd Stack of Layers-------------------------#
    
# =============================================================================
#     Decoder_3rd_Stack=Correlation_Layer_Transpose_TF(Tensor_Order=4,
#                                                      Correlation_Transpose_Subsection=Convolution_Transpose_Shape_Decoder,
#                                                      Filter_Size=Filter_Size_Convolution_Transpose_Decoder,
#                                                      Number_of_Filters=Number_of_Filters_Convolution_Transpose_Decoder_3rd_Stack,
#                                                      Correlation_Transpose_Strides=Strides_Convolution_Transpose_Decoder,
#                                                      Weights_Initializer="glorot_uniform",
#                                                      Bias_Initializer="zeros",
#                                                      name="Conv_4DT_3")(Decoder_2nd_Stack)
# =============================================================================
    
    
#---------------------------------Prediction Layer----------------------------#
    
    Neural_Network_Output=Correlation_Layer_TF(Tensor_Order=4,
                                               Correlation_Subsection=Convolution_Shape_Last,
                                               Filter_Size=Filter_Size_Convolution_Last,
                                               Number_of_Filters=Number_of_Filters_Convolution_Last,
                                               Correlation_Strides=Strides_Convolution_Last,
                                               Weights_Initializer="glorot_uniform",
                                               Bias_Initializer="zeros",
                                               name="Conv_Final")(Decoder_2nd_Stack)
    
    Neural_Network_Output=Activation('sigmoid',
                                     name="Sigmoid_Final")(Neural_Network_Output)
# =============================================================================
#     Neural_Network_Output=Activation('relu',
#                                      name="ReLU_Final")(Neural_Network_Output)
# =============================================================================
    
#--------------------------------Reshape Layer--------------------------------#
    
    Neural_Network_Output=Reshape(target_shape=Neural_Network_Output.shape[1:3]+Neural_Network_Output.shape[-2],
                                  name="Reshape_Final")(Neural_Network_Output)
    
#--------------------------------Model Creation-------------------------------#
    
    Model_Architecture=Model(inputs=Neural_Network_Input,
                             outputs=Neural_Network_Output,
                             name="ETC_Model_4D")
    
#-----------------------------------------------------------------------------#
    
    print("\nNeural Network Model was successfully built with Functional API!\n")
    
#-----------------------------------------------------------------------------#
    
    return Model_Architecture