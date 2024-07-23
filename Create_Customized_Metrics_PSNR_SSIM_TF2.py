#%% All Required Imports

#-------------------------Keras' General Imports------------------------------#

import tensorflow as tf

# import keras
# from tensorflow import keras

#----------------------Tensorflow Image Metrics' Imports----------------------#

from tensorflow import image

#%% Function Definition

def SSIM_TF(y_true,y_pred):
    
#%% Customize SSIM-Metric via Tensorflow Metrics Options
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    # print("\nCustomizing SSIM-Metric via Tensorflow Metrics Options...\n")
    
#-----------------------------------------------------------------------------#
    
    Images_Dynamic_Range=1
    # Images_Dynamic_Range=2
    # Images_Dynamic_Range=255
    
#-----------------------------------------------------------------------------#
    
    Gaussian_Filter_Size=11
    
#-----------------------------------------------------------------------------#
    
    Gaussian_Filter_Width=1.5
    
#-----------------------------------------------------------------------------#
    
    Kappa_1=0.01
    
#-----------------------------------------------------------------------------#
    
    Kappa_2=0.03
    
#-----------------------------------------------------------------------------#
    
    # print("\nSSIM-Metric via Tensorflow Metrics Options were customized successfully!\n")
    
#%% Create Customized SSIM-Metric via Tensorflow Metrics
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    # print("\nCreating Customized SSIM-Metric via Tensorflow Metrics...\n")
    
#-----------------------------------------------------------------------------#
    
    SSIM_Metric_Tensorflow=tf.reduce_mean(image.ssim(y_true,y_pred,
                                                     max_val=Images_Dynamic_Range,
                                                     filter_size=Gaussian_Filter_Size,
                                                     filter_sigma=Gaussian_Filter_Width,
                                                     k1=Kappa_1,
                                                     k2=Kappa_2))
    
#-----------------------------------------------------------------------------#
    
    # print("\nCustomized SSIM-Metric via Tensorflow Metrics was created successfully!\n")
    
#-----------------------------------------------------------------------------#
    
#%% Return the required values
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    # print('\nReturning the required values...\n')
    
#-----------------------------------------------------------------------------#
    
    return SSIM_Metric_Tensorflow
    
#-----------------------------------------------------------------------------#
    
    # print('\nThe required values were returned successfully!\n')

#%% Function Definition

def PSNR_TF(y_true,y_pred):
    
#%% Customize PSNR-Metric via Tensorflow Metrics Options
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    # print("\nCustomizing PSNR-Metric via Tensorflow Metrics Options...\n")
    
#-----------------------------------------------------------------------------#
    
    Images_Dynamic_Range=1
    # Images_Dynamic_Range=2
    # Images_Dynamic_Range=255
    
#-----------------------------------------------------------------------------#
    
    # print("\nPSNR-Metric via Tensorflow Metrics Options were customized successfully!\n")
    
#%% Create Customized PSNR-Metric via Tensorflow Metrics
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    # print("\nCreating Customized PSNR-Metric via Tensorflow Metrics...\n")
    
#-----------------------------------------------------------------------------#
    
    PSNR_Metric_Tensorflow=tf.reduce_mean(image.psnr(y_true,y_pred,
                                                     max_val=Images_Dynamic_Range))
    
#-----------------------------------------------------------------------------#
    
    # print("\nCustomized PSNR-Metric via Tensorflow Metrics was created successfully!\n")
    
#-----------------------------------------------------------------------------#
    
#%% Return the required values
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    # print('\nReturning the required values...\n')
    
#-----------------------------------------------------------------------------#
    
    return PSNR_Metric_Tensorflow
    
#-----------------------------------------------------------------------------#
    
    # print('\nThe required values were returned successfully!\n')