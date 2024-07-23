#%% All Required Imports

#-------------------------Keras' General Imports------------------------------#

# import tensorflow as tf

# import keras
# from tensorflow import keras

#----------------------Keras Regression Metrics' Imports----------------------#

from tensorflow.keras.metrics import MeanSquaredError 
from tensorflow.keras.metrics import RootMeanSquaredError

from tensorflow.keras.metrics import MeanAbsoluteError

#------------------Keras Super-Resolution Metrics' Imports--------------------#

from Create_Customized_Metrics_PSNR_SSIM_TF2 import SSIM_TF

from Create_Customized_Metrics_PSNR_SSIM_TF2 import PSNR_TF

#---------------------Keras Correlation Metrics' Imports----------------------#

from Create_Customized_Metrics_PLCC_TF2 import PLCC_TF

#%% Function Definition

def Create_Customized_Metrics_Regression(Number_of_Metrics_to_Compute):
    
#%% Customize Regression Metrics Options
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    print("\nCustomizing Regression Metrics Options...\n")
    
#-----------------------------------------------------------------------------#
    
#-----------------------------Regression Metrics------------------------------#
    
    MSE_Metric=MeanSquaredError(name="MSE")
    RMSE_Metric=RootMeanSquaredError(name="RMSE")
    
    MAE_Metric=MeanAbsoluteError(name="MAE")
    
#-----------------------------------------------------------------------------#
    
    print("\nRegression Metrics Options were customized successfully!\n")
    
#%% Customize Super-Resolution Metrics Options
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    print("\nCustomizing Super-Resolution Metrics Options...\n")
    
#-----------------------------------------------------------------------------#
    
#--------------------------Super-Resolution Metrics---------------------------#
    
    SSIM_Metric=SSIM_TF
    
    PSNR_Metric=PSNR_TF
    
#-----------------------------------------------------------------------------#
    
    print("\nSuper-Resolution Metrics Options were customized successfully!\n")
    
#%% Customize Correlation Metrics Options
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    print("\nCustomizing Correlation Metrics Options...\n")
    
#-----------------------------------------------------------------------------#
    
    PLCC_Metric=PLCC_TF
    
#-----------------------------------------------------------------------------#
    
    print("\nCorrelation Metrics Options were customized successfully!\n")
    
#%% Create Customized Regression Metrics
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    print("\nCreating Customized Regression Metrics...\n")
    
#-----------------------------------------------------------------------------#
    
    if Number_of_Metrics_to_Compute==1:
        Customized_Metrics=[MSE_Metric]
    elif Number_of_Metrics_to_Compute==2:
        Customized_Metrics=[MSE_Metric,RMSE_Metric]
    elif Number_of_Metrics_to_Compute==6:
        Customized_Metrics=[MSE_Metric,RMSE_Metric,
                            MAE_Metric]
    elif Number_of_Metrics_to_Compute==7:
        # Customized_Metrics=[MSE_Metric,RMSE_Metric,
        #                     MAE_Metric,
        #                     SSIM_Metric,PSNR_Metric]
        Customized_Metrics=[MSE_Metric,RMSE_Metric,
                            MAE_Metric,
                            SSIM_Metric,PSNR_Metric,
                            PLCC_Metric]
    else:
        pass
    
#-----------------------------------------------------------------------------#
    
    print("\nCustomized Regression Metrics were created successfully!\n")
    
#%% Return the required values
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    print('\nReturning the required values...\n')
    
#-----------------------------------------------------------------------------#
    
    return Customized_Metrics
    
#-----------------------------------------------------------------------------#
    
    print('\nThe required values were returned successfully!\n')