#%% Function Definition

def Compute_Transpose_Convolution_Output_Length(dim_size,
                                                stride_size,kernel_size,
                                                padding,output_padding,
                                                dilation=1):
    
    """Determines output length of a transposed convolution given input length.
    # Arguments
        dim_size: Integer, the input length.
        stride_size: Integer, the stride along the dimension of `dim_size`.
        kernel_size: Integer, the kernel size along the dimension of `dim_size`.
        padding: One of `"same"`, `"valid"`, `"full"`.
        output_padding: Integer, amount of padding along the output dimension, can be set to `None` in which case the output length is inferred.
        dilation: dilation rate, integer.
    # Returns
        The output length (integer).
    """
    
#%% Assert that the Padding argument has allowable value
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    # print('\nAsserting that the Padding argument has allowable value...\n')
    
#-----------------------------------------------------------------------------#
    
    assert padding in {"full","same","valid"},"Padding argument MUST be one of the following: full-same-valid"
    
#-----------------------------------------------------------------------------#
    
    # print('\nThe Padding argument assertion of having allowable value was performed successfully!\n')
    
#%% Assert that the Input-Length argument has allowable value
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    # print('\nAsserting that the Input-Length argument has allowable value...\n')
    
#-----------------------------------------------------------------------------#
    
    if dim_size is None:
        return None
    
#-----------------------------------------------------------------------------#
    
    # print('\nThe Input-Length argument assertion of having allowable value was performed successfully!\n')
    
#%% Get the Dilated Kernel Size
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    # print('\nGetting the Dilated Kernel Size...\n')
    
#-----------------------------------------------------------------------------#
    
    kernel_size=kernel_size+(kernel_size-1)*(dilation-1)
    
#-----------------------------------------------------------------------------#
    
    # print('\nThe Dilated Kernel Size was get successfully!\n')
    
#%% Infer Length if Output-Padding is None,else compute the Exact Length
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    # print('\nInferring Length if Output-Padding is None,else computing the Exact Length...\n')
    
#-----------------------------------------------------------------------------#
    
    if output_padding is None:
        if padding=="full":
            dim_size=dim_size*stride_size-(stride_size+kernel_size-2)
        elif padding=="same":
            dim_size=dim_size*stride_size
        elif padding=="valid":
            dim_size=dim_size*stride_size+max(kernel_size-stride_size,0)
    else:
        if padding=="full":
            pad=kernel_size-1
        elif padding=="same":
            pad=kernel_size//2
        elif padding=="valid":
            pad=0
        
        dim_size=((dim_size-1)*stride_size+kernel_size-2*pad+output_padding)
    
#-----------------------------------------------------------------------------#
    
    # print('\nIf Output-Padding is None the Length was inferred successfully,else the Exact Length was computed successfully!\n')
    
#%% Return the required values
    
# =============================================================================
#     print("\014")
#     print("\033[H\033[J")
# =============================================================================
    
#-----------------------------------------------------------------------------#
    
    # print('\nReturning the required values...\n')
    
#-----------------------------------------------------------------------------#
    
    return dim_size

#-----------------------------------------------------------------------------#
    
    # print('\nThe required values were returned successfully!\n')