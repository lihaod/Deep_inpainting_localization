import numpy as np

def bilinear_upsample_weights(factor, out_channels, in_channels):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    filter_size = 2 * factor - factor % 2
    center = (factor - 1) if filter_size % 2 == 1 else factor - 0.5
    og = np.ogrid[:filter_size, :filter_size]
    upsample_kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weights = np.tile(upsample_kernel[:,:,np.newaxis,np.newaxis],(1,1,out_channels,in_channels))

    return weights 