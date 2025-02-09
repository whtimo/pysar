import numpy as np

def goldstein_filter(interferogram:np.ndarray, alpha=0.8, fft_size=32) -> np.ndarray:
    """
    Apply the Goldstein filter to an InSAR interferogram to reduce phase noise.

    Is that even a Goldstein filter? It was developed by DeepSeek-R1 and I think it is something else

    Parameters:
    interferogram (np.ndarray): Complex 2D array representing the interferogram.
    alpha (float): Filter parameter (0 to 1), higher values increase filtering strength.
    fft_size (int): Size of the blocks to process (must be a power of two for FFT efficiency).

    Returns:
    np.ndarray: Filtered complex interferogram.
    """
    rows, cols = interferogram.shape

    # Calculate padding to make dimensions multiples of fft_size
    pad_rows = (fft_size - (rows % fft_size)) % fft_size
    pad_cols = (fft_size - (cols % fft_size)) % fft_size

    # Pad the interferogram using reflect mode to minimize edge artifacts
    padded = np.pad(interferogram, ((0, pad_rows), (0, pad_cols)), mode='reflect')
    padded_rows, padded_cols = padded.shape

    # Initialize filtered array with complex dtype
    filtered = np.zeros_like(padded, dtype=np.complex128)

    # Process each block
    for i in range(0, padded_rows, fft_size):
        print(f'Processing row {i}/{rows}')
        for j in range(0, padded_cols, fft_size):
            # Extract the current block
            block = padded[i:i + fft_size, j:j + fft_size]

            # Compute 2D FFT
            fft_block = np.fft.fft2(block)

            # Calculate magnitude spectrum
            magnitude = np.abs(fft_block)

            # Handle zero magnitude case to avoid division by zero
            max_mag = np.max(magnitude)
            if max_mag == 0:
                max_mag = 1e-10

            # Normalize magnitude and apply Goldstein filter
            normalized = magnitude / max_mag
            filt = normalized ** alpha

            # Filter the FFT coefficients and compute inverse FFT
            filtered_fft = fft_block * filt
            ifft_block = np.fft.ifft2(filtered_fft)

            # Store the result (complex)
            filtered[i:i + fft_size, j:j + fft_size] = ifft_block

    # Crop to the original dimensions
    return filtered[:rows, :cols]