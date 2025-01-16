#Required Libraries
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np

# Load the image using PIL
img = Image.open('images/cat.jpeg')

# Smooth effect using GaussianBlur
smooth_img = img.filter(ImageFilter.GaussianBlur(radius=100))

# Edge enhancement with EDGE_ENHANCE_MORE
edge_img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)

# Convert to numpy arrays
smooth_array = np.array(smooth_img)
edge_array = np.array(edge_img)

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Left subplot
ax1.imshow(smooth_array)
ax1.set_title('Gaussian Smoothing')
ax1.axis('off')

# Right subplot
ax2.imshow(edge_array)
ax2.set_title('Edge Enhancement')
ax2.axis('off')

# Adjust spacing between subplots
plt.tight_layout()
plt.show()