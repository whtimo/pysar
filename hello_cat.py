#Required Libraries
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load the image using PIL
img = Image.open('images/cat.jpeg')

# Convert to numpy array and display
img_array = np.array(img)
print(f'Image shape: {img_array.shape}')

plt.figure(figsize=(8, 10))
plt.imshow(img_array)
plt.show()