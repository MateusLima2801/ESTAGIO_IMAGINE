from PIL import Image
import math

pixelation_factor = 7  # Adjust this value to control pixel size
img = Image.open('img1.jpeg')
# Calculate the new dimensions (smaller)
new_width = math.floor(img.width / pixelation_factor)
new_height = math.floor(img.height / pixelation_factor)

# Resize the image to the smaller dimensions
smaller_img = img.resize((new_width, new_height), resample=Image.BILINEAR)

# Save or display the pixelated image
smaller_img.save('pixelated_img.jpeg')
smaller_img.show()