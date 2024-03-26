import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('C:\Image_Processing_2\Crop_field_cropped.jpg', cv2.IMREAD_COLOR)
if img is None:
    print("Error: Could not open or read the image file.")
    exit()
edges = cv2.Canny(img, 550, 690)
indices = np.where(edges != [0])
x = indices[1]
y = -indices[0]
y, x = indices 
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.subplot(122)
plt.imshow(edges, cmap='gray')
plt.title('Edges')
plt.show()
plt.figure()
plt.scatter(x, y, marker='.', color='b')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of Edge Points')
plt.gca().invert_yaxis()  
plt.show()
coefficients = np.polyfit(x, y, 1)
slope = coefficients[0]
angle_degrees = np.arctan(slope) * 180 / np.pi
print("Estimated crop field angle (in degrees):", angle_degrees)

