import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('C:\Image_Processing_2\Crop_field_cropped.jpg', cv2.IMREAD_COLOR)
if img is None:
    print("Error: Could not open or read the image file.")
    exit()
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_img, 550, 690)
indices = np.nonzero(edges)
y, x = indices  
coefficients = np.polyfit(x, y, 1)
slope = coefficients[0]
intercept = coefficients[1]
fit_line_y = slope * x + intercept
angle_degrees = np.arctan(slope) * 180 / np.pi
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img[:,:,::-1])  
plt.title('Original Image')
plt.subplot(122)
plt.imshow(edges, cmap='gray')
plt.title('Edges')
plt.show()
plt.figure()
plt.scatter(x, y, marker='.', color='b')
plt.plot(x, fit_line_y, color='r', linewidth=2)  
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of Edge Points with Least Squares Fit Line')
plt.gca().invert_yaxis()  
plt.show()
print("Estimated crop field angle (in degrees):", angle_degrees)
