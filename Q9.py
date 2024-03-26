import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
img = cv2.imread('C:\Image_Processing_2\Crop_field_cropped.jpg', cv2.IMREAD_COLOR)
if img is None:
    print("Error: Could not open or read the image file.")
    exit()
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_img, 550, 690)
indices = np.nonzero(edges)
y, x = indices  
ransac = RANSACRegressor()
ransac.fit(x.reshape(-1, 1), y)
ransac_slope = ransac.estimator_.coef_[0]
ransac_intercept = ransac.estimator_.intercept_
ransac_fit_line_y = ransac.predict(x.reshape(-1, 1))
ransac_angle_degrees = np.arctan(ransac_slope) * 180 / np.pi
plt.figure()
plt.scatter(x, y, marker='.', color='b')
plt.plot(x, ransac_fit_line_y, color='r', linewidth=2)  
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of Edge Points with RANSAC Fit Line')
plt.gca().invert_yaxis()  
plt.show()
print("Estimated crop field angle (in degrees) using RANSAC Fit:", ransac_angle_degrees)

