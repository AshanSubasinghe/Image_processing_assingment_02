import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, Data, RealData
img = cv2.imread('C:\Image_Processing_2\Crop_field_cropped.jpg', cv2.IMREAD_COLOR)
if img is None:
    print("Error: Could not open or read the image file.")
    exit()
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_img, 550, 690)
indices = np.nonzero(edges)
y, x = indices  
x = np.array(x, dtype=float)
y = np.array(y, dtype=float)
def tls_model(B, x):
    return B[0] * x + B[1]
data = RealData(x, y)
model = Model(tls_model)
odr = ODR(data, model, beta0=[1.0, 0.0])
tls_result = odr.run()
tls_slope = tls_result.beta[0]
tls_intercept = tls_result.beta[1]
tls_fit_line_y = tls_slope * x + tls_intercept
tls_angle_degrees = np.arctan(tls_slope) * 180 / np.pi
plt.figure()
plt.scatter(x, y, marker='.', color='b')
plt.plot(x, tls_fit_line_y, color='r', linewidth=2) 
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of Edge Points with Total Least Squares Fit Line')
plt.gca().invert_yaxis()  
plt.show()

print("Estimated crop field angle (in degrees) using Total Least Squares Fit:", tls_angle_degrees)
