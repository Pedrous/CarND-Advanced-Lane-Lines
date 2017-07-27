import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

def Calibrate(img, objpoints, imgpoints):
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None)
	return mtx, dist

# A pipeline for the images
dist_pickle = pickle.load( open( "dist.p", "rb" ) )
objpoints = dist_pickle[0]
imgpoints = dist_pickle[1]

# Acquire the filepaths for the calibration images
cal_paths = glob.glob('camera_cal/*.jpg')
cal_paths.sort()
#print(cal_paths)

# Calibrate the camera
mtx, dist = Calibrate(cv2.imread('camera_cal/calibration1.jpg'), objpoints, imgpoints)

# Obtain the filepaths for the test images
test_paths = glob.glob('test_images/straight_lines1.jpg')

# Plot the original test images	
f, ax = plt.subplots(1, 1, figsize=(24, 10))
#for idx in ax:
ax.imshow(cv2.cvtColor(cv2.imread(test_paths.pop(0)), cv2.COLOR_BGR2RGB))	
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

plt.show()
