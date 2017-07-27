import pickle
import cv2
import numpy as np
import glob

nx = 9 # amount of corners in x direction
ny = 6 # amount of corners in y direction
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

filepaths = glob.glob('camera_cal/*.jpg')

for filepath in filepaths:
	img = cv2.imread(filename)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Find the corners
	ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

	if ret == True:
		imgpoints.append(corners)
		objpoints.append(objp)
		
		# Draw and display the corners
		cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
		cv2.imshow('img',img)
		cv2.waitKey(500)

dist_pickle = [objpoints, imgpoints]

pickle.dump( dist_pickle, open( "dist.p", "wb" ) )
