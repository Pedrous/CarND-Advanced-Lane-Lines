import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

def Calibrate(img, objpoints, imgpoints):
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None)
	return mtx, dist


def Undistort(img, mtx, dist):
	return cv2.undistort(img, mtx, dist, None, mtx)
	

def Warp(img, inv=False):
	img_size = (img.shape[1], img.shape[0])
	src = np.float32(
		[[ 685, 451],
		 [1042, 677],
		 [ 267, 677],
		 [ 594, 451]])
	dst = np.float32(
		[[956,   0],
		 [956, 719],
		 [350, 719],
		 [350,   0]])
	if inv:
		Minv = cv2.getPerspectiveTransform(dst, src)
		warped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
	else:
		M = cv2.getPerspectiveTransform(src, dst)
		warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
		
	
	return warped
	

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output
    

def Grayscale(img, thresh=(200, 255)):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray_binary = np.zeros_like(gray)
	gray_binary[(gray >= thresh[0]) & (gray <= thresh[1])] = 1
    
	return gray_binary
    
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(20, 100)):
	# Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    grad_binary = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return grad_binary


def mag_thresh(img, sobel_kernel=3, thresh=(30, 100)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    
    return mag_binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    
    return dir_binary

    
def hls_select(img, thresh=(0, 255), channel=2): 
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # Apply a threshold to the S channel
    Chn = hls[...,channel] #default S channel
    binary_output = np.zeros_like(Chn)
    binary_output[(Chn > thresh[0]) & (Chn <= thresh[1])] = 1
    
    return binary_output
    
    
def find_window_centroids(warped, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
	    # convolve the window into the vertical slice of the image
	    image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
	    conv_signal = np.convolve(window, image_layer)
	    # Find the best left centroid by using past left center as a reference
	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
	    offset = window_width/2
	    l_min_index = int(max(l_center+offset-margin,0))
	    l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
	    # Find the best right centroid by using past right center as a reference
	    r_min_index = int(max(r_center+offset-margin,0))
	    r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
	    # Add what we found for that layer
	    window_centroids.append((l_center,r_center))

    return window_centroids
    
	
def sliding_window(warped):
	# window settings
	window_width = 50 
	window_height = 80 # Break image into 9 vertical layers since image height is 720
	margin = 100 # How much to slide left and right for searching
	
	window_centroids = find_window_centroids(warped, window_width, window_height, margin)

	# If we found any window centers
	if len(window_centroids) > 0:

		# Points used to draw all the left and right windows
		l_points = np.zeros_like(warped)
		r_points = np.zeros_like(warped)

		# Go through each level and draw the windows 	
		for level in range(0,len(window_centroids)):
		    # Window_mask is a function to draw window areas
			l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
			r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
			# Add graphic points from window mask here to total pixels found 
			l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
			r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

		# Draw the results
		template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
		zero_channel = np.zeros_like(template) # create a zero color channel
		template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
		warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
		output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
	 
	# If no window centers found, just display orginal road image
	else:
		output = np.array(cv2.merge((warped,warped,warped)),np.uint8)
	
	return output
	
	
def sliding_window2(binary_warped):
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Draw the windows on the visualization image
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
		    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
		    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	
	# Generate x and y values for plotting
	ploty = np.int_(np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0]))
	left_fitx = np.int_(left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2])
	right_fitx = np.int_(right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2])

	'''
	y_eval = np.max(ploty)
	left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
	offset = binary_warped.shape[1] - left_fitx[-1] - right_fitx[-1]
	rad = min(abs(1000 - left_curverad), abs(1000 - right_curverad))
	'''
	
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension

	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
	# Calculate the new radii of curvature
	y_eval = np.max(ploty)
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	if abs(1000 - left_curverad) < abs(1000 - right_curverad):
		rad = int(left_curverad)
	else:
		rad = int(right_curverad)
	# calculate the offset
	offset = round(-(binary_warped.shape[1] - left_fitx[-1] - right_fitx[-1])*xm_per_pix, 2)
	
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	
	# Image to return
	out_img2 = np.zeros_like(out_img)
	for p, l, r in zip(ploty, left_fitx, right_fitx):
		out_img2[p,l:r] = [0, 50, 0]
	
	'''
	image = plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	'''
	
	return [out_img, out_img2], [rad, offset]
    
    
def Combined(imgs):
	comb = []
	comb.append(abs_sobel_thresh(imgs, orient='x', sobel_kernel=5, thresh=(30, 255)))
	comb.append(abs_sobel_thresh(imgs, orient='y', sobel_kernel=5, thresh=(30, 255)))
	comb.append(mag_thresh(imgs, sobel_kernel=3, thresh=(20, 255)))
	comb.append(Grayscale(imgs, thresh=(150, 255)))
	#comb.append(dir_threshold(imgs, sobel_kernel=3, thresh=(3*np.pi/8, 4*np.pi/8)))
	comb.append(hls_select(imgs, (15,100), 0))
	comb.append(hls_select(imgs, (100,245), 1))
	comb.append(hls_select(imgs, (130,255), 2))
	model = np.zeros_like(comb[-1])
	model[(comb[0] == 1) & (comb[1] == 1)] = 1
	comb.append(model)
	model = np.zeros_like(comb[-1])
	model[(comb[0] == 1) & (comb[3] == 1)] = 1
	comb.append(model)
	model = np.zeros_like(comb[-1])
	model[(comb[2] == 1) & (comb[5] == 1)] = 1
	comb.append(model)
	model = np.zeros_like(comb[-1])
	model[(comb[5] == 1) & (comb[6] == 1)] = 1
	comb.append(model)
	model = np.zeros_like(comb[-1])
	model[((comb[0] == 1) & (comb[1] == 1)) | (((comb[0] == 1) & (comb[3] == 1)) | ((comb[5] == 1) & (comb[6] == 1)))] = 1
	comb.append(model)
	
	return comb
    
    
def Pipeline(img, mtx, dist):
	pipe = []
	pipe.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	pipe.append(Undistort(pipe[-1], mtx, dist))
	pipe.append(cv2.GaussianBlur(pipe[-1], (5, 5), 0))
	comb = Combined(pipe[-1])
	pipe.extend(comb)
	'''
	pipe.append(comb[0])
	pipe.append(comb[1])
	pipe.append(comb[2])
	pipe.append(comb[3])
	pipe.append(comb[4])
	pipe.append(comb[5])
	pipe.append(comb[6])
	pipe.append(comb[7])
	pipe.append(comb[8])
	pipe.append(comb[9])
	pipe.append(comb[10])
	pipe.append(comb[11])
	'''
	pipe.append(Warp(pipe[-1]))
	lane, values = sliding_window2(pipe[-1])
	pipe.extend(lane)
	pipe.append(Warp(pipe[-1], inv=True))
	pipe.append(cv2.addWeighted(pipe[0], 1., pipe[-1], 0.9, 0.))
	pipe.append(np.zeros((720,1280,3)))
	out = pipe[-2]
	
	extras = ['Original', 'Undistorted', 'Gaussian Filtered', 'Sobel X', 'Sobel Y', 'Sobel Magnitude', 'Thresholded GrayScale', 'H channel', 'L channel', 'S channel', 'Sobel X & Y combined', 'Magnitude & Grayscale', 'L channel & Magnitude', 'L & S channels']
	titles = ['Combined Edge Detection', 'Warped Image', 'Sliding window', 'Curvature Fit', 'Unwarped Lane', 'Lane Prediction']
	extras.extend(titles)
	titles = extras
	f, ax = plt.subplots(4, 5, figsize=(24, 10))
	string1 = 'Curve Radius: {}(m)'.format(values[0])
	string2 = 'Vehicle is {}(m) left of center'.format(values[1])
	#print(string)
	for row in ax:
		for col in row:
			col.imshow(pipe.pop(0), cmap='gray')
			col.set_title(titles.pop(0), fontsize=30)	
			col.text(10, 60, string1, color='white', fontsize=15)
			col.text(10, 120, string2, color='white', fontsize=15)
	plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
	
	return out
	

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

'''
# Obtain the original and undistorted images
original = []
undistorted = []
for filepath in cal_paths:
	original.append(cv2.imread(filepath))
	undistorted.append(Undistort(original[-1], mtx, dist))

# Plot the both image sets into two figures
f, ax = plt.subplots(1, 2, figsize=(24, 10))
ax[0].imshow(original[15], cmap='gray')
ax[1].imshow(undistorted[15], cmap='gray')
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
'''

'''
# Plot the both image sets into two figures
f, ax = plt.subplots(1, 5, figsize=(24, 10))

for row in ax:
	for col in row:
		col.imshow(undistorted.pop(0), cmap='gray')
		col.set_title(cal_paths.pop(0), fontsize=30)
		
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

f2, ax2 = plt.subplots(4, 5, figsize=(24, 10))
for row in ax2:
	for col in row:
		col.imshow(original.pop(0), cmap='gray')
		col.set_title(cal_paths.pop(0), fontsize=30)
		
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
'''

# Obtain the filepaths for the test images
test_paths = glob.glob('test_images/*.jpg')
original = []
pipelined = []

for filepath in test_paths:
	original.append(cv2.imread(filepath))
	pipelined.append(Pipeline(original[-1], mtx, dist))

# Realize the plots
plt.show()
