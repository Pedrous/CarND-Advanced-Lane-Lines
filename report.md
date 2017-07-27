**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration.png "Undistorted"
[image2]: ./output_images/figure_1.png "Test image 5"
[image3]: ./output_images/figure_2.png "Test image 1"
[image4]: ./output_images/figure_3.png "Test image straight lines 2"
[image5]: ./output_images/figure_4.png "Test image 3"
[image6]: ./output_images/figure_5.png "Test image straight lines 1"
[image7]: ./output_images/figure_6.png "Test image 4"
[image8]: ./output_images/figure_7.png "Test image 6"
[image9]: ./output_images/figure_8.png "Test image 2"
[video1]: ./test_videos_output/project.mp4 "Video"

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

First I extracted the corner coordinates from the calibration images in `dist.py` and saved them to `dist.p` with pickle together with the object points. I did this so that I don't need to extract the points each time that I run my main code, because it is time consuming. 

Then in `findlane2.py` I loaded the `objpoints` which have the (x, y, z) coordinates of the chessboard corners in real world and the `imgpoints` which have the pixel coordinates of the chessboard coordinates. After this I first calibrated the camera using the `cv2.calibrateCamera()` function and then undistorted the image by using the `cv2.undistort()` function, this is done in line 420. Example of `calibration5.jpg` is shown below:

![alt text][image1]

### Pipeline (single images)

The steps that we're taken to process the test images, are shown in the end of this chapter. Each step is shown for each test image.

#### 1. Provide an example of a distortion-corrected image.

The undistortion of the images takes place in the row 392 inside the `Pipeline()` function. basically it is done by calling the `Undistort()` function in the `Pipeline()` function. The `Undistort()` only performs the `cv2.undistort()` operation.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 372 through 385 in `findline2.py`). These rows take advantage of the functions such than `Grayscale()`, `abs_sobel_thresh()`, `mag_thresh()`, `hsl_select()`. Ultimately as my combined threshold considers the pixels which were found together by the X and Y gradient or X gradient and Magnitude gradient or L and S channels.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `Warp()`, which appears in lines 45 through 64 in the file `findlane2.py` (./findlane2.py).  The `Warp()` function takes as inputs an image (`img`), as well as the information if a the transformations is an inverse or not.  I chose to hardcode the source and destination points inside the function the following manner (The points were measured by looking from the image with a sample of straight road):

```python
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
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 685, 451      | 956, 0        | 
| 1042, 677     | 956, 719      |
| 267, 677      | 350, 719      |
| 594, 451      | 350, 0        |

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then in the `sliding_window2()` function I first calculated the histogram of the lower half of the image to find where the lane lines are starting from and then used the sliding window technique to find the pixels corresponding to the lane lines. After that the 2nd order polynomials were used to fit a model for the both lines. This takes place in the code between lines 231 - 316.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 323 through 342 in my code in `findlane2.py`. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `Pipeline()` function in lines 397 through 398 in my code in `findlane2.py` and it is taking advantage of the `Warp()` function to do the inverse perspective transformation for the found lane and the `cv2.addWeighted()` function after that to overlay the found lane in green with the undistorted image.  Here are examples of my results on all test images:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_videos_output/project.mp4)

I think it is pretty good solution, there is just small wobbling during the lighter parts of the road but it is not catastrophic.
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most crucial part here is the to detect the lane lines correctly but in a continuous video bad detection can be boosted a little bit by the fact that the previous prediction uses the last measured curve as a starting point for the next prediction to make the process more robust. 

It was relatively hard to detect the lane lines and avoid the unwanted noise and to find suitable threshold levels for each detection and combine them. So for example in the challenge videos my edge detection fails quite catastrophically since it detects the sharp edge of the shadow in the left side and fails to detect the lines against the light road. So this is where my pipeline will fail.

Some improvements could be to use some intelligent masking of the image before the detection. Also it would be interesting to have use some algorithm or machine learning to find the optimal paraters for the detection thresholds so that would make the job quite much easier. Also I could have improved how the previous measurements are used to increase the possibility of correct prediction for the current image.
