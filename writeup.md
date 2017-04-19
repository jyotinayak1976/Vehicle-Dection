# Vehicle-Detection
## These are the following steps taken

### 1. Manual Detection of vehicle using bounding rectangles

I followed the tutorial and created a function called 'draw_boxes' which takes pixel position as imput and created bounding boxed around the identified vehicles in the image. I have shown an example by taking an image test6 from test folder.

### 2. Different color space exploration

I have explored different color spaces for catr and non car images to identify distinguishable differences. I have taken the sample images from udacity only. It does not seem to provide any noticeable difference.

### 3. HOG computation

I defined a function `get_hog_features` which uses the OpenCV function `hogDescriptor()` to get Histogram of Oriented features from an image. The functions computes the HOG features from each of the 3 color channels in an image and returns them as a single feature vector. 

### 4. HOG feature extraction and build training dataset

I defined a function `extract_features` which calls the function `get_hog_features` and appends the HOG features from each image to a dataset of training features. After building the datsets of car and noncar images and labelling them appropriately I normalized the features to have zero mean and unit variance using the Scikit-learn function `StandardScaler`, and used `train_test_split`to split them in to training and testing datasets. 

After testing the performance of the HOG features from several color spaces like HSV,YUV,LUV,HLS I concluded that the YUV color space has the best performance for identifying vehicles and I chose to use that space for feature extraction.

### 5. Training a classifier

I used linear SVM and neural netowrk classifer to do the classification. Neural network classifier gave me better accuracy.

### 6. Vehicle detection in images

I  implemented a sliding window approach where I looked at one slice of the image at a time and made predictions on the HOG features from that particular window. In order to minimize the search area and speed up the pipeline I only searched for cars in the lower half of the image. Additionally, my algorithm searches for vehicles in windows of multiple scales, with an 80% overlap, in order to identify vehicles which can be either near or far in the image and will appear to have different sizes.

In order to ensure a high confidence for my predictions and minimize the instance of false positives, I made use of the neural network's method `predict_proba` which returns a probability score for each possible class. I chose to threshold my predictions by looking for windows which were classified as vehicle with a probability score higher than 0.99.

The coordinates of windows which are classified as vehicle will be appended to a list called `detected` and after all windows are searched, I used the `draw_boxes` function to draw the boxes in `detected` on to a blank `mask` image with the same dimensions as the input image. 

Next I used the OpenCV function `cv2.findContours` to find all of the objects in the `mask` image and once the contours are found I used the OpenCV function `cv2.boundingRect` on each contour to get the coordinates of the bounding rect for each vehicle.

Finally, I create a copy of the original image, called `result`, on which I drew the bounding rectangles.

### 7. Tracking vehicles in Video

Finally, to track vehicles across frames in a video stream, I decided to create a class `boxes` to store all of the bounding rectangles from each of the previous 12 frames in a list. In each frame, I then combine the lists of bounding rectangles from current and previous frames, and then use the OpenCV function `cv2.groupRectangles` to combine overlapping rectangles in to consolidated bounding boxes. Within the group rectangles function I set the parameter `groupThreshold` equal to 10 which means that it will only look for places where there are greater than 10 overlapping boxes and it will ignore everything else. 

The group rectangles function takes care of the problem of false positives because if any part of the image is classified as a vehicle in fewer than 10 out of 12 consecutive frames, it will be filtered out and will not be included in the final bounding rectangles which are annotated on to the output video. 

### 8. Discussion

This was one of the most difficult challenge for me. Due to lack of time, I could not try any other classifier apart from SVM and Neural network and I could not also do much of the parameter tuning. Hence the bounding box does not seem to be stable and in some instances it is identifying other objects as vehicles. I strongly believe that deep learning methods will provide better result, though they may become computationally expensive. But, I think it may also become little bit challenging to build the pipeline for deep learning methods.

I will work further on this project by applying other classifiers(including deep learning) and by applying any other feature engineering if possible in future.