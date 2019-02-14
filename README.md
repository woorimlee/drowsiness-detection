# Drowsiness driving detection system with OpenCV & KNN
***
: In this repository, a program was developed to **identify the driver's drowsiness based on real-time camera image and image processing techniques**, and this program makes warning alarms go off for each level of drowsiness when it detects drowsiness driving.
## Description
: Based on the real-time Vision System, drivers' face and eye detection techniques were added, as well as **removing lighting effects** due to the eye detection false positives, drowsiness detection techniques, and **supervised learning algorithms** to identify drowsiness level.
  
The Histogram of Oriented Gradients technology and the learned Face Landmark estimation techniques were used to detect faces and eyes.
 
In order to eliminate the effects of lighting, **the light channels of the original images were separated and reversed, and then composed with the grayscale images of original images**. 
 
Furthermore the concept of **Eye Aspect Ratio was used** to detect drivers' drowsiness. 

Finally, the **KNN algorithm was used** to divide the drivers' level of drowsiness into three stages, and differential alarms go off for each stages.

Through these works, we could research and make technology of intelligent vehicle systems and vision computing, which is gaining much attention recently.
   
***This code is in Python 3.6***

## System diagram
 
<img src="https://user-images.githubusercontent.com/36785390/52612113-b61b6c00-2ecb-11e9-90f2-da6a51e00a7d.png" width="40%">
    
    
Get face images from the camera -> Grayscaling -> Light processing -> HOG & find face -> Face Landmark Estimation -> Detect drowsiness driving. 
   
+ In detail

<img src="https://user-images.githubusercontent.com/36785390/52612116-ba478980-2ecb-11e9-9851-0e037d0db792.png" width="80%">
  
0 : The filming.
  
10 : Lightness preprocessing.
  
100 : Detecting drowsiness.
  
110 : Getting face's image.
  
120 : Finding eyes region.
  
130 : Determining the value of the EAR normally.
  
140 : Determining drowsiness driving.
  
141 : Calculating the value of the EAR.
   
142 : Calculating the amount of time eyes are closed.
    
143 : Calculating the amount of time eyes are opened.
    
144 : Determining the level of the drowsiness.



## Extracting face and eye region
+ Using the **HOG face pattern**, to find the face from the Grayscaled-HOG-input-image. 
  
<img src="https://user-images.githubusercontent.com/36785390/52613168-3b088480-2ed0-11e9-8651-97afc34f4bae.png" width="60%">
   
+ Use the **Face Landmark Estimation algorithm** to locate the landmarks on the face.
  
<img src="https://user-images.githubusercontent.com/36785390/52613175-3d6ade80-2ed0-11e9-9290-ee5dc2f2d525.png" width="30%">
<img src="https://user-images.githubusercontent.com/36785390/52613176-3f34a200-2ed0-11e9-8f3f-94998fd2ab63.png" width="30%">
  


## Preprocessing
 
: **Invert the lightness channel** detached from the original image and **composed it with the original grayscale image** to produce a clear image.
 
<img src="https://user-images.githubusercontent.com/36785390/52613306-bb2eea00-2ed0-11e9-9b64-5c45981e953e.png" width="40%">
  
+ Converting color to grayscale using **Luma Coding**
  
<img src="https://user-images.githubusercontent.com/36785390/52613343-dc8fd600-2ed0-11e9-93f6-e154e20df31d.png" width="35%">
  
<img src="https://user-images.githubusercontent.com/36785390/52613308-bc601700-2ed0-11e9-999e-40a2782932c9.png" width="40%">
  
+ There are many different models in Color Space, the **LAB color space model** is the best way to separate Lightness. [Median filtering](https://en.wikipedia.org/wiki/Median_filter) is applied to convert the value of lightness(L) obtained by using the LAB color space to match the actual lighting conditions because it differs from the actual lighting conditions.
+ The pictures below are the original image, image that separates L channel, image with Median filter applied, and inverted images from left to right. Drowsiness detection method
    
<img src="https://user-images.githubusercontent.com/36785390/52613441-35f80500-2ed1-11e9-9c6c-819b9e92b150.png" width="70%">
   
+ Results of preprocessing
   
<img src="https://user-images.githubusercontent.com/36785390/52613443-385a5f00-2ed1-11e9-94e3-e325b3436041.png" width="20%">
    
     
## Drowsiness detection method
+ Each eye is represented by 6 (x, y)-coordinates
<img src="https://user-images.githubusercontent.com/36785390/52702447-83eb3680-2fbf-11e9-985f-f96ec72f5b26.png" width="20%">
   
+ The EAR equation
   
<img src="https://user-images.githubusercontent.com/36785390/52702578-cb71c280-2fbf-11e9-9a06-d4434250d622.png" width ="30%">

+ Calculated EAR
<img src="https://user-images.githubusercontent.com/36785390/52702645-ee9c7200-2fbf-11e9-9757-975fa22da6e1.png" width="60%">

+ The calculated EAR will have a value more than zero when the eyes are open, and a value close to zero when the eyes are closed.
+ This program has **set a 50% value from the average EAR value to the threshold value**. So, 1) measures the average EAR value when the eyes are open, 2) measures the average EAR value when the driver is closing his eyes, and 3) sets the threshold using the above two results.
+ .1) == 과정 1),   2) == 과정 2),   3) == 과정 3) (in drowsiness_detector code)
<img src="https://user-images.githubusercontent.com/36785390/52703067-ded15d80-2fc0-11e9-9b64-1fdbf554c12a.png">

  
## Drowsiness level selection
+ Conditions :
  1. 30 FPS
  2. Prescribed speed : 100km/h, Retention distance between vehicles >= 100m
  3. The time which takes a person to push the brakes 0.45 (response time) + 0.2 (brake pushing time) + 0.05 (time to start braking) = 0.7 seconds
  4. The braking distance of a vehicle running at 100 km/h is 56 meters (the driver has 44 meters of free distance)
+ Under the above conditions, the drivers has almost 0.9 seconds of free time (100km/h -> 27m/s == 1.63s of free time. 1.63 - 0.7 = 0.9 s). 
+ 30 FPS -> 27 frame = 0.9s.
  + **if EAR < threshold for 27 frame? then going alarm off.**
+ Now I separated the drowsiness phase into three steps.
  
<img src="https://user-images.githubusercontent.com/36785390/52762348-8058bd80-305a-11e9-9256-905e8de77740.png" width="45%">
  
+ Drowsiness levels are identified by the following conditions.
  1. The first alarm will sound(approximately 0.9 seconds) between level 1 and 2 of the drowsy phase.
  2. If you are dozing (sleeping and waking again and again) in less than 15 seconds, the drowsiness phase starts at level 1 and then the next alarm goes up to 0.
  3. The first alarm is level 2 and the second alarm is level 1 and the third alarm makes level 0 sound when driving drowsy between 15 and 30 seconds.
    
<img src="https://user-images.githubusercontent.com/36785390/52762615-b0549080-305b-11e9-872a-127992397496.png" width="50%">
   
+ To distinguish drowsiness level, I used K-Nearest Neighbor(KNN) supervised learning algorithm.
  
  1. Create arrays with random (x, y)-coordinates.
<img src="https://user-images.githubusercontent.com/36785390/52762829-82bc1700-305c-11e9-97cb-b41e35dfb9e6.png" width="30%">
  
  2. Labeling
<img src="https://user-images.githubusercontent.com/36785390/52762830-8485da80-305c-11e9-96db-f24a7a1ebdd6.png" width="40%">
  
  3. Define K value.
<img src="https://user-images.githubusercontent.com/36785390/52762904-e6dedb00-305c-11e9-952c-f201390eb9bd.png" width="50%">
  
  4. Test KNN algorithm.
<img src="https://user-images.githubusercontent.com/36785390/52762907-e8a89e80-305c-11e9-8928-9409bd4eaa7a.png" width="50%">
  
  
## Synthesis
<img src="https://user-images.githubusercontent.com/36785390/52762972-36bda200-305d-11e9-99a6-314dfae8f3c7.png" width="50%">

## Test
+ Before applying preprocessing
[![BeforePreprocessing](https://img.youtube.com/vi/8yLHAP6gmOA/0.jpg)](https://www.youtube.com/watch?v=8yLHAP6gmOA)
+ After applying preprocessing
[![AfterPreprocessing](https://img.youtube.com/vi/7iCVzF3LI6o/0.jpg)](https://www.youtube.com/watch?v=7iCVzF3LI6o)

  
## Execution
+ I run drowsiness_detector.ipynb just typing CTRL+ENTER.
  
## References
+ [Machine Learning is Fun! Part 4: Modern Face Recognition with Deep Learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)
+ [Real-Time Eye Blink Detection using Facial Landmarks](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)
+ [Eye blink detection with OpenCV, Python, and dlib](https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/)
+ [Histograms of Oriented Gradients for Human Detection](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)
