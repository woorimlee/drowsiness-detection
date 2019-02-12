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
이미지 추가
<이미지>
 
In detail
<이미지>

## Extracting face and eye region
+ Using the **HOG face pattern**, to find the face from the Grayscaled-HOG-input-image. 
+ Use the **Face Landmark Estimation algorithm** to locate the landmarks on the face.
  
<그림 5, 7, 8> 추가


## Preprocessing
 
: **Invert the lightness channel** detached from the original image and **composed it with the original grayscale image** to produce a clear image.
 
<그림 9> 추가
  + Converting color to grayscale using **Luma Coding**
<공식> 추가
  
<그림 10> 추가
+ There are many different models in Color Space, the **LAB color space model** is the best way to separate Lightness. [Median filtering](https://en.wikipedia.org/wiki/Median_filter) is applied to convert the value of lightness(L) obtained by using the LAB color space to match the actual lighting conditions because it differs from the actual lighting conditions.
+ The pictures below are the original image, image that separates L channel, image with Median filter applied, and inverted images from left to right. Drowsiness detection method
    
<그림 13> 추가
+ Results of preprocessing
<그림 14> 추가
 
## Drowsiness detection method