# Smartforms #
Dataset, code and pre-trained models are released. Library and end-to-end code will be released soon.
</br></br>
Handwritten Digit Dataset - ✅ </br>
Code and pretrained models - ✅ </br>
Library for designing new forms - ❌ </br>
End-to-end Code for homography based roi extraction and recognition - ❌ </br>

Paper link- 

## Requirements ##
1. Python >= 3.0
2. Tensorflow >= 2.0
3. tensorflow-addons
4. OpenCV

## Dataset Description ##
The dataset consists of a grid of 16X10 cells, in which each cell contains a handwritten digit of size 32X32. The size of the digit image is 30X30. There is a white(255) boundary of 1 pixel around the digit. A cell can be empty if there is no digit. 

<p align="center">
  <img src="https://github.com/pantDevesh/Smartforms/blob/main/Sample/661.png"  />
</p>

Download the datasets from here- <a href="https://drive.google.com/file/d/1fX4LIAZlF645cSXxQPkJufSZUbbR_6s0/view?usp=sharing" target="_blank">Gdrive</a>

## Training ##
Train the model by running the train.sh file. This file specifies the location of the image folder, the ground truth file, the train/val/test split file, and the directory to which the weights should be saved. You may modify the paths in this file as necessary. If None is specified for the train/val/test split file, the code will generate a new split.

## Testing ##
Run the test.sh file to validate the model, weights for the trained models are included in the Weights folder. You may modify the paths in this file as necessary.
