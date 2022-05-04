# Smartforms #
## Abstract ##

Initiation, monitoring, and evaluation of development programmes can involve field-based data collection about project activities. This
data collection through digital devices may not always be feasible though, for reasons like unaffordability of smartphones and tablets
by field-based cadre, or shortfalls in their training and capacity building. Paper-based data collection has been argued to be more
appropriate in several contexts, with automated digitization of the paper forms through OCR (Optical Character Recognition) and
OMR (Optical Mark Recognition) techniques. We contribute with providing a large dataset of handwritten digits, and deep learning
based models and methods built using this data, that are effective in real-world environments. We demonstrate the deployment of
these tools in the context of a maternal and child health and nutrition awareness project, which uses IVR (Interactive Voice Response)
systems to provide awareness information to rural women SHG (Self Help Group) members in north India. Paper forms were used to
collect phone numbers of the SHG members at scale, which were digitized using the OCR tools developed by us, and used to push
almost 4 million phone calls. We are in the process of releasing the data, model, and code in the open-source domain.

</br></br>
This repository contains the code for the handwritten digit prediction using Triplet Loss in a LeNet-5 based model. 
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
Run train.sh file to train the model. This file contains the path of the image folder, ground truth file, train/val/test split file and the target directory for saving the weights. You can update the paths in this file accordingly. If train/val/test split file is None, the code will create a new split. 

## Testing ##
To test the model, run the test.sh file. Weights for trained models are included in the Weights folder. You can edit the paths contained in this file.

