# facemaskclassifier
Computer Vision Project: Face Detection and Mask Classification

This was a brief project to develop skills in computer vision, which covered loading in models, usinging transfer learning thorugh fine tuning and feature extraction, and creating a pipeline that combines multiple models together.

The design for the final functionality is to accept an image, and returns it with dectection boxes added to faces, and a classification with percent confident in its decision on weather the face is wearing a mask or not.

I would like to acknowledge the PyImageSearch blog, by Adrian Rosebrock. Many of his posts were used as a refence for each task.

## Content

### /images/ 
Stores demonstration images post-processed.

### /mask-classifier-images/ 
folder that holds files to prepare the training and testing data for training the mask-classifier, taken from a [simliar post](https://github.com/aome510/Mask-Classifier)

### .gitignore 
designed to avoid uploading large files/images

### PlayAroundWithPretrainModels.ipynb
This was the first notebook in this project, which focused on exploring the options available to preform predicitons on images. 

### TransferLearning-FeatureExtraction.ipynb
This notebook focuses on using logisitic regression on a headless pretrained model to make mask classifications

### TransferLearning-FineTurning.ipynb
This notebook focuses on training new fully connected layers and the formerly headless pretrained model to make mask classifications.

### predict.ipynb
File which produces final results. It uses a [yolo-v3 model pretrained on faces](https://github.com/sthanhng/yoloface) to create detection boxes on faces in an image, then continues by cropping the face and push it into the second model which returns the classification and percent certainty. Finally, the boxes have labels added to it using the second model.

## Results

Below are four images that were passed through the models.

![demo_image_1](https://github.com/alexretana/facemaskclassifier/blob/master/images/demo_image_1(processed).jpg)

This first images is from a few year ago before the 2020 covid-19 pandemic, so we are all maskless in this image, and the result correctly detected all 4 faces and classified them as maskless.

![demo_image_2](https://github.com/alexretana/facemaskclassifier/blob/master/images/demo_image_2(processed).jpg)

Also a few years back. It correctly detect/classifies of the subjects of the image. The face detection picked up a couple extra faces in the background, and incorrectly classified them as masked. In defense of the prediciton, the lighting is odd in the background, and the confidence didn't reach about 90% as it normally would, so there are indicators to not trust the result altogether.

![demo_image_3](https://github.com/alexretana/facemaskclassifier/blob/master/images/demo_image_3(processed).jpg)

A more current image during the pandemic. Here the model correctly detected my face and classified it correctly, but with lower percent confidence than usual. That could be accounted for by the fact that the face is wearing sunglasses.

![demo_image_4](https://github.com/alexretana/facemaskclassifier/blob/master/images/demo_image_4(processed).jpg)

This final demo is to see how it preforms with many faces. The results are generally good; Thre are 13 faces detected, and 12 are correctly classified, with the only exception being a face half-covered by another face.