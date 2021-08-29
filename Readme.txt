The codes and data are for manuscript:
	A novel deep convolutional neural network approach for large area satellite time series land cover classification

The codes are written by Hankui Zhang in R

The codes require library of Tensorflow version 1.9 installed for R

There are four main functions:
	NLCD training and testing with input txt data: ./NLCD/Pro.training.metric.nlcd.r
	NLCD model application with input image file:  ./NLCD/Pro.load.predict.image.NLCD.v1.1.r
	CDL training and testing with input txt data: ./CDL/Pro.training.metric.cdl.r
	CDL model application with input image file:  ./CDL/Pro.load.predict.image.CDL.v1.1.r
	
There are 2 input text files (csv) storing the 3,314,439 NLCD and 484,476 training samples:
	NLCD training: ./NLCD/metric.ard.nlcd.Mar01.18.40.txt
	CDL training: ./CDL/metric.ard.nlcd.Mar01.18.40.txt

The data are stored in https://zenodo.org/record/5333519 due to github storage limitation