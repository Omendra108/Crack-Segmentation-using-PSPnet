# Crack Segmentation using PSPnet

Many constructions and infrastructures—such as buildings, pavements, bridges, and tunnels—are deteriorating across the globe, making it challenging to inspect them all manually. 
As an alternative, automated inspection offers a promising solution for efficiently assessing structural conditions. Although various sensors, including falling weight deflectometers, pavement density profilers, and ground-penetrating radar, can be used for such inspections, this solution specifically focuses on crack segmentation in images captured by standard cameras for visual assessment. 
In this project we aims to develop a crack segmentation network that takes a crack image as input and predicts its corresponding binary segmentation mask. The proposed deep learning model will address challenges such as class imbalance and the accurate detection of fine cracks. 
This project is part of our computer vision course project, in which we implemented crack segmentation network based on the reserach paper **"Joint Learning of Blind Super-Resolution and Crack Segmentation for Realistic Degraded Images"** [1].

## SYSTEM OVERVIEW

![image](https://github.com/user-attachments/assets/41d6b76a-ccb4-418b-b729-5fcc2aefed94)

We are resizing all the input images to 448x448 because our model is trained best on this resolution.
The crack segmentation model is based in PSPnet(Pyramid Scene Parsing network) architecture which is explained later in this README.

## DATASET

The dataset plays a critical role in developing an effective neural network. As referenced in the research paper, the dataset used is provided by Khanhha and includes both pavement and concrete crack images, enhancing the model’s generalization across diverse surfaces. It contains labeled samples of both crack and non-crack images, each paired with corresponding binary segmentation masks. All images are uniformly sized at 448×448 pixels, ensuring consistency during training. Below are the dataset access links: 
**Original Dataset:**[Khanhha Dataset](https://drive.google.com/file/d/1xrOqv0-3uMHjZyEUrerOYiYXW_E8SUMP/view?pli=1)
**Github:**[Khanhha Github](https://github.com/khanhha/crack_segmentation?tab=readme-ov-file)
**Our Training Dataset:**[Omendra Kaggle](https://www.kaggle.com/datasets/omendrakumarupadhyay/crack-segmentation-datasetimage-mask)












[1] Y. Kondo and N. Ukita, "Joint Learning of Blind Super-Resolution and Crack Segmentation for Realistic Degraded Images," in IEEE Transactions on Instrumentation and Measurement, vol. 73, pp. 1-16, 2024, Art no. 5013816, doi: 10.1109/TIM.2024.3374293.


