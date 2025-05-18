# Crack Segmentation using PSPnet
Many constructions and infrastructures—such as buildings, pavements, bridges, and tunnels—are deteriorating across the globe, making it challenging to inspect them all manually. 
As an alternative, automated inspection offers a promising solution for efficiently assessing structural conditions. Although various sensors, including falling weight deflectometers, pavement density profilers, and ground-penetrating radar, can be used for such inspections, this solution specifically focuses on crack segmentation in images captured by standard cameras for visual assessment. 
In this project we aims to develop a crack segmentation network that takes a crack image as input and predicts its corresponding binary segmentation mask. The proposed deep learning model will address challenges such as class imbalance and the accurate detection of fine cracks.  
This project is part of our computer vision course project, in which we implemented crack segmentation network based on the reserach paper **"Joint Learning of Blind Super-Resolution and Crack Segmentation for Realistic Degraded Images"** [1](https://ieeexplore.ieee.org/document/10462152)

## SYSTEM OVERVIEW

![image](https://github.com/user-attachments/assets/41d6b76a-ccb4-418b-b729-5fcc2aefed94)

We are resizing all the input images to 448x448 because our model is trained best on this resolution.
The crack segmentation model is based in PSPnet(Pyramid Scene Parsing network) architecture which is explained later in this README.

## DATASET
The dataset plays a critical role in developing an effective neural network. As referenced in the research paper, the dataset used is provided by Khanhha and includes both pavement and concrete crack images, enhancing the model’s generalization across diverse surfaces. It contains labeled samples of both crack and non-crack images, each paired with corresponding binary segmentation masks. All images are uniformly sized at 448×448 pixels, ensuring consistency during training. Below are the dataset access links:  
**Original Dataset:**[Khanhha Dataset](https://drive.google.com/file/d/1xrOqv0-3uMHjZyEUrerOYiYXW_E8SUMP/view?pli=1)  
**Github:**[Khanhha Github](https://github.com/khanhha/crack_segmentation?tab=readme-ov-file)  
**Our Training Dataset:**[Omendra Kaggle](https://www.kaggle.com/datasets/omendrakumarupadhyay/crack-segmentation-datasetimage-mask)  

The dataset consists of real world crack images of wall cracks and pavement cracks at different abstraction levels for making the model more robust and versatile. 

## CRACK SEGMENTATION NETWORK
The research paper has given comparative data on the performance of multiple semantic segmentation architectures like U-net[2](https://arxiv.org/abs/1505.04597), PSPnet[3](https://ieeexplore.ieee.org/document/8100143), CrackFormer[4](https://ieeexplore.ieee.org/document/9711107), HRnet+OCR[5](https://arxiv.org/abs/1909.11065).  











[1] Y. Kondo and N. Ukita, "Joint Learning of Blind Super-Resolution and Crack Segmentation for Realistic Degraded Images," in IEEE Transactions on Instrumentation and Measurement, vol. 73, pp. 1-16, 2024, Art no. 5013816, doi: 10.1109/TIM.2024.3374293.  
[2] Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In: Navab, N., Hornegger, J., Wells, W., Frangi, A. (eds) Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. MICCAI 2015. Lecture Notes in Computer Science(), vol 9351.  
[3] H. Zhao, J. Shi, X. Qi, X. Wang and J. Jia, "Pyramid Scene Parsing Network," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, USA, 2017, pp. 6230-6239, doi: 10.1109/CVPR.2017.660.  
[4] H. Liu, X. Miao, C. Mertz, C. Xu and H. Kong, "CrackFormer: Transformer Network for Fine-Grained Crack Detection," 2021 IEEE/CVF International Conference on Computer Vision (ICCV), Montreal, QC, Canada, 2021, pp. 3763-3772, doi: 10.1109/ICCV48922.2021.00376.  
[5] Yuan, Yuhui, et al. "Segmentation Transformer: Object-Contextual Representations for Semantic Segmentation. 2021." arXiv preprint arXiv:1909.11065 (1909).  



