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
After the quantative comparision of different model's mentioned in Table II in the research paper[1] we choose PSPnet as the model architecture as it has the highest IoU score.  
PSPNet(Pyramid Scene Parsing network) is a deep learning model used for semantic segmentation(2 class segmentation), where each pixel in an image is classified into a category (e.g., road, crack, wall, etc.). It’s especially known for capturing global context using its pyramid pooling module (PPM).  
Following are the components of the PSPnet based crack segmentation network:  
1. Encoder: The encoder serves as a feature extractor for the input image. In our implementation, we use ResNet-18, pre-trained on the ImageNet dataset, as the backbone for the encoder.   
2. PPM: Pyramid Pooling Module helps the network understand the local and global details of the image. PPM applies pooling at different scales (1x1,2x2,...6x6 etc) and forms layers of low resolution feature maps. 
  
3. Decoder: After PPM, the layers of the feature map are upsampled to the size of the input image and concatenated to form a segmentation mask of the input image.  
4. Output: The convolution layers are converted to segmentation mask using sigmoid activation function.The segmentation mask is a probability map, where each pixel has a probability score, which tells how likely the pixel is belonging to the crack pixel class.Thresholding function is applied on the segmentation mask to convert the probability score map to binary mask.

![image](https://github.com/user-attachments/assets/7c4f4386-328c-465e-9ad9-a3845a51c04c)  

## LOSS FUNCTION
Loss functions are mathematical tool for assisting the model to learn on the dataset by reducing the errors between ground truth and predicted value. In our project we have used a combination of BC(Boundary Combo)) loss function, which itself is a weighted combination of Boundary Loss, GDice(Generalised Dice) loss and Weighted Cross Entropy loss(WCE). Boundary loss function ensures that the boundary of fine cracks is highlighted properly. Ensures Crack contours are well defined.GDice loss ensures the cracks are segmented properly. WCE loss Gives more weight to fine crack pixels compared to the background pixels. In the BC loss function 𝜶 and 𝜸 are the hyperparameters, where 𝜸=0.5 and 𝜶 is dynamically calculated.

![image](https://github.com/user-attachments/assets/cf907ce7-ed26-42ef-b1be-8ada9f472bcc)  

## TRAINING
The model was trained for 50 epochs, during which both training and validation losses were monitored at each epoch to evaluate performance and track generalization. To mitigate overfitting and ensure optimal 
model selection, a checkpointing strategy was employed wherein the model state was saved only if the current epoch yielded a lower validation loss than all previous epochs. The final model with the lowest 
recorded validation loss was saved to Google Drive as **“PSPnet_best_model.pth”**.

![image](https://github.com/user-attachments/assets/abddbdae-4832-4989-81a2-e2efd11294c7)  

## MODEL EVALUATION
For model evaluation, a randomly selected crack image from the test dataset was passed through the trained network to generate its corresponding predicted segmentation mask. To qualitatively assess the model’s performance, the input image, ground truth mask, and predicted mask were visualized side by side. For quantitative analysis, a pixel-level confusion matrix was computed and plotted, enabling detailed assessment of true positives, false positives, false negatives, and true negatives—consistent with the pixel-wise nature of semantic segmentation in crack detection. 

![image](https://github.com/user-attachments/assets/bbdba5f4-ccd7-42b8-90bf-05e64390358c)  
![image](https://github.com/user-attachments/assets/37461aa1-ff0f-4a14-848a-b1a08f2e2a3a)  

To quantitatively assess the model's performance, segmentation masks were predicted for all test images. For each prediction IoU, Precision, Accuracy, Recall, and F1 Score were computed. The final performance of the model was determined by averaging these metrics across the entire test set, providing a comprehensive evaluation of its segmentation capability. 

![image](https://github.com/user-attachments/assets/fcadf662-906f-42a2-8900-cdbe105a3153)  

## CONCLUSION 
We successfully developed a crack segmentation model based on the PSPNet architecture, capable of performing pixel-wise semantic segmentation. Based on the quantitative evaluation metrics, the following observations can be made:  
1) The model demonstrates high recall, effectively identifying the majority of crack pixels 
with minimal false negatives.  
2) It generalizes well across different types of crack patterns, indicating strong versatility.  
3) The model is highly sensitive to fine-grained crack structures, enabling the detection of even subtle crack formations.  
The crack segmentation network build in this project is baised and sensitive towards crack pixels, where it predicts almost all true crack pixels but also labels some background pixels as crack pixels.  
With certain refinement and fine-tuning we can improve models performance and so, therefore we are providing the source code publicly for any person to use and produce a better model than us ✌️

## REFERENCES
[1] Y. Kondo and N. Ukita, "Joint Learning of Blind Super-Resolution and Crack Segmentation for Realistic Degraded Images," in IEEE Transactions on Instrumentation and Measurement, vol. 73, pp. 1-16, 2024, Art no. 5013816, doi: 10.1109/TIM.2024.3374293.  

[2] Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In: Navab, N., Hornegger, J., Wells, W., Frangi, A. (eds) Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. MICCAI 2015. Lecture Notes in Computer Science(), vol 9351.  

[3] H. Zhao, J. Shi, X. Qi, X. Wang and J. Jia, "Pyramid Scene Parsing Network," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, USA, 2017, pp. 6230-6239, doi: 10.1109/CVPR.2017.660.  

[4] H. Liu, X. Miao, C. Mertz, C. Xu and H. Kong, "CrackFormer: Transformer Network for Fine-Grained Crack Detection," 2021 IEEE/CVF International Conference on Computer Vision (ICCV), Montreal, QC, Canada, 2021, pp. 3763-3772, doi: 10.1109/ICCV48922.2021.00376.  

[5] Yuan, Yuhui, et al. "Segmentation Transformer: Object-Contextual Representations for Semantic Segmentation. 2021." arXiv preprint arXiv:1909.11065 (1909).  



