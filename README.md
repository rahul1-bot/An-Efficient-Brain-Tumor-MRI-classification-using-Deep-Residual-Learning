# _An Efficient Brain Tumor MRI classification using Deep Residual Learning_

## _Authors_
1) _Rahul Sawhney_
2) _Leah Pathan Khan_
3) _Dr. Shilpi Sharma (Associate Professor)_
  
## _Abstract_
Brain tumor categorization is essential for evaluating tumors as well as determining treatment choices established on their classifications. To identify brain tumors, a variety of imaging methods are employed. Oppositely, MRI is widely utilized because of its improved image quality and the fact of matter is that it does not employ ionizing radiation. Deep learning is a subset of machine learning that lately has demonstrated exceptional performance, particularly in classification and segmentation. In this study, we used a deep residual network to classify distinct kinds of tumors which are present in brain using images datasets. The MRI scans create a massive quantity of data. The radiologist examines these scans. Meningioma, glioma and pituitary tumor are the three main categories of tumors which are present in brain. Because of the intricacies involved in brain tumors, a manual examination might be an error prone. Automated classification machine learning-based approaches have consistently outperformed manual categorization. As a result, we propose a system that performs detection as well as classification using deep residual networks based on CNN and other different models such as ResNet, AlexNet, Inception, VGG16.

## _Keywords_
1) Artificial Neural Networks
2) Convolutional Neural Networks
3) Image and Pattern Recognition
4) Brain-Tumor MRI Images classification

## _Methodology_
![image](https://user-images.githubusercontent.com/65220704/132106660-75869364-19cc-4a17-8a68-58939ba24bd9.png)

## _Sneek Peek at the Dataset_
![Capture](https://user-images.githubusercontent.com/65220704/132455110-e1d062a8-917a-462d-b3a7-00d3b04ef3bb.PNG)

## _Model Architecture_
![image](https://user-images.githubusercontent.com/65220704/132106720-2c68f29e-1c5e-4d6d-bd27-a0a054bfb20a.png)

![Original-ResNet-18-Architecture](https://user-images.githubusercontent.com/65220704/132106510-02b931d3-7e48-459f-8977-22dbce19ef79.png)

A residual network, or ResNet for short, is an artificial neural network that aids in the construction of deeper neural networks by employing skip connections or shortcuts to bypass some levels. You will see how skipping aids in the construction of deeper network layers while avoiding the issue of disappearing gradients. ResNet comes in a variety of versions, such as ResNet-18, ResNet-34, ResNet-50, and so on. Even though the design is the same, the numbers represent layers. In our work we used the ResNet 18 model. It contains 18 layers. The layers which are involved in different ResNet Model. First there is a conv layer of (7x7) filter, then 16 conv layers are used with filter size of (3x3) and at last average pooling is done then the output is passed through the fully connected layers and output from the fully connected layer is passed through SoftMax activation function.

## _Results_
![image](https://user-images.githubusercontent.com/65220704/132457573-466d37fd-0283-4b26-8007-002c11eff876.png)

![image](https://user-images.githubusercontent.com/65220704/132456968-291cc5f2-bc92-43ee-b73f-ed17134f3ddc.png)

![Capture](https://user-images.githubusercontent.com/65220704/132455430-9ae4acfb-7558-442b-8425-be4a7604e77d.PNG)
![Capture](https://user-images.githubusercontent.com/65220704/132455525-97577fc3-52c6-49d6-bbd1-bb2da686723b.PNG)


## _Conclusion_
Without a doubt, Brain tumor is a fatal disease which becomes very challenging to diagnose with high accuracy and precision. Therefore, creating this model would be highly beneficial in the clinical trials. Thus, we developed a CNN based deep neural network which observes and classify brain tumor MRI images in 4 classes. That CNN model begins by reconstructing frontal Brain tumor MRI images into compressed size and classify them whether an individual is tainted with either of Glioma, Meningioma or Pituitary tumor. This powerful model will be applied to various healthcare firms, which is a much better approach than any other traditional clinical trials. Nonetheless, there are lots of problems which must be solved in this astonishing field. In future, this architecture will exhibit various application in different domains as well.     

## _Dataset Link_
https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri

## _Acknowledgements for Dataset_
1) _Navoneel Chakrabarty_
2) _Swati Kanchan_

