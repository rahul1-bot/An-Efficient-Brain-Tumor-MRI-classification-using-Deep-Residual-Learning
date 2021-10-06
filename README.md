# _An Efficient Brain Tumor MRI classification using Deep Residual Learning_

## _Authors_
* Rahul Sawhney
* Aabha Malik 
* Shambhavi Lau
* Sumanyu Dutta
* Dr. Shilpi Sharma (Associate Professor)

## _Abstract_
Brain tumor categorization is essential for evaluating tumors as well as determining treatment choices established on their classifications. To identify brain tumors, a variety of imaging methods are employed. Oppositely, MRI is widely utilized because of its improved image quality and the fact of matter is that it does not employ ionizing radiation. Deep learning is a subset of machine learning that lately has demonstrated exceptional performance, particularly in classification and segmentation. In this study, we used a deep residual network to classify distinct kinds of tumors which are present in brain using images datasets. The MRI scans create a massive quantity of data. The radiologist examines these scans. Meningioma, glioma and pituitary tumor are the three main categories of tumors which are present in brain. Because of the intricacies involved in brain tumors, a manual examination might be an error prone. Automated classification machine learning-based approaches have consistently outperformed manual categorization. As a result, we propose a system that performs detection as well as classification using deep residual networks based on CNN.

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
![image](https://user-images.githubusercontent.com/65220704/136267578-e298cba1-4e69-45c9-b6b2-63e708fca99d.png)


For our paper, we began by creating a ResNet model. Residual Networks, or ResNets, instead of learning from unreferenced functions, they learn residual functions with reference to the layer inputs. Residual nets allow these layers to suit a residual mapping rather than expecting that each few stacked layers directly match a desired underlying mapping. They are built by stacking residual blocks on top of one other. The problems which are solved from these kinds of networks is that when deeper networks begin to converge, a degradation problem emerges as network depth increases, accuracy becomes saturated and subsequently rapidly declines. In the worst-case scenario, the early levels of a deeper model can be substituted with a shallow network, while the remaining layers can simply serve as an identity function and in the rewarding scenario these extra layers of a deeper network better match the mapping than their shallower counterparts, which in turns results in reduction in error by a substantial margin.

ResNet comes in many different architectures, such as ResNet18, ResNet34, ResNet50, and so on. Even though the design is the same, the numbers represent layers. In our work we used the ResNet-18 model which contains 18 layers. In these networks, firstly there is a conv layer of (7x7) filter, then 16 conv layers are used with filter size of (3x3) and at last average pooling is done then the output is passed through the dense layers and then it is passed through SoftMax activation function.

## _Results_
![image](https://user-images.githubusercontent.com/65220704/136268672-018864d1-0250-4569-9535-3cf16429124a.png)
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

