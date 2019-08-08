# Tensorflow Fast-MPNCOV
![](https://camo.githubusercontent.com/f2cdc5f25d743e922fd2c23e8a2a42e1f25c1e36/687474703a2f2f7065696875616c692e6f72672f70696374757265732f666173745f4d504e2d434f562e4a5047)
## Introduction
This repository contains the source code under ***TensorFlow2.0 framework*** and models trained on ImageNet 2012 dataset for the following paper:<br>
```
@InProceedings{Li_2018_CVPR,
           author = {Li, Peihua and Xie, Jiangtao and Wang, Qilong and Gao, Zilin},
           title = {Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix Square Root Normalization},
           booktitle = { IEEE Int. Conf. on Computer Vision and Pattern Recognition (CVPR)},
           month = {June},
           year = {2018}
     }
```
This paper concerns an iterative matrix square root normalization network (called fast MPN-COV), which is very efficient, fit for large-scale datasets, as opposed to its predecessor (i.e., [MPN-COV](https://github.com/jiangtaoxie/MPN-COV)) published in ICCV17) that performs matrix power normalization by Eigen-decompositon. The code on bilinear CNN (B-CNN), compact bilinear pooling and global average pooling etc. is also released for both training from scratch and finetuning. If you use the code, please cite this [fast MPN-COV](http://peihuali.org/iSQRT-COV/iSQRT-COV_bib.htm) work and its predecessor (i.e., [MPN-COV](http://peihuali.org/iSQRT-COV/iSQRT-COV_bib.htm)).           
## Classification results
#### Classification results (single crop 224x224, %) on ImageNet 2012 validation set
<table>
<tr>                                      
    <td rowspan="3" align='center'>Network</strong></td>
    <td rowspan="3" align='center'>Dim</td>
    <td colspan="3" align='center'>Top1_err/Top5_err</td>
    <td colspan="2" rowspan="2" align='center'>Pre-trained models<br>(tensorflow)</td>
</tr>
<tr>
    <td rowspan="2" align='center'>paper</td>
    <td colspan="2" align='center'>reproduce</td>
</tr>
<tr>
    <td align='center'><strong>tensorflow</strong></td>
    <td align='center'><a href="https://github.com/jiangtaoxie/fast-MPN-COV" title="标题">pytorch</a></td>
    <td align='center'>GoogleDrive</td>
    <td align='center'>BaiduDrive</td>
</tr>
<tr>
    <td>mpncov_resnet50</td>
    <td rowspan="2" align='center'> 32K</td>
    <td align='center'>22.14/6.22</td>
    <td align='center'><strong>21.57/6.14</strong></td>
    <td align='center'>21.71/6.13</td>
    <td align='center'><a href="https://drive.google.com/file/d/1kXi3PGixfn7QZaxtLK2DkiZ6h-zoGpfq/view?usp=sharing" title="标题">GoogleDrive</a></td>
    <td align='center'>BaiduDrive</td>
</tr>
<tr>
    <td>mpncov_resnet101</td>
    <td align='center'>21.21/5.68</td>
    <td align='center'><strong>20.50/5.45</strong></td>
    <td align='center'>20.99/5.56</td>
    <td align='center'><a href="https://drive.google.com/file/d/1RFdw2oEZLe03SCDFanwQKHUY13OeEzp0/view" title="标题">GoogleDrive</a></td>
    <td align='center'><a href="https://pan.baidu.com/s/1fj0-vukSbRz1ihTDtAbUdA" title="标题">BaiduDrive</a></td>
</tr>
</table>

#### Fine-grained classification results (top-1 accuracy rates, %)
<table>
<tr>                                      
    <td rowspan="2" align='center'>Backbone Model</td>
    <td rowspan="2" align='center'>Dim</td>
    <td colspan="2" align='center'><a href="http://www.vision.caltech.edu/visipedia/CUB-200-2011.html" title="标题">CUB</a></td>
    <td colspan="2" align='center'><a href="http://ai.stanford.edu/~jkrause/cars/car_dataset.html" title="标题">Aircraft</a></td>
    <td colspan="2" align='center'><a href="http://www.robots.ox.ac.uk/~vgg/data/oid/" title="标题">Cars</a></td>
</tr>
<tr>
    <td align='center'>paper</td>
    <td align='center'>reproduce<br>(tensorflow)</td>
    <td align='center'>paper</td>
    <td align='center'>reproduce<br>(tensorflow)</td>
    <td align='center'>paper</td>
    <td align='center'>reproduce<br>(tensorflow)</td>
</tr>
<tr>
    <td>resnet50</td>
    <td rowspan="2"> 32K</td>
    <td align='center'>88.1</td>
    <td align='center'><strong>TODO</strong></td>
    <td align='center'>90.0</td>
    <td align='center'><strong>TODO</strong></td>
    <td align='center'>92.8</td>
    <td align='center'><strong>TODO</strong></td>
</tr>
<tr>
    <td>resnet101</td>
    <td align='center'>88.7</td>
    <td align='center'><strong>88.1</strong></td>
    <td align='center'>91.4</td>
    <td align='center'><strong>91.8</strong></td>
    <td align='center'>93.3</td>
    <td align='center'><strong>93.9</strong></td>
</tr>
</table>

* Our method uses neither bounding boxes nor part annotations<br>
* The reproduced results are obtained by simply finetuning our pre-trained fast MPN-COV-ResNet model with a small learning rate, which do not perform SVM as our paper described.<br>
## Implementation details
We implement our Fast MPN-COV (i.e., iSQRT-COV) [meta-layer](https://github.com/XuChunqiao/Tensorflow-Fast-MPNCOV/blob/master/src/representation/MPNCOV.py) under ***Tensorflow2.0*** package. We release two versions of code:<br> 

* The backpropagation of our meta-layer without using autograd package;<br>
* The backpropagation of our meta-layer with using autograd package(**TODO**).<br>

For making our Fast MPN-COV meta layer can be added in a network conveniently, we divide any network for three parts: <br>
* features extractor;<br>
* global image representation;<br>
* classifier. <br>

As such, we can arbitrarily combine a network with our Fast MPN-COV or some other global image representation methods (e.g.,Global average pooling, Bilinear pooling(**TODO**), Compact bilinear pooling(**TODO**), etc.) 
## Installation and Usage
1. Install [Tensorflow (2.0.0b0)](https://tensorflow.google.cn/install)
2. type ```git clone https://github.com/jiangtaoxie/fast-MPN-COV ```
3. prepare the dataset as follows
```
.
├── train
│   ├── class1
│   │   ├── class1_001.jpg
│   │   ├── class1_002.jpg
|   |   └── ...
│   ├── class2
│   ├── class3
│   ├── ...
│   ├── ...
│   └── classN
└── val
    ├── class1
    │   ├── class1_001.jpg
    │   ├── class1_002.jpg
    |   └── ...
    ├── class2
    ├── class3
    ├── ...
    ├── ...
    └── classN
```
### train from scratch
1. ``` cp ./trainingFromScratch/imagenet/imagenet_tfrecords.py ./ ```
2. modify the dataset path and run ``` python imagenet_tfrecords.py ``` to create tfrecord files
3. modify the parameters in train.sh ```sh train.sh```
### finetune fast-MPNCOV models
1. modify the parameters in finetune.sh
2. ```sh finetune.sh```
## Other Implementations
* [MatConvNet Implementation](https://github.com/jiangtaoxie/matconvnet.fast-mpn-cov)
* [PyTorch Implementation](https://github.com/jiangtaoxie/fast-MPN-COV)
