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
This paper concerns an iterative matrix square root normalization network (called fast MPN-COV), which is very efficient, fit for large-scale datasets, as opposed to its predecessor (i.e., MPN-COV published in ICCV17) that performs matrix power normalization by Eigen-decompositon. The code on bilinear CNN (B-CNN), compact bilinear pooling and global average pooling etc. is also released for both training from scratch and finetuning. If you use the code, please cite this fast MPN-COV work and its predecessor (i.e., MPN-COV).           
## Classification results
#### Classification results (single crop 224x224, %) on ImageNet 2012 validation set
<table>
          <center>
<tr>                                      
    <td rowspan="3"> Network</td>
    <td rowspan="3"> Dim</td>
    <td colspan="3"><center>Top1_err/Top5_err</center></td>
    <td colspan="2" rowspan="2">Pre-trained models(tensorflow)</td>
</tr>
                     </cenetr>

<tr>
    <td rowspan="2">paper</td>
    <td colspan="2">reproduce</td>
</tr>
<tr>
    <td>tensorflow</td>
    <td>pytorch</td>
    <td>GoogleDrive</td>
    <td>BaiduDrive</td>
</tr>
<tr>
    <td>mpncov_resnet50</td>
    <td rowspan="2"> 32K</td>
    <td>22.14/6.22</td>
    <td><strong>TODO</strong></td>
    <td>21.71/6.13</td>
    <td>GoogleDrive</td>
    <td>BaiduDrive</td>
</tr>
<tr>
    <td>mpncov_resnet101</td>
    <td>21.21/5.68</td>
    <td><strong>20.50/5.45</strong></td>
    <td>20.99/5.56</td>
    <td>GoogleDrive</td>
    <td>BaiduDrive</td>
</tr>
</table>

#### Fine-grained classification results (top-1 accuracy rates, %)
<table>
<tr>                                      
    <td rowspan="2">Backbone Model</td>
    <td rowspan="2">Dim</td>
    <td colspan="2">CUB</td>
    <td colspan="2">Aircraft</td>
    <td colspan="2">Cars</td>
</tr>
<tr>
    <td>paper</td>
    <td>reproduce</td>
    <td>paper</td>
    <td>reproduce</td>
    <td>paper</td>
    <td>reproduce</td>
</tr>
<tr>
    <td>resnet50</td>
    <td rowspan="2"> 32K</td>
    <td>88.1</td>
    <td><strong>TODO</strong></td>
    <td>90.0</td>
    <td><strong>TODO</strong></td>
    <td>92.8</td>
    <td><strong>TODO</strong></td>
</tr>
<tr>
    <td>resnet101</td>
    <td>88.7</td>
    <td><strong>88.1</strong></td>
    <td>91.4</td>
    <td><strong>91.8</strong></td>
    <td>93.3</td>
    <td><strong>93.9</strong></td>
</tr>
</table>

* Our method uses neither bounding boxes nor part annotations<br>
* The reproduced results are obtained by simply finetuning our pre-trained fast MPN-COV-ResNet model with a small learning rate, which do not perform SVM as our paper described.<br>
## Implementation details
We implement our Fast MPN-COV (i.e., iSQRT-COV) meta-layer under ***Tensorflow2.0*** package. We release two versions of code:<br> 

* The backpropagation of our meta-layer without using autograd package;<br>
* The backpropagation of our meta-layer with using autograd package(TODO).<br>

For making our Fast MPN-COV meta layer can be added in a network conveniently, we reconstruct pytorch official demo imagenet/ and models/. In which, we divide any network for three parts: 1) features extractor; 2) global image representation; 3) classifier. As such, we can arbitrarily combine a network with our Fast MPN-COV or some other global image representation methods (e.g.,Global average pooling, Bilinear pooling, Compact bilinear pooling, etc.) 
## Installation and Usage

## Other Implementations
* [MatConvNet Implementation](https://github.com/jiangtaoxie/matconvnet.fast-mpn-cov)
* [PyTorch Implementation](https://github.com/jiangtaoxie/fast-MPN-COV)
