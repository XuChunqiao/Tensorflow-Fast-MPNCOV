# Fine-tune pre-trained model in Tensorflow
By using our code, we reproduce the results of three popular fine-grained benchmarks.(i.e., Bird, Aircrafts and Cars) We will keep updating the results of this page.

## Our experiments are running on
* Tensorflow2.0.0b0<br>
* 1 × 1080<br>
* Cuda 10.0 with CuDNN 7.5<br>
# Results (top-1 accuracy rates, %)
All the reproduced results use neither bounding boxes nor part annotations, and the SVM classifier is not performed.
## MPNCOV
<table>
<tr>                                      
    <td rowspan="2"> Backbone Model</td>
    <td rowspan="2"> Dim</td>
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
    <td>TODO</td>
    <td>90.0</td>
    <td>TODO</td>
    <td>92.8</td>
    <td>TODO</td>
</tr>
<tr>
    <td>resnet101</td>
    <td>88.7</td>
    <td>TODO</td>
    <td>91.4</td>
    <td>91.8</td>
    <td>93.3</td>
    <td>93.9</td>
</tr>
</table>

