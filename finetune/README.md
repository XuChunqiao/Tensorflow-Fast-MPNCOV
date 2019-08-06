## Fine-tune pre-trained model in Tensorflow
By using our code, we reproduce the results of three popular fine-grained benchmarks.(i.e., Bird, Aircrafts and Cars) We will keep updating the results of this page.

### Our experiments are running on
* Tensorflow2.0.0b0<br>
* 1 × 1080<br>
* Cuda 10.0 with CuDNN 7.5<br>
## Results (top-1 accuracy rates, %)
All the reproduced results use neither bounding boxes nor part annotations, and the SVM classifier is not performed.
### MPNCOV
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
    <td align='center'>reproduce</td>
    <td align='center'>paper</td>
    <td align='center'>reproduce</td>
    <td align='center'>paper</td>
    <td align='center'>reproduce</td>
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

### Bilinear CNN
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
    <td align='center'>reproduce</td>
    <td align='center'>paper</td>
    <td align='center'>reproduce</td>
    <td align='center'>paper</td>
    <td align='center'>reproduce</td>
</tr>
<tr>
    <td>VGG16</td>
    <td rowspan="2"> 262</td>
    <td align='center'>84.0</td>
    <td align='center'><strong>TODO</strong></td>
    <td align='center'>86.9</td>
    <td align='center'><strong>TODO</strong></td>
    <td align='center'>90.6</td>
    <td align='center'><strong>TODO</strong></td>
</tr>
</table>

### Compact Bilinear CNN
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
    <td align='center'>reproduce</td>
    <td align='center'>paper</td>
    <td align='center'>reproduce</td>
    <td align='center'>paper</td>
    <td align='center'>reproduce</td>
</tr>
<tr>
    <td>VGG16</td>
    <td rowspan="2"> 8k</td>
    <td align='center'>84.0</td>
    <td align='center'><strong>TODO</strong></td>
    <td align='center'>-</td>
    <td align='center'><strong>TODO</strong></td>
    <td align='center'>-</td>
    <td align='center'><strong>TODO</strong></td>
</tr>
</table>

