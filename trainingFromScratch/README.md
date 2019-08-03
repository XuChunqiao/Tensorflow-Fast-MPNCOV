## Train from scratch
By using our code, we reproduce the results of our Fast MPN-COV ResNet models on ImageNet 2012.

### Our experiments are running on
 * Tensorflow 2.0.0b0<br>
 * 2 x 1080Ti<br>
 * Cuda 10.0 with CuDNN 7.5<br>
 
 ## Results
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
    <td align='center'>tensorflow</td>
    <td align='center'>pytorch</td>
    <td align='center'>GoogleDrive</td>
    <td align='center'>BaiduDrive</td>
</tr>
<tr>
    <td>mpncov_resnet50</td>
    <td rowspan="2" align='center'> 32K</td>
    <td align='center'>22.14/6.22</td>
    <td align='center'><strong>TODO</strong></td>
    <td align='center'>21.71/6.13</td>
    <td align='center'>GoogleDrive</td>
    <td align='center'>BaiduDrive</td>
</tr>
<tr>
    <td>mpncov_resnet101</td>
    <td align='center'>21.21/5.68</td>
    <td align='center'><strong>20.50/5.45</strong></td>
    <td align='center'>20.99/5.56</td>
    <td align='center'>GoogleDrive</td>
    <td align='center'>BaiduDrive</td>
</tr>
</table>
