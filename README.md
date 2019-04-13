# SSF-DAN 
The code and resouces for SSF-DAN , We have uploaded all the models and the evaluate code.  
We will publish the paper and upload the train code soon.

![the result of the cityscape val dataset can be seen as follow](https://github.com/JingangTan/S2-DAN/blob/master/pictures/result.jpg)

![the mIoU comparison with the state-of-art methods](https://github.com/JingangTan/S2-DAN/blob/master/pictures/mIoU_comparison.png)

## System requirements
System : ubuntu 16.04  
Hardware: Nvidia Tesla P100  
Software: Pytorch 0.4.1, CUDA-8.0, cuDNN 5.1 or 6.0, python 3.5  
For more packages information, we prepared them in the requires.sh file. 

## Pretrained model
We released our models in google drive, they can be downloaded [[here]](https://drive.google.com/open?id=1dJuBAqw3XosXSMbRbteUKarQWVqhQd5I)

## DataSet
In the experiments, we use GTA5, cityscape, Synthia, Crosscity datasets,you can download them in the follow links:  
[[GTA5]](https://download.visinf.tu-darmstadt.de/data/from_games/)
[[Cityscape]](https://www.cityscapes-dataset.com/)
[[Synthia]](http://synthia-dataset.net/)
[[Crosscity]](https://yihsinchen.github.io/segmentation_adaptation/#Dataset)

## Usage
1.make a project dir:  
```
mkdir ~/SSF-DAN
```
2.clone the SSF-DAN repository:  
```
git clone https://github.com/JingangTan/SSF-DAN
```
3.download the GTA5 dataset and put it in ```dataset/GTA5/``` and download Cityscape dataset put it in ```dataset/Cityscapes/```. For other dataset, you can take the same operation.

4.create a new virtual enviromen, if you want to use system python enviromen,you can ignore this.
```
sudo pip install virtualenv
virtualenv -p /usr/bin/python3.5 ~/vir-ssf-dan
cd ~/vir-ssf-dan
source bin/activate
cd ~/SSF-DAN
```
5.install the packages in requires.sh:  
```
sh requires.sh
```
## evaluate the model on Cityscape val dataset
```
python evaluate.py
iou.py
```
