# Deflecting Adversarial Attacks with Pixel Deflection

![deflecting pixels](https://i.imgur.com/BhxmVwx.png)

Code for paper: https://arxiv.org/abs/1801.08926 

Blog with demo: https://iamaaditya.github.io/2018/02/demo-for-pixel-deflection/

Requirements:

1. Keras 2.0+

2. Scipy 1.0+

(Older version of scipy wavelet transform does not have BayesShrink)

## Example


```python
» python main.py                                                                                                  02:24:51 on 2018-02-01
Image: images/n02447366_00008562.png, True Class: 'badger'
Before Defense :
Predicted Class  skunk:0.59 , badger:0.15 , polecat:0.07 , wood_rabbit:0.02 , weasel:0.01
After Defense :
Predicted Class  badger:0.90 , skunk:0.08 , polecat:0.01 , weasel:0.00 , mink:0.00
```
---------------------------------------------------------------------

```
» python main.py -process_batch -directory ./images

After recovery Top 1 accuracy is 66.67 and Top 5 accuracy is 100.0

```
---------------------------------------------------------------------



## Usage

### Single image

```python
» python main.py -image <image_path> -map <map_path>
```


### Batch usage

```python
» python main.py -process_batch -directory <directory_containing_images>
```


In batch usage the map file is expected to have same name as image file but inside './maps' directory
To generate map see this https://github.com/iamaaditya/image-compression-cnn/blob/master/generate_map.py

### Without map
To use without a map, pass in '-disable_map' argument, e.g:

```python
» python main.py -disable_map                                                                                     02:26:02 on 2018-02-01
Image: images/n02447366_00008562.png, True Class: 'badger'
Before Defense :
Predicted Class  skunk:0.59 , badger:0.15 , polecat:0.07 , wood_rabbit:0.02 , weasel:0.01
After Defense :
Predicted Class  badger:0.88 , skunk:0.11 , polecat:0.01 , weasel:0.00 , mink:0.00
```


### Detailed usage

```
» python main.py --help
  -h, --help            show this help message and exit
  -image
  -map 
  -directory
  -process_batch
  -disable_map
  -classifier 
    options: resnet50, inception_v3, vgg19, xception
  -denoiser 
    options: wavelet, TVM, bilateral, deconv, NLM
  -batch_size 
  -sigma 
  -window 
  -deflections 
```  

### Impact of Pixel Deflection & localization of attacks

<center>
<img src="https://raw.githubusercontent.com/iamaaditya/iamaaditya.github.io/master/images/image_combine.jpg" /> 
</center>


For any issues please contact aprakash@brandeis.edu
