# Deflecting Adversarial Attacks with Pixel Deflection

Code for paper: https://arxiv.org/abs/1801.08926

Requirements:

1. Keras 2.0+

2. Scipy 1.0+

(Older version of scipy wavelet transform does not have BayesShrink)

## Example

```
» python main.py -image images/n02443114_00000055.png -map maps/n02443114_00000055.png     
Image: images/n02443114_00000055.png, True Class: 'polecat'
Before Defense ----------------
Predicted Class  black-footed_ferret:0.98 , weasel:0.01 , polecat:0.01 , hamster:0.00 , mink:0.00
After Defense -----------------
Predicted Class  polecat:0.57 , black-footed_ferret:0.37 , weasel:0.02 , hamster:0.01 , mink:0.01
```
---------------------------------------------------------------------

```
» python main.py -process_batch -directory ./images

After recovery Top 1 accuracy is 66.67 and Top 5 accuracy is 100.0

```


## Usage

### Single image

```python
» python main.py -image <image_path> -map <map_path>
```


### Batch usage

```python
» python main.py -process_batch --directory <directory_containing_images>
```


In batch usage the map file is expected to have same name as image file but inside './maps' directory
To generate map see this https://github.com/iamaaditya/image-compression-cnn/blob/master/generate_map.py

### Without map
To use without a map, pass in '-disable_map' argument, e.g:

```python
» python main.py -disable_map -image images/n02443114_00000055.png
Image: images/n02443114_00000055.png, True Class: 'polecat'
Before Defense ----------------
Predicted Class  black-footed_ferret:0.98 , weasel:0.01 , polecat:0.01 , hamster:0.00 , mink:0.00
After Defense -----------------
Predicted Class  polecat:0.55 , black-footed_ferret:0.36 , weasel:0.02 , hamster:0.01 , mink:0.01
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

For any issues please contact aprakash@brandeis.edu


### Impact of Pixel Deflection & localization of attacks

<center>
![pd](https://raw.githubusercontent.com/iamaaditya/iamaaditya.github.io/master/images/image_combine.jpg) 
</center>
