# Deflecting Adversarial Attacks with Pixel Deflection

Code for paper: https://arxiv.org/abs/1801.08926

## Example

```
» python main.py -image images/n02443114_00000055.png -map maps/n02443114_00000055.png     

Image: images/n02443114_00000055.png, True Class: 'polecat'

Before Defense ----------------

Predicted Class [('black-footed_ferret', 0.976), ('weasel', 0.012), ('polecat', 0.008), ('hamster', 0.002), ('mink', 0.000)]

After Defense -----------------

Predicted Class [('polecat', 0.580), ('black-footed_ferret', 0.351), ('weasel', 0.020), ('hamster', 0.008), ('mink', 0.006)]

```

---------------------------------------------------------------------


```
» python main.py -process_batch -directory ./images

After recovery Top 1 accuracy is 66.6666666667 and Top 5 accuracy is 100.0

```


## Usage

### Single image

```python
» python main.py -image <image_path> -map <map_path>
```


### Batch usage

» python main.py -process_batch --directory <directory_containing_images>

In batch usage the map file is expected to have same name as image file but inside './maps' directory


To generate map see this https://github.com/iamaaditya/image-compression-cnn/blob/master/generate_map.py


### Detailed usage

```
» python main.py --help
  -h, --help            show this help message and exit
  
  -image
  
  -map 
  
  -directory
  
  -process_batch

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

