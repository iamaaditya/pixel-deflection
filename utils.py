import matplotlib
import numpy as np
from keras.preprocessing import image
from scipy.misc import imread
import json

def batches(array, batch_size):
    return (array[i:i + batch_size] for i in range(0, len(array), batch_size))

def rgb2ycc(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return ycbcr

def ycc2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    return rgb.dot(xform.T)


def get_img(file):
    img = image.load_img(file, target_size=(224, 224))
    return image.img_to_array(img)

def get_map(file):
    norm = matplotlib.colors.Normalize()
    return norm(imread(file))

def get_imagenet_labels():
    imagenet = json.load(open('./imagenet_labels.json'))
    return {i[0]:i[1] for i in imagenet.values()}

