from __future__ import division
import os, sys, argparse, multiprocessing
from joblib import Parallel, delayed
from glob import glob
import numpy as np
from utils import *
from methods import pixel_deflection, denoiser
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silences the TF INFO messages

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-image'            , type=str ,  default= 'images/n02447366_00008562.png')
    parser.add_argument('-map'              , type=str ,  default= 'maps/n02447366_00008562.png')
    parser.add_argument('-directory'        , type=str ,  default= './images/')
    parser.add_argument('-disable_map'      , action='store_true')
    parser.add_argument('-process_batch'    , action='store_true')
    parser.add_argument('-classifier'       , type=str ,  default= 'resnet50',    help='options: resnet50, inception_v3, vgg19, xception')
    parser.add_argument('-denoiser'         , type=str ,  default= 'wavelet',     help='options: wavelet, TVM, bilateral, deconv, NLM')
    parser.add_argument('-batch_size'       , type=int,   default= 64)
    parser.add_argument('-sigma'            , type=float, default= 0.04)
    parser.add_argument('-window'           , type=int,   default= 10)
    parser.add_argument('-deflections'      , type=int,   default= 200)
    return parser.parse_args()

def classify_images(images, class_names, supress_print=False):
    total, top1, top5 = 0,0,0
    images = preprocess_input(np.stack(images,axis=0))
    predictions = decode_predictions(model.predict(images),top=5)
    for p,true_class in zip(predictions,class_names):
        ans= [' {0}:{1:.2f} '.format(i[1],i[2]) for i in p]
        if not supress_print: print('Predicted Class {}'.format(','.join(ans)))
        total += 1
        r = [i[0] for i in p]
        top1 += int(true_class  == r[0])
        top5 += int(true_class in r)
    return top1*100.0/total, top5*100.0/total

def process_image(args, image_name, defend=True):
    image = get_img(image_name)
    # assumes map is same name as image but inside maps directory
    if not args.disable_map:
        map   = get_map('./maps/'+image_name.split('/')[-1])
    else:
        map   = np.zeros((image.shape[0],image.shape[1]))

    if defend:
        img = pixel_deflection(image, map, args.deflections, args.window, args.sigma)
        return denoiser(args.denoiser, img/255.0, args.sigma)*255.0
    else:
        return image

def process_image_parallel(args):
    num_cores = multiprocessing.cpu_count()
    scores = []
    for image_names in batches(glob(args.directory+'/*'), args.batch_size):
        images = Parallel(n_jobs=num_cores)(delayed(process_image)(args, image_name) for image_name in image_names)
        class_names = [image_name.split('/')[-1].split('_')[0] for image_name in image_names]
        scores.append(classify_images(images, class_names, supress_print=True))

    t1,t5 = sum([i[0] for i in scores])/len(scores), sum([i[1] for i in scores])/len(scores)
    print('After recovery Top 1 accuracy is {0:.2f} and Top 5 accuracy is {1:.2f}'.format(t1,t5))

if __name__ == '__main__':
    args = get_arguments()
    if args.classifier == 'resnet50':
        from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
        model = ResNet50(weights='imagenet')
    elif args.classifier == 'inception_v3':
        from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
        model = InceptionV3(weights='imagenet')
    elif args.classifier == 'vgg19':
        from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
        model = VGG19(weights='imagenet')
    elif args.classifier == 'xception':
        from keras.applications.xception import Xception, preprocess_input, decode_predictions
        model = Xception(weights='imagenet')
    else:
        raise Exception('Incorrect classifier mentioned. Options: resnet50, inception_v3, vgg19, xception')
    
    imagenet_labels = get_imagenet_labels()
    if args.process_batch:
        images = process_image_parallel(args)
    else:
        class_name = args.image.split('/')[-1].split('_')[0]
        print("Image: {}, True Class: '{}'".format(args.image, imagenet_labels[class_name]))
        print('Before Defense :')
        image = process_image(args, args.image, defend=False)
        classify_images([image], [class_name])
        print('After Defense :')
        image = process_image(args, args.image, defend=True)
        classify_images([image], [class_name])
