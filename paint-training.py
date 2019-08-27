from __future__ import print_function
from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
import cv2 as cv
from keras.layers import AveragePooling2D
from keras.applications import vgg19
from keras import backend as K
from IPython import embed
import os
############## Initializing ####################

base_image_path = "input/painting/balloon1/input.jpg"
style_reference_image_path = "input/painting/balloon1/background.jpg"
mask_image_path = "input/painting/balloon1/bw_mask.jpg"
save_path = "input/painting/balloon1/result.jpg"
iterations = 30

total_variation_weight = 1
style_weight = 1
content_weight = 0.05

# dimensions of the generated picture.
height, width = load_img(base_image_path).size
# img_nrows = width // 5
# img_ncols = height // 5
img_nrows = 150
img_ncols = int(height / (width / img_nrows))
# You can divide more if OOM

def preprocess_image(image_path, ifMask = False):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    if ifMask:
        # masked part should be black 
        img[np.where(img != 255)] = 1
        img[np.where(img == 255)] = 0
        img = np.expand_dims(img, axis=0)
    else:
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

base_image = K.variable(preprocess_image(base_image_path))
mask_image = K.variable(preprocess_image(mask_image_path, ifMask=True))
style_reference_image = K.variable(preprocess_image(style_reference_image_path))
combination_image = K.variable(preprocess_image(base_image_path))

input_tensor = K.concatenate([base_image, style_reference_image, combination_image], axis=0)

# embed()
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)

print('Model loaded.')

# Get features from all layers
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
print(outputs_dict)
# we use the pre-trained parameters on imagenet, so freeze training
for layer in model.layers:
	layer.trainable = False

def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


# resize the mask into the right size for each conv
def halve_mask(mask):
    h, w, ch = mask.shape
    return cv.resize(mask, (w // 2, h // 2))

def convol(mask, times):
    x = K.variable(mask)
    if K.ndim(x) == 3:
        x = K.expand_dims(x, axis=0)
    # embed()
    for i in range(times):
        x = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    # return K.eval(K.squeeze(x, axis=0))
    return K.eval(K.squeeze(x, axis=0))

def get_mask_ftrs(mask):
    ftrs = []
    mask = halve_mask(convol(mask,2))
    mask = halve_mask(convol(mask,2))
    mask = convol(mask,1)
    ftrs.append(K.variable(mask[:, :, 0][:, :, np.newaxis]))
    mask = halve_mask(convol(mask,3))
    mask = convol(mask,1)
    ftrs.append(K.variable(mask[:, :, 0][:, :, np.newaxis]))
    # mask = halve_mask(convol(mask,3))
    # mask = convol(mask,1)
    # ftrs.append(K.variable(mask[:, :, 0][:, :, np.newaxis]))
    return ftrs

mask_features = get_mask_ftrs(mask_image)
# embed()

# style loss
# style_feature_layers = ['block3_conv1','block4_conv1', 'block5_conv1']
style_feature_layers = ['block3_conv1','block4_conv1']
# content loss
content_feature_layer = 'block4_conv1'
# extract features from layers:

input_features = [outputs_dict[layer_name][0, :, :, :] for layer_name in style_feature_layers]
style_features = [outputs_dict[layer_name][1, :, :, :] for layer_name in style_feature_layers]


def get_patches(x, ks=3, stride=1, padding=1):
    # embed()
    ch, n1, n2 = x.shape
    y = np.zeros((ch, n1  +2 * padding, n2 + 2 * padding))
    y[:, padding:n1 + padding, padding:n2 + padding] = x
    start_idx = np.array([j + (n2 + 2 * padding) * i for i in range(0,n1 - ks + 1 + 2 * padding,stride) for j in range(0, n2 - ks + 1 + 2 * padding,stride) ])
    grid = np.array([j + (n2 + 2 * padding) * i + (n1 + 2 * padding) * (n2 + 2 * padding) * k for k in range(0, ch) for i in range(ks) for j in range(ks)])
    mask = start_idx[:,None] + grid[None,:]
    return np.take(y, mask)


def match_ftrs(inp_ftrs,sty_ftrs):
    res = []
    for l_inp,s_inp in zip(inp_ftrs,sty_ftrs):
        # embed()
        l_inp = get_patches(K.eval(l_inp))
        s_inp = get_patches(K.eval(s_inp))
        scals = np.dot(l_inp, s_inp.T)
        norms_in = np.sqrt((l_inp ** 2).sum(1))
        norms_st = np.sqrt((s_inp ** 2).sum(1))
        # embed()
        cosine_sim = scals / (1e-15 + norms_in[:, np.newaxis] * norms_st[np.newaxis, :])
        ind_max = np.argmax(cosine_sim, axis=0)
        res.append(ind_max)
    return res

map_features = match_ftrs(input_features, style_features)
# embed()
def map_style(style_features, map_features):
    res = []
    for sf, mapf in zip(style_features, map_features):
        sf = K.eval(sf)
        ori_shape = sf.shape
        sf = sf.reshape(list(sf.shape)[0],-1)
        sf = sf[:,mapf]
        sf = sf.reshape(ori_shape)
        res.append(K.variable(sf))
    return res

sty_ftrs = map_style(style_features, map_features)

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image



def style_loss(style, combination, mask_feature):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style * mask_feature)
    C = gram_matrix(combination * mask_feature)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image


def content_loss(base, combination, mask_features):
    return K.sum(K.square(combination * mask_features - base * mask_features))

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent
# this loss function should not be used in the first pass

def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


######################## Optimization Part ########################

loss = K.variable(0.0)
content_layer_feature = outputs_dict[content_feature_layer]
base_image_feature = content_layer_feature[0, :, :, :]
combination_feature = content_layer_feature[2, :, :, :]
loss += content_weight * content_loss(base_image_feature,
                                      combination_feature,
                                      mask_features[1]) # mask should match the content feature

for layer_name, mask_feature, style_reference_feature in zip(style_feature_layers, mask_features, sty_ftrs):
    # embed()
    layer_feature = outputs_dict[layer_name]
    combination_feature = layer_feature[2, :, :, :]
    sl = style_loss(style_reference_feature, combination_feature, mask_feature)
    loss += (style_weight / len(style_feature_layers)) * sl

# Don't use this loss function in the first pass
# loss += total_variation_weight * total_variation_loss(combination_image)

# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)


def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()


x = preprocess_image(base_image_path)

if not os.path.exists('mask_result'):
    os.makedirs('mask_result')

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # save current generated image
    img = deprocess_image(x.copy())
    fname = 'mask_result/result_at_iteration_%d.png' % i
    save_img(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))

save_img(save_path, img)