from __future__ import print_function
import os
import tensorflow as tf
import glob
import numpy as np
import random
import operator

def get_file_list(data_dir, pattern):
    assert os.path.exists(data_dir), 'Directory {} not found.'.format(data_dir)

    file_list = []
    file_glob = os.path.join(data_dir, pattern)
    file_list.extend(glob.glob(file_glob))

    assert file_list, 'No file found in {}.'.format(file_glob)

    file_list.sort()

    return file_list


def tf_read_image(file_name, channels=3, img_type=tf.float32, div_val=255.0, outputsize=None, random_flip_key=None):
    image_string = tf.read_file(file_name)
    image_decoded = tf.div(tf.cast(tf.image.decode_png(image_string, channels=channels), img_type), div_val)
    if outputsize:
        image_decoded = tf.cast(tf.image.resize_images(image_decoded, outputsize, align_corners=True, method=0), img_type)
    if random_flip_key is not None:
        image_decoded = tf.cond(tf.less(random_flip_key[0], .5), lambda: tf.image.flip_up_down(image_decoded), lambda: image_decoded)
        image_decoded = tf.cond(tf.less(random_flip_key[1], .5), lambda: tf.image.flip_left_right(image_decoded), lambda: image_decoded)
        image_decoded = tf.cond(tf.less(random_flip_key[2], .5), lambda: tf.image.transpose_image(image_decoded), lambda: image_decoded)
    return image_decoded


def read_dataset(label_value, data_dir, pattern='*', shuffle_seed=None, subset=None, begin=0):

    file_names = get_file_list(data_dir, pattern)
    if shuffle_seed:
        random.seed(shuffle_seed)
        random.shuffle(file_names)
    if subset:
        file_names = file_names[begin:begin+subset]
    instance_num = len(file_names)
    labels = tf.constant(label_value, shape=[instance_num])
    dataset = tf.data.Dataset.from_tensor_slices((file_names, labels))

    print('Read {} instances from {}'.format(instance_num, data_dir))

    return dataset, instance_num


def read_dataset_withmsk(data_dir, pattern, msk_replace, shuffle_seed=None, subset=None):

    image_names = get_file_list(data_dir, pattern)
    if shuffle_seed:
        random.seed(shuffle_seed)
        random.shuffle(image_names)
    if subset:
        image_names = image_names[:subset]
    instance_num = len(image_names)
    label_names = image_names
    for entry in msk_replace:
        label_names = [name.replace(entry[0], entry[1], 1) for name in label_names]
    dataset = tf.data.Dataset.from_tensor_slices((image_names, label_names))

    print('Read {} instances from {}'.format(instance_num, data_dir))

    return dataset, instance_num 


def read_dataset_with2msk(data_dir, pattern, msk_replace, shuffle_seed=None, subset=None):

    image_names = get_file_list(data_dir, pattern)
    if shuffle_seed:
        random.seed(shuffle_seed)
        random.shuffle(image_names)
    if subset:
        image_names = image_names[:subset]
    instance_num = len(image_names)
    label1_names = image_names
    label2_names = image_names
    for entry in msk_replace[0]:
        label1_names = [name.replace(entry[0], entry[1], 1) for name in label1_names]
    for entry in msk_replace[1]:
        label2_names = [name.replace(entry[0], entry[1], 1) for name in label2_names]

    dataset = tf.data.Dataset.from_tensor_slices((image_names, label1_names, label2_names))

    print('Read {} instances from {}'.format(instance_num, data_dir))

    return dataset, instance_num 


def read_image(file_name, label, outputsize=None, random_flip=False):
    random_flip_key = tf.random_uniform([3,], 0, 1.0) if random_flip else None
    image_decoded = tf_read_image(file_name, outputsize=outputsize, random_flip_key=random_flip_key)
    return image_decoded, label


def read_image_withmsk(image_name, label_name, outputsize=None, random_flip=False):
    random_flip_key = tf.random_uniform([3,], 0, 1.0) if random_flip else None
    image_decoded = tf_read_image(image_name, outputsize=outputsize, random_flip_key=random_flip_key)
    label_decoded = tf_read_image(label_name, channels=1, img_type=tf.int32, div_val=255, outputsize=outputsize, random_flip_key=random_flip_key)
    return image_decoded, label_decoded, image_name


def read_image_with2msk(image_name, label1_name, label2_name, outputsize=None, random_flip=False):
    random_flip_key = tf.random_uniform([3,], 0, 1.0) if random_flip else None
    image_decoded = tf_read_image(image_name, outputsize=outputsize, random_flip_key=random_flip_key)
    label1_decoded = tf_read_image(label1_name, channels=1, img_type=tf.int32, div_val=255, outputsize=outputsize, random_flip_key=random_flip_key)
    label2_decoded = tf_read_image(label2_name, channels=1, img_type=tf.int32, div_val=255, outputsize=outputsize, random_flip_key=random_flip_key)
    return image_decoded, label1_decoded, label2_decoded, image_name


        