# addapted from github.com/kwotsin/create_tfrecords/dataset_utils.py

import math
import os
import sys
import tensorflow as tf
import csv
from tensorflow.contrib import slim


def int64_feature(values):
    '''
    :param values: list of int64 values
    :return: TF-Feature of int64's
    '''
    if not isinstance(values, (list, tuple)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_feature(values):
    '''
    :param values: list of floats
    :return: a TF-Feature of floats
    '''
    if not isinstance(values, (list, tuple)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def bytes_feature(values):
    '''
    :param values: a string to be converted to bytes
    :return: TF-Feature of bytes
    '''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format,
                       # height, width,
                       label, left_eye, right_eye):
    '''
    :param image_data: (string) the data that is read from the image
    :param image_format: (string) image file type
    :param height: (int) image height
    :param width:  (int) image width
    :param label: (list or tuple) of (floats) with x, y positions of
                    tobii gaze in that order
    :param left_eye: (list or tuple) of (floats) with x, y positions of
                    left eye in that order
    :param right_eye: (list or tuple) of (floats) with x, y positions of
                    right eye in that order
    :return: A tf.train.Example with the info for the image
    '''
    return tf.train.Example(features=tf.train.Features(
        feature={
            'image/encoded': bytes_feature(image_data),
            'image/format': bytes_feature(bytes(image_format, 'utf-8')),
            'image/label_x': float_feature(label[0]),
            'image/label_y': float_feature(label[1]),
            'image/left_eye_x': int64_feature(left_eye[0]),
            'image/left_eye_y': int64_feature(left_eye[1]),
            'image/right_eye_x': int64_feature(right_eye[0]),
            'image/right_eye_y': int64_feature(right_eye[1])
        }
    ))


def get_eye_centers(dataset_dir, list_of_dirs, csv_name='gazePredictions.csv'):
    '''
    :param dataset_dir: (string) directory that contains the actual data
    :param list_of_dirs: (list of strings) of directories in the dataset_dir
                         from {train, test}_1430_1.txt
    :param csv_name: (string) name of csv file (default=gazePredictions.csv)
    :return: (dictionary) with keys {'left', 'right'} which map to lists of
             dictionaries with keys {'x', 'y'} with approximate
             centers of eyes in images
    '''
    centers = {'left':[], 'right':[]}
    for directory in list_of_dirs:
        path = os.path.join(dataset_dir, os.path.normpath(directory))
        csv_path = os.path.join(path, csv_name)
        with tf.gfile.Open(csv_path, 'r') as f:
            readCSV = csv.reader(f, delimiter=',')
            for line in readCSV:
                clmTracker = line[8:len(line) - 1]
                clmTracker = [float(i) for i in clmTracker]
                clmInt = [int(i) for i in clmTracker]

                centers['left'].append({'x':clmInt[54], 'y':clmInt[55]})
                centers['right'].append({'x':clmInt[64], 'y':clmInt[65]})
    return centers

def make_label_dict(dataset_dir, list_of_dirs, csv_name='gazePredictions.csv'):
    '''
    :param dataset_dir: (string) directory that contains the actual data
    :param list_of_dirs: (list of strings) of directories in the dataset_dir
                         from {train, test}_1430_1.txt
    :param csv_name: (string) name of csv file (default=gazePredictions.csv)
    :return: (dictionary) with keys {'x', 'y'} which map to lists of tobii eyetracking data
    '''
    labels = {'x':[], 'y':[]}
    for directory in list_of_dirs:
        path = os.path.join(dataset_dir, os.path.normpath(directory))
        csv_path = os.path.join(path, csv_name)
        with tf.gfile.Open(csv_path, 'r') as f:
            readCSV = csv.reader(f, delimiter=',')
        for line in readCSV:
            leftX = float(line[2])
            leftY = float(line[3])
            rightX = float(line[4])
            rightY = float(line[5])

            labelX = (leftX + rightX) / 2.
            labelY = (leftY + rightY) / 2.

            labels['x'].append(labelX)
            labels['y'].append(labelY)

    return labels


def get_filenames(dataset_dir, list_of_dirs, csv_name='gazePredictions.csv'):
    '''
    :param dataset_dir: (string) directory that contains the actual data
    :param list_of_dirs: (list of strings) of directories in the dataset_dir
                         from {train, test}_1430_1.txt
    :param csv_name: (string) name of csv file (default=gazePredictions.csv)
    :return: (list) with path to image files
    '''
    image_filenames = []
    for directory in list_of_dirs:
        path = os.path.join(dataset_dir, os.path.normpath(directory))
        csv_path = os.path.join(path, csv_name)
        with tf.gfile.Open(csv_path, 'r') as f:
            readCSV = csv.reader(f, delimiter=',')
        for line in readCSV:
            image_filenames.append(os.path.join(dataset_dir, os.path.normpath(line[0])) )

    return image_filenames


def get_dataset_filename(dataset_dir, split_name, shard_num, tfrecord_filename, num_shards):
    '''
    :param dataset_dir: (string) location of data directory
    :param split_name: (string) which split
    :param shard_num: (int) which shard
    :param tfrecord_filename: (string) base name of data
    :param num_shards: (int) number of shards to break the data set up into
    :return: string of filename for data file
    '''
    output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
        tfrecord_filename, split_name, shard_num, num_shards)
    return os.path.join(dataset_dir, output_filename)


def convert_dataset(split_name, filenames, labels, eye_centers, dataset_dir,
                    tfrecord_filename, num_shards):
    '''
    :param split_name: (string) which split
    :param filenames: (list) of image file names
    :param labels: (dictionary) of (tuples or lists) with the x and y tobii labels
    :param eye_centers: (dictionary) of (lists) of (dictionaries with the x and y
                        pixel coordinate of left and right eyes
    :param dataset_dir: (string) location of data directory
    :param tfrecord_filename: (string) base name of data
    :param num_shards: (int) number of shards to break the data set up into

    Make TFRecord files of dataset

    '''
    assert split_name in ['train', 'test']

    num_per_shard = int(math.ceil(len(filenames) / float(num_shards)))

    with tf.Graph().as_default():

        with tf.Session('') as sess:

            for shard_num in range(num_shards):
                output_filename = get_dataset_filename(dataset_dir, split_name, shard_num,
                                                       tfrecord_filename=tfrecord_filename,
                                                       num_shards=num_shards)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_idx = shard_num * num_per_shard
                    end_idx = min((shard_num+1) * num_per_shard, len(filenames))
                    for i in range(start_idx, end_idx):
                        if i % 1000 == 0:
                            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                                i+1, len(filenames), shard_num))
                            sys.stdout.flush()

                        image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()

                        label = (labels['x'][i], labels['y'][i])
                        left_eye = (eye_centers['left'][i]['x'], eye_centers['left'][i]['y'])
                        right_eye = (eye_centers['right'][i]['x'], eye_centers['right'][i]['y'])

                        example = image_to_tfexample(image_data, 'png', label,
                                                     left_eye=left_eye, right_eye=right_eye)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()

def dataset_exists(dataset_dir, num_shards, output_filename):
    '''
    :param dataset_dir: (string) location of data directory
    :param num_shards: (int) number of shards to break the data set up into
    :param output_filename: (string) format of dataset files
    :return: boolean indicating whether or not the dataset files exists in dataset_dir
    '''
    for split_name in ['train', 'test']:
        for shard_num in range(num_shards):
            tfrecord_filename = get_dataset_filename(dataset_dir,
                                    split_name, shard_num, output_filename, num_shards)
            if not tf.gfile.Exists(tfrecord_filename):
                return False
    return True
