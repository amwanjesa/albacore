import argparse
from json import loads
from os import listdir, mkdir
from os.path import isdir, join
from shutil import copy

import numpy as np

from sklearn.model_selection import train_test_split
from tqdm import tqdm


def read_and_split_data(data_folder):
    truths_file_path = join(data_folder, 'truth.jsonl')

    def get_id_and_class(truth_row):
        truth_row = loads(truth_row)
        return [truth_row['id'], truth_row['truthClass']]

    with open(truths_file_path, 'r') as truths_file:
        truth_class_per_instance = np.array(
            list(map(get_id_and_class, truths_file)))

    ids, classes = truth_class_per_instance[:,
                                            0], truth_class_per_instance[:, 1]
    train_ids, test_ids, _, _ = train_test_split(
        ids, classes, test_size=0.3, random_state=42, stratify=classes)

    return list(train_ids), list(test_ids)


def split_and_store_data(train_ids, test_ids, data_folder, output_train_folder, output_test_folder):
    input_instances_file_path = join(data_folder, 'instances.jsonl')
    input_truths_file_path = join(data_folder, 'truth.jsonl')

    if not isdir(output_train_folder):
        mkdir(output_train_folder)

    if not isdir(output_test_folder):
        mkdir(output_test_folder)

    output_train_instances_file_path = join(
        output_train_folder, 'instances.jsonl')
    output_test_instances_file_path = join(
        output_test_folder, 'instances.jsonl')

    train_image_ids, test_image_ids, ignore_ids = get_image_ids(input_instances_file_path, train_ids, test_ids)

    print("Number of training images: {}".format(len(train_image_ids)))
    print("Number of testing images: {}".format(len(test_image_ids)))

    split_and_store_images(train_image_ids, test_image_ids, data_folder, output_train_folder, output_test_folder)

    output_train_truths_file_path = join(output_train_folder, 'truth.jsonl')
    output_test_truths_file_path = join(output_test_folder, 'truth.jsonl')

    store_sampled_data(train_ids, input_instances_file_path,
                       output_train_instances_file_path, ignore_ids=ignore_ids)
    store_sampled_data(train_ids, input_truths_file_path,
                       output_train_truths_file_path, ignore_ids=ignore_ids)

    store_sampled_data(test_ids, input_instances_file_path,
                       output_test_instances_file_path, ignore_ids=ignore_ids)
    store_sampled_data(test_ids, input_truths_file_path,
                       output_test_truths_file_path, ignore_ids=ignore_ids)

def get_image_ids(instances_path, train_ids, test_ids):
    train_image_ids, test_image_ids = [], []
    ignore_ids = []
    with open(instances_path, 'r', encoding="utf8") as instances_file:
        for line in instances_file:
            line_as_dict = loads(line)
            image_ids = line_as_dict["postMedia"]
            if image_ids:
                if line_as_dict['id'] in train_ids:
                    for image_id in image_ids:
                        train_image_ids.append(image_id.split('/')[1])
                elif line_as_dict['id'] in test_ids:
                    for image_id in image_ids:
                        test_image_ids.append(image_id.split('/')[1])
                else:
                    continue
            else:
                ignore_ids.append(line_as_dict['id'])
    return train_image_ids, test_image_ids, ignore_ids

def split_and_store_images(train_ids, test_ids, data_folder, output_train_folder, output_test_folder):
    input_images_folder = join(data_folder, 'media')
    output_train_images_folder = join(output_train_folder, 'media')
    output_test_images_folder = join(output_test_folder, 'media')

    if not isdir(output_train_images_folder):
        mkdir(output_train_images_folder)

    if not isdir(output_test_images_folder):
        mkdir(output_test_images_folder)

    store_sampled_images(train_ids, input_images_folder,
                         output_train_images_folder)
    store_sampled_images(test_ids, input_images_folder,
                         output_test_images_folder)


def store_sampled_data(ids, input_path, output_path, ignore_ids=[]):
    with open(input_path, 'r', encoding="utf8") as input_file, open(output_path, 'w', encoding="utf8") as output_file:
        for line in input_file:
            line_as_dict = loads(line)
            if line_as_dict['id'] in ids and not line_as_dict['id'] in ignore_ids:
                output_file.write(line)


def store_sampled_images(ids, input_folder, output_folder):
    for image_file in tqdm(listdir(input_folder), desc="Storing images"):
        if image_file in ids:
            src = join(input_folder, image_file)
            dest = join(output_folder, image_file)
            copy(src, dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-folder', help="Folder containing data we want to sample from")
    parser.add_argument('--output-train-folder',
                        help="Folder containing training data we acquired by sampling")
    parser.add_argument('--output-test-folder',
                        help="Folder containing test data we acquired by sampling")
    args = parser.parse_args()

    train_ids, test_ids = read_and_split_data(args.data_folder)
    split_and_store_data(train_ids, test_ids, args.data_folder,
                         args.output_train_folder, args.output_test_folder)
