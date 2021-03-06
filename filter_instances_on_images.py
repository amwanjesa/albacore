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
    

    return ids, []


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

    output_train_truths_file_path = join(output_train_folder, 'truth.jsonl')
    output_test_truths_file_path = join(output_test_folder, 'truth.jsonl')

    store_sampled_data(train_ids, input_instances_file_path,
                       output_train_instances_file_path, check_for_images=True)
    store_sampled_data(train_ids, input_truths_file_path,
                       output_train_truths_file_path)

    store_sampled_data(test_ids, input_instances_file_path,
                       output_test_instances_file_path, check_for_images=True)
    store_sampled_data(test_ids, input_truths_file_path,
                       output_test_truths_file_path)


def store_sampled_data(ids, input_path, output_path, check_for_images=False):
    with open(input_path, 'r', encoding="utf8") as input_file, open(output_path, 'w', encoding="utf8") as output_file:
        for line in input_file:
            line_as_dict = loads(line)
            if line_as_dict['id'] in ids:
                if check_for_images and not line_as_dict['postMedia']:
                    continue
                output_file.write(line)


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
