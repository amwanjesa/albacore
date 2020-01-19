import argparse
import os
import re
import numpy as np
from json import loads
from os.path import isdir, join
import tensorflow as tf
from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

checkpoint_path = 'Pretrained-Show-and-Tell-model/model.ckpt-2000000'
vocab_file = 'Pretrained-Show-and-Tell-model/word_counts.txt'

def caption_image(id_folder, output_folder, data_folder):
    images = []
    for file in os.listdir(data_folder):
        images.append(join(data_folder, file))

    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
      model = inference_wrapper.InferenceWrapper()
      restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                             checkpoint_path)
    g.finalize()

    # Create the vocabulary.
    vocab = vocabulary.Vocabulary(vocab_file)

    with tf.Session(graph=g) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)

        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        generator = caption_generator.CaptionGenerator(model, vocab)

        for filename in images:
            with tf.gfile.GFile(filename, "rb") as f:
                image = f.read()
            captions = generator.beam_search(sess, image)

            # Just take the first caption and display it
            caption = captions[0]
            sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
            sentence = " ".join(sentence)

            for file in os.listdir(data_folder):
                ids_dict = {}
                with open(id_folder, 'r', encoding='utf8') as instances:
                    for instance in instances:
                        instance = loads(instance)
                        photo = instance["postMedia"]
                        photo_name = re.split("[_']", str(photo))
                        #for i in image_name:
                        if len(photo_name) == 7:
                            if photo_name[2] == file:
                                ids_dict[instance["id"]] = sentence
                            elif photo_name[5] == file:
                                ids_dict[instance["id"]] = sentence
                        elif len(photo_name) == 4:
                            if photo_name[2] == file:
                                ids_dict[instance["id"]] = sentence
                        else:
                            continue

    with open(output_folder, 'w', encoding="utf8") as output_file:
        output_file.write(str(ids_dict))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-folder', help="Folder containing the images we want to caption")
    parser.add_argument(
        '--id-folder', help="Folder containing the ids of the images")
    parser.add_argument('--output-folder',
                        help="Folder containing the captions")
    args = parser.parse_args()

    caption_image(args.id_folder, args.output_folder, args.data_folder)