from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import os

import tensorflow as tf
import pandas as pd

from medium_show_and_tell_caption_generator.caption_generator import CaptionGenerator
from medium_show_and_tell_caption_generator.model import ShowAndTellModel
from medium_show_and_tell_caption_generator.vocabulary import Vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("model_path", "../model/show-and-tell.pb", "Model graph def path")
tf.flags.DEFINE_string("vocab_file", "../etc/word_counts.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "../imgs/",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

idcol =[]
predol1 =[]
predol2 =[]
predol3 =[]
predol4 =[]
temp =[]

def main(_):
    model = ShowAndTellModel(FLAGS.model_path)
    vocab = Vocabulary(FLAGS.vocab_file)
    filenames = _load_filenames()

    generator = CaptionGenerator(model, vocab)

    for filename in filenames:
        with tf.gfile.GFile(filename, "rb") as f:
            image = f.read()
        captions = generator.beam_search(image)
        print("Captions for image %s:" % os.path.basename(filename))
        for i, caption in enumerate(captions):
            # Ignore begin and end tokens <S> and </S>.
            sentence = [vocab.id_to_token(w) for w in caption.sentence[1:-1]]
            sentence = " ".join(sentence)
            if i == 0:
                # f1.write("%s \n" % sentence)
                print("this is---",os.path.basename(filename)[:os.path.basename(filename).find(".")], sentence)
                idcol.append(os.path.basename(filename)[:os.path.basename(filename).find(".")])
                predol1.append(sentence)
            if i == 1:
                predol2.append(sentence)
            if i == 2:
                predol3.append(sentence)
            if i == 3:
                predol4.append(sentence)
            print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

    print(idcol,predol1)

    predDF = pd.DataFrame()
    predDF['id'] = idcol
    predDF['pred_caption1'] = predol1
    predDF['pred_caption2'] = predol2
    predDF['pred_caption3'] = predol3
    predDF['pred_caption4'] = predol4

    predDF.to_csv('MpredCapt.csv')


def _load_filenames():
    filenames = []
    for file_pattern in FLAGS.input_files.split(","):
        filenames.extend(tf.gfile.Glob(file_pattern))
    logger.info("Running caption generation on %d files matching %s",
                len(filenames), FLAGS.input_files)
    return filenames


def _load_filenames():
    filenames = []
    fn =[]
    directory = FLAGS.input_files
    for filename in os.listdir(directory):
        if filename.endswith(".JPG") or filename.endswith(".png") or filename.endswith(".jpg"):
            pathname = os.path.join(directory, filename)
            filenames.append(pathname)
            print(filenames)
            continue
        else:
            continue
    return filenames

if __name__ == "__main__":
    tf.app.run()

