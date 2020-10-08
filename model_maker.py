import gpt_2_simple as gpt2
import os
import requests
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import device_lib

###########################################
#IDEAL TO USE THIS SCRIPT WITH A MACHINE WITH GPU
#MODEL CAN BE TRAINED USING GOOGLE COLLAB
#https://github.com/minimaxir/gpt-2-simple
##############################################

model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
	print(f"Downloading {model_name} model...")
	gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/

# def read_file():
#     file = open('./lyrics_txt_files/john_mayer.txt', 'r')
#
#     return file.read()
#
# file_name = read_file()

#select the file you want to train the model on
file_name = './lyrics_txt_files/john_mayer.txt'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction=0.77
config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
sess = tf.compat.v1.Session(config=config)
#code above adapted from https://www.youtube.com/watch?v=LjkubM5IIos

# sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              file_name,
              model_name=model_name,
              steps=1000)   # steps is max number of training steps

gpt2.generate(sess)
