# -*- coding: utf-8 -*-
# Copyright 2019 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from nltk.corpus import wordnet as wn
from pytorch_pretrained_biggan.utils import IMAGENET

from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample, one_hot_from_names
from transformers import is_tf_available, AutoTokenizer
from transformers import BertForSequenceClassification, BertTokenizer

if is_tf_available():
    from transformers import TFAutoModel

    MODEL_CLASS = TFAutoModel
else:
    from transformers import AutoModel

    MODEL_CLASS = AutoModel

vocab_size = 1000
MODELPATH = "./model/pytorch_model.bin"
CONFIGPATH = "./model/config.json"
MAX_LEN = 30  # todo
MODELFOLDER = "./model/"


def generate_image(dense_class_vector=None, name=None, noise_seed_vector=None, truncation=0.4,
                   gan_model=None, pretrained_gan_model_name='biggan-deep-128'):
    """ Utility function to generate an image (numpy uint8 array) from either:
        - a name (string): converted in an associated ImageNet class and then
            a dense class embedding using BigGAN's internal ImageNet class embeddings.
        - a dense_class_vector (torch.Tensor or np.ndarray with 128 elements): used as a replacement of BigGAN internal
            ImageNet class embeddings.
        
        Other args:
            - noise_seed_vector: a vector used to control the seed (seed set to the sum of the vector elements)
            - truncation: a float between 0 and 1 to control image quality/diversity tradeoff (see BigGAN paper)
            - gan_model: a BigGAN model from pytorch_pretrained_biggan library.
                If None a model is instanciated from a pretrained model name given by `pretrained_gan_model_name`
                List of possible names: https://github.com/huggingface/pytorch-pretrained-BigGAN#models
            - pretrained_gan_model_name: shortcut name of the GAN model to instantiate if no gan_model is provided. Default to 'biggan-deep-128'
    """
    seed = int(noise_seed_vector.sum().item()) if noise_seed_vector is not None else None
    noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1, seed=seed)
    noise_vector = torch.from_numpy(noise_vector)

    if gan_model is None:
        gan_model = BigGAN.from_pretrained(pretrained_gan_model_name)


    if name is not None:
        class_vector = one_hot_from_names([name], batch_size=1)
        class_vector = torch.from_numpy(class_vector)
        dense_class_vector = gan_model.embeddings(class_vector)
        # input_vector = torch.cat([noise_vector, gan_class_vect.unsqueeze(0)], dim=1)
        # dense_class_vector = torch.matmul(class_vector, gan.embeddings.weight.t())
    else:
        if isinstance(dense_class_vector, np.ndarray):
            dense_class_vector = torch.tensor(dense_class_vector)
        dense_class_vector = dense_class_vector.view(1, 128)

    input_vector = torch.cat([noise_vector, dense_class_vector], dim=1)

    # Generate an image
    with torch.no_grad():
        output = gan_model.generator(input_vector, truncation)
    output = output.cpu().numpy()
    output = output.transpose((0, 2, 3, 1))
    output = ((output + 1.0) / 2.0) * 256
    output.clip(0, 255, out=output)
    output = np.asarray(np.uint8(output[0]), dtype=np.uint8)
    return output

def print_image(numpy_array):
    """ Utility function to print a numpy uint8 array as an image
    """
    img = Image.fromarray(numpy_array)
    plt.imshow(img)
    plt.show()
    time.sleep(1.5)
    plt.close()

    return img



def load_sentence_model(modelpath=None, config_path=None, modelfolder=MODELFOLDER):
    """Loads fine-tuned mapping model"""

    if modelpath is None:
        modelpath = MODELPATH

    if config_path is None:
        config_path = CONFIGPATH

    if modelfolder is None:
        modelfolder = MODELFOLDER

    model = BertForSequenceClassification.from_pretrained(modelpath, config=config_path)
    tokenizer = BertTokenizer.from_pretrained(modelfolder)

    print("Sequence classification model succesfully loaded.")

    return model, tokenizer


def one_hot(index, vocab_size=vocab_size):
    index = np.asarray(index)
    if len(index.shape) == 0:
        index = np.asarray([index])
    assert len(index.shape) == 1
    num = index.shape[0]
    output = np.zeros((num, vocab_size), dtype=np.float32)
    output[np.arange(num), index] = 1
    return output

def mapping_model(text, gan_model, model, tokenizer, max_len=MAX_LEN, pretrained_gan_model_name='biggan-deep-128',
                  verbose=0):

    if gan_model is None:
        gan_model = BigGAN.from_pretrained(pretrained_gan_model_name)

    inputs = tokenizer(text, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt")

    # predict:
    with torch.no_grad():
        logits = model(**inputs).logits

    # Imagenet label:
    label_id = int(np.argmax(logits[0]))

    if verbose:
        class_to_synset = dict((v, wn.synset_from_pos_and_offset('n', k)) for k, v in IMAGENET.items())
        label = class_to_synset[label_id]
        print(f"{text} --> {label}")

    one_hot_label = one_hot(label_id)
    class_embed = gan_model.embeddings(torch.tensor(one_hot_label))  # shape (batch size, 128)

    return class_embed, inputs["input_ids"]


def text_to_image(text, mapping_model=mapping_model,
                  lm_model=None, lm_tokenizer=None, gan_model=None, pretrained_gan_model_name='biggan-deep-128',
                  truncation=0.4, verbose=0,
                  noise_seed=None): # todo define max token len
    """ Utility function to generate an image from text using a mapping model"""

    if lm_tokenizer is None or lm_model is None:
        lm_model, lm_tokenizer = load_sentence_model()

    if gan_model is None:
        gan_model = BigGAN.from_pretrained(pretrained_gan_model_name)

    mapping_output, tokens = mapping_model(text, gan_model=gan_model, model=lm_model, tokenizer=lm_tokenizer,
                                           verbose=verbose)

    tokens = tokens.numpy()
    # Now generate an image (a numpy array)
    numpy_image = generate_image(mapping_output,
                                 gan_model=gan_model,
                                 pretrained_gan_model_name=pretrained_gan_model_name,
                                 truncation=truncation,
                                 noise_seed_vector=tokens if noise_seed is None else noise_seed)
    img = print_image(numpy_image)

    return img
