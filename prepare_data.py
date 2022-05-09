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

import argparse
import logging
import os

import nltk
import numpy as np
from nltk.corpus import wordnet as wn
from pytorch_pretrained_biggan.utils import IMAGENET
from tqdm import tqdm
from transformers import AutoTokenizer, is_torch_available
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from keytotext import pipeline

nltk.download('wordnet')
nltk.download('punkt')
from nltk import sent_tokenize

# Initiate models we will use to generate sentences
kw2sent = pipeline("k2t-new")  # loading pre-trained key 2 text model
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=gpt_tokenizer.eos_token_id)

if is_torch_available():
    import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PATTERNS = [
    'i saw a <WORD>.',
]


def pos_filter(sentence, word, target_pos=["NN", "NNS"]):
    """
    :param sentence: input sentence
    :param word: target word
    :param target_pos: target part of speech (in our case nouns)
    :return: POS check for target word in a sentence
    """

    wordsList = nltk.word_tokenize(sentence.lower())
    tagged = nltk.pos_tag(wordsList)
    word_tagged = [x for x in tagged if word in x[0]]

    if not word_tagged:
        return None
    if word_tagged[0][-1] in target_pos:
        return 1
    return 0


def sentences_from_kw2sent(words, model_):
    """
    :param words: a list of words to generate sentence with
    :param model_: keyword2sentence model
    :return: a list of sentences containing our target words
    """

    if not model_:
        model_ = kw2sent

    generated = []
    for n in [1, 4, 10]:  # try with different n most probable words
        params = {"do_sample": False, "num_beams": n, "no_repeat_ngram_size": 1,
                  "early_stopping": True}  # decoding params
        generated += [model_(words, **params)]
        params = {"do_sample": True, "num_beams": n, "no_repeat_ngram_size": 1, "early_stopping": True}
        generated += [model_(words, **params)]

    generated = list(set(generated))
    generated = [x for x in generated if all(word in x.lower() for word in words)]
    # Filter out sentences which do not contain our target word (keep same POS, different form is fine
    # - e.g. word = animal. Keep sent with 'animals' , remove sent with 'animalic'
    relevant = [x for x in generated if all(pos_filter(x, word) for word in words)]

    return relevant


def sentences_from_model(word, tokenizer_=None, model_=None, vowels=['a', 'e', 'i', 'o', 'u']):
    """
    :param word: word we want to generate a sentence with
    :param tokenizer_: tokenizer to use for sentence generation
    :param model_: model to use for sentence generation (e.g. gpt2)
    :return: a list of sentences generated from model containing target word filtered for POS match
    # Note: Only works for individual words
    """

    if not tokenizer_:
        tokenizer_ = gpt_tokenizer
    if not model_:
        model_ = gpt_model

    # Better sentences generated when inputting A + word rather than simply our word:
    if word[0] in vowels:
        input_text = "An " + word
    else:
        input_text = "A " + word

    input_ids = tokenizer_.encode(input_text, return_tensors='tf')

    beam_output = model_.generate(
        input_ids,
        max_length=25,
        num_beams=5,
        no_repeat_ngram_size=1,
        early_stopping=True
    )
    res = tokenizer_.decode(beam_output[0], skip_special_tokens=True)

    sentences = sent_tokenize(res)
    sentences = [s for s in sentences if pos_filter(s, word)]  # filter using POS

    return sentences


def sentences_from_Wordnet(word):
    """
    :param word: a word
    :return: all example sentences from Wordnet which contain our target word
    """
    sentences = []
    synsets = wn.synsets(word, pos=wn.NOUN)
    for synset_ in synsets:
        sentences += [s for s in synset_.examples() if word in s]

    return sentences


def generate_dataset(patterns=None, lm_tokenizer=None, pretrained_lm_model_name='distilbert-base-uncased'):
    """ Utility function to generate a dataset of string sequence associated to each imagenet class Args: - patterns:
    a list of string to use as patterns in which <WORD> will be replaced by a word related to each ImageNet class (
    e.g. dog, fish...). Example of pattern: 'i saw a <WORD>.' will be converted in 'i saw a dog' for the ImageNet
    class associated to dog. - lm_tokenizer: a tokenizer from Transformers library. If None a tokenizer is
    instanciated from a pretrained model vocabulary given by `pretrained_lm_model_name` List of possible names:
    https://huggingface.co/transformers/pretrained_models.html - pretrained_lm_model_name: shortcut name of the
    pretrained language model tokenizer to instantiate if no lm_tokenizer is provided. Default to
    'distilbert-base-uncased'

        Returns: tuple with:
            - tokens_list: list (batch) of list (sequence) of int (language model voc indices)
            - labels_list: list (batch) of int (imagenet class labels)
            - texts_list: list (batch) of string (text before tokenization)
    """
    if lm_tokenizer is None:
        lm_tokenizer = AutoTokenizer.from_pretrained(pretrained_lm_model_name)
    if patterns is None:
        patterns = PATTERNS

    # The IMAGENET dict provided in pytorch_pretrained_biggan is
    # a dict of {wordnet.synset_offset: ImageNet class index}
    # For more information on ImageNet classes relation with wordnet:
    # see http://www.image-net.org/download-API
    # Let's inverse the dict and link it with wordnet synsets:
    class_to_synset = dict((v, wn.synset_from_pos_and_offset('n', k)) for k, v in IMAGENET.items())

    # %%
    # Some classes in ImageNet are associated to complex and rare words like breeds of dog
    # To make it simpler for us in the first step, we'll filter the words
    # to associate a more common word to each class or remove the class if we can't find a
    # common word or if all the common words associated to the class are already taken.
    # A common word here is a word that is in Bert tokenizer vocabulary.
    logger.info('Building a list of simple words associated to ImageNet classes')
    words_dataset = {}
    all_words = set()
    for i, synset in tqdm(class_to_synset.items()):
        current_synset = synset
        while current_synset:
            for lemma in current_synset.lemmas():
                name = lemma.name().replace('_', ' ').lower()
                if name in all_words:
                    continue  # Word is already assigned
                if lm_tokenizer.convert_tokens_to_ids(name) != lm_tokenizer.unk_token_id:
                    # Word is in Bert tokenizer vocabulary
                    words_dataset[i] = name
                    all_words.add(name)
                    current_synset = False  # We're good
                    break
            if current_synset and current_synset.hypernyms():
                current_synset = current_synset.hypernyms()[0]
            else:
                current_synset = False

    # Now let's build a simple sentence for each ImageNet category
    # to use as input to our language model
    logger.info('Building a list of sentences associated to each selected ImageNet class')
    # Here we only provide one simple pattern but better ways
    # to build a diverse dataset could be used.
    examples_dataset = {}
    for i, word in tqdm(words_dataset.items()):
        logger.info(i)
        examples_dataset[i] = [pat.replace('<WORD>', word) for pat in patterns]
        examples_dataset[i] += sentences_from_Wordnet(word)
        examples_dataset[i] += sentences_from_kw2sent([word], model_=kw2sent)
        examples_dataset[i] += sentences_from_model(word)

    return examples_dataset


def tokenize_dataset(dataset, lm_tokenizer=None, pretrained_lm_model_name='distilbert-base-uncased', max_len=30):
    """ Utility function to tokenize a dataset
        Args:
            - dataset: a dict {ImageNet class index: list of examples as string related to the class}
            - lm_tokenizer: a tokenizer from Transformers library.
                If None a tokenizer is instanciated from a pretrained model vocabulary given by
                `pretrained_lm_model_name`
                List of possible names: https://huggingface.co/transformers/pretrained_models.html
            - pretrained_lm_model_name: shortcut name of the pretrained language model tokenizer to instantiate if no
                lm_tokenizer is provided. Default to 'distilbert-base-uncased'
        
        Returns: tuple with:
            - tokens_list: list (batch) of list (sequence) of int (language model voc indices)
            - labels_list: list (batch) of int (imagenet class labels)
            - texts_list: list (batch) of string (text before tokenization)
    """
    if lm_tokenizer is None:
        lm_tokenizer = AutoTokenizer.from_pretrained(pretrained_lm_model_name)

    labels_list = []
    tokens_list = []
    texts_list = []

    # import pickle # todo del
    # with open("./data/dataset_dic.pkl", "rb") as h:
    #     dataset = pickle.load(h)

    for class_index, examples in tqdm(dataset.items()):
        examples_tokens = lm_tokenizer(examples, padding="max_length", max_length=max_len, truncation=True)["input_ids"]
        tokens_list.extend(examples_tokens)
        labels_list += [class_index] * len(examples_tokens)
        texts_list += examples

    return (tokens_list, labels_list, texts_list)


def generate_simple_dataset(args):
    """ Utility function to generate a very simple dataset of string sequence associated to each imagenet class.
        The sentence are tokenized and gathered in a torch.Tensor of shape (num of examples, input sequence)
        Args:
            - args.pretrained_lm_model_name: shortcut name of a tokenizer from Transformers library.
                List of possible names: https://huggingface.co/transformers/pretrained_models.html
            - args.output_dir: Output directory to save the encoded dataset to.
        
        Returns: Nothing (save dataset to output_dir)
    """
    lm_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_lm_model_name)

    dataset = generate_dataset(patterns=PATTERNS,
                               lm_tokenizer=lm_tokenizer)

    # print(f"{np.sum([len(x) for k, x in dataset.items()])} sentences generated.")

    tokenized_output = tokenize_dataset(dataset,
                                        lm_tokenizer=lm_tokenizer)
    tokens_list, labels_list, texts_list = tokenized_output

    os.makedirs(args.output_dir, exist_ok=True)

    # We are cheating a bit here:
    # since all our input have the same length (only one simple pattern) we can just build
    # a big tensor from them without having to worry about padding.
    # Update this if you built a more diverse input dataset.
    if args.framework == "pt":
        labels_tensor = torch.tensor(labels_list)
        tokens_tensor = torch.tensor(tokens_list)

        # Save encoded inputs, labels and associated texts
        torch.save(labels_tensor, os.path.join(args.output_dir, 'labels_tensor.bin'))
        torch.save(tokens_tensor, os.path.join(args.output_dir, 'tokens_tensor.bin'))
    else:
        labels_tensor = np.array(labels_list)
        tokens_tensor = np.array(tokens_list)

        # Save encoded inputs, labels and associated texts
        np.save(os.path.join(args.output_dir, 'labels_tensor.npy'), labels_tensor)
        np.save(os.path.join(args.output_dir, 'tokens_tensor.npy'), tokens_tensor)
    with open(os.path.join(args.output_dir, 'input_texts.txt'), 'w') as f:
        for item in texts_list:
            f.write("%s\n" % item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_lm_model_name", default='distilbert-base-uncased', type=str,
                        help="Pretrained language model vocabulary to use for pre-processing. "
                             "See full list at https://huggingface.co/transformers/pretrained_models.html")
    parser.add_argument("--output_dir", default='./data/', type=str,
                        help="Output directory to save the generated dataset to. "
                             "We save labels and tokens as torch.Tensors or NumPy arrays and the associated sentences "
                             "as a text file with one example per line.")
    parser.add_argument("--framework", default="pt", type=str, choices=["np", "pt", "tf"],
                        help="The framework (np for NumPy, pt for PyTorch, tf for TensorFlow) to use to save the "
                             "data. Note that selecting TensorFlow will save it as NumPy arrays.")

    args = parser.parse_args()
    generate_simple_dataset(args)
