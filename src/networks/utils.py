from os import path, makedirs
from networks.translator import Translator

import tensorflow as tf

from nltk.translate.bleu_score import sentence_bleu


def create_csv_logger_cb(folder_name: str):
    '''
    dynamically creates a csvlogger and tensorboard logger
    '''
    # check if dir exists
    if not path.exists(folder_name + '/historyLogs/'):
        makedirs(folder_name + '/historyLogs/')

    # checkfirst, if history file exists.
    logName = folder_name + '/historyLogs/history_001_'
    count = 1
    while path.isfile(logName + '.csv'):
        count += 1
        logName = folder_name + \
                  '/historyLogs/history_' + str(count).zfill(3) + '_'

    logFileName = logName + '.csv'
    # create logger callback
    f = open(logFileName, "a")

    return f, logFileName


def create_test_output_files(folder_name: str):
    '''
    dynamically creates a csvlogger and tensorboard logger
    '''
    # check if dir exists
    if not path.exists(folder_name + '/test_output/'):
        makedirs(folder_name + '/test_output/')

    # checkfirst, if history file exists. #pt
    logName = folder_name + '/test_output/pt_in_001_'
    count = 1
    while path.isfile(logName + '.txt'):
        count += 1
        logName = folder_name + '/test_output/pt_in_' + str(count).zfill(3) + '_'

    logFileName = logName + '.txt'
    # create logger callback
    f_pt_in = logFileName

    # checkfirst, if history file exists. #en_pred
    logName = folder_name + '/test_output/en_pred_001_'
    count = 1
    while path.isfile(logName + '.txt'):
        count += 1
        logName = folder_name + '/test_output/en_pred_001_' + str(count).zfill(3) + '_'

    logFileName = logName + '.txt'
    # create logger callback
    f_en_pred = logFileName

    # checkfirst, if history file exists. #en_ref
    logName = folder_name + '/test_output/en_ref_001_'
    count = 1
    while path.isfile(logName + '.txt'):
        count += 1
        logName = folder_name + '/test_output/en_ref_001_' + str(count).zfill(3) + '_'

    logFileName = logName + '.txt'
    # create logger callback
    f_en_ref = logFileName

    return f_pt_in, f_en_pred, f_en_ref


def list_of_lists_to_string(list_o_lists: list) -> str:
    res = ""
    for i in list_o_lists:
        for j in i:
            for k in j:
                if type(k) == list:
                    for l in k:
                        res = res + ";" + str(l)
                else:
                    res = res + ";" + str(k)

    return res


def test_transformer(transformer, tokenizers, test_examples, filename):
    # Test the translator
    translator = Translator(tokenizers, transformer)

    f_pt, f_en_pred, f_en_ref = create_test_output_files(filename)

    n = len(test_examples)

    cumulative_bleu = 0.0
    for (batch, (inp, tar)) in enumerate(test_examples):
        translated_text, translated_tokens, attention_weights = translator(tf.constant(inp))

        # make list of words
        pre_list = str(translated_text.numpy()).split(" ")
        tar_list = str(tar.numpy()).split(" ")

        # compute bleu score
        score = sentence_bleu(tar_list, pre_list)
        cumulative_bleu += score
        with open(f_pt, "a") as log:
            log.write(str(inp.numpy()) + "\n")
        with open(f_en_pred, "a") as log:
            log.write(str(score) + " | " + str(translated_text.numpy()) + "\n")
        with open(f_en_ref, "a") as log:
            log.write(str(tar.numpy()) + "\n")
        print("tested on example:" + str(batch) + " of " + str(n))

    with open(f_en_pred, "a") as log:
        log.write(str(cumulative_bleu) + " | +++++++++++ END OF FILE +++++++++++ \n")

    return 0
