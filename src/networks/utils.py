from os import path, makedirs


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
