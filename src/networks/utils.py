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
