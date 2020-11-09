import logging

logger = logging.getLogger('normal')
logger.setLevel(logging.INFO)

logfile = './console.txt'
fh = logging.FileHandler(logfile, mode='a')
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)


logger_acc = logging.getLogger('acc')
logger_acc.setLevel(logging.INFO)

logfile = './acc.txt'
fh_acc = logging.FileHandler(logfile, mode='a')
fh_acc.setLevel(logging.INFO)

formatter = logging.Formatter('%(message)s')
fh_acc.setFormatter(formatter)
logger_acc.addHandler(fh_acc)