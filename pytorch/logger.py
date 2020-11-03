import logging

# 第一步，创建一个logger
logger = logging.getLogger('normal')
logger.setLevel(logging.INFO)  # Log等级总开关

# 第二步，创建一个handler，用于写入日志文件
logfile = './console.txt'
fh = logging.FileHandler(logfile, mode='a')
fh.setLevel(logging.INFO)  # 用于写到file的等级开关

# 第三步，再创建一个handler,用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)  # 输出到console的log等级的开关

# 第四步，定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# 第五步，将logger添加到handler里面
logger.addHandler(fh)
logger.addHandler(ch)


# 写入loss和acc的logger
logger_acc = logging.getLogger('acc')
logger_acc.setLevel(logging.INFO)  # Log等级总开关

# 第二步，创建一个handler，用于写入日志文件
logfile = './acc.txt'
fh_acc = logging.FileHandler(logfile, mode='a')
fh_acc.setLevel(logging.INFO)  # 用于写到file的等级开关

# 第三步，再创建一个handler,用于输出到控制台
# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)  # 输出到console的log等级的开关

# 第四步，定义handler的输出格式
formatter = logging.Formatter('%(message)s')
fh_acc.setFormatter(formatter)
# ch.setFormatter(formatter)

# 第五步，将logger添加到handler里面
logger_acc.addHandler(fh_acc)
# logger.addHandler(ch)