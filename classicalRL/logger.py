import logging
import sys
def get_logger(name='logger',fname='logger.log'):
  logger = logging.getLogger(name)
  logger.setLevel(logging.INFO)
  fileHandler = logging.FileHandler(fname,mode='w')
  stdHandler = logging.StreamHandler(stream=sys.stdout)
  formatter = logging.Formatter('%(message)s')
  fileHandler.setFormatter(formatter)
  stdHandler.setFormatter(formatter)
  if not logger.hasHandlers():
      logger.addHandler(fileHandler)
      logger.addHandler(stdHandler)
  return logger