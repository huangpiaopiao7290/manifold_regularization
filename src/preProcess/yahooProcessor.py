import os

from src.utils.utility import Utility


work_space = os.getcwd()

# 解压yahooAnswers.tar.gz
yahoo = os.path.join(work_space, "data/raw/yahooAnswers/yahoo_answers_csv.tar.gz")
yahoo_dir = os.path.join(work_space, "data/processed/yahooAnswers")
Utility.un_tar(yahoo, yahoo_dir)
