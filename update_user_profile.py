"""
更新用户画像
@Author Aaron
@Time 2020.11.9
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR))

from offline import SessionBase
from setting.config import *
import pandas as pd
from datetime import datetime, timedelta
import logging
import jieba
jieba.load_userdict(UserDictPath)


class UpdateUser(SessionBase):
    """
    更新用户画像
    """
    def __init__(self, online='0'):
        self.mysql = self._create_mysql_session(online)


    def get_based_info(self):
        """
        获取用户的基本信息
        """
        mycursor = self.mysql.cursor()
        sql_ = "select customer_id, info_gender, info_birthday, register_channel from px_customer_info"
        mycursor.execute(sql_)
        myresult = mycursor.fetchall()
        # 用户id 性别 生日 注册渠道
        df_based_info = pd.DataFrame(myresult, columns=['customer_id', 'info_gender', 'info_birthday', 'register_channel'])

    def get_text_profile(self):
        """
        获取用户文本的画像
        :return:
        """
        # 获取浏览记录