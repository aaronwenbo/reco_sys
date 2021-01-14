"""
帖子点击率计算留存
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR))

from offline import SessionBase
import pandas as pd
import logging
import datetime
from setting.config import version, EXPOSURE_NUMBER_URL
import requests
logger = logging.getLogger('offline')



class UpdatePostCtr(SessionBase):
    def __init__(self, online='0'):
        self.mysql = self._create_mysql_session(online)
        # self.engine = self._create_mysql_engine(online)

    def update_post_ctr(self):
        mycursor = self.mysql.cursor()
        sql_order = 'select count(post_behavior_id) from (select post_behavior_id from px_post_behavior where create_time > CURDATE() and behavior_type = "0" GROUP BY customer_id, post_id) aa'
        mycursor.execute(sql_order)
        myresult = mycursor.fetchall()
        # 点击数
        clicked_times = myresult[0][0]
        # 曝光数
        response = requests.get(EXPOSURE_NUMBER_URL)
        r = response.json()
        if r['success']:
            exposures_times = r['data']
        else:
            exposures_times = 0
        # 计算点击率
        CTR = round(clicked_times / (exposures_times+0.00001), 6)
        today = datetime.date.today()
        sql_ = "INSERT INTO px_ctr_post_rs (calculate_date, click_through_rate, create_time, version_id) VALUES('%s', '%s', '%s', '%s')" % (str(today), float(CTR), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), version)
        mycursor.execute(sql_)
        self.mysql.commit()
        logger.info("update_post_ctr finished, clicked_times:{0}, exposures_times:{1}, CTR:{2}".format(clicked_times, exposures_times, CTR))


if __name__ == '__main__':
    import setting.logging as lg
    lg.create_logger()
    logger = logging.getLogger('offline')

    upc = UpdatePostCtr()
    upc.update_post_ctr()

