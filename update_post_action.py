import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR))

from offline import SessionBase
import pandas as pd
# from sqlalchemy import create_engine
import logging
import datetime
from setting.config import version

logger = logging.getLogger('offline')

# user = 'root'
# password = 'Pxsj_20190408'
# host = '192.168.1.254'
# db = 'thirdmeasuretest'
#
# engine = create_engine(str(r"mysql+pymysql://%s:" + '%s' + "@%s/%s") % (user, password, host, db), pool_pre_ping=True, pool_recycle=25200)


class UpdateAction(SessionBase):
    def __init__(self, online='0'):
        self.mysql = self._create_mysql_session(online)
        self.engine = self._create_mysql_engine(online)

    def update_user_post_action_rs(self):
        mycursor = self.mysql.cursor()
        # 查询用户行为表的数据
        sql_order = r"select customer_id, post_id, behavior_type, duration, create_time from px_post_behavior where create_time > CURDATE()"
        # sql_order = r"select customer_id, article_id, behavior_type, duration, create_time from px_customer_behavior where create_time"
        mycursor.execute(sql_order)
        myresult = mycursor.fetchall()
        # 行为表的数据
        df_behavior = pd.DataFrame(myresult,
                                   columns=['customer_id', 'article_id', 'behavior_type', 'duration', 'create_time'])
        list_behavior = []
        for index, row in df_behavior.iterrows():
            class Temp(object):
                clicked = False
                praised = False
                collected = False
                shared = False
                read_time = 0
                notinterested = False

            _tp = Temp()
            if row['behavior_type'] == '0':
                _tp.clicked = True
            elif row['behavior_type'] == '1':
                _tp.praised = True
            elif row['behavior_type'] == '2':
                _tp.collected = True
            elif row['behavior_type'] == '3':
                _tp.shared = True
            elif row['behavior_type'] == '4':
                _tp.clicked = True
                _tp.read_time = int(row['duration'])
            elif row['behavior_type'] == '5':
                _tp.notinterested = True
            else:
                pass
            list_behavior.append(
                [row['customer_id'], row['create_time'], row['article_id'], _tp.clicked, _tp.praised, _tp.collected,
                 _tp.shared, _tp.notinterested, True, _tp.read_time])

        # 查询用户曝光表的数据
        sql_order_to_exposure = r"select customer_id, article_id, exposure_time FROM px_article_exposure WHERE del_flag='0' and exposure_time > CURDATE() GROUP BY customer_id, article_id"
        # sql_order_to_exposure = r"select customer_id, article_id, exposure_time FROM px_article_exposure WHERE del_flag='0' and exposure_time GROUP BY customer_id, article_id"
        mycursor.execute(sql_order_to_exposure)
        myresult = mycursor.fetchall()
        # 曝光表的数据
        df_exposure = pd.DataFrame(myresult, columns=['customer_id', 'post_id', 'exposure_time'])
        for index, row in df_exposure.iterrows():
            list_behavior.append(
                [row['customer_id'], row['exposure_time'], row['article_id'], False, False, False, False, False, True,
                 0])
        # 行为表
        df_result = pd.DataFrame(list_behavior,
                                 columns=['user_id', 'action_time', 'post_id', 'clicked', 'praised', 'collected',
                                          'shared', 'notinterested', 'exposured', 'read_time'])

        pd.io.sql.to_sql(df_result, 'px_user_action_rs_temptable', self.engine, if_exists='append')
        # 插入前 首先清空表
        mycursor.execute("truncate table px_user_action_rs")
        # 更新行为表 update_user_action_rs
        sql_ = "insert into px_user_action_rs (user_id, action_time, article_id, clicked, praised, collected, shared, notinterested, exposured, read_time) select user_id, max(action_time) as action_time, article_id, max(clicked) as clicked, max(praised) as praised, max(collected) as collected, max(shared) as shared, max(notinterested) as notinterested, max(exposured) as exposured, max(read_time) as read_time from px_user_action_rs_temptable group by user_id, article_id"
        mycursor.execute(sql_)
        self.mysql.commit()
        logger.info("update_user_action_rs finished, add {} items user actions".format(len(df_result)))


if __name__ == '__main__':
    import setting.logging as lg
    lg.create_logger()
    logger = logging.getLogger('offline')
    ua = UpdateAction(online='0')
    ua.update_user_post_action_rs()
