"""
初始化
参数配置
"""
import mysql.connector
from sqlalchemy import create_engine

class SessionBase(object):
    def _create_mysql_session(self, online='0'):
        """
        建立与mysql之间的联系
        :param online: 0: debug 1:online
        :return:
        """
        if online == '1':
            pass
        else:
            pass

        return mydb

    def _create_mysql_engine(self, online='0'):
        """
        建立与mysql之间的数据引擎 目的：使DataFrame数据可插入mysql数据库
        :param online: 0: debug 1:online
        :return:
        """
        if online == '1':
            pass
        else:
            pass
        engine = create_engine(str(r"mysql+pymysql://%s:" + '%s' + "@%s/%s") % (user, password, host, db), pool_pre_ping=True, pool_recycle=25200)
        return engine