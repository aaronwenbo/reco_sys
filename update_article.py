"""
根据tfidf值相似度的算法
更新所有文章的tfidf值
@Author Aaron
@Time 2020-6-1
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR))

from offline import SessionBase
import pandas as pd
import codecs
from setting.config import *
import re
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from datetime import datetime
from datetime import timedelta
import logging

logger = logging.getLogger('offline')

class UpdateArticle(SessionBase):
    """
    更新文章画像import mysql.connector
    """

    def __init__(self, online='0'):
        self.mysql = self._create_mysql_session(online)

    def judge_update(self):
        _yester = datetime.today().replace(minute=0, second=0, microsecond=0)
        start = datetime.strftime(_yester + timedelta(days=0, hours=-1, minutes=0), "%Y-%m-%d %H:%M:%S")
        end = datetime.strftime(_yester, "%Y-%m-%d %H:%M:%S")
        mycursor = self.mysql.cursor()
        sql_order = "SELECT article_id, REPLACE(REPLACE(article_title, CHAR(10), ''), CHAR(13), '') as article_title, REPLACE(REPLACE(article_content, CHAR(10), ''), CHAR(13), '') as article_content FROM px_customer_article where release_status='1' AND del_flag='0' AND status='2' AND post_time >='{}' AND post_time<='{}'".format(
            start, end)
        mycursor.execute(sql_order)
        myresult = mycursor.fetchall()
        # turn to DataFrame
        df = pd.DataFrame(myresult, columns=['article_id', 'article_title', 'article_content'])
        logger.info("update the article profile, True or False: {0}".format(not df.empty))
        return df

    def merge_article_data(self):
        """
            合并业务中增量更新的文章数据
            :return:
        """
        # 获取文章相关数据, 指定过去一个小时整点到整点的更新数据
        # 如：26日：1：00~2：00，2：00~3：00，左闭右开
        mycursor = self.mysql.cursor()
        sql_order = "SELECT article_id, REPLACE(REPLACE(article_title, CHAR(10), ''), CHAR(13), '') as article_title, REPLACE(REPLACE(article_content, CHAR(10), ''), CHAR(13), '') as article_content FROM px_customer_article where release_status='1' AND del_flag='0' AND status='2'"
        mycursor.execute(sql_order)
        myresult = mycursor.fetchall()
        # turn to DataFrame
        df = pd.DataFrame(myresult, columns=['article_id', 'article_title', 'article_content'])
        if df.empty:
            return df
        df['article_title'].fillna('无标', inplace=True)
        df['article_content'].fillna('无容', inplace=True)
        df['sentence'] = df['article_id'].map(str) + df['article_title'] + df['article_content']

        # load stopwords
        def get_stopwords_list(stopwords_path):
            stopwords_list = [i.strip() for i in codecs.open(stopwords_path, encoding='utf-8').readlines()]
            return stopwords_list

        stopwords_list = get_stopwords_list(STOPWORDS_PATH)

        # 分词 去标签等操作
        def clean_text(text):
            def cut_sentence(sentence):
                seg_list = pseg.lcut(sentence)
                seg_list = [i for i in seg_list if i.flag not in stopwords_list]
                filtered_words_list = []
                for seg in seg_list:
                    if len(seg.word) <= 1:
                        continue
                    elif seg.flag == "eng":
                        if len(seg.word) <= 2:
                            continue
                        else:
                            filtered_words_list.append(seg.word)
                    elif seg.flag.startswith("n"):
                        filtered_words_list.append(seg.word)
                    elif seg.flag.startswith("v"):
                        filtered_words_list.append(seg.word)
                    elif seg.flag in ["eng", "x"]:
                        filtered_words_list.append(seg.word)
                article = ''
                for i in filtered_words_list:
                    article = article + i + " "
                return article

            text = re.sub("<.*?>", "", text)  # 替换掉标签数据
            text = re.sub("&lt;.*?&gt;", "", text)  # 替换掉标签数据
            text = re.sub('[a-zA-z]', '', text)  # 这里替换所有英文 待商榷
            words = cut_sentence(text)
            return words

        df['sentence_clean'] = df['sentence'].apply(clean_text)
        logger.info("INFO: merge_article_data complete")
        return df

    def compute_article_similar(self, df):
        import pickle
        def save_model(vec, path):
            with open(path, 'wb') as fw:
                pickle.dump(vec.vocabulary_, fw)

        df.set_index('article_id', inplace=True)
        tf = TfidfVectorizer()
        tfidf_matrix = tf.fit_transform(df['sentence_clean'])
        save_model(tf, TfidfVectorizer_path)
        # tfidf_matrix_array = tfidf_matrix.toarray()
        # word_indices = pd.Series(tf.get_feature_names())
        cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
        indices = pd.Series(df.index)  # df.index是文章id

        def save_redis(indices, cosine_similarities, df):
            import redis
            pool = redis.ConnectionPool(host=REDIS_ADDRESS, port=REDIS_PORT, decode_responses=True)
            r = redis.Redis(connection_pool=pool)
            for _, id_ in indices.items():
                idx = indices[indices == id_].index[0]
                score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
                # x, p = [], []
                # for i, v in score_series.iloc[1:11].items():
                #     x.append(list(df.index)[i])
                #     p.append(v)
                # result = list(zip(x, p))
                # # 放到推荐列表中
                # r.set("ar"+str(id_), str(result))
                for i, v in score_series.iloc[1:11].items():
                    v_key = str(list(df.index)[i])
                    mapping = {v_key: v, }
                    r.zadd('ar'+str(id_), mapping)

        logger.info("INFO: compute tfidf complete")
        save_redis(indices, cosine_similarities, df)
        logger.info("INFO: compute_article_similar complete")


if __name__ == '__main__':
    ua = UpdateArticle(online='1')
    df = ua.judge_update()
    print('update the article profile, True or False: ', not df.empty)
    if df.empty:
        sentence_df = ua.merge_article_data()
        print('merge_article_data is over')
        ua.compute_article_similar(sentence_df)
        print('compute_article_similar is over')
    print('it is over')
