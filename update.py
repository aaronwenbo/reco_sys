"""
更新各个圈子中最热最新的帖子排序结果(新版)
@Author Aaron
@Time 2020.10.29
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR))

from offline import SessionBase
from setting.config import *
import pandas as pd
from datetime import datetime, timedelta
import math
import redis
import logging
import re
import jieba
jieba.load_userdict(UserDictPath)
import jieba.posseg as pseg
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


logger = logging.getLogger('offline')

class UpdatePost(SessionBase):
    """
    更新各个圈子中最热最新的帖子排序结果
    """
    def __init__(self, online='0'):
        self.mysql = self._create_mysql_session(online)

    def based_on_praise_and_time(self):
        mycursor = self.mysql.cursor()
        # 获取所有帖子的postid以及创建时间
        _sql_get_all_id_and_creatTime = "SELECT p.post_id,p.topic_circle_id,c.count, p.create_time FROM px_post p LEFT JOIN (SELECT l.post_id,COUNT(post_id) count FROM px_post_like l GROUP BY l.post_id) c ON p.post_id = c.post_id WHERE p.check_status='1' and p.del_flag = '0' and p.top_status = '0'"
        mycursor.execute(_sql_get_all_id_and_creatTime)
        myresult = mycursor.fetchall()
        # 帖子id 所属圈子id 点赞量 帖子创建时间
        df_id_and_time = pd.DataFrame(myresult, columns=['post_id', 'topic_circle_id', 'like', 'create_time'])
        df_id_and_time['like'].fillna(0, inplace=True)
        df_id_and_time['diff'] = pd.DataFrame((datetime.now() - df_id_and_time['create_time']).dt.days)

        df_id_and_time = df_id_and_time.drop(['create_time'], axis=1)
        res = df_id_and_time.fillna(value=0)
        res['value'] = res[['diff', 'like']].apply(lambda x: (x['like'] * 100 + 10) / (math.exp(0.233 * x['diff'])), axis=1)
        res = res.sort_values(by="value", ascending=False)
        rank_list = res['post_id'].tolist()
        # print(rank_list)
        pool = redis.ConnectionPool(host=REDIS_ADDRESS, port=REDIS_PORT, decode_responses=True)
        r = redis.Redis(connection_pool=pool)
        r.delete("hotPost")
        for i in rank_list:
            r.rpush("hotPost", str(i))
        logger.info("based_on_praise_and_time finish, the length of rank_list is {0}".format(len(rank_list)))

    def judge_update_post(self):
        _yester = datetime.today().replace(minute=0, second=0, microsecond=0)
        start = datetime.strftime(_yester + timedelta(days=0, hours=-1, minutes=0), "%Y-%m-%d %H:%M:%S")
        end = datetime.strftime(_yester, "%Y-%m-%d %H:%M:%S")
        mycursor = self.mysql.cursor()
        sql_order = "SELECT post_id, REPLACE(REPLACE(post_title, CHAR(10), ''), CHAR(13), '') as article_title, REPLACE(REPLACE(post_content, CHAR(10), ''), CHAR(13), '') as post_content FROM px_post where check_status='1' AND del_flag='0' AND top_status='0' AND create_time >='{}' AND create_time<='{}'".format(
            start, end)
        mycursor.execute(sql_order)
        myresult = mycursor.fetchall()
        # turn to DataFrame
        df = pd.DataFrame(myresult, columns=['post_id', 'post_title', 'post_content'])
        logger.info("update the post profile, True or False: {0}".format(not df.empty))
        return df

    def merge_post_data(self):
        """
            合并增量更新的帖子数据
            :return:
        """
        # 获取帖子相关数据, 指定过去一个小时整点到整点的更新数据
        # 如：26日：1：00~2：00，2：00~3：00，左闭右开
        mycursor = self.mysql.cursor()
        sql_order = "SELECT post_id, REPLACE(REPLACE(post_title, CHAR(10), ''), CHAR(13), '') as article_title, REPLACE(REPLACE(post_content, CHAR(10), ''), CHAR(13), '') as post_content FROM px_post where check_status='1' AND del_flag='0' AND top_status='0'"
        mycursor.execute(sql_order)
        myresult = mycursor.fetchall()
        # turn to DataFrame
        df = pd.DataFrame(myresult, columns=['post_id', 'post_title', 'post_content'])
        if df.empty:
            return df
        df['post_title'].fillna('无标', inplace=True)
        df['post_content'].fillna('无容', inplace=True)
        df['sentence'] = df['post_id'].map(str) + df['post_title'] + df['post_content']

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
                    elif seg.flag == "eng" and seg.word not in filtered_words_list:
                        if len(seg.word) <= 2:
                            continue
                        else:
                            filtered_words_list.append(seg.word)
                    elif seg.flag.startswith("n") and seg.word not in filtered_words_list:
                        filtered_words_list.append(seg.word)
                    elif seg.flag.startswith("v") and seg.word not in filtered_words_list:
                        filtered_words_list.append(seg.word)
                    elif seg.flag.startswith("a") and seg.word not in filtered_words_list:
                        filtered_words_list.append(seg.word)
                    elif seg.flag in ["f", "s", "t", "PER", "LOC", "ORG"] and seg.word not in filtered_words_list:
                        filtered_words_list.append(seg.word)
                article = ''
                for i in filtered_words_list:
                    article = article + i + " "
                return article

            text = re.sub("<.*?>", "", text)  # 替换掉标签数据
            text = re.sub("&lt;.*?&gt;", "", text)  # 替换掉标签数据
            # text = re.sub('[a-zA-z]', '', text)  # 这里替换所有英文 待商榷
            words = cut_sentence(text)
            return words

        df['sentence_clean'] = df['sentence'].apply(clean_text)
        logger.info("INFO: merge_post_data complete")
        return df

    def compute_post_similar(self, df):
        import pickle
        def save_model(vec, path):
            with open(path, 'wb') as fw:
                pickle.dump(vec.vocabulary_, fw)

        df.set_index('post_id', inplace=True)
        tf = TfidfVectorizer()
        tfidf_matrix = tf.fit_transform(df['sentence_clean'])
        save_model(tf, post_TfidfVectorizer_path)
        # tfidf_matrix_array = tfidf_matrix.toarray()
        # word_indices = pd.Series(tf.get_feature_names())
        cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
        indices = pd.Series(df.index)  # df.index是文章id

        def save_redis(indices, cosine_similarities, df):
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
                if r.zcard('pr'+str(id_)) > 25:
                    r.delete('pr'+str(id_))
                for i, v in score_series.iloc[1:11].items():
                    v_key = str(list(df.index)[i])
                    mapping = {v_key: v, }
                    r.zadd('pr' + str(id_), mapping)

        logger.info("INFO: compute post tfidf complete")
        save_redis(indices, cosine_similarities, df)
        logger.info("INFO: compute_post_similar complete")

if __name__ == '__main__':
    up = UpdatePost(online='0')
    up.based_on_praise_and_time()
    df = up.judge_update_post()
    print('update the post profile, True or False: ', not df.empty)
    if df.empty:
        sentence_df = up.merge_post_data()
        print('merge_post_data is over')
        up.compute_post_similar(sentence_df)
        print('compute_post_similar is over')
    print('it is over')

