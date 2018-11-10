#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 14:37:15 2018

@author: nzhang
"""

import pandas as pd   
import re
from sqlalchemy import create_engine
import jieba_fast.analyse as jiebanalyse
import reader

if __name__ == '__main__':

        sql = '''select wwsywz,
                        title, 
                        keywords,
                        description,
                        unicode,
                        label
                from df_p2p ;
        '''
        p2p= reader.fetch_data_db(sql)
        
        p2p.drop_duplicates( keep = 'first',inplace =True)
        
        p2p=p2p.fillna('')
        
        #p2p['meta'] = p2p['title'] + p2p['keywords']+ p2p['description']
        
        p2p['corpus'] = p2p['unicode'].map(lambda s: re.sub('p2p','个贷',str(s),flags = re.IGNORECASE))
        p2p['corpus'] = p2p['corpus'].map(lambda s: re.sub(r'[^\u4E00-\u9FA5]','',str(s)))
        p2p['corpus'] = p2p['corpus'].map(lambda s: s.replace('个贷','p2p'))
        
        p2p['label'].value_counts()
        
        p2p['td_idf'] = p2p['corpus'].map(lambda s : jiebanalyse.extract_tags(s, withWeight=False, topK=20))
        
        
        ## 必须转行成string， list 入MYSQL会导致格式错误
        p2p['tag_words'] = p2p['td_idf'].map(lambda s : ' '.join(s))
        
        p2p.columns
        p2p = p2p[['wwsywz',
                   'title',
                   'keywords',
                   'description',
                   'unicode',
                   'corpus',
                   'tag_words',
                   'label']]
        
        y = p2p['label']
        X = p2p.iloc[:,0:7]
        
        X.columns
        
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size = 0.25, 
                                                            random_state = 0)
        
        trainset = pd.concat([X_train,y_train],axis = 1)
        testset = pd.concat([X_test,y_test],axis=1)
        
        
        trainset['class'] = 'trainset'
        testset['class']= 'testset'
        
        dataset = pd.concat([trainset, testset], axis = 0, ignore_index = True)
        dataset.reset_index(inplace = True, drop =True)
        
        dataset.columns
        
        engine = create_engine("mysql+pymysql://root:Rzx@1218!@!#@202.108.211.109:51037/funds_task?charset=utf8",encoding = 'utf-8')
        
        
        from sqlalchemy.dialects.mysql import LONGTEXT, TEXT
        
        dtypedict = {
                'wwsywz':TEXT,
                'title':TEXT,
                'keywords':TEXT,
                'description':TEXT,
                'unicode':TEXT,
                'corpus':LONGTEXT,
                'tag_words':TEXT,
                'label':TEXT,
                'class':TEXT
                }
        
                      
        pd.io.sql.to_sql(dataset,
                         name='p2p_model',
                         con=engine,
                         schema= 'funds_task',
                         if_exists='append',
                         index= True,
                         dtype = dtypedict
                         )
        
        
        dataset['label'].value_counts()
        











