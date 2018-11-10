#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:24:02 2018

@author: nzhang
"""

import pandas as pd   
import os
import re
import time
from sqlalchemy import create_engine
import reader


if __name__ == '__main__':
    
    data = pd.read_excel('/Users/nzhang/Desktop/资管及多互金非持牌在运营网站_YANG2.xlsx',sheet_name = 'Sheet2')
    url = (','.join('"'+item +'"' for item in list(data['网址'])))
    sql = '''select q.wzid, 
                q.wzmc,
                q.wwsywz, 
                q.nwsywz,
                q.scfxsj
           from qrjrwz q where q.wwsywz in ({})
      '''.format(url)
    
    p2p = reader.ssh_con_fetch_data(sql)

    
    print('The shape of output is {} '.format(p2p.shape))

    
    p2p['file_name'] = p2p['nwsywz'].map(lambda s : s.split('/')[-1])
    p2p['update_time'] = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) 
    files_name =p2p['nwsywz'].tolist()
    
    #localpath = '/home/zhangnan/latte/data'
    localpath = '/Users/nzhang/P2P/addon'
    
    if not os.path.exists(localpath):
        os.makedirs(localpath)

    for remotepath in files_name: 
        reader.sftp_download_files(remotepath,os.path.join(localpath,remotepath.split('/')[-1]))
        
    data = reader.read_snapshot(localpath)
    
    data['words'] = data['unicode'].map(lambda s: re.sub(r'[^\u4E00-\u9FA5]','',str(s)))    
    
    df = p2p.merge(data,how ='left',on='file_name')
   
    df['title'] = df['unicode'].map(lambda s :reader.extract_title(s))
    df['keywords'] = df['unicode'].map(lambda s : reader.extract_keywords(s))
    df['description'] = df['unicode'].map(lambda s : reader.extract_description(s))

    df.to_csv(os.path.join(localpath,'df.csv'))
    
    #data['clean'].to_csv(os.path.join(localpath,'data.txt'),index=None)
    
    df.drop('nwsywz',axis =1, inplace =True)
    df.drop('file_name',axis =1, inplace =True)
#    df.drop('unicode',axis =1, inplace =True)
    
    engine = create_engine("mysql+pymysql://root:Rzx@1218!@!#@202.108.211.109:51037/funds_task?charset=utf8",encoding = 'utf-8')
    
    import sqlalchemy.types as types 
    from sqlalchemy.dialects.mysql import LONGTEXT
    
    dtypedict = {'id':types.Integer,
                'scfxsj':types.DateTime,
                'wwsywz':types.VARCHAR(50),
                'wzid':types.BigInteger,
                'wzmc':types.VARCHAR(50),
                'update_time':types.DateTime,
                'unicode':LONGTEXT,
                'words':types.Text,
                'title':types.Text,
                'keywords':types.Text,
                'description':types.Text}
                  
    pd.io.sql.to_sql(df,
                     name='df_p2p_supplement',
                     con=engine,
                     schema= 'funds_task',
                     if_exists='append',
                     index= False,
                     dtype = dtypedict)



   # df.to_csv('df_p2p_supplement.csv',index=None)







    


