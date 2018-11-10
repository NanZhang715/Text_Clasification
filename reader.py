#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:24:02 2018

@author: nzhang
"""

import pandas as pd   
import pymysql
import paramiko
import codecs
import os
import re
import time
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder


def fetch_data_db(sql):

    ''' 
    IP/ICP Connect to the database  样本库

    '''

    connection = pymysql.connect(host='202.108.211.109',
                                 user='root',
                                 password='Rzx@1218!@!#',
                                 db='funds_task',
                                 charset='utf8mb4',
                                 port=51037,
                                 cursorclass=pymysql.cursors.DictCursor,
                                 connect_timeout=86400)

    try:
        with connection.cursor() as cursor:
            # Read a single record
            print(sql + ' is running')
            cursor.execute(sql)
            result = cursor.fetchall()
            result = pd.DataFrame(result)
            
    except ValueError:
        print("Error: Oopus, No Data, Drink a coffee")
        return None

    
    finally:
        connection.close()
        print(sql + 'is obtained')
        
    return result


def ssh_con_fetch_data(sql):
    
    '''
    
    SSH 代理登陆 MYSQL 服务器
    
    '''
    
    with SSHTunnelForwarder(
            ('202.108.211.109',51007),  ## 堡垒机地址
            ssh_username = 'root',
            ssh_password = 'rzx@1218!@!#',
            remote_bind_address = ('10.130.21.150',3308)  ## mysql服务器地址
            ) as server:
        
        print(server.local_bind_port)       
        
        # Connect to the database
        connection = pymysql.connect(host='127.0.0.1',  ##必须是127.0.0.1 
                                     user='rzx',
                                     password='rzx@1218!@!#',
                                     db='funds_task',
                                     charset='utf8mb4',
                                     port=server.local_bind_port,
                                     cursorclass=pymysql.cursors.DictCursor,
                                     connect_timeout=86400)
    
        try:
            with connection.cursor() as cursor:
                # Read a single record
                print(sql + ' is running')
                cursor.execute(sql)
                result = cursor.fetchall()
                result = pd.DataFrame(result)
                
        except ValueError:
            print("Error: Oopus, No Data, Drink a coffee")
            return None
       
        finally:
            connection.close()
            print(sql + 'is obtained')
            
        return result


def sftp_download_files(remotepath, localpath):
    try:
        t = paramiko.Transport(("202.108.211.109",51007))
        t.connect(username = 'root', password ='rzx@1218!@!#')
        sftp = paramiko.SFTPClient.from_transport(t)
        
    #remotepath='/datanfs/rzx/toinner/snapshot'
    #localpath='/tmp/system.log'  
    
        sftp.get(remotepath, localpath)
        print('{} is obtained successfully'.format(remotepath.split('/')[-1]))
        
    except IOError:
         print('{} is not exists'.format(remotepath.split('/')[-1]))
        
    finally:
        t.close()
        
        return
    
    
def read_snapshot(file_path):
    
    dir_list=[file for file in os.listdir(file_path) if not file.startswith('.')]
    data = {}
    data['unicode'] = []
    data['file_name'] = []
    for file in dir_list:
        with codecs.open(os.path.join(file_path,file), "r", "utf-8")as f:
            data['unicode'].append(f.read())
            data['file_name'].append(file)
            
    return pd.DataFrame.from_dict(data)


def extract_title(html_doc):
    
    try:
        soup = BeautifulSoup(html_doc, 'html.parser')
        title = soup.title.string
        
    except:       
        title = None
    
    return title


def extract_keywords(html_doc):
    
    try:
        soup = BeautifulSoup(html_doc, 'html.parser')
        keywords = soup.find(attrs={"name": re.compile('keywords',re.IGNORECASE)})['content']

        
    except:       
        keywords = None
    
    return keywords


def extract_description(html_doc):
    
    try:
        soup = BeautifulSoup(html_doc, 'html.parser')
        description = soup.find(attrs={"name":re.compile('description',re.IGNORECASE)})['content']
        
    except:
        description = None
    
    return description      
    

if __name__ == '__main__':
    
    sql = '''select q.wzid, 
                    q.wzmc,
                    q.wwsywz, 
                    q.nwsywz,
                    q.scfxsj
            from qrjrwz q left join wz_jryt y on q.wzid= y.wzid 
            where y.ytfl in (21) and q.nwsywz is not null and q.scfxsj > '2017-00-00 00:00:00';
    '''
    
    p2p= fetch_data_db(sql)
    
    
    print('The shape of output is {} '.format(p2p.shape))

    
    p2p['file_name'] = p2p['nwsywz'].map(lambda s : s.split('/')[-1])
    p2p['update_time'] = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) 
    files_name =p2p['nwsywz'].tolist()
    
    #localpath = '/home/zhangnan/latte/data'
    localpath = '/Users/nzhang/P2P/Data'
    
    if not os.path.exists(localpath):
        os.makedirs(localpath)

    for remotepath in files_name: 
        sftp_download_files(remotepath,os.path.join(localpath,remotepath.split('/')[-1]))
        
    data = read_snapshot(localpath)
    
    data['words'] = data['unicode'].map(lambda s: re.sub(r'[^\u4E00-\u9FA5]','',str(s)))
    
    
    df = p2p.merge(data,how ='left',on='file_name')

    
    df['title'] = df['unicode'].map(lambda s :extract_title(s))
    df['keywords'] = df['unicode'].map(lambda s : extract_keywords(s))
    df['description'] = df['unicode'].map(lambda s : extract_description(s))

    df.to_csv(os.path.join(localpath,'df.csv'))
    
    #data['clean'].to_csv(os.path.join(localpath,'data.txt'),index=None)
    
    df.drop('title',axis =1, inplace =True)
    
    engine = create_engine("mysql+pymysql://root:rzx@1218!@!#@202.108.211.109:51024/funds_task?charset=utf8",encoding = 'utf-8')
        
    pd.io.sql.to_sql(df,
                     name='df_p2p',
                     con=engine,
                     schema= 'funds_task',
                     if_exists='append',
                     chunksize=100)


    

