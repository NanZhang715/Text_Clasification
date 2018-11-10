#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:24:02 2018

@author: nzhang
"""

from __future__ import print_function

from pprint import pprint
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import reader
import jieba_fast as jieba

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

sql_train= '''
            select wwsywz as url, 
                    corpus, 
                    tag_words,
                    label 
            from
                    p2p_model 
            where 
                    class='trainset'
         '''
         
sql_test= '''
            select wwsywz as url, 
                    corpus, 
                    tag_words,
                    label 
            from
                    p2p_model 
            where 
                    class='testset'
         '''         
     
train_data = reader.fetch_data_db(sql_train) 
test_data = reader.fetch_data_db(sql_test) 

print('{} samples'.format(train_data.shape[0]))

trainset = train_data['corpus'].map(lambda s : ' '.join(list(jieba.cut(s))))
label =train_data['label'].tolist()

testset = test_data['corpus'].map(lambda s : ' '.join(list(jieba.cut(s))))
y_true =test_data['label'].tolist()


# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    'clf__max_iter': (5,),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    # 'clf__max_iter': (10, 50, 80),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, cv=5,
                               n_jobs=-1, verbose=1)  # n_jobs: 使用说有线程
    
    

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(trainset, label)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
        
    
    y_pred = grid_search.predict(testset)
    
    
    target_names = ['non_p2p', 'p2p']
    #classification_report
    print(classification_report(y_true, y_pred, target_names=target_names))  
    
    #confusion map 
    print(confusion_matrix(y_true, y_pred))
    
