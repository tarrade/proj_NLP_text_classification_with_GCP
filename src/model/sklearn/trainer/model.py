import os
import psutil
import subprocess
import datetime
import joblib
import copy
import pprint
import google.cloud.bigquery as bigquery
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

def create_queries(eval_size):
    #query = """
    #SELECT
    #*
    #FROM
    #`nlp-text-classification.stackoverflow.posts_preprocessed_selection_subset`
    #"""

    query = """
    SELECT
      *
    FROM
      `nlp-text-classification.stackoverflow.posts_preprocessed`    
    """
    
    eval_query = "{} WHERE MOD(ABS(FARM_FINGERPRINT(CAST(id as STRING))),100) < {}".format(query, eval_size)
    train_query  = "{} WHERE MOD(ABS(FARM_FINGERPRINT(CAST(id as STRING))),100)>= {}".format(query, eval_size)
  
    return train_query, eval_query
    

def query_to_dataframe(query):
    
    client = bigquery.Client()
    df = client.query(query).to_dataframe()
    
    # label
    df['label'] = df['tags'].apply(lambda row: ",".join(row))
    del df['tags']
    
    # features
    df['text'] = df['title'] + df['text_body'] + df['code_body']
    del df['code_body']
    del df['title']
    del df['text_body']
    
    # use BigQuery index
    df.set_index('id',inplace=True)
    
    return df


def create_dataframes(frac, eval_size):   

    # split in df in training and testing
    #train_df, eval_df = train_test_split(df, test_size=0.2, random_state=101010)
    
    # small dataset for testing
    if frac > 0 and frac < 1:
        sample = " AND RAND() < {}".format(frac)
    else:
        sample = ""

    train_query, eval_query = create_queries(eval_size)
    train_query = "{} {}".format(train_query, sample)
    eval_query =  "{} {}".format(eval_query, sample)

    train_df = query_to_dataframe(train_query)
    eval_df = query_to_dataframe(eval_query)
    
    print('size of the training set  :',len(train_df ))
    print('size of the evaluation set:',len(eval_df ))
    print(train_df.info(memory_usage="deep"))
    
    print('### CPU (count) {}'.format(psutil.cpu_count()))
    print('### CPU (count) {}'.format(psutil.cpu_count(logical=False)))
    print('### CPU (count) {}'.format(len(psutil.Process().cpu_affinity())))
    
    mem = psutil.virtual_memory()
    print(mem)
    print('### Memory total     {:.2f} Gb'.format(mem.total/1024**3))
    print('### Memory percent   {:.2f} %'.format(mem.percent))
    print('### Memory available {:.2f} Gb'.format(mem.available/1024**3))
    print('### Memory used      {:.2f} Gb'.format(mem.used/1024**3))
    print('### Memory free      {:.2f} Gb'.format(mem.free/1024**3))
    print('### Memory active    {:.2f} Gb'.format(mem.active/1024**3))
    print('### Memory inactive  {:.2f} Gb'.format(mem.inactive/1024**3))
    print('### Memory buffers   {:.2f} Gb'.format(mem.buffers/1024**3))      
    print('### Memory cached    {:.2f} Gb'.format(mem.cached/1024**3))    
    print('### Memory shared    {:.2f} Gb'.format(mem.shared/1024**3))   
    print('### Memory slab      {:.2f} Gb'.format(mem.slab/1024**3)) 
    print(' ')
    
    return train_df, eval_df


def input_fn(df):
    #df = copy.deepcopy(input_df)
    
    # features, label
    label = df['label']
    del df['label']
    
    features = df['text']
    return features, label

def train_and_evaluate(eval_size, frac, max_df, min_df, norm, alpha):
    
    # transforming data type from YAML to python
    if norm=='None': norm=None 
    if min_df==1.0: min_df=1
    
    # get data
    train_df, eval_df = create_dataframes(frac, eval_size)
    train_X, train_y = input_fn(train_df)
    eval_X, eval_y = input_fn(eval_df)
    del train_df
    del eval_df
    
    mem = psutil.virtual_memory()
    print('----> memory before scikit-learn training ...')
    print(mem) 
    print('### Memory total     {:.2f} Gb'.format(mem.total/1024**3))
    print('### Memory percent   {:.2f} %'.format(mem.percent))
    print('### Memory available {:.2f} Gb'.format(mem.available/1024**3))
    print('### Memory used      {:.2f} Gb'.format(mem.used/1024**3))
    print('### Memory free      {:.2f} Gb'.format(mem.free/1024**3))
    print('### Memory active    {:.2f} Gb'.format(mem.active/1024**3))
    print('### Memory inactive  {:.2f} Gb'.format(mem.inactive/1024**3))
    print('### Memory buffers   {:.2f} Gb'.format(mem.buffers/1024**3))      
    print('### Memory cached    {:.2f} Gb'.format(mem.cached/1024**3))    
    print('### Memory shared    {:.2f} Gb'.format(mem.shared/1024**3))   
    print('### Memory slab      {:.2f} Gb'.format(mem.slab/1024**3)) 
    print(' ')
    
    # train
    pipeline=Pipeline([('Word Embedding', CountVectorizer(max_df=max_df,min_df=min_df)),
                       ('Feature Transform', TfidfTransformer(norm=norm)),
                       ('Classifier', MultinomialNB(alpha=alpha))])
    pipeline.fit(train_X, train_y)
    
    print('the list of steps and parameters in the pipeline\n')
    for k, v in pipeline.named_steps.items():
        print('{}:{}\n'.format(k,v))
        
    # print the lenght of the vocabulary
    has_index=False
    if 'Word Embedding' in pipeline.named_steps.keys():
        # '.vocabulary_': dictionary item (word) and index 'world': index
        # '.get_feature_names()': list of word from (vocabulary)
        voc=pipeline.named_steps['Word Embedding'].vocabulary_
        voc_list=sorted(voc.items(), key=lambda kv: kv[1], reverse=True)
        print(' --> length of the vocabulary vector : \n{} {} \n'.format(len(voc), len(pipeline.named_steps['Word Embedding'].get_feature_names())))  
        
        # looking at the word occurency after CountVectorizer
        vect_fit=pipeline.named_steps['Word Embedding'].transform(eval_X)
        counts=np.asarray(vect_fit.sum(axis=0)).ravel().tolist()
        df_counts=pd.DataFrame({'term':pipeline.named_steps['Word Embedding'].get_feature_names(),'count':counts})
        df_counts.sort_values(by='count', ascending=False, inplace=True)
        print(' --> df head 20')
        print(df_counts.head(20))
        print(' --> df tail 20')
        print(df_counts.tail(20))
        print(' --- ')
        n=0
        for i in voc_list:
            n+=1
            print('    ',i)
            if (n>20):
                break
        print(' --> more frequet words: \n{} \n'.format(voc_list[0:20]))
        print(' --- ')
        print(' --> less frequet words: \n{} \n'.format(voc_list[-20:-1]))
        print(' --- ')
        print(' --> longest word: \n{} \n'.format(max(voc, key=len)))
        print(' ---)')
        print(' --> shortest word: \n{} \n'.format(min(voc, key=len)))
        print(' --- ')
        index=pipeline.named_steps['Word Embedding'].get_feature_names()
        has_index=True
              
    # print the tfidf values
    if 'Feature Transform' in pipeline.named_steps.keys():
        tfidf_value=pipeline.named_steps['Feature Transform'].idf_
        #print('model\'s methods: {}\n'.format(dir(pipeline.named_steps['tfidf'])))
        if has_index:
            # looking at the word occurency after CountVectorizer
            tfidf_fit=pipeline.named_steps['Feature Transform'].transform(vect_fit)
            tfidf=np.asarray(tfidf_fit.mean(axis=0)).ravel().tolist()
            df_tfidf=pd.DataFrame({'term':pipeline.named_steps['Word Embedding'].get_feature_names(),'tfidf':tfidf})
            df_tfidf.sort_values(by='tfidf', ascending=False, inplace=True)
            print(' --> df head 20')
            print(df_tfidf.head(20))
            print(' --> df tail 20')
            print(df_tfidf.tail(20))
            print(' --- ')
            tfidf_series=pd.Series(data=tfidf_value,index=index)
            print(' --> IDF:')
            print(' --> Smallest idf:\n{}'.format(tfidf_series.nsmallest(20).index.values.tolist()))
            print(' {} \n'.format(tfidf_series.nsmallest(20).values.tolist()))
            print(' --- ')
            print(' --> Largest idf:\n{}'.format(tfidf_series.nlargest(20).index.values.tolist()))
            print('{} \n'.format(tfidf_series.nlargest(20).values.tolist()))
            print(' --- ')
   
    mem = psutil.virtual_memory()
    print('----> memory after scikit-learn training ...')
    print(mem) 
    print('### Memory total     {:.2f} Gb'.format(mem.total/1024**3))
    print('### Memory percent   {:.2f} %'.format(mem.percent))
    print('### Memory available {:.2f} Gb'.format(mem.available/1024**3))
    print('### Memory used      {:.2f} Gb'.format(mem.used/1024**3))
    print('### Memory free      {:.2f} Gb'.format(mem.free/1024**3))
    print('### Memory active    {:.2f} Gb'.format(mem.active/1024**3))
    print('### Memory inactive  {:.2f} Gb'.format(mem.inactive/1024**3))
    print('### Memory buffers   {:.2f} Gb'.format(mem.buffers/1024**3))      
    print('### Memory cached    {:.2f} Gb'.format(mem.cached/1024**3))    
    print('### Memory shared    {:.2f} Gb'.format(mem.shared/1024**3))   
    print('### Memory slab      {:.2f} Gb'.format(mem.slab/1024**3)) 
    print(' ')
    
    # evaluate
    train_y_pred = pipeline.predict(train_X)
    
    # define the score we want to use to evaluate the classifier on
    acc_train = accuracy_score(train_y,train_y_pred)
    
    del train_X
    
    # evaluate
    eval_y_pred = pipeline.predict(eval_X)
    
    # define the score we want to use to evaluate the classifier on
    acc_eval = accuracy_score(eval_y,eval_y_pred)
    
    del eval_X
    
    print('accuracy on test set: \n {} % \n'.format(acc_eval))
    print('accuracy on train set: \n {} % \n'.format(acc_train))

    return pipeline, acc_eval

def save_model(estimator, gcspath, name):
    
    model = 'model.joblib'
    joblib.dump(estimator, model)
    model_path = os.path.join(gcspath, datetime.datetime.now().strftime('export_%Y%m%d_%H%M%S'), model)
    subprocess.check_call(['gsutil', 'cp', model, model_path])
    return model_path