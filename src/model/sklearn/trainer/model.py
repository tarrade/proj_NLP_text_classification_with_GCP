import google.cloud.bigquery as bigquery
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

#PROJECT = None
#KEYDIR  = 'trainer'

def create_queries():
    query = """
    SELECT
    *
    FROM
    `nlp-text-classification.stackoverflow.posts_preprocessed_selection_subset`
    """
    
    return query

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
    
def create_dataframes():   

    df = query_to_dataframe(create_queries())

    # split in df in training and testing
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=101010)
    
    return train_df, eval_df

def input_fn(input_df):
    df = copy.deepcopy(input_df)
    
    # features, label
    label = df['label']
    del df['label']
    
    features = df['text']
    return features, label

def train_and_evaluate():
    
    # get data
    train_df, eval_df = create_dataframes()
    train_X, train_y = input_fn(train_df)
    
    # train
    pipeline=Pipeline([('Word Embedding', CountVectorizer()),
                       ('Feature Transform', TfidfTransformer()),
                       ('Clasifier', MultinomialNB(alpha=0.1))])
    pipeline.fit(train_X, train_y)
   
    # evaluate
    eval_X, eval_y = input_fn(eval_df)
    eval_y_pred = pipeline.predict(eval_X)
    train_y_pred = pipeline.predict(train_X)

    print('accuracy on test set: \n {} % \n'.format(accuracy_score(eval_y,eval_y_pred)))
    print('accuracy on train set: \n {} % \n'.format(accuracy_score(train_y,train_y_pred)))
    
    return pipeline

def save_model(estimator, gcspath, name):
    from sklearn.externals import joblib
    import os, subprocess, datetime
    model = 'model.joblib'
    joblib.dump(estimator, model)
    model_path = os.path.join(gcspath, datetime.datetime.now().strftime('export_%Y%m%d_%H%M%S'), model)
    subprocess.check_call(['gsutil', 'cp', model, model_path])
    return model_path