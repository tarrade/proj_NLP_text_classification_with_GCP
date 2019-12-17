from collections import Counter
import operator
import google.cloud.bigquery as bigquery

def create_queries(eval_size):
    
    query = """
    SELECT
      *
    FROM
      `nlp-text-classification.stackoverflow.posts_preprocessed`    
    """
    
    eval_query = "{} WHERE MOD(ABS(FARM_FINGERPRINT(CAST(id as STRING))),100) < {}".format(query, eval_size)
    train_query  = "{} WHERE MOD(ABS(FARM_FINGERPRINT(CAST(id as STRING))),100)>= {}".format(query, eval_size)
  
    return train_query, eval_query
    
def create_queries_subset(eval_size):
    query = """
    SELECT
      *
    FROM
      `nlp-text-classification.stackoverflow.posts_preprocessed_selection_subset`
    """
    
    eval_query = "{} WHERE MOD(ABS(FARM_FINGERPRINT(CAST(id as STRING))),100) < {}".format(query, eval_size)
    train_query  = "{} WHERE MOD(ABS(FARM_FINGERPRINT(CAST(id as STRING))),100)>= {}".format(query, eval_size)
  
    return train_query, eval_query

def build_tag(row, list_tags):
    new_list=[]
    for idx, val in enumerate(row):
        if val in list_tags:
            new_list.append(val)
    del row
    return new_list

def query_to_dataframe(query, is_training, tags, nb_label):
    
    client = bigquery.Client()
    df = client.query(query).to_dataframe()
    
    # label
    if is_training:
        tags=df['tags'].sum()
        unique_tags = dict(Counter(tags))
        unique_tags = sorted(unique_tags.items(), key=operator.itemgetter(1))
        unique_tags.reverse()
        max_nb_label=len(unique_tags)+1
        if nb_label>max_nb_label: nb_label=max_nb_label
        keep_tags=[x[0] for x in unique_tags][0:nb_label]
    else:
        keep_tags=tags
    
    if is_training:
        print('list of labels to be used\n',keep_tags)
        print('number of labels',len(keep_tags))
        print('max number of labels set',nb_label)
        
    #print(df['tags'])
    df['tags'] = df['tags'].apply(lambda x: build_tag(x, keep_tags))
    #print(df['tags'])
    df['label'] = df['tags'].apply(lambda x: x[0] if len(x)>0 else 'other-tags')
    #print(df['label'])
    #df['label'] = df['tags'].apply(lambda row: ",".join(row))
    del df['tags']
    
    #print('list tags {}'.format(df['label'].unique()))
    
    # features
    df['text'] = df['title'] + df['text_body'] + df['code_body']
    del df['code_body']
    del df['title']
    del df['text_body']
    
    # use BigQuery index
    df.set_index('id',inplace=True)
    
    return keep_tags, df


def create_dataframes(frac, eval_size, nb_label):   
    
    # small dataset for testing
    if frac > 0 and frac < 1:
        sample = " AND RAND() < {}".format(frac)
    else:
        sample = ""

    train_query, eval_query = create_queries(eval_size)
    train_query = "{} {}".format(train_query, sample)
    eval_query =  "{} {}".format(eval_query, sample)
    
    keep_tags,train_df = query_to_dataframe(train_query, True, '', nb_label)
    _, eval_df = query_to_dataframe(eval_query, False, keep_tags, nb_label)
    
    print('size of the training set          : {:,}'.format(len(train_df )))
    print('size of the evaluation set        : {:,}'.format(len(eval_df)))
    
    print('number of labels in training set  : {}'.format(len(train_df['label'].unique())))
    print('number of labels in evaluation set: {}'.format(len(eval_df['label'].unique())))
                                                              
    return train_df, eval_df


def input_fn(df):
    
    # features, label
    label = df['label']
    del df['label']
    
    features = df['text']
    return features, label