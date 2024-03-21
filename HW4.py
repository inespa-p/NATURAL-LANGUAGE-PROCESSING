# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 2023
Author: Ines Perez

"""
"""
1.  clean_text
2.  remove stopwords
3.  perform stemming
4.  call up vectorizer and transform
5.  call up the pca function and
6.  call up the model and classify/prediction
"""

from  utils import *
import pandas as pd
import praw
import pandas as pd
import pytz
from datetime import datetime
import pickle

out_path="/Users/inesperezalvarez-pallete/Desktop/NLP/"

#define our subreddit channel of interest
subreddit_channel = 'politics'

#load our pca, trained model  and vectorizesr
with open('/Users/inesperezalvarez-pallete/Desktop/NLP/pca.pk', 'rb') as f:
    pca=pickle.load(f)

reddit = praw.Reddit(
     client_id="P810r9LiXccsOcC4rkYZ6g",
     client_secret="hqhomAaWu8B54aEsutXfgxX9kz7fqA",
     user_agent="testscript by u/fakebot3",
     username="ines_PAP",
     password="Pie18974355!",
     check_for_async=False
 )



def conv_time(var):
    tmp_df = pd.DataFrame()
    tmp_df = tmp_df.append(
        {'created_at': var},ignore_index=True)
    tmp_df.created_at = pd.to_datetime(
        tmp_df.created_at, unit='s').dt.tz_localize(
            'utc').dt.tz_convert('US/Eastern') 
    return datetime.fromtimestamp(var).astimezone(pytz.utc)

def get_reddit_data(var_in):
    import pandas as pd
    tmp_dict = pd.DataFrame()
    tmp_time = None
    try:
        tmp_dict = tmp_dict.append({"created_at": conv_time(
                                        var_in.created_utc)},
                                    ignore_index=True)
        tmp_time = tmp_dict.created_at[0] 
    except:
        print ("ERROR")
        pass
    tmp_dict = {'msg_id': str(var_in.id),
                'author': str(var_in.author),
                'body': var_in.body, 'datetime': tmp_time}
    return tmp_dict
    

#we create a function here that serves to predict the sentiment 
#of incoming reddit messages, following a series of preprocessing steps 
def predict(message):
    
    # Completing text preprocessing, including stemming and vectorization
    #cleaning the text
    x_text = clean_txt(message)
  
    #remove stopwords
    text_without_sw = rem_sw(x_text)  # Renamed variable to avoid conflict
    apply_stem = stem_fun(text_without_sw)
    print(apply_stem)
    #vectorize the data
    vec_f = read_pickle(out_path, "vectorizer")
    x_data = vec_f.transform([apply_stem]).toarray()
    print(x_data)
    
    #apply pca for dimension reduction
    #loaded_pca = read_pickle(out_path, "pca")
    pca_applied = pca.transform(x_data)
    print(len(pca_applied))    
    #apply our pretrained model for feature classification
    
    model_f = read_pickle(out_path, "my_model")
    the_pred = model_f.predict(pca_applied)[0]

    the_pred_proba = pd.DataFrame(model_f.predict_proba(pca_applied))
    
    the_pred_proba.columns = model_f.classes_
    
    print("Corpus from this Reddit comment belongs to", the_pred,
          "with likelihood of", max(the_pred_proba.iloc[0]))
    print(the_pred_proba)
    return the_pred, max(the_pred_proba.iloc[0])

#iterate for incoming reddit messages, and compute sentiment score and classifcation
for comment in reddit.subreddit(subreddit_channel).stream.comments():
    tmp_df = get_reddit_data(comment)
    # Perform text classification and print class label prediction and likelihood score
    predict(tmp_df['body'])
    


