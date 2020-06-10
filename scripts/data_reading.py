import pandas as pd

def read_train(base_path):
    train=pd.read_csv(base_path + '/input/tweet-sentiment-extraction/train.csv')
    train['text']=train['text'].astype(str)
    train['selected_text']=train['selected_text'].astype(str)
    return train

def read_test(base_path):
    test=pd.read_csv(base_path + '/input/tweet-sentiment-extraction/test.csv')
    test['text']=test['text'].astype(str)
    return test

def read_submission(base_path):
    test=pd.read_csv(base_path + '/input/tweet-sentiment-extraction/sample_submission.csv')
    return test

def read_data(params):
    train_df = read_train(params["base_path"])
    test_df = read_test(params["base_path"])
    submission_df = read_submission(params["base_path"])
    
    return train_df, test_df, submission_df