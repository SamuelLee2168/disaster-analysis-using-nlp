import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import string
import os
import shutil
import plotly_express as px
import plotly.graph_objects as go

st.sidebar.subheader('Table of Contents')
st.sidebar.write('1. ','<a href=#introduction>Introduction</a>', unsafe_allow_html=True)
st.sidebar.write('2. ','<a href=#data-exploration>Data Exploration</a>', unsafe_allow_html=True)
st.sidebar.write('3. ','<a href=#calculating-the-vocabulary>Calculating the Vocabulary</a>', unsafe_allow_html=True)
st.sidebar.write('4. ','<a href=#data-preprocessing-for-lstm>Data Preprocessing for LSTM</a>', unsafe_allow_html=True)
st.sidebar.write('5. ','<a href=#hyperparameter-tuning-using-grid-search>Hyperparameter Tuning using Grid Search</a>', unsafe_allow_html=True)
st.sidebar.write('6. ','<a href=#creating-the-final-model>Creating the Final Model</a>', unsafe_allow_html=True)


st.title("Disaster Analysis on Twitter Data using Natural Language Processing")
st.header("Introduction")

introduction_text_1 = """
Looking at Twitter is a great way to extract information about things like the newest trending movie, or the current support rate for a presidential candidate. 

In this article, I created a nueral network to perform natural language processing on historical Twitter data, in order to detect disasters that are shared by people on Twitter. This can be used for disaster relief organizations and news agencies to automatically monitor Twitter. It can also be used to estimate the amount of disasters in the past by looking at historical tweets.

The goal of the nueral network is to output 1 (disaster) or 0 (not disaster) given a tweet from Twitter. For example, the nueral network should output 1 if the tweet is "I can see that fire in the forest" or if the tweet is "The government is evacuating the entire city", while outputting 0 if the tweet is "What's up man?" or "It's pretty sunny today". Also, if the tweet is "If you drink bleach, it can cure COVID", that won't only indicate that there's a disaster going on (COVID outbreak), but also cause a disaster (Don't drink bleach!). 

The nueral network has an average of 80% accuracy in analyzing tweets.
"""
st.write(introduction_text_1)



#def sentence_to_index(raw_text):
#    cv = CountVectorizer()
#    cv.fit_transform(raw_text)

#    vocab = cv.vocabulary_.copy()

#    def lookup_key(string):
#        s = string.lower()
#        return [vocab[w] for w in s.split()]

#    return list(map(lookup_key, raw_text))


df = pd.read_csv("data/original_train.csv")
#raw_train, raw_val_and_test = model_selection.train_test_split(df,train_size=0.7,random_state=1)
#raw_val, raw_test = model_selection.train_test_split(raw_val_and_test,train_size=0.5,random_state=1)


#intersecting_keywords = list(set(raw_train['keyword'].dropna().unique()) & set(raw_val['keyword'].dropna().unique()) & set(raw_test['keyword'].dropna().unique()))
#intersecting_locations = list(set(raw_train['location'].dropna().unique()) & set(raw_val['location'].dropna().unique()) & set(raw_test['location'].dropna().unique()))
#keyword_value_counts = df['keyword'].value_counts()[intersecting_keywords]
#location_value_counts = df['location'].value_counts()[intersecting_locations]
#frequent_keywords = pd.Series(keyword_value_counts.loc[keyword_value_counts>=5].index).map(add_keyword_prefix)
#frequent_locations = pd.Series(location_value_counts.loc[location_value_counts>=1000].index).map(add_location_prefix)
#extra_features_to_use = pd.concat([frequent_keywords,frequent_locations])

#options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
#final_model = tf.keras.models.load_model('models/final_model_v2',options=options)
#my_pkl = open("models/final_model_v2.pkl",'rb')
#final_model = pickle.load(my_pkl)
#my_pkl.close()

#input_text = st.text_input('Input a sentence','')

#if input_text != '':
#    prediction_result = predict_sentence(input_text,final_model,vocab)
#    if prediction_result == 1:
#        st.write("The inputted sentence indicates a disaster!")
#    else:
#        st.write("The inputted sentence does not indiciate a disaster!")

st.header("Data Exploration")
exploration_text_1 = """
Let's explore the data to get a better understanding of it. The twitter tweet dataset is downloaded from kaggle:

https://www.kaggle.com/competitions/nlp-getting-started/data

The dataset has 7613 rows, each representing one tweet from twitter, and 5 columns:

id - An unique arbitrary number that is not useful for classifying the tweet, although it can be used to match up the target column with the other columns.

text - The literal text of the tweet

keyword - Disaster related keywords that are in the tweet. These keywords are manually labeled. For example, "ablaze" , "accident", and "wrecked" are all keywords.

location - The location the tweet was sent from.

target - Whether the tweet indicates disaster or not. 1 = disaster and 0 = no disaster.
"""
st.write(exploration_text_1)
st.dataframe(df.iloc[50:75])

exploration_text_2 = """
43% of the tweets are indicating disaster, while 57% of the tweets are non-disaster, so the dataset is pretty balanced.

Some users don't fill in their location for their profile, and some sentences don't have important keywords, so 33% of the rows in the location column are nan, while 1% of the keywords are nan. All other columns don't have missing values.

Here is a pie chart for the most frequent locations that the tweets come from (all locations that appear less than 15 times are not shown)
"""
st.write(exploration_text_2)

frequent_locations = df['location'].value_counts().loc[df['location'].value_counts()>15]
fig = go.Figure(data=[go.Pie(labels=list(frequent_locations.to_dict().keys()), values=list(frequent_locations), textinfo='label+percent',
                             insidetextorientation='radial'
                            )])
st.plotly_chart(fig)

st.header("Calculating the Vocabulary")
vocab_calculation_text_1 = """
If the nueral network model needs to learn how to recognize ALL of the words that appear in the dataset, the model will take too long to train. We can reduce the training time by limiting the model's vocabulary to only the words that are important. To decide which words are important, we can first train a lasso model on the dataset since lasso models are fast to train, and then we can look at the coefficients of the lasso model and only choose the words with high absolute coefficients. We can also do the same thing with a random forest and then combine the 2 sets of words from lasso and random forest to form the vocabulary.

Here are the steps to preprocess for lasso and random forest:

1. Split the dataset into training, validation, and test sets. I gave the training set 70% of the data, validation set 15% of the data, and test set 15% of the data. 

2. Transform every word into lower case.

3. Mark special text like mentions, hashtags and websites by replacing the punctuation with words. (ie, #teamseas will be converted to hashtagteamseas). This way the model won't get confused on whether the @ and # symbols are normal punctuation or special text. 

4. Remove all punctuation.

5. Count vectorize the words using sklearn.

6. Add the target column back into the dataset (this is not done in the main clean function since in serving data we won't have the target column)
"""

st.write(vocab_calculation_text_1)
with st.expander("Click to view code implementation"):
    code = """
#Import raw dataset
df = pd.read_csv("data/original_train.csv")
#Split dataset into train, validation, and test sets
raw_train, raw_val_and_test = model_selection.train_test_split(df,train_size=0.7,random_state=1)
raw_val, raw_test = model_selection.train_test_split(raw_val_and_test,train_size=0.5,random_state=1)

#marks all special text like mentions, hashtags and websites links
#by replacing punctuation with one connected word
#so that the new special word can be treated as a word by the vocabulary
#for example, #teamseas will get converted to "hashtagteamseas"
def mark_special_text(sentence):
    new_sentence = sentence
    mentions_in_sentence = re.findall(r"@\w+[\s.,?:!-]|@\w+$",new_sentence)
    for mention in mentions_in_sentence:
        new_sentence = new_sentence.replace(mention, "mention"+mention.replace("@",""))

    hashtags_in_sentence = re.findall(r"#\w+[\s.,?:!-]|#\w+$",new_sentence)
    for hashtag in hashtags_in_sentence:
        new_sentence = new_sentence.replace(hashtag, "hashtag"+hashtag.replace("#",""))

    links_in_sentence = re.findall(r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])",new_sentence)
    for link in links_in_sentence:
        new_sentence = new_sentence.replace(link[0]+"://"+link[1]+link[2], "link"+link[2])

    return new_sentence

#count_vectorize takes a df with text and id columns, and
#uses sklearn to count vectorize the sentences,
#meaning each word that appears in the dataset
#gets a column
def count_vectorize(df):
    count_vectorizer = feature_extraction.text.CountVectorizer()
    counted_df = pd.DataFrame.sparse.from_spmatrix(count_vectorizer.fit_transform(df['text']))
    counted_df.columns = ["text_"+word for word in count_vectorizer.get_feature_names_out()]
    counted_df['id'] = df['id']
    return counted_df

#count_vectorize takes a df with a text column, 
#and count vectorizes the text column, 
#meaning each word has it's own one-hot column for each row.
def count_vectorize(df):
    count_vectorizer = feature_extraction.text.CountVectorizer()
    counted_df = pd.DataFrame.sparse.from_spmatrix(count_vectorizer.fit_transform(df['text']))
    counted_df.columns = ["text_"+word for word in count_vectorizer.get_feature_names_out()]
    counted_df = pd.concat([df.reset_index(drop=True),counted_df],axis=1)
    return counted_df

#add_target_using_id takes a dataframe 
#that has an id column and corresponding target column
#and adds the target column to df_to_get_added
#making sure to match the rows using the id column
def add_target_using_id(df_to_get_added,df_with_targets):
    def id_to_target(id):
        return df_with_targets.loc[df_with_targets['id']==id].reset_index()['target'][0]

    new_df = df_to_get_added.copy()
    new_df['target'] = new_df['id'].map(id_to_target)
    return new_df

#main clean function for data that will be inputted to lasso and random forest
def clean_for_lasso(df):
    cleaned_df = pd.DataFrame()
    cleaned_df['id'] = df['id']
    cleaned_df['text'] = df['text'].str.lower()
    cleaned_df['text'] = cleaned_df['text'].map(mark_special_text)
    cleaned_df['text'] = cleaned_df['text'].str.translate(str.maketrans('','',string.punctuation))
    cleaned_df = count_vectorize(cleaned_df)
    return cleaned_df
count_vectorized_train = clean_for_lasso(raw_train)
#We need to use add_target_using_id AFTER clean_for_lasso, since some serving datasets might not have a target column
#This way we can choose whether to add target after clean_for_lasso or not.
count_vectorized_train = add_target_using_id(count_vectorized_train,raw_train)
    """
    st.code(code)

vocab_calculation_text_2 = """
Now we can use grid search in order to find the most optimal hyperparameters for the lasso and random forest models. 

To save computational resources, we can grid search "in waves" by first conducting an overall grid search, and then use the results of the overall grid search to conduct a more narrow grid search, and repeat. For example, if we are tuning the C for lasso, we can first pick C like 0.001,0.01,0.1,1,10,100,1000,10000 to grid search. If we get 100 as the result, we can then grid search 50,70,100,150,200,500 for C and continue the cycle.
"""
st.write(vocab_calculation_text_2)

with st.expander("Click to view code implementation"):
    code = """
params = {
    "penalty":['l1'],
    "solver":["liblinear"],
    "C":[0.001,0.01,0.1,1,10,100,1000,10000],
    "max_iter":[1000]
}
  
lr = model_selection.GridSearchCV(linear_model.LogisticRegression(),param_grid=params,verbose=1)
lr.fit(count_vectorized_train.drop(['id','text','target'],axis=1).to_numpy(),count_vectorized_train['target'].to_numpy())

params = {
    "penalty":['l1'],
    "solver":["liblinear"],
    "C":[0.5,0.7,1,1.3,1.5,2,3,5,7],
    "max_iter":[1000]
}
  
lr = model_selection.GridSearchCV(linear_model.LogisticRegression(),param_grid=params,verbose=1)
lr.fit(count_vectorized_train.drop(['id','text','target'],axis=1).to_numpy(),count_vectorized_train['target'].to_numpy())

#We can first tune the hyperparameters of a decision tree, since it has similar parameters to a random forest, but takes much less time
dt_params = {
    'min_samples_leaf':np.arange(3,10,3),
    'min_samples_split':np.arange(3,10,3),
    'max_depth':np.arange(4,23,8),  
    'random_state':[1],
    'max_leaf_nodes':[100,500,1500,2200],
    'max_features':[0.6,0.8,1,"sqrt","log2"],
    'ccp_alpha':[0,0.2,0.5,0.8,1]
}

dt = model_selection.GridSearchCV(estimator=tree.DecisionTreeClassifier(),param_grid=dt_params,verbose=1)
dt.fit(count_vectorized_train.drop(['id','text','target'],axis=1),count_vectorized_train['target'])

#Now we can use the hyperparameters from the decision tree, and tune the n_estimators hyperparameter
rf_params = {
    'ccp_alpha':[0], 
    'max_depth':[20], 
    'max_features':[0.6],
    'max_leaf_nodes':[500], 
    'min_samples_leaf':[9],
    'min_samples_split':[3], 
    'random_state':[1],
    'n_estimators':[50,100,200,500]
}

rf = model_selection.GridSearchCV(estimator=ensemble.RandomForestClassifier(),param_grid=rf_params,verbose=1)
rf.fit(count_vectorized_train.drop(['id','text','target'],axis=1),count_vectorized_train['target'])

#keep this out
rf = ensemble.RandomForestClassifier(ccp_alpha=0, max_depth=20, max_features=0.6,max_leaf_nodes=500, min_samples_leaf=9,min_samples_split=3, random_state=1)
rf.fit(count_vectorized_train.drop(['id','text','target'],axis=1),count_vectorized_train['target'])

#Convert coefficients into readable dataframes
lr_features = pd.DataFrame({'word':lr.feature_names_in_,'importance':lr.coef_[0]})
rf_features = pd.DataFrame({'word':rf.feature_names_in_,'importance':rf.feature_importances_})
    """
    st.code(code)

vocab_calculation_text_3 = """
Now that we have the feature importance values for each word from lasso and random forest, we can choose the words that have above 0 importance as our vocab for the final model. We can also append "unk_" and "pad_" to vocab. "unk_" can be used to replace all words in the data that are not in our vocab, while "pad_" can be used to increase the length of shorter sentences so that all sentences have the same length.
"""
st.write(vocab_calculation_text_3)

with st.expander("Click to view code implementation"):
    code = """
#Removes "text_", which is needed later since each word has a "text_" prefix
def remove_text_(text):
    return text.replace("text_","")

#Drop all features that have importance of 0
top_lr = lr_features.loc[(lr_features['importance'] > 0)]
top_rf = rf_features.loc[(rf_features['importance'] > 0)]

#Add together the lasso words and random forest words, remove "text_" prefix, and append "unk_" and "pad_" into vocabulary.
vocab = pd.concat([top_lr['word'],top_rf['word']]).unique()
vocab = np.array(pd.Series(vocab).map(remove_text_))
vocab = np.append(vocab,"unk_")
vocab = np.append(vocab,"pad_")
    """
    st.code(code)
    
vocab_calculation_text_4 = """
We will need to one-hot encode the "keyword" and "location" column later, so we should drop the useless keywords and locations. I chose to keep any keyword that appears more than 5 times, and any location that appears more than 10 times in the dataset since keywords are more important.
"""

with st.expander("Click to view code implementation"):
    code = """
    #Adds "keyword_" prefix to a string, this is needed to differentiate keywords from locations
    def add_keyword_prefix(string):
        return "keyword_"+string
    
    #Adds "location_" prefix to a string, this is needed to differentiate locations from keywords
    def add_location_prefix(string):
        return "location_"+string
    
    #Only keep keywords and locations that exist in the train, val and test sets
    #since if a keyword/location doesn't exist in one of the sets, it's useless
    intersecting_keywords = list(set(raw_train['keyword'].dropna().unique()) & set(raw_val['keyword'].dropna().unique()) & set(raw_test['keyword'].dropna().unique()))
    intersecting_locations = list(set(raw_train['location'].dropna().unique()) & set(raw_val['location'].dropna().unique()) & set(raw_test['location'].dropna().unique()))
    #Get value counts of keywords and locations
    keyword_value_counts = df['keyword'].value_counts()[intersecting_keywords]
    location_value_counts = df['location'].value_counts()[intersecting_locations]
    #Only keep keywords that appear more than 4 times, and locations that appear more than 9 times
    frequent_keywords = pd.Series(keyword_value_counts.loc[keyword_value_counts>=5].index).map(add_keyword_prefix)
    frequent_locations = pd.Series(location_value_counts.loc[location_value_counts>=10].index).map(add_location_prefix)
    extra_features_to_use = pd.concat([frequent_keywords,frequent_locations])
    """

st.header("Data Preprocessing for LSTM")

data_preprocesing_text_1 = """
Now that we have our vocab decided, we can clean the data for the LSTM model using this process:
1. Transform all words into lower case
2. Mark all of the special words like mentions, hashtags, and links. 
3. Split the sentences into lists of words.
4. Apply the vocabulary by converting each word into the corresponding index of the word in the vocabulary. 
5. Pad the sentences with the "unk_" string so that every sentence is of the same length.
6. One hot encode the extra features like the location and keyword. 
7. Add the target column back (this is not done in the main clean function since in serving data we won't have the target column).
8. The model will be made using tensorflow, so we need to convert the data into tensor format instead of a dataframe. I also converted the data into numpy format since I will be using the sci-keras library to grid-search the hyperparameters of the model.
"""
st.write(data_preprocesing_text_1)

with st.expander("Click to view code implementation"):
    code = """
#Takes a vocabulary of list format, and turns the vocab into a dictionary
#Where the keys are words, and the values are the corresponding index from the original list.
def convert_to_dict(vocab):
    vocab_dict = {}
    for i in np.arange(len(vocab)):
        vocab_dict[vocab[i]] = i
    return vocab_dict

#Replaces any word in the inputted sentence with "unk_" if it is not in the inputted vocabulary
#and extends the sentence to the inputted sentence_len by appending "pad_" at the end of the sentence
def filter_with_vocab_and_pad_for_sentence(sentence,vocab,sentence_len):
    new_sentence = []
    for word in sentence:
        if word in vocab:
            new_sentence.append(word)
        else:
            new_sentence.append("unk_")
    
    actual_sentence_len = len(new_sentence)
    if (actual_sentence_len >= sentence_len):
        new_sentence = new_sentence[0:sentence_len]
        return new_sentence
    
    for i in np.arange(sentence_len-actual_sentence_len):
        new_sentence.append("pad_")
        
    return new_sentence

#Applies filter_with_vocab_and_pad_for_sentence for entire dataframe
def apply_vocab_and_pad_for_df(df,vocab,sentence_len):
    new_df = df.copy()
    new_sentences = []
    for sentence in df['text']:
        new_sentences.append(filter_with_vocab_and_pad_for_sentence(sentence,vocab,sentence_len))
    new_df['text'] = new_sentences
    new_df['text'] = sentence_to_index_for_df(new_df,vocab)
    return new_df

#Takes a sentence (formatted as a list of strings), and returns a list of the corresponding indexes of the words using the inputted vocabulary
def sentence_to_index(sentence,vocab):
    vocab_dict = convert_to_dict(vocab)
    new_sentence = []
    for word in sentence:
        if (word in vocab_dict.keys()):
            new_sentence.append(vocab_dict[word])
        else:
            new_sentence.append(len(vocab_dict))
    return new_sentence

#Applies sentence_to_index for entire dataframe
def sentence_to_index_for_df(df,vocab):
    new_sentences = []
    for sentence in df['text']:
        new_sentences.append(sentence_to_index(sentence,vocab))
        
    return new_sentences

#Takes a dataframe, extra features to add to the dataframe, and a list of extra features to use (some extra features are useless)
#One hot encodes extra_features_df, while only keeping extra_features_to_use, and concatenates to df
def add_extra_features_to_df(df,extra_features_df,extra_features_to_use):
    dummies = pd.get_dummies(extra_features_df.fillna("_nan"))[extra_features_to_use]
    new_df = pd.concat([df,dummies],axis=1)
    return new_df  

def split_sentence(sentence):
    return sentence.split()

#Main clean function for embedding LSTM model
def clean(df,vocab,sentence_len,extra_features_to_use):
    new_df = pd.DataFrame()
    new_df['id'] = df['id']
    new_df['text'] = df['text'].str.lower()
    new_df['text'] = new_df['text'].map(mark_special_text)
    new_df['text'] = new_df['text'].str.translate(str.maketrans('','',string.punctuation))
    new_df['text'] = new_df['text'].map(split_sentence)
    new_df = apply_vocab_and_pad_for_df(new_df,vocab,sentence_len)
    new_df = add_extra_features_to_df(new_df,df[["keyword","location"]],extra_features_to_use)
    
    return new_df

max_sentence_len = max(raw_train['text'].map(len))
train = clean(raw_train,vocab,max_sentence_len,extra_features_to_use)
train = add_target_using_id(train,raw_train)
val = clean(raw_val,vocab,max_sentence_len,extra_features_to_use)
val = add_target_using_id(val,raw_val)
test = clean(raw_test,vocab,max_sentence_len,extra_features_to_use)
test = add_target_using_id(test,raw_test)

def list_to_array(input_list):
    return np.array(input_list)

#The current "text" column is a series of lists, so instead of just using np.array for the series, 
#we need to convert each individual list into an array
def convert_to_array(data):
    array = data.map(list_to_array).to_numpy()
    new_array = np.array([])
    for row in array:
        new_array = np.concatenate([new_array,row])
    
    return new_array

sentence_len = len(train['text'][0])
train_len = train.shape[0]
val_len = val.shape[0]
test_len = test.shape[0]
#Converts the dataframes into tensors
tensor_train_sentences = tf.convert_to_tensor(list(train['text']))
tensor_train_y = tf.convert_to_tensor(list(train['target']))
tensor_train_extra = tf.convert_to_tensor(np.array(train.drop(['id','text','target'],axis=1)))
tensor_val_sentences = tf.convert_to_tensor(list(val['text']))
tensor_val_y = tf.convert_to_tensor(list(val['target']))
tensor_val_extra = tf.convert_to_tensor(np.array(val.drop(['id','text','target'],axis=1)))
tensor_test_sentences = tf.convert_to_tensor(list(test['text']))
tensor_test_y = tf.convert_to_tensor(list(test['target']))
tensor_test_extra = tf.convert_to_tensor(np.array(test.drop(['id','text','target'],axis=1)))
#Converts the dataframes into arrays
numpy_train_sentences = convert_to_array(train['text']).reshape(train_len,sentence_len)
numpy_train_y = train['target'].to_numpy()
numpy_val_sentences = convert_to_array(val['text']).reshape(val_len,sentence_len)
numpy_val_y = val['target'].to_numpy()
numpy_test_sentences = convert_to_array(test['text']).reshape(test_len,sentence_len)
numpy_test_y = test['target'].to_numpy()
    """
    st.code(code)
    
st.header("Hyperparameter Tuning using Grid Search")

hyperparameter_text_1 = """
To grid search the hyperparameters, I am using sci-keras because it combines sklearn functions with tensorflow models, meaning we can use the GridSearchCV function from sklearn on a tensorflow model. To do so, we can first define a function that returns a tensorflow sequential model, and then wrap the tensorflow model with sci-keras. To save computational resources, I conducted "4 waves" of grid searches by first conducting an overall grid search, and then narrowing down the search. This way we don't have to conduct a giant grid search that takes weeks.
"""
st.write(hyperparameter_text_1)

with st.expander("Click to view code implementation"):
    code = """
#Returns the sequential embedding LSTM model. 
#Note that since this is a sequential model, we cannot use the extra input 
#since the extra input needs to be concatenated with the result of the LSTM layer, which is not possible with a sequential model
#later we will use tensorflow functional api to create the final model, which can use the extra input
def get_sequential_model(vocab_len,embedding_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_len,output_dim=embedding_dim,mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

#Uses "KerasClassifier" wrapper from sci-keras to allow tensorflow model to use sklearn grid searching
sci_keras_model = KerasClassifier(
    model=get_sequential_model,
    vocab_len = len(vocab),
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
    fit__validation_data = (numpy_val_sentences,numpy_val_y),
)

#Custom scorer for sklearn to keep track of f1_score
f1_scorer = metrics.make_scorer(metrics.f1_score, greater_is_better=True)

#We first grid search the optimizer. We get Adamax as the result, which is an optimizer that works well for models with embedding layers
gs_1_params = {
    "optimizer":["SGD","RMSprop","Adam","Adadelta","Adagrad","Adamax","Nadam","Ftrl"],
    "epochs":[25],
    "model__embedding_dim":[64]
}

gs_1 = GridSearchCV(sci_keras_model, gs_1_params, cv=5, scoring=f1_scorer,verbose=2)

gs_1.fit(numpy_train_sentences, numpy_train_y)
print(gs_1.best_score_, gs_1.best_params_)

#We can define 2 early stop callbacks and choose between the 2 using grid search
early_stop_1 = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=8
)

early_stop_2 = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=15
)

#Now we can grid search the learning rate, the number of epochs, and which early stop to use
gs_2_params = {"optimizer__learning_rate":[0.0002,0.001,0.005],
    "epochs":[25,50,100],
    "callbacks":[early_stop_1,early_stop_2],
    "model__embedding_dim":[64]
}

gs_2 = GridSearchCV(sci_keras_model, gs_2_params, cv=5, scoring=f1_scorer,verbose=2)

gs_2.fit(numpy_train_sentences, numpy_train_y)
print(gs_2.best_score_, gs_2.best_params_)

#For the third wave, we grid search the learning rate and number of epochs further
gs_3_params = {"optimizer":["adamax"],
    "optimizer__learning_rate":[0.0002,0.001,0.005],
    "epochs":[15,25,35],
    "callbacks":[early_stop_1],
    "model__embedding_dim":[64]
}

gs_3 = GridSearchCV(sci_keras_model, gs_3_params, cv=5, scoring=f1_scorer,verbose=2)

gs_3.fit(numpy_train_sentences, numpy_train_y)
print(gs_3.best_score_, gs_3.best_params_)

#Finally, we grid search the embedding dimension, and the learning rate
gs_4_params = {
    "optimizer":["adamax"],
    "optimizer__learning_rate":[0.0003,0.001,0.003],
    "epochs":[35],
    "model__embedding_dim":[32,64,128,200],
    "callbacks":[early_stop_1]
}

gs_4 = GridSearchCV(sci_keras_model, gs_4_params, cv=5, scoring=f1_scorer,verbose=2)

gs_4.fit(numpy_train_sentences, numpy_train_y)
print(gs_4.best_score_, gs_4_part_1.best_params_)
    """
    st.code(code)
    
hyperparameter_text_2 = """
We get from that grid search that the most optimal hyperparameters for the model are:

Optimizer:Adamax

Learning rate:0.001

Epochs:35

Embedding dimension:32

Early stop callback:Early stop 1 (patience of 8)
"""
st.write(hyperparameter_text_2)

st.header("Creating the Final Model")

final_model_text_1 = """
With the hyperparameters determined, we can define the final model using tensorflow functional api in order to include the extra features like location and keyword as a part of the input data.

The model is made up of 4 layers:
1. Embedding layer - Transforms the lists of indices of each word into a list of embeddings, allowing the model to understand the meaning of each individual word.
2. Bidirectional LSTM layer - Processes the embeddings from the embedding layer as an ordered sequence and makes a prediction.
3. Dense layer - Combines the output of the LSTM layer with the extra input (the keywords and locations)
4. Dense Layer - Final layer to wrap up the model
"""
st.write(final_model_text_1)

with st.expander("Click to view code implementation"):
    code = """
    #Returns final model
    def get_final_model(sentence_input,extra_input,vocab_len,embedding_dim):
        #Define tensorflow input objects
        input_1 = keras.Input(shape=(sentence_input.shape[1]))
        input_2 = keras.Input(shape=(extra_input.shape[1]))
        #Process sentence input through embedding and then bidirectional lstm layers
        X = tf.keras.layers.Embedding(input_dim=(vocab_len),output_dim=embedding_dim,mask_zero=True)(input_1)
        X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim))(X)
        #Combine output of bidirectional lstm layer with extra input
        combined_X = tf.keras.layers.concatenate([X,input_2])
        #Run through dense layers to get the final output
        X = tf.keras.layers.Dense(embedding_dim, activation='relu')(combined_X)
        output = tf.keras.layers.Dense(1)(X)
        
        model = keras.Model(inputs = [input_1,input_2],outputs=output)
        return model
    
    final_model = get_final_model(sentence_input=tensor_train_sentences,extra_input=tensor_train_extra,vocab_len=len(vocab),embedding_dim=32)
    
    final_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adamax(0.001),
                  metrics=['accuracy',tfa.metrics.F1Score(num_classes=1)])
    
    epochs=35
    final_model.fit(x=(tensor_train_sentences,tensor_train_extra),
        y=tensor_train_y,
        validation_data=((tensor_val_sentences,tensor_val_extra),tensor_val_y),
        epochs=epochs,
        callbacks=[early_stop_1])
        """
    st.code(code)
    
evaluation_text_1 = """
Evaluating the model with the test set, we see that the model achieves:

Train accuracy:
0.85

Train F1 score:
0.79

Validation accuracy:
0.81

Validation F1 score:
0.68

Test accuracy:
0.80

Test F1 score:
0.67
"""
st.write(evaluation_text_1)

with st.expander("Click to view code implementation"):
    code = """
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

#Calculates f1 score using the f1 score formula
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#Converts predicted outputs of model to probabilities (0 or 1)
def convert_to_probs(preds):
    probs = np.exp(preds)/(1+np.exp(preds))
    return np.round(probs).flatten()

def evaluate_accuracy(true_y,pred_y):
    return np.mean(pred_y==np.array(true_y))

#Prints out the accuracy and f1-score of a model for train, val, and test sets
def eval_model(model,train_x,train_y,val_x,val_y,test_x,test_y):
    train_preds = convert_to_probs(model.predict(train_x))
    val_preds = convert_to_probs(model.predict(val_x))
    test_preds = convert_to_probs(model.predict(test_x))
    print("Train accuracy:")
    print(evaluate_accuracy(train_y,train_preds))
    print("Train F1 score:")
    print(f1_score(train_y,train_preds))
    print("Val accuracy:")
    print(evaluate_accuracy(val_y,val_preds))
    print("Val F1 score:")
    print(f1_score(val_y,val_preds))
    print("Test accuracy:")
    print(evaluate_accuracy(test_y,test_preds))
    print("Test F1 score:")
    print(f1_score(test_y,test_preds))
    
eval_model(final_model,(tensor_train_sentences,tensor_train_extra),tensor_train_y,(tensor_val_sentences,tensor_val_extra),tensor_val_y,(tensor_test_sentences,tensor_test_extra),tensor_test_y)
    """
    st.code(code)
