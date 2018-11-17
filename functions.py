import numpy as np
import pandas as pd

from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict

import pickle

#import enchant 
pd.options.mode.chained_assignment = None

def clean(airbnb_data):
    """
    Method that removes nan values and imputes them
    
    Input: dataframe
    Output: cleaned dataframe
    
    """
    #replace NAN with 0
    airbnb_data.average_rate_per_night.replace(np.nan, '$0',inplace=True)
    #convert to int and remove $
    airbnb_data.average_rate_per_night=airbnb_data.average_rate_per_night.replace('[\$]', '', regex=True).astype(int)

    #replace NAN with'unknown'

    airbnb_data.description.replace(np.nan,'unknown',inplace=True)
    airbnb_data.title.replace(np.nan,'unknown',inplace=True)

    airbnb_data.latitude.replace(np.nan,'unknown',inplace=True)
    airbnb_data.longitude.replace(np.nan,'unknown',inplace=True)

    #check where bedrooms_count doesn't have a value and save indexes of those records to a list
    null_value_idx=airbnb_data[airbnb_data.bedrooms_count.isnull()].index
    #if the word studio is mentioned in description then it is a studio otherwise 'unknown'
    for idx in null_value_idx:
        if 'studio' in airbnb_data.iloc[idx].description.split():
            airbnb_data.bedrooms_count[idx]='Studio'
        else:
            airbnb_data.bedrooms_count[idx]='unknown'
        
    return airbnb_data

def create_tsv_documents(airbnb_data):
    """
    Method that creates different .tsv files for each record in the airbnb_data 
    
    Input: dataframe
    """   
    #clean data
    airbnb_data=clean(airbnb_data)
    
    #for each index make a dataframe of airbnb_data and store it into new tsv file
    for i in airbnb_data.index:
        pd.DataFrame(airbnb_data.loc[i]).transpose().to_csv('data/doc_'+str(i)+'.tsv',sep='\t')
        
def preprocessing_text(df):
    #remove upper cases
    df=df.lower()
    #replacing new line sign '\n' with a whitespace ' '    
    df=df.replace('\\n',' ')

    #removing stop words and punctuation
    stop_words = set(stopwords.words('english')) 

    #for removing punctuations
    tokenizer = RegexpTokenizer(r'\w+')
    
    #to tokenize the string
    word_tokens = tokenizer.tokenize(df) 
    
    #stemming
    ps = PorterStemmer()
    filtered_words = [ps.stem(w) for w in word_tokens if not w in stop_words] 

    return filtered_words



def build_vocabulary(airbnb_data):
    #set for vocabulary (values of the set will be the keys fo vocabulary_dict)
    vocabulary_lst=[]
    #building a dictionary which will be used for making an inverted index
    doc_vocabs=defaultdict(list)

    for i in airbnb_data.index:
        #take one file
        df=pd.read_csv('data/doc_'+str(i)+'.tsv',sep='\t',usecols=['description','title'],encoding='ISO-8859-1')
        #preprocessing 
        df=df.description[0]+' '+df.title[0]
        filtered_words=preprocessing_text(df)
        temp_vocabulary_set=set()
        for word in filtered_words:
            temp_vocabulary_set.add(word)
        vocabulary_lst.append(temp_vocabulary_set)
        doc_vocabs[i]=list(temp_vocabulary_set)
    vocabulary_set=set.union(*vocabulary_lst)
    #mapping words into integers
    vocabulary={}
    for k,v in enumerate(vocabulary_set):
        vocabulary[v]= k
    return vocabulary,doc_vocabs


def save_vocabulary(vocabulary,file_name): 
    """
    method that converts vocabulary into a dataframe and saves it into a csv file
    
    input: vocabulary(dictionary, key='term',value='term_id')
    """
    vocabulary_dataframe=pd.DataFrame()
    vocabulary_dataframe['word']=vocabulary.keys()
    vocabulary_dataframe.to_csv(str(file_name)+'.csv')
     
    
#conjunctive query
    
def finalize_output(result_set):
    df=pd.DataFrame()
    for i,val in enumerate(result_set):
        pd.set_option('display.max_colwidth', -1)
        df=df.append(pd.read_csv('data/doc_'+str(val)+'.tsv',sep='\t',usecols=['description','title','city','url']
                                 ,encoding='ISO-8859-1',index_col=False))
        df.reset_index().drop('index',axis=1,inplace=True)
    return df    
    
    
def search_engine(vocabulary,inverted_idx):
    user_query=str(input())
    #input()

    user_query=preprocessing_text(user_query)

    list_term_idx=[]
    result_set=[]
    for word in user_query:
        #if word exist in the vocabulary
        if word in vocabulary.keys():
            list_term_idx.append(set(inverted_idx[vocabulary[word]]))
        else:
            list_term_idx.append({'x'})
            break
    result_set=list(set.intersection(*list_term_idx))
    if 'x' in result_set or not result_set:
        result_set='No results! Try again!'
        return result_set
        
    print(result_set)
    result_set=finalize_output(result_set)
    return result_set


def compute_inverted_idx(doc_vocabs,vocabulary):
    """
    method that computes an inverted index
    
    input: doc_vocabs(dictionary), vocabulary(dictionary of all unique words, key=term, value=term_id)
    output: inverted_idx(dictionary, key=term_id, value=list of document_ids) 
    """
    #initialize defaultdict for making an inverted index
    inverted_idx = defaultdict(list)
    #in every document look for every word and assign document id to the words which belong to it
    for idx in doc_vocabs.keys():
        for word in doc_vocabs[idx]:
            inverted_idx[vocabulary[word]].append(idx)
    return inverted_idx

def save_inverted_idx(inverted_idx):
    #save it into a file named inverted_idx.p
    pickle.dump(inverted_idx, open("inverted_idx.p", "wb"))  
def load_inverted_idx():
    #load file named inverted_idx.p
    return pickle.load(open("inverted_idx.p", "rb"))
