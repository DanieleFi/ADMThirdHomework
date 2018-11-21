#%% LIBRARIES
import pandas as pd
import numpy as np

from collections import defaultdict
#import re
#import nltk

from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
#from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer

from sklearn.metrics.pairwise import cosine_similarity
from heapq import heappush, nlargest

#import enchant 
import pickle

pd.options.mode.chained_assignment = None
#%% DATA CLEANING

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
        
        
    airbnb_data.url=airbnb_data.url.apply(lambda x:x.split('?')[0])
    airbnb_data.drop_duplicates(subset='url',inplace=True)
    
    return airbnb_data
#%% Making of .tsv files

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

#%% Making of vocabulary
       
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
     
 #%% First search engine with a conjunctive query   
    
def finalize_output(result_set):
    df=pd.DataFrame()
    for i,val in enumerate(result_set):
        pd.set_option('display.max_colwidth', -1)
        df=df.append(pd.read_csv('data/doc_'+str(val)+'.tsv',sep='\t',usecols=['description','title','city','url']
                                 ,encoding='ISO-8859-1'))
        df.reset_index().drop('index',axis=1,inplace=True)
    return df    
    
    
def search_engine(vocabulary,inverted_idx):
    user_query=str(input())

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


 #%% Second search engine with a conjunctive query 


# First way
#TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
#IDF(t) = log_e(Total number of documents / Number of documents with term t in it)

def calculate_tf_idf(airbnb_data,inverted_idx,vocabulary):
    tf_idf_dic=dict()
    total_num_docs=airbnb_data.shape[0]
    result_df=pd.DataFrame()
    for i in airbnb_data.index:
        #take one file
        df=pd.read_csv('data/doc_'+str(i)+'.tsv',sep='\t',usecols=['description','title'],encoding='ISO-8859-1')
        #preprocessing 
        df=df.description[0]+' '+df.title[0] 
        filtered_words=preprocessing_text(df)
        tf_series=pd.Series(filtered_words)
        tf_series=((tf_series.value_counts())/len(tf_series)).sort_index()
        idf_series=pd.Series(list(set(filtered_words))).sort_values()
        idf_calc=idf_series.apply(lambda x: np.log(total_num_docs/len(inverted_idx[vocabulary[x]])))
        result_df=pd.concat([pd.Series(idf_series.values),pd.Series(tf_series.values),pd.Series(idf_calc.values)],axis=1)#.reset_index()
        result_df['tf_idf']=result_df[1]*result_df[2]

        for idx in range(result_df.shape[0]):
            tf_idf_dic[result_df[0][idx],i]=result_df['tf_idf'][idx]
    return tf_idf_dic        

# Second way--to check if it is the same like the 1st-for double checking the results
def calculate_tf_idf2(airbnb_data,inverted_idx,vocabulary):
    idf_dic2={}
    tf_dic2={}
    proba={}
    total_num_docs=airbnb_data.shape[0]

    for i in airbnb_data.index:
        #take one file
        df=pd.read_csv('data/doc_'+str(i)+'.tsv',sep='\t',usecols=['description','title'],encoding='ISO-8859-1')
        #preprocessing 
        df=df.description[0]+' '+df.title[0] 
        filtered_words=preprocessing_text(df)
        tf_series=pd.Series(filtered_words)
        tf_series=((tf_series.value_counts())/len(tf_series)).sort_index()
        idf_series=pd.Series(list(set(filtered_words))).sort_values()
        idf_calc=idf_series.apply(lambda x: np.log(total_num_docs/len(inverted_idx[vocabulary[x]])))
       
        for idx in range(len(tf_series)):
            idf_dic2[idf_series[idx],i]=idf_calc[idx] 
        for index,value in tf_series.iteritems():
            tf_dic2[index,i]=value
        for k in tf_dic2.keys():
            proba[k]=tf_dic2[k]*idf_dic2[k]
    return proba        

def compute_inverted_idx2(inverted_idx,vocabulary,tf_idf_dic):
    """
    method that computes the second inverted index
    
    input:   
    output: inverted_idx2(dictionary, key=term_id, value=list of tuples (document_id,tf_idf value) )
    """
    inverted_idx2=defaultdict(list)
    for term_id in inverted_idx.keys():
        for k,v in vocabulary.items():#k->term, v->term_id
            if v==term_id:
                term=k
        for doc_id in inverted_idx[term_id]:
            inverted_idx2[term_id].append((doc_id,tf_idf_dic[term,doc_id]))
    return inverted_idx2


def search_engine2(k,vocabulary,inverted_idx,inverted_idx2):
    user_query=str(input())
    user_query=preprocessing_text(user_query)
    user_query_tfidf=np.ones(len(user_query))

    list_term_idx=[]
    #list of dataframes
    list_tf_idf=[]

    result_set=[]

#    result_tf_idf_dic=defaultdict(list)
    for word in user_query:
        #if word exist in the vocabulary
        if word in vocabulary.keys():
            list_term_idx.append(set(inverted_idx[vocabulary[word]]))
            list_tf_idf.append((inverted_idx2[vocabulary[word]]))#[:,1])
            #result_tf_idf_dic
        else:
            list_term_idx.append({'x'})
            break
    result_set=list(set.intersection(*list_term_idx))
    if 'x' in result_set or not result_set:
        result_set='No results! Try again!'
        return result_set
    tf_idf_dic=defaultdict(list)

    for tf_idf_1doc in list_tf_idf:
        for tuple_pair in tf_idf_1doc:
            if tuple_pair[0] in result_set:
                tf_idf_dic[tuple_pair[0]].append(tuple_pair[1])

    print(result_set)
    result_set=finalize_output2(result_set,user_query_tfidf,tf_idf_dic,k)
    return result_set

def cosine_sim_tuples(user_query_tfidf,tf_idf_dic):
    cosine_sim_lst_tuples=[]
    for key,value in tf_idf_dic.items():
        tf_idf_val=cosine_similarity([user_query_tfidf],[value])[0][0]
        cosine_sim_lst_tuples.append((tf_idf_val,key))
    return cosine_sim_lst_tuples

def heapify_tuples(cosine_sim_lst_tuples,k):
    heap = []
    for item in cosine_sim_lst_tuples:
         heappush(heap, item)
    return wanted_doc(nlargest(k,heap))

def wanted_doc(heap_k_docs):
    wanted_doc_ids={}
    for tup in (heap_k_docs):
        wanted_doc_ids[tup[1]]=round(tup[0],2)
    return wanted_doc_ids

def finalize_output2(result_set,user_query_tfidf,tf_idf_dic,k):
    cosine_sim_lst_tuples=cosine_sim_tuples(user_query_tfidf,tf_idf_dic)
    wanted_doc_ids=heapify_tuples(cosine_sim_lst_tuples,k)
    result_set=wanted_doc_ids.keys()
    df=pd.DataFrame()

    for i,val in enumerate(result_set):
        pd.set_option('display.max_colwidth', -1)
        df=df.append(pd.read_csv('data/doc_'+str(val)+'.tsv',sep='\t',usecols=['description','title','city','url']
                                 ,encoding='ISO-8859-1'))
        df.reset_index().drop('index',axis=1)
    df['similarity']=wanted_doc_ids.values()
    df=df[['title','description','city','url','similarity']]
    return df