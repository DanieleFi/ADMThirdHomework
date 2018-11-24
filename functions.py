#%% LIBRARIES
import pandas as pd
import numpy as np
from collections import defaultdict

from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

from sklearn.metrics.pairwise import cosine_similarity
from heapq import heappush, nlargest

from collections import OrderedDict
import pickle
pd.options.mode.chained_assignment = None
#%% DATA CLEANING

def clean(airbnb_data):
    """
    Method that removes nan values and imputes them
    
    Input: dataframe
    Output: cleaned dataframe
    """
    #replace NAN values with 0 for average_rate_per_night column
    airbnb_data.average_rate_per_night.replace(np.nan, '$0',inplace=True)
    #convert to int average_rate_per_night and remove $
    airbnb_data.average_rate_per_night=airbnb_data.average_rate_per_night.replace('[\$]', '', regex=True).astype(int)

    #replace NAN values with'unknown' for description, title and latitude and longitude
    airbnb_data.description.replace(np.nan,'unknown',inplace=True)
    airbnb_data.title.replace(np.nan,'unknown',inplace=True)

    airbnb_data.latitude.replace(np.nan,'unknown',inplace=True)
    airbnb_data.longitude.replace(np.nan,'unknown',inplace=True)

    
    #check where bedrooms_count doesn't have a value and save indexes of those records to a list
    null_value_idx=airbnb_data[airbnb_data.bedrooms_count.isnull()].index
    #if the word studio is mentioned in the description then it is a studio otherwise 'unknown'
    for idx in null_value_idx:
        if 'studio' in airbnb_data.iloc[idx].description.split():
            airbnb_data.bedrooms_count[idx]='Studio'
        else:
            airbnb_data.bedrooms_count[idx]='unknown'
        
    #remove duplicate houses based on the url
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
    """
    Method that returns filtered words from the text input 
    
    Input: string(text)
    Output: list(bag of words)
    """  
    #remove upper cases
    df=df.lower()
    #replacing new line sign '\n' with a whitespace ' '    
    df=df.replace('\\n',' ')

    #for removing stop words
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
    """
    Method that creates vocabulary
    
    Input: dataframe in order to access number of files made by that airbnb dataframe
    Output: vocabulary list and doc_vocabs(dictionary, key='doc_id',value=list of unique words belonging to that document)
    """  
    #list for vocabulary 
    vocabulary_lst=[]
    #building a dictionary which will be used for making an inverted index
    doc_vocabs=defaultdict(list)

    for i in airbnb_data.index:
        #take one file
        df=pd.read_csv('data/doc_'+str(i)+'.tsv',sep='\t',usecols=['description','title'],encoding='ISO-8859-1')
        #preprocessing description and title
        df=df.description[0]+' '+df.title[0]
        filtered_words=preprocessing_text(df)
        #temporary variable set used for making vocabulary with unique words
        temp_vocabulary_set=set()
        for word in filtered_words:
            temp_vocabulary_set.add(word)
        vocabulary_lst.append(temp_vocabulary_set)
        doc_vocabs[i]=list(temp_vocabulary_set)
    #union of content of vocabulary_lst
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
    """
    method that is used for creating the result dataframe with the columns 'title','description','city','url'
    
    Input: result_set - list of document indices
    Output: df - result dataframe
    """
    #initialization od result df
    df=pd.DataFrame()
    #iterate through result document indices
    for i,val in enumerate(result_set):
        pd.set_option('display.max_colwidth', -1)
        df=df.append(pd.read_csv('data/doc_'+str(val)+'.tsv',sep='\t',usecols=['description','title','city','url']
                                 ,encoding='ISO-8859-1'))
        #reset index 
        df.reset_index(inplace=True)
        #drop column "index" which appears when index is being reseted cause previous index
        #becomes new column named "index"
        df.drop('index',axis=1,inplace=True)
        #return columns in this order
        df=df[['title','description','city','url']]
    return df    
    
    
def search_engine(vocabulary,inverted_idx):
    """
    method that prints the result dataframe with the columns 'title','description','city','url'
    based on user query 
    
    Input:  vocabulary-dictionary of all words,(key='term',value='term_id')
            inverted_idx-dictionary(key='term_id',value=list of doument id's containing that term)
    Output: doc_id_lst-list of document id's which are result documents of the query
            result_set-dataframe which is result of the query presented to the user
    """
    
    user_query=str(input())
    #preprocess text user inputed(same process like in making of vocabulary)
    user_query=preprocessing_text(user_query)

    list_term_idx=[]#list of sets of doc_ids containing inputed words,for each word one set
    result_set=[]
    for word in user_query:
        #if word exist in the vocabulary
        if word in vocabulary.keys():
            list_term_idx.append(set(inverted_idx[vocabulary[word]]))
        else:
            list_term_idx.append({'x'})
            break
    #intersection of sets containing doc_ids
    result_set=list(set.intersection(*list_term_idx))
    doc_id_lst=result_set
    #if intersection is empty set end the method
    if 'x' in result_set or len(result_set) == 0:
        result_set='No results! Try again!'
        print(result_set)
        return doc_id_lst,result_set
    result_set=finalize_output(result_set)
    return doc_id_lst,result_set


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
    """
    method that computes an inverted index
    
    input:  airbnbdata-just for using the number of files we made
            inverted_idx(dictionary, key=term_id, value=list of document_ids)
            vocabulary(dictionary of all unique words, key=term, value=term_id)
    output: tf_idf_dic(dictionary of tf_idf_values for all docs, key=tuple(term,doc_id ), value=tf_idf value)
    """
    tf_idf_dic=dict()
    #number of .tsv files which were made
    total_num_docs=airbnb_data.shape[0]
    result_df=pd.DataFrame()
    for i in airbnb_data.index:
        #take one file
        df=pd.read_csv('data/doc_'+str(i)+'.tsv',sep='\t',usecols=['description','title'],encoding='ISO-8859-1')
        #preprocessing 
        df=df.description[0]+' '+df.title[0] 
        filtered_words=preprocessing_text(df)
        tf_series=pd.Series(filtered_words)
        #series of tf values
        tf_series=((tf_series.value_counts())/len(tf_series)).sort_index()
        idf_series=pd.Series(list(set(filtered_words))).sort_values()
        #idf calculation
        idf_calc=idf_series.apply(lambda x: np.log(total_num_docs/len(inverted_idx[vocabulary[x]])))
        #combine tf and idf in one result_df dataframe
        result_df=pd.concat([pd.Series(idf_series.values),pd.Series(tf_series.values),pd.Series(idf_calc.values)],axis=1)#.reset_index()
        #multiply tf and idf and create tf_idf column
        result_df['tf_idf']=result_df[1]*result_df[2]
        #key=tuple(term,doc_id), value=tf_idf value
        for idx in range(result_df.shape[0]):
            tf_idf_dic[result_df[0][idx],i]=result_df['tf_idf'][idx]
    return tf_idf_dic        

# Second way--to check if it is the same like the 1st-for double checking the results
def calculate_tf_idf2(airbnb_data,inverted_idx,vocabulary):
    """
    method that computes an inverted index(stores it differently than the first one just for comparison)
    
    input:  airbnbdata-just for using the number of files we made
            inverted_idx(dictionary, key=term_id, value=list of document_ids)
            vocabulary(dictionary of all unique words, key=term, value=term_id)
    output: proba(dictionary of tf_idf_values for all docs)
    """
    #store separately tf and idf values into dictionaries
    idf_dic2={}
    tf_dic2={}
    #dictionary for tf_idf values
    proba={}
    total_num_docs=airbnb_data.shape[0]

    for i in airbnb_data.index:
        #take one file
        df=pd.read_csv('data/doc_'+str(i)+'.tsv',sep='\t',usecols=['description','title'],encoding='ISO-8859-1')
        #preprocessing 
        df=df.description[0]+' '+df.title[0] 
        #preprocessed words
        filtered_words=preprocessing_text(df)
        #tf values calculations
        tf_series=pd.Series(filtered_words)
        tf_series=((tf_series.value_counts())/len(tf_series)).sort_index()
        #idf values calculations
        idf_series=pd.Series(list(set(filtered_words))).sort_values()
        idf_calc=idf_series.apply(lambda x: np.log(total_num_docs/len(inverted_idx[vocabulary[x]])))
        #store idf values into dict
        for idx in range(len(tf_series)):
            idf_dic2[idf_series[idx],i]=idf_calc[idx] 
        #store tf values into dict
        for index,value in tf_series.iteritems():
            tf_dic2[index,i]=value
        #combine tf and idf ito a new dictionary by their multiplication using the same key
        for k in tf_dic2.keys():
            proba[k]=tf_dic2[k]*idf_dic2[k]
    return proba        

def compute_inverted_idx2(inverted_idx,vocabulary,tf_idf_dic):
    """
    method that computes the second inverted index
    
    input:  inverted_idx(dictionary, key=term_id, value=list of document_ids)
            vocabulary(dictionary of all unique words, key=term, value=term_id)
            tf_idf_dic(dictionary of tf_idf_values for all docs, key=tuple(term,doc_id ), value=tf_idf value)
    output: inverted_idx2(dictionary, key=term_id, value=list of tuples (document_id,tf_idf value))
    """
    
    inverted_idx2=defaultdict(list)
    #for every term_id from the first inverted index
    for term_id in inverted_idx.keys():
        #find term from vocabulary
        for k,v in vocabulary.items():#k->term, v->term_id
            if v==term_id:
                term=k
        #for every document AKA doc_id from the first inverted index
        for doc_id in inverted_idx[term_id]:
            inverted_idx2[term_id].append((doc_id,tf_idf_dic[term,doc_id]))
    return inverted_idx2


def search_engine2(k,vocabulary,inverted_idx,inverted_idx2):
    """
    method that prints the result dataframe with the columns 'title','description','city','url'
    based on user query
    
    input:  k - number of top documents that should be returned
            inverted_idx(dictionary, key=term_id, value=list of document_ids)
            vocabulary(dictionary of all unique words, key=term, value=term_id)
            inverted_idx2(dictionary, key=term_id, value=list of tuples (document_id,tf_idf value))
    output: result_set dataframe based on the user query
    """
    #text user searches and preprocessing of that text input
    user_query=str(input())
    user_query=preprocessing_text(user_query)
    #tf_idf values for the user query are array of 1, they have 1 as a tf_idf value
    user_query_tfidf=np.ones(len(user_query))
    
    #list of document indices
    list_term_idx=[]
    #list of dataframes
    list_tf_idf=[]

    result_set=[]

    #for every word in user query
    for word in user_query:
        #if word exist in the vocabulary
        if word in vocabulary.keys():
            #append a list of document indices
            list_term_idx.append(set(inverted_idx[vocabulary[word]]))
            #append a list of tuples from inverted_index2
            list_tf_idf.append((inverted_idx2[vocabulary[word]]))            
        else:
            list_term_idx.append({'x'})
            break
    #result will be intersection of all sets od document ids     
    result_set=list(set.intersection(*list_term_idx))
    if 'x' in result_set or not result_set:
        result_set='No results! Try again!'
        return result_set
    tf_idf_dic=defaultdict(list)
    #making of tf_idf_dic dictionary, where key=document_id,value=tf_idf value
    for tf_idf_1doc in list_tf_idf:
        for tuple_pair in tf_idf_1doc:
            if tuple_pair[0] in result_set:
                tf_idf_dic[tuple_pair[0]].append(tuple_pair[1])

    print(result_set)
    result_set=finalize_output2(result_set,user_query_tfidf,tf_idf_dic,k)
    return result_set

def cosine_sim_tuples(user_query_tfidf,tf_idf_dic):
    """
    method that calculates cosine similarity between user query and every document of 
    the result set of the query
    
    Input:  user_query_tfidf - tf_idf values for the user query are array of 1
            tf_idf_dic - dictionary(key=document_id,value=tf_idf value)
    Output: cosine_sim_lst_tuples - list of tuples with calculated cosine similarities tuple(cosine similarity,document_id)
    """
    cosine_sim_lst_tuples=[]
    for key,value in tf_idf_dic.items():
        tf_idf_val=cosine_similarity([user_query_tfidf],[value])[0][0]
        cosine_sim_lst_tuples.append((tf_idf_val,key))#tuple(cosine similarity,document_id)
    return cosine_sim_lst_tuples

def heapify_tuples(cosine_sim_lst_tuples,k):
    """
    method that makes heap from list of tuples and returns K largest values based on
    the cosine similarity value in heap
    
    Input:  cosine_sim_lst_tuples - list of tuples with calculated cosine similarities tuple(cosine similarity,document_id)
            k - number of top documents that should be returned
    Output: wanted_doc-list of document id's with biggest cosine similarity values
    """
    heap = []
    for item in cosine_sim_lst_tuples:
         heappush(heap, item)
    return wanted_doc(nlargest(k,heap))

def wanted_doc(heap_k_docs):
    """
    method that returns list of document id's with biggest cosine similarity values
    
    Input:  heap_k_docs - list of document indices with K biggest cosine similarity values
    Output: df - wanted_doc_ids - list of document id's with biggest cosine similarity values
    """
    wanted_doc_ids={}
    for tup in (heap_k_docs):
        wanted_doc_ids[tup[1]]=round(tup[0],2)
    return wanted_doc_ids

def finalize_output2(result_set,user_query_tfidf,tf_idf_dic,k):
    """
    method that is used for creating the result dataframe with the columns 'title','description','city','url','similarity'
    
    Input:  result_set - list of document indices
            user_query_tfidf - tf_idf values for the user query are array of 1
            tf_idf_dic - dictionary(key=document_id,value=tf_idf value)
            k - number of top documents that should be returned
    Output: df - result dataframe
    """
    #list of tuples of cosine similarity betewwn user_query and every document of the result set
    cosine_sim_lst_tuples=cosine_sim_tuples(user_query_tfidf,tf_idf_dic)
    #check if result set is smaller than top K values that should be returned
    #if it is smaller return whole result set
    if len(result_set)<k:
        k=len(result_set)
    #'HEAPIFY' list of tuples and return top K document id's from the heap
    wanted_doc_ids=heapify_tuples(cosine_sim_lst_tuples,k)
    result_set=wanted_doc_ids.keys()
    df=pd.DataFrame()

    for i,val in enumerate(result_set):
        #display whole text in the columns
        pd.set_option('display.max_colwidth', -1)
        df=df.append(pd.read_csv('data/doc_'+str(val)+'.tsv',sep='\t',usecols=['description','title','city','url']
                                 ,encoding='ISO-8859-1'))
        df.reset_index(inplace=True)
        df.drop('index',axis=1,inplace=True)
    #add column with similarity values
    df['similarity']=wanted_doc_ids.values()
    df=df[['title','description','city','url','similarity']]
    return df

def calculate_room_nums(doc_id_rs):
    """
    method that returns list of possible room numbers based on first user input and
    returns it as a second question so the results can be sorted by their significance
    input:list of document id's of the results from the first query
    output:list of possible number of rooms user can choose based on the first result
    """
    result_df=pd.DataFrame()
    l=[]
    for i in doc_id_rs:
            #take one file
            df=pd.read_csv('data/doc_'+str(i)+'.tsv',sep='\t',usecols=['bedrooms_count'],encoding='ISO-8859-1')
            result_df=result_df.append(df)
    temp=np.unique(result_df.bedrooms_count.values)
    l=[*temp]
    del result_df
    return l

def example_score():
    """
    a simple method to show an example of the new score calculation for sorting result documents
    output:example of user input and dataframe of BR score calculation
    """
    user_input=pd.DataFrame(columns=['chosen_avg_price','chosen_no_rooms'])
    user_input.chosen_no_rooms=[3]
    user_input.chosen_avg_price=[150]
    explanatory_df=pd.DataFrame(columns=['bedrooms_count','B score','average_rate_per_night','R score','BR score'])
    explanatory_df.average_rate_per_night=[50,100,250,550]
    explanatory_df.bedrooms_count=[2,3,3,4]
    explanatory_df['B score']=(0,0.25,0.25,0)
    explanatory_df['R score']=(0.45,0.25,0.05,0)
    explanatory_df['BR score']=explanatory_df['B score']+explanatory_df['R score']
    return user_input,explanatory_df

def calculate_score(row,chosen_avg_price,chosen_no_rooms):
    """
    a method that calculates new score("BR SCORE") which will be used 
    for sorting result documents
    
    input:  row - one row(document) of result set
            chosen_avg_price - price user chose in the additional question
            chosen_no_rooms - number of rooms user chose in the additional question
    output: calculated score for one row
    """
    temp_rate=0
    temp_beds=0
    
    #rules of R score calculation
    if row['average_rate_per_night']in range(0,int((chosen_avg_price/2)+1)):
        temp_rate=0.45
    if row['average_rate_per_night']in range(int(chosen_avg_price/2)+1,int(chosen_avg_price)+1):
        temp_rate=0.25 
    if row['average_rate_per_night'] in range(int(chosen_avg_price),int(chosen_avg_price+101)):
        temp_rate=0.05
    #rules of B score calculation
    if row['bedrooms_count'] == chosen_no_rooms:
        temp_beds=0.25   
    #BR score=B score+ R score
    return temp_rate+temp_beds

def heapify_tuples_BR(BR_score_tuples):
    """
    method that makes heap from list of tuples 
    
    Input:  BR_score_tuples - list of tuples with calculated BR score->tuple(BR_score,document_id)
    Output: list converted to heap structure
    """
    heap = []
    for item in BR_score_tuples:
         heappush(heap, item)
    return heap

def new_score(doc_id_rs,chosen_avg_price,chosen_no_rooms):
    """
    method that is used for creating list of BR scores for each result row and 
    heapified list of tuples 
    
    Input:  doc_id_rs - list of document id's of the results from the first query
            chosen_avg_price - price user chose in the additional question
            chosen_no_rooms - number of rooms user chose in the additional question
    Output: score_lst - list of scores with calculated BR score for each row in the result set
            heapified_tuples - heapified list of tuples where ->tuple(BR_score,document_id)
    """
    #make subdataframe with columns 'average_rate_per_night','bedrooms_count' which will be used 
    #for BR score calculation
    calc_result_df=pd.DataFrame()
    for i in doc_id_rs:
        #take one file
        df=pd.read_csv('data/doc_'+str(i)+'.tsv',sep='\t',usecols=['average_rate_per_night','bedrooms_count'],encoding='ISO-8859-1')
        calc_result_df=calc_result_df.append(df)

    #initialize list for BR score for each result set row
    score_lst=[]
    for idx in range(len(calc_result_df)):
        score_lst.append(calculate_score(calc_result_df.iloc[idx],chosen_avg_price,chosen_no_rooms))
    
    #initialize list for BR score for each result set row
    BR_score_tuples=[]
    for idx,val in enumerate(score_lst):
        #BR_score_tuples (BR score,doc_id)
        BR_score_tuples.append((round(val,2),doc_id_rs[idx]))#idx index of row
        heapified_tuples=heapify_tuples_BR(BR_score_tuples)
    return score_lst,heapified_tuples

def ranking_BR_score(heapified_tuples):
    """
    method that is used for sorting BR scores and creating ranks for result dataframe
    
    Input:  heapified_tuples - heapified list of tuples where ->tuple(BR_score,document_id)
    Output: ranking_dict - dictionary(key=doc_id,value=rank)
    """
    #select nlargest or in this case all len(heapified_tuples) from the list which basically selects all but sorted
    sorted_scores=nlargest(len(heapified_tuples),heapified_tuples)
    sorted_docs_dic=defaultdict(list)
    
    
    for tup in sorted_scores:
        sorted_docs_dic[tup[0]].append(tup[1])
    sorted_docs_rank_dic=defaultdict(list)
    
    #making of rank so the same score has the same rank
    counter_rank=1
    for k,v in sorted_docs_dic.items():
        k=counter_rank
        sorted_docs_rank_dic[k]=v
        counter_rank=counter_rank+1
    #put it in the OrderedDict so the order doesn't change     
    ranking_dict=OrderedDict()
    for k,v in (sorted_docs_rank_dic.items()):
        for list_val in v:
            ranking_dict[list_val]=k
    return ranking_dict

