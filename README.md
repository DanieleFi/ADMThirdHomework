# Homework 3 - Find the perfect place to stay in Texas!

<p align="center">
<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQaaU_U07OhDmVBWK8ezLMYxzp1SbzuL1rCVvACQYPKub6mvTGc">
</p>


The goal of this project was analyzing the text of property listings and creating three different search engines using the [airbnb data](https://www.kaggle.com/PromptCloudHQ/airbnb-property-data-from-texas) that, given as input a query, return the houses that match the query. 



Instructions for project utilization:

	1. Download airbnb data
 	2. Use files: functions.py and Homework 3 - group #19.ipynb
 	3. Cells that are markdown cells in the .ipynb file put them as code cells 
	and run them (they should be executed only once and then just saved as files and loaded from the working directory)
 	4. There should be a folder named 'data' where .tsv files are created and stored
 
 
The repository consists of the following files:
1. __`Homework 3 - group #19.ipynb`__: 
     > A Jupyter notebook which provides the following: 
	
       Search Engine 1 - Conjunctive query
    	    The first Search Engine evaluated queries based on the `description` and `title` of each document. It also uses inverted index to return the result of the query. Inverted index is in the form of dictionary(key=term_id, value=list of document_ids). 
 
       Search Engine 2 - Conjunctive query & Ranking score
	   In the new Search Engine, given a query, top-k documents related to the query should be returned 
	   sorted based on the calculated _Cosine similarity_  
	   Based on the second inverted index it will return the result of the query. Second inverted index is in the form of		            dictionary(key=term_id, value=list of tuples(doc_id,dict{key=(term,doc_id), value=tf_idf value}). 
	   Afterwards the values were stored and sorted using the heap structure. It was also used to return top-5 houses.
	   
       Search Engine 3 - Conjunctive query & a new score
			
2. __`functions.py`__:
      > A python script which provides all the functions used in the `Homework 3 - group #19.ipynb` notebook. 

3. __`Maps_radius.html`__:
      > A map that shows the houses in the radius user chose based on the location he entered. The code is in the `Homework 3 - group #19.ipynb` notebook. 

Team members: * Dusica Stepic * Giulia Maslov * Daniele Figoli *
  


 
 
 
