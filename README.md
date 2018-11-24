# Homework 3 - Find the perfect place to stay in Texas!
 
The goal of this project was to create two different search engines using the [airbnb data](https://www.kaggle.com/PromptCloudHQ/airbnb-property-data-from-texas) that, given as input a query, return the houses that matches the query.


Instructions for project utilization:

	1. Download airbnb data
 	2. Use files: functions.py and Homework 3 - group #19.ipynb
 	3. Cells that are markdown cells in the .ipynb file put them as code cells and run them (they should be executed only once and then just saved as files and loaded from the working directory)
 	4. There should be a folder named 'data' where .tsv files are created and stored

To do list:
  * Should we remove (non-english words)
  * OUR SCORE 
 !!!!!* Return in output *k* documents, or all the documents with non-zero similarity with the query when the results are less than _k_. You __must__ use a heap data structure (you can use Python libraries) for maintaining the *top-k* documents.

 
 
The user will give a text query, we will first get the query relatex documents with the search engine of Step 3.1 (?)


 
 
The repository consists of the following files:
1. __`Homework 3 - group #19.ipynb`__: 
     > A Jupyter notebook which provides the following: 
	
       Search Engine 1 - Conjunctive query
    	    The first Search Engine evaluated queries based on the `description` and `title` of each document.  
 
       Search Engine 2 - Conjunctive query & Ranking score
	   In the new Search Engine, given a query, top-k documents related to the query should be returned 
	   sorted based on the calculated _Cosine similarity_  
	   
       Search Engine 3 - Conjunctive query & a new score
			
2. __`functions.py`__:
      > A python script which provides all the functions used in the `Homework 3 - group #19.ipynb` notebook. 

 

 

 
 
 
