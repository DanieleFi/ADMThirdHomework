# Homework 3 - Find the perfect place to stay in Texas!
 
The goal of this project was to create two different search engines using the [airbnb data](https://www.kaggle.com/PromptCloudHQ/airbnb-property-data-from-texas) that, given as input a query, return the houses that matches the query.


Instructions for project utilization:
	1. Download airbnb data
 	2. Use files: functions.py and Homework 3 - group #19.ipynb
 	3. Cells that are markdown cells in the .ipynb file put them as code cells and run them (they should be executed only once and then just saved as files and loaded from the working directory)
 	4. There should be a folder named 'data' where .tsv files are created and stored

To do list:
  * Should we remove (non-english words)
  * OUR SCORE, rating/#of bedrooms/price with heap data structure python to maintain top k documents in  a query 
 !!!!!* Return in output *k* documents, or all the documents with non-zero similarity with the query when the results are less than _k_. You __must__ use a heap data structure (you can use Python libraries) for maintaining the *top-k* documents.

 
 
The user will give a text query, we will first get the query relatex documents with the search engine of Step 3.1 (?)

Then, to sort the according the new score we will define a new variable from existing ones, this variable (AverageBedrooms) is defined by the average_rate_per_night over the number of bedrooms in each airbnb. This will give us a "normalized" value according to which we can compare all the documents. 


Thus we will create a new inverted index, where each documents will have its corresponding AB score, which we can precalculate because, like the tfidf score previously viewed, it is invariant with respect to the query.

Since our key idea is to show to the consumer the most convenient result, we will have the results ranked by their "Conveniency", that is a percentage based on the maximum and minimum value of AB. The minimum value of AB will be the document that has 100% , because it would be the chapest solution on one bedroom, the maximum value of AB  we will assign 1% and we will rank all the other documents in respect to their "conveniency", given the max and min value. 
Giving the percentage we will have the output ranking as requested. 

We will realize that by calculatine the value which will count as 100% and divide it by 10. so we will have 10 chunks and when we make a  the query we will see which chunk is related to that value and we will give it the value of 10%,20%..etc. Given that we will have the output ranking


-to do: think about how to study and realize the "Conveniency" ranking.
 
 
The repository consists of the following files:
1. __`Homework 3 - group #19.ipynb`__: 
     > A Jupyter notebook which provides the following: 
	
       Search Engine 1 - Conjunctive query
    	    The first Search Engine evaluated queries based on the `description` and `title` of each document.  
 
       Search Engine 2 - Conjunctive query & Ranking score
	   In the new Search Engine, given a query, top-k documents related to the query should be returned 
	   sorted based on the calculated _Cosine similarity_  
       Search Engine 2 - Conjunctive query & Our new score
			
2. __`functions.py`__:
      > A python script which provides all the functions used in the `Homework 3 - group #19.ipynb` notebook. 

 

 

 
 
 
