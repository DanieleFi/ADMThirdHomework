# Homework 3 - Find the perfect place to stay in Texas!
 
Instructions:
-- use just functions.py file and Homework 3 - group #19
-- cells that are markdown put them as code cells and run them (they should be executed only once and then just saved as files and loaded from the working directory)
-- there should be a folder named 'data' where .tsv files are created and stored

To do list:
  * Should we remove (non-english words)
  * OUR SCORE, rating/#of bedrooms/price with heap data structure python to maintain top k documents in  a query 
 
 
 
The user will give a text query, we will first get the query relatex documents with the search engine of Step 3.1 (?)

Then, to sort the according the new score we will define a new variable from existing ones, this variable (AverageBedrooms) is defined by the average_rate_per_night over the number of bedrooms in each airbnb. This will give us a "normalized" value according to which we can compare all the documents. 


Thus we will create a new inverted index, where each documents will have its corresponding AB score, which we can precalculate because, like the tfidf score previously viewed, it is invariant with respect to the query.

Since our key idea is to show to the consumer the most convenient result, we will have the results ranked by their "Conveniency", that is a percentage based on the maximum and minimum value of AB. The minimum value of AB will be the document that has 100% , because it would be the chapest solution on one bedroom, the maximum value of AB  we will assign 1% and we will rank all the other documents in respect to their "conveniency", given the max and min value. 
Giving the percentage we will have the output ranking as requested. 

We will realize that by calculatine the value which will count as 100% and divide it by 10. so we will have 10 chunks and when we make a  the query we will see which chunk is related to that value and we will give it the value of 10%,20%..etc. Given that we will have the output ranking


-to do: think about how to study and realize the "Conveniency" ranking.
