# Assignment 02 - K-Means Clustering Evaluation
We will be analysing all types of combinations for k-means. we have to analyse for k=5, 9 and 13
- all are with lowercase conversion, so this is not a combination

### Lets count all the combinations
- (2) stop words removal: with or without
- (2) stemming vs lemmatization
- (2) n-grams: unigrams or bigrams

so thats 8 preprocessing techniques on every combination (4 on unigram and 4 on bigram and so on)

- text vectorization: BOW (count vectorizer(tp or tf) or tfidf)

so thats 8 + 8 on countvectorizer and 8 on tfidf which is 24 total

- text vectorization: truncatedSVD on TF-IDF vectors (try on 50, 100, 200)

so thats 8 on 50, 8 on 100, 8 on 200 so 24 + 24 = 48 total

and then we have to do each of these 48 for k=5, 9, 11 so 48 + 48 + 48 = 144 total

so lets begin!

## Step 1
lets make an excel sheet displaying all these entries to get a good visual idea first

## Step 2
writing the code and testing every part of it to make sure it works

## Step 3
run code for all cases

## Step 4 
analyse. all analysing is done in A2_Assessment.pdf

assignment completed!

# Assignment 2 – K-Means Clustering Assessment

**Course:** Introduction to Text Analytics  
**Name:** Zuha Aqib  
**ID:** 26106  

## Overview

This assignment involves evaluating K-Means clustering performance on text data using different preprocessing and vectorization techniques. The experiments are conducted for three different values of **k** (number of clusters): 5, 9, and 13. For each value of k, ten experiments were performed, using random seeds set to my ERP ID (26106) to ensure consistency.

## Experiment Setup

For each k, experiments varied based on:
- **Vectorization techniques:** 
  - Bag of Words (BOW) with Term Frequency (TP) and Term Frequency (TF)
  - TF-IDF
  - TruncatedSVD (Dimensionality Reduction with 50, 100, 200 components)
- **Text preprocessing steps:**
  - Stemming (Yes/No)
  - Lemmatization (Yes/No)
  - N-gram usage (Unigram/Bigram)
  - Stopword removal (Yes/No)

Each configuration was evaluated based on:
- **Silhouette Score** (higher is better)
- **Within-cluster Sum of Squares (WSS)** (lower is better)

## Evaluations
This table shows that the smallest WSS was found in case 43  for  all  k’s,  (best  was  k=13).  That  case  was:  LSA  50, lemmatization, bigrams and removed stop words.  The best silhouette score was found again in cases 43 and 31. Again, the best was k=5 in case 43. Case 31 was  LSA 50, stemming, bigrams, and not removing stop words.

So the best embedding technique was hands down LSA 50.  
- LSA 50 gave WSS in 50-100, whereas  
- BOW was in 3500-4000.  
- LSA 100 was 100-200 and  
- LSA 200 was 200-300.  
- TFIDF was 400-500.  
 
So the best was LSA 50. The silhouette scores were also the highest in LSA 50. 
- I  noticed  that  when  we  applied  bigrams,  we  got  more  columns,  and  silhouette  falled  into  a negative FOR BOW. However for the rest bigrams performed better, it gave a higher silhouette and lower WSS 
- However removing stop words was ALWAYS good, whenever I didn’t remove, I had more columns and very high WSS and low silhouette 
- Lemmatization performed slightly better than stemming 
- LSA 50 was faster than LSA 200. It took slightly longer to run LSA 200.  