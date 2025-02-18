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

