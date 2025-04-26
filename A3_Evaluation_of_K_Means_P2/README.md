# Assignment 03 – Introduction to Text Analytics

## Student Information
- **Name:** Zuha Aqib
- **ID:** 26106
- **Institute:** Institute of Business Administration

## Objective
This assignment explores the clustering effectiveness of **Word2Vec** and **Doc2Vec** embeddings using **K-Means** clustering.  
We perform experiments for three different numbers of clusters: **k = 5, 9, and 13**.

For each `k`, ten experiments are performed with varying hyperparameters:
- **Vector Size**
- **Window Size**
- **Epochs**
- **Model Type:** CBOW/Skipgram (for Word2Vec) and DM/DBOW (for Doc2Vec)

The **random seed** was set to the student's ERP ID for consistency across experiments.

Metrics recorded:
- **Silhouette Score (SIL)**
- **Within-Cluster Sum of Squares (WSS)**

## Dataset Preprocessing
Following the best practices from Assignment 02, the text data was:
- **Cleaned**
- **Tokenized**
- **Stopwords Removed**
- **Lemmatized**
- **Unigrams extracted**

## Experiments Summary

### Word2Vec
- **Approaches tested:** CBOW and Skipgram
- **Key Observations:**
  - **Skipgram** produced lower WSS values but suffered from poor SIL scores.
  - **CBOW** generally achieved **higher Silhouette Scores** and better clustering compactness overall.
  - Smaller **epochs** led to better results.
  - **Vector size** beyond 400 worsened performance for Skipgram.
  - CBOW models with a **window size of 10** and lower **epochs (5-50)** performed best.

### Doc2Vec
- **Approaches tested:** DBOW and DM
- **Key Observations:**
  - **DBOW** performed best with large **vector sizes** (up to 4000).
  - **Window size** changes from 3 to 20 **did not affect** performance at all.
  - Increasing **epochs** worsened performance drastically and made training slow.
  - **DM** model showed even better SIL scores at higher vector sizes (~2500–3000), outperforming DBOW slightly.

## Best Results
- **Word2Vec CBOW** achieved strong performance with reasonable training time.
- **Doc2Vec DM** achieved the highest Silhouette Score (0.3176) with very compact clusters (WSS ~0.122).
- Overall, **Doc2Vec with DM architecture** gave the best clustering results.

## Comparison with Previous Assignment (Assignment 02)
- In Assignment 02, our best scores were:
  - **Silhouette Score ~0.56**
  - **WSS ~52**
- In this assignment:
  - **Silhouette Score ~0.31**
  - **WSS ~0.12**

Thus, **Word2Vec** and **Doc2Vec** embeddings produced **significantly better clustering** compared to the TF-IDF-based approaches used previously.

## Analysis of Hyperparameter Tuning
| Hyperparameter | Word2Vec Behavior | Doc2Vec Behavior |
|:---------------|:------------------|:-----------------|
| **Vector Size** | Better results at moderate sizes (200–400) | Better results at very high sizes (up to 4000) |
| **Window Size** | Medium window sizes (8–10) best | No impact observed |
| **Epochs** | Fewer epochs (5–50) better | Fewer epochs (50) better |

## Final Note
This assignment demonstrates that using **semantic embeddings** like **Word2Vec** and **Doc2Vec** can drastically improve clustering performance compared to traditional bag-of-words models.

> *Doc2Vec DM model with vector_size ~3000 and epochs ~50 was the best combination.*