# Topic Models from Ubuntu Dialog Corpus

Detecting topics from multi-user chats (unstructured data) from Ubuntu Dialog Corpus.


## Pre-requisites

- Python 3 (Anaconda)
- NLTK
- scikit-learn
- Gensim
- NumPy

## Data Resource

- Ubuntu Dialog Corpus: http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/
- Raw dialogue files (two-way conversation, no pre-processing):Ubuntu dialogues (527M)
- Folder: 4

## Methodology

Data pre-processing: 

1) Removed " ' " from words, to make words like 'don't' and 'dont' as the same.
2) Lower-casing all the words.
3) RegexTokenizer.
4) Removing general stop words of english language, using stopwords from nltk.corpus.
5) Removing list of stop words specific to this data.
6) Using alphabetic strings.
7) WordNet Lemmatizer
7) For LDA: Filtered out words that occur less than 20 documents, or more than 50% of the documents.

ML models:

Since we do not know the topics and every document can fall under two or more topics, simpler models such as K-means clusterning are not useful here. Rather generative models such as Latent Dirichlet Allocation (LDA), where we have more control over data mean and variance are more helpful. We do not need to know number of topics and these allow the data to fall under more than one topics. Another such technique for approximate topic modeling is Non-negative Matrix Factorization (NMF).

Question 1: Finding the 10 most common topics:
- Used ngrams (unigrams/bigrams) to identify the common topics in the corpus. (Better data-processing techniques e.g. identifying better collocations will help in identifying more useful topics.)

- Future step: identify these using word frequency in LDA.

Trained LDA for entire corpus (Folder 4 data).

Question 2: Topic Detector for any file:
- Used LDA
- Used NMF

## Next steps

1. Using Wikipedia Dataset, to identify topic names.
2. More data pre-processing: removing user names from the chat conversations (implemented it for single file processing), finding better collocations etc.
3. Using Part-of-Speech tagging to identify some structure.
4. Using frequency distribution, finding most common topics using LDA.
5. Use entire corpus for more powerful analysis (using deep learning framework for faster computations).
