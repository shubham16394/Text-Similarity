# Text Similarity using BM25 & WordNet

# Prerequisite for running code

1) Python 2.x - https://askubuntu.com/questions/101591/how-do-i-install-the-latest-python-2-7-x-or-3-x-on-ubuntu
2) Numpy - pip install numpy
3) Scipy - pip install scipy
4) NLTK - pip install nltk
5) Pattern - pip install pattern
6) Sklearn - pip install sklearn

# Command for running code

python execute.py

# OUTPUT

![alt text](https://github.com/shubham16394/WeCP_Task/blob/master/output.JPG)

# Algorithms

Syntactic Similarity -

I used BM25(Best Matching) algorithm for syntactic similarity, it generates the similarity score between two sentences.

BM25 Algorithm - 
  
    bm25_score(CD,QD) = 𝚺(i=1 to n) idf(qi)*(f(qi,CD)*k1+1)/(f(qi,CD)+k1*(1-b+(b*|CD|/avgdl)))
    idf(t) = 1 + log(C/1+df(t))
    
    Where,
    
        CD = corpus document, e.g.- list of all the answers
        QD = query document, e.g.- list of model answer
        idf(qi) = inverse document frequency (IDF) of the term qi in CD
        C = count of the total number of documents in CD
        df(t) = frequency of the number of documents in which the term t is present
        f(qi, CD) = frequency of the term qi in CD
        |CD| = total length of the CD
        avgdl = average document length of CD
        k1, b = Constants
        
Semantic Similarity - 

I used NLTK's WordNet corpus for generating the semantic similarity score between two sentences. I used synsets function to get all the lexnames of a word then calulated the path similarity between words then took the maximum value among all the lexnames for a single word. After that I calculated the average of all scores for a single sentence and that is the value of semantic similarity score. 

Final Score is the average of bm25_score and semantic_similarity_score.







