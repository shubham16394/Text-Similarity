
from normalization import normalize_corpus
from utils import build_feature_matrix
from bm25 import compute_corpus_term_idfs
from bm25 import compute_bm25_similarity
import numpy as np


def run():
	answers=['Functions are used as one-time processing snippet for inling and jumbling the code.', 
		'Functions are used for reusing, inlining and jumbling the code.', 
		'Functions are used as one-time processing snippet for inlining and organizing the code.', 
		'Functions are used as one-time processing snippet for modularizing and jumbling the code.', 
		'Functions are used for reusing, inling and organizing the code.', 
		'Functions are used as one-time processing snippet for modularizing and organizing the code.', 
		'Functions are used for reusing, modularizing and jumbling the code.', 
		'Functions are used for reusing, modularizing and organizing the code.']

	model_answer = ["Functions are used for reusing, modularizing and organizing the code."]


	# normalize answers
	norm_corpus = normalize_corpus(answers, lemmatize=True)
	                                                        
	# normalize model_answer
	norm_model_answer =  normalize_corpus(model_answer, lemmatize=True)            

	vectorizer, corpus_features = build_feature_matrix(norm_corpus,feature_type='frequency')

	# extract features from model_answer
	model_answer_features = vectorizer.transform(norm_model_answer)

	doc_lengths = [len(doc.split()) for doc in norm_corpus]   
	avg_dl = np.average(doc_lengths) 
	corpus_term_idfs = compute_corpus_term_idfs(corpus_features, norm_corpus)
	                 
	for index, doc in enumerate(model_answer):
	    
	    doc_features = model_answer_features[index]
	    bm25_scores = compute_bm25_similarity(doc_features,corpus_features,doc_lengths,avg_dl,corpus_term_idfs,k1=1.5, b=0.75)
	    print 'Model Answer',':', doc
	    print '-'*40 
	    doc_index=0
	    for sim_score in bm25_scores:
	    	if(sim_score<1):
	    		sim_score=0
	    	elif(1<=sim_score<=2):	
	    		sim_score=1
	    	elif(2<sim_score<=4):
	    		sim_score=2
	    	elif(4<sim_score<=6):
	    		sim_score=3
	    	elif(6<sim_score<=8):
	    		sim_score=4
	    	elif(8<sim_score<=10):	
	    		sim_score=5					
	        print 'Ans num: {} Score: {}\nAnswer: {}'.format(doc_index+1, sim_score, answers[doc_index])  
	        print '-'*40       
	        doc_index=doc_index+1
	    print

run()	    