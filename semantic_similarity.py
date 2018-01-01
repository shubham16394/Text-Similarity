from normalization import penn_to_wn_tags
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn

def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn_tags(tag)
    if wn_tag is None:
        return None
 
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None
 
def sentence_similarity(sentence1, sentence2):
    
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))

    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
 
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
 
    score, count = 0.0, 0

    for synset in synsets1:
        best_score = max([synset.path_similarity(ss) for ss in synsets2])
 
        if best_score is not None:
            score += best_score
            count += 1
 
    score /= count
    return score