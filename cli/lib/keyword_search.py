from xml.dom import INDEX_SIZE_ERR
from lib.search_utils import load_movies,load_stop_words,CACHE_PATH
import string
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
import pickle
import os
import math

ps = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("","",string.punctuation))
    return text

def tokenise_text(text):
    text = clean_text(text)
    stop_words = load_stop_words()

    tokens = []
    
    def _filter(token):
        if token and token not in stop_words:
            return True
        return False
    
    for tok in text.split():
        if _filter(tok):
            tok = ps.stem(tok)
            tokens.append(tok)
        
    return tokens

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_freaquencies = defaultdict(Counter)

        self.index_path = CACHE_PATH/'index.pkl'
        self.docmap_path = CACHE_PATH/'docmap.pkl' #mapping id to document
        self.term_freaquencies_path = CACHE_PATH/'term_freq.pkl'
    
    def __add_document(self, doc_id, text):
        tokens = tokenise_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)

        self.term_freaquencies[doc_id].update(tokens)

    def get_documents(self, term):
        return sorted(list(self.index[term]))
    
    def get_tf(self,doc_id,term):
        token = tokenise_text(term)

        if len(token) != 1:
            raise ValueError("Only 1 token is allowed")
        
        return self.term_freaquencies[doc_id][token[0]]

    def get_idf(self,term):
        token = tokenise_text(term)

        if len(token) != 1:
            raise ValueError("Only 1 token is allowed")
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[token[0]])

        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    
    def get_tf_idf(self,doc_id, term):

        tf = self.get_tf(doc_id,term)
        idf = self.get_idf(term)

        return round(tf * idf,2)


    def build(self):
        movies = load_movies()

        for movie in movies:
            doc_id = movie['id']
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id,text)
            self.docmap[doc_id] = movie

    def save(self):
        os.makedirs(CACHE_PATH,exist_ok=True)

        with open(self.index_path,'wb') as f:
            pickle.dump(self.index,f)

        with open(self.docmap_path,'wb') as f:
            pickle.dump(self.docmap,f)

        with open(self.term_freaquencies_path,'wb') as f:
            pickle.dump(self.term_freaquencies,f)


    def load(self):
        with open(self.index_path,'rb') as f:
            self.index = pickle.load(f)

        with open(self.docmap_path,'rb') as f:
            self.docmap = pickle.load(f)

        with open(self.term_freaquencies_path,'rb') as f:
            self.term_freaquencies = pickle.load(f)


def has_matching_tokens(query_tokens, movie_tokens):
    for query_tok in query_tokens:
        for movie_tok in movie_tokens:
            if query_tok in movie_tok:
                return True
    return False

def search_command(key,n_results=5):

    query_tokens = tokenise_text(key)
    idx = InvertedIndex()
    idx.load()

    seen, results = set(), []

    for token in query_tokens:
        matching_doc_ids = idx.get_documents(token)

        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            movie = idx.docmap.get(doc_id)
            results.append(movie)
            seen.add(doc_id)
    
            if len(results) >= n_results:
                return results
    
    return results

def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()

def tf_command(doc_id,term):
    idx = InvertedIndex()
    idx.load()

    print(idx.get_tf(doc_id,term))

def idf_command(term):
    idx = InvertedIndex()
    idx.load()
    print(idx.get_idf(term))

def tfidf_command(doc_id,term):
    idx = InvertedIndex()
    idx.load()
    tf_idf = idx.get_tf_idf(doc_id,term)
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")



