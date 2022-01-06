import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import signal
import multiprocessing
import pickle
from itertools import product
from datetime import datetime as dt
import tqdm


#Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.models import TfidfModel

#spacy
import spacy
from nltk.corpus import stopwords

RANDOM_STATE = 100

stopwords = stopwords.words("english")
news_accounts = ['foxnews', 'nytimes', 'CNN', 'NBCNews', 'voxdotcom', 'washingtonpost', 'WSJ', 'AP', 'Reuters', 'newsmax', 'OANN', 'BreitbartNews']
badwords = ['https_co', 'https', 'co', 'new', 'be', 'say', 'in', 'make', 'go', 'tell', 'write', 'want', 'watch', 'get', 'see', 'use', 'like', 'city', 'state', 'country', 'one', 'two', 'use', 'breaking_news', 'new_york_times', 'government', 'icymi', 'biden', 'president', 'president_biden'] + [i.lower() for i in news_accounts] + stopwords

def merge_retweet_full_text(data: pd.DataFrame) -> pd.DataFrame:
    """Replaces truncated retweet text in full_text with the true full text"""
    data = data.copy()
    data['is_retweet'] = ~pd.isna(data['retweeted_status.full_text'])
    data.loc[~data['retweeted_status.full_text'].isna(),'full_text'] = data.loc[~pd.isna(data['retweeted_status.full_text']),'retweeted_status.full_text']
    return data

def load_data() -> List[List[str]]:
    
    data = pd.read_excel("../data/nytimes_foxnews_tweets.xlsx")

    # Clean data
    data = merge_retweet_full_text(data)
    data = data.dropna(axis=1)
    data = data[data['lang'] == "en"]
    data = data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    texts = list(data['full_text'])
    texts = [" ".join([t for t in text.split(" ") if "https:" not in t]) for text in texts]
    return texts

def lemmatization(texts, allowed_postags=["PROPN", "NOUN", "ADJ", "VERB", "ADP", "NUM"]) -> List[List[str]]:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    for text in texts:
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags and token.is_alpha:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return (texts_out)

def gen_words(texts) -> List[List[str]]:
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return (final)

def clean_words(texts) -> List[List[str]]:
    return [[ele for ele in sub if ele not in badwords] for sub in texts]

def make_ngrams(texts) -> List[List[str]]:
    bigram_phrases = gensim.models.Phrases(texts, min_count=2, threshold=5)
    trigram_phrases = gensim.models.Phrases(bigram_phrases[texts], threshold=5)

    bigram = gensim.models.phrases.Phraser(bigram_phrases)
    trigram = gensim.models.phrases.Phraser(trigram_phrases)

    def make_bigrams(texts):
        return([bigram[doc] for doc in texts])

    def make_trigrams(texts):
        return ([trigram[bigram[doc]] for doc in texts])

    data_bigrams = make_bigrams(texts)
    return make_trigrams(data_bigrams)

def tf_idf_removal(texts) -> Tuple[List,List]:
    id2word = corpora.Dictionary(texts)

    corpus = [id2word.doc2bow(text) for text in texts]

    tfidf = TfidfModel(corpus, id2word=id2word)

    low_value = 0.03
    words  = []
    words_missing_in_tfidf = []
    for i in range(0, len(corpus)):
        bow = corpus[i]
        low_value_words = [] #reinitialize to be safe. You can skip this.
        tfidf_ids = [id for id, _ in tfidf[bow]]
        bow_ids = [id for id, _ in bow]
        low_value_words = [id for id, value in tfidf[bow] if value < low_value]
        drops = low_value_words+words_missing_in_tfidf
        for item in drops:
            words.append(id2word[item])
        words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] # The words with tf-idf socre 0 will be missing

        new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
        corpus[i] = new_bow

    return id2word, corpus

def process_texts(texts) -> Tuple[List,List]:
    texts = gen_words(lemmatization(texts))
    texts = make_ngrams(clean_words(texts))
    return tf_idf_removal(clean_words(texts))

def notebook_process() -> Tuple[List,List]:
    texts = load_data()
    return process_texts(texts)

def init_worker():
    ''' Add KeyboardInterrupt exception to multiprocessing workers '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def train_model(n, params) -> Dict:
    id2word, corpus, texts, n_topics, n_iter, decay, minimum_phi_value = params

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=n_topics,
                                           random_state=RANDOM_STATE,
                                           update_every=1,
                                           chunksize=100,
                                           passes=n_iter,
                                           decay=decay,
                                           per_word_topics=True,
                                           minimum_phi_value=minimum_phi_value,
                                           alpha="auto")

    coherence_model_train = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, processes=1)
    coherence_train = coherence_model_train.get_coherence()

    print(f"Model {n}: n_topics = {n_topics}, n_iter = {n_iter}, decay = {decay}, min_phi_value = {minimum_phi_value}, coherence = {coherence_train:.4f}")
    
    return {"lda_model":lda_model, "n_topics":n_topics, "n_iter":n_iter, "decay":decay, "minimum_phi_value":minimum_phi_value, "coherence":coherence_train}

def star_train_model(args):
    return train_model(*args)

def train_multiprocess(params, n_workers) -> Dict:
    with multiprocessing.Pool(n_workers, init_worker) as pool:
        return list(tqdm.tqdm(pool.imap(star_train_model, enumerate(params)), total=len(params)))

def main_hyper():
    print("Processing text...")

    n_topics = [13]
    n_iters = [50, 100, 200, 300]
    decay = [0.5, 0.6, 0.7]
    minimum_phi_value = [0.01, 0.03, 0.05]

    start = dt.now()

    texts = load_data()
    texts = gen_words(lemmatization(texts))
    texts = make_ngrams(clean_words(texts))
    id2word, corpus = tf_idf_removal(clean_words(texts))

    params = [args for args in product([id2word], [corpus], [texts], n_topics, n_iters, decay, minimum_phi_value)]

    print(f"Training {(n_models := len(params))} models...")

    lda_models = train_multiprocess(params, 8)

    best = np.argmax([model['coherence'] for model in lda_models])

    print(f"Best model: {lda_models[best]}")

    with open("../data/models.pickle", "wb") as f:
        pickle.dump(lda_models, f, protocol=pickle.HIGHEST_PROTOCOL)

    end = dt.now()

    print(f"Execution time for {n_models} models: {(elapsed := end - start)}, avg: {elapsed / n_models}")

    print("Done")

def main_topics():
    print("Processing text...")

    n_topics = range(1, 61)
    n_iters = [50]
    decay = [0.5]
    minimum_phi_value = [0.01]

    start = dt.now()

    texts = load_data()
    texts = gen_words(lemmatization(texts))
    texts = make_ngrams(clean_words(texts))
    id2word, corpus = tf_idf_removal(clean_words(texts))

    params = [args for args in product([id2word], [corpus], [texts], n_topics, n_iters, decay, minimum_phi_value)]

    print(f"Training {(n_models := len(params))} models...")

    lda_models = train_multiprocess(params, 10)

    best = np.argmax([model['coherence'] for model in lda_models])

    print(f"Best model: {lda_models[best]}")

    with open("../data/models.pickle", "wb") as f:
        pickle.dump(lda_models, f, protocol=pickle.HIGHEST_PROTOCOL)

    end = dt.now()

    print(f"Execution time for {n_models} models: {(elapsed := end - start)}, avg: {elapsed / n_models}")

    print("Done")

def main():
    main_topics()

if __name__ == "__main__":
    main()


    