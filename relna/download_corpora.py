"""
Downloads the necessary NLTK corpora for relna.

Usage: ::

    $ python -m relna.download_corpora

"""
if __name__ == '__main__':
    import nltk
    from spacy.en import download

    CORPORA = ['stopwords']

    for corpus in CORPORA:
        nltk.download(corpus)

    # See: https://github.com/explosion/spaCy/blob/master/spacy/en/download.py -- we don't need Glove, but hey, maybe in the future!
        # glove global vectors for word representation bibtex
        # http://nlp.stanford.edu/pubs/glove.pdf
    download.main(data_size='parser', force=False)
