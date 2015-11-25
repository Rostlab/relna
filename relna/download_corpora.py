"""
Downloads the necessary NLTK corpora for nala.

Usage: ::

    $ python -m relna.download_corpora

"""
if __name__ == '__main__':
    import nltk

    CORPORA = ['stopwords']

    for corpus in CORPORA:
        nltk.download(corpus)
