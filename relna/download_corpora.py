"""
Downloads the necessary NLTK corpora for relna.

Usage: ::

    $ python -m relna.download_corpora

"""
if __name__ == '__main__':
    import nltk
	import spacy

    CORPORA = ['stopwords']

    for corpus in CORPORA:
        nltk.download(corpus)
		
	spacy.en.download()
