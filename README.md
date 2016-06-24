# relna - Biomedical Text Mining for Relation Extraction

Relna is a Text Mining (TM) tool for relation extraction for transcription factors and gene / gene products. To the best of our knowledge, it is the first text mining tool for relation extraction of transcriptor factors and associated proteins. It is part of a thesis at Technical University, Munich. This tool is built on the _nalaf_ framework, developed as part of two other theses done at Technical University, Munich. The tool is generic enough that it can be extended by people with their own modules, eg. parsers, features, taggers etc. The method uses Support Vector Machines, and allows for the use of Tree Kernels.

_nalaf_ framework is well documented [here](https://github.com/carstenuhlig/thesis-alex-carsten).

As part of the thesis, an associated corpus by the same name (_relna_) was annotated using [tagtog](https://www.tagtog.net). The _relna_ corpus consists of 140 documents that have been semi-automatically annotated using GNormPlus for named entities and manually annotated for relations. The reason for relation extraction for transcription factors and gene / gene products, and corpus statistics is documented [here](https://github.com/ashishbaghudana/relna/wiki/Corpus).

Using our method, we achieve an F-measure of 69.3% on the _relna_ corpus. The full results of our experiments are available [here](https://github.com/ashishbaghudana/relna/wiki/Results).

![Brief Results](https://cloud.githubusercontent.com/assets/3714983/11531733/438b2460-98ff-11e5-8332-e01d575570cc.png)

* [relna corpus on tagtog](http://pubannotation.org/projects/relna)
* [relna corpus on PubAnnotation](http://pubannotation.org/projects/relna)
* [relna tool on Elixir bio.tools](https://bio.tools/tool/RostLab/relna/0.1.0)

The pipeline used by _relna_ is as follows:

![Pipeline diagram](https://cloud.githubusercontent.com/assets/3714983/11531709/231f59c6-98ff-11e5-8b0e-427cc35c5d80.png)

# Install

##  Requirements

* Python 3
* [SWIG](http://www.swig.org) (>= 3.0.7, sanity check after installation: `swig -version`)
* [PCRE](http://www.pcre.org) (not PCRE2, but 8.x series, >=8.37; sanity check after installation: `pcre-config --version`)
* [BLLIP Parser](https://github.com/BLLIP/bllip-parser)
* [SVMLight-TK-1.2](http://disi.unitn.it/moschitti/TK1.2-software/Tree-Kernel.htm)
    * The easiest way to install it is to download compiled binaries from the [official website.](http://disi.unitn.it/moschitti/TK1.2-software/download.html)
      * You will have to fill up a form to get this, and make the build using the given Makefile.
      * If you are **ABSOLUTELY SURE** that you do not need to use Tree Kernels (TK), you can also get precompiled binaries from the original page: [SVMLight](http://svmlight.joachims.org)
      * Place the binaries `svm_classify` and `svm_learn` in your `$PATH`

## Install Code

* Installation of _nalaf_

```
git clone https://github.com/Rostlab/nalaf
cd nalaf
python3 setup.py install
python3 -m nalaf.download_corpora
```

* Installation of _BLLIP Parser_ (needed to be compiled now locally since their PyPi module doesn't run on Python 3 - follow issue [here](https://github.com/BLLIP/bllip-parser/issues/45))
```
git clone https://github.com/BLLIP/bllip-parser
cd bllip-parser
python3 setup.py install
```

* Installation of _relna_

```
git clone https://github.com/Rostlab/relna.git
cd relna
python3 setup.py install
python3 -m relna.download_corpora
```

Eventually, when the package is registered on PyPi, you can simply install _relna_ by:

    pip3 install relna

# Examples
Run:
* `relna.py` for a simple example how to use _relna_ just for prediction with a pre-trained model
    * `python3 relna.py -c [PATH SVMLight BIN DIR] -p 10383460`
    * `python3 relna.py -c [PATH SVMLight BIN DIR] -s "Conclusion: we find that Ubc9 interacts with the androgen receptor (AR), a member of the steroid receptor family of ligand-activated transcription factors. In transiently transfected COS-1 cells, AR-dependent but not basal transcription is enhanced by the coexpression of Ubc9."`
    * `python3 relna.py -c [PATH SVMLight BIN DIR] -d example.txt`

# Future Work

## Important:
* Implement neural networks (Theano or TensorFlow, when they release for Python 3) for training and classifying data and evaluate performance on that.
* Implement bootstrapping for relation extraction (similar to nalaf, where it has been done for entities)
* Implement multiple sentence models, looking at relations at a distance of one sentence and beyond

## Not-So-Important:
* Implement corereference resolution (might increase performance slightly)
* Check performance on PUBMED data with Tree Kernels (since the precision is really high, we might potentially come across new relationships, even though the recall is low.)
* SpaCy [plans to](https://github.com/spacy-io/spaCy/issues/59) implement its own constituent parser, replace BLLIP with SpaCy for speed and efficiency (no linking to external C/C++ libraries)
