# relna - Biomedical Text Mining for Relation Extraction
Relna is a Text Mining (TM) tool for relation extraction for transcription factors and gene / gene products. It is part of a thesis at Technical University, Munich. This tool is built on the _nalaf_ framework, developed as part of two other theses done at Technical University, Munich. The tool is generic enough that it can be extended by people with their own modules, eg. parsers, features, taggers etc. The method uses Support Vector Machines, and allows for the use of Tree Kernels.

_nalaf_ framework is well documented [here](https://github.com/carstenuhlig/thesis-alex-carsten).

As part of the thesis, an associated corpus by the same name (_relna_) was annotated using TagTog. The _relna_ corpus consists of 140 documents that have been semi-automatically annotated using GNormPlus for named entities and manually annotated for relations. The reason for relation extraction for transcription factors and gene / gene products, and corpus statistics is documented [here](https://github.com/ashishbaghudana/relna/wiki/Corpus).

Using our method, we achieve an F-measure of 69.3% on the _relna_ corpus. The full results of our experiments are available [here](https://github.com/ashishbaghudana/relna/wiki/Results).

<!-- ![Pipeline diagram](https://www.lucidchart.com/publicSegments/view/558052b8-fcf0-4e3b-a6b4-05990a008f2c/image.png) -->

# Install

##  Requirements

* Requires Python 3
* Requires a working installation of SVMLight
    * The easieast way to install it is to download compiled binaries from the [official website.](http://disi.unitn.it/moschitti/TK1.2-software/download.html)
      * You will have to fill up a form to get this, and make the build using the given Makefile.
      * If you are **ABSOLUTELY SURE** that you do not need to use Tree Kernels, you can also get precompiled binaries from the following links
        * Linux (32-bit): http://download.joachims.org/svm_light/current/svm_light_linux32.tar.gz
        * Linux (64-bit): http://download.joachims.org/svm_light/current/svm_light_linux64.tar.gz
        * Windows (32-bit): http://download.joachims.org/svm_light/current/svm_light_windows32.zip
        * Windows (64-bit): http://download.joachims.org/svm_light/current/svm_light_windows64.zip
        * Mac OS X (old): http://download.joachims.org/svm_light/current/svm_light_osx.tar.gz
        * Mac OS X (new): http://download.joachims.org/svm_light/current/svm_light_osx.8.4_i7.tar.gz
      * Place the binaries in `resources/svmlight/`

## Install Code

* Installation of _nalaf_

    git clone https://github.com/carstenuhlig/thesis-alex-carsten.git
    cd thesis-alex-carsten
    python3 setup.py install
    python3 -m nala.download_corpora

* Installation of _relna_

    git clone https://github.com/ashishbaghudana/relna.git
    cd relna
    python3 setup.py
    python3 -m relna.download_corpora

Eventually, when the package is registered on PyPi, you can simply install _relna_ by:

    pip3 install relna

# Examples
Run:
* `relna.py` for a simple example how to use _relna_ just for prediction with a pre-trained model
    * `python3 relna.py -c [PATH SVMLight BIN DIR] -p 15878741 12625412`
    * `python3 relna.py -c [PATH SVMLight BIN DIR] -s "Ubc9 interacts with androgen receptor (AR)."`
    * `python3 relna.py -c [PATH SVMLight BIN DIR] -d example.txt`
