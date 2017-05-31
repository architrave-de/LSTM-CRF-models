#!/usr/bin/env bash

# install python dependencies
pip install Theano nltk tqdm sklearn Gensim
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

# download nltk sentence splitter
python -c "import nltk; nltk.dowload('punkt')"

# download data dependencies
wget --no-parent -r http://public.architrave.de/ExtractionData/data/
mv public.architrave.de/ExtractionData/data data
rm -rf public.architrave.de


