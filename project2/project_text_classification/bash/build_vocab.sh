#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat ./data/pos_train.txt ./data/neg_train.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab.txt
