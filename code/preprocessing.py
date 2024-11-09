import os
import re
import pandas as pd
from langdetect import detect
import nltk
nltk.download('punkt_tab')

parent = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))

##### elevator pitches
ep_path = os.path.join(parent, 'text-gen-llm', 'data', 'ep', 'ep_text.txt')
ep_out = os.path.join(parent, 'text-gen-llm', 'data', 'ep', 'ep_text_cleaned.txt')
ep_file = open(ep_path, 'r')
if os.path.isfile(ep_out): open(ep_out, 'w').close()
to_write = open(ep_out, 'w')
lines = ep_file.readlines()
for line in lines:
    sents = nltk.sent_tokenize(line)
    for sent in sents:
        to_write.write(sent + '\n')
to_write.close()
ep_file.close
