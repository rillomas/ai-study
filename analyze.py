#!/usr/bin/env python
import click
import pdb
from sklearn.feature_extraction.text import CountVectorizer

def tokenizer(text):
    # We assume an input in which the morphenes are divided in comma
    l = text.split(',')
    return l
    

@click.command()
@click.option('--input','-i',default=None)
def analyze(input):
    corpus = open(input).read().split("\n")
    cv = CountVectorizer(tokenizer=tokenizer)
    x = cv.fit_transform(corpus)
    fn = cv.get_feature_names()
    pdb.set_trace()
    print(fn)

if __name__ == "__main__":
    analyze()
