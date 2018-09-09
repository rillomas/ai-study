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
    ignored_words = [
        '。',
        '、',
        '，',
    ]
    cv = CountVectorizer(tokenizer=tokenizer, stop_words=ignored_words)
    x = cv.fit_transform(corpus)
    fn = cv.get_feature_names()
    #pdb.set_trace()
    print(fn)
    digit_count = 0
    for w in fn:
        if w.isdigit():
            # all letters are number numbers
            digit_count += 1
    print("全体: {}".format(len(fn)))
    print("数字: {}".format(digit_count))

if __name__ == "__main__":
    analyze()
