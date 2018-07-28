#!/usr/bin/env python
from lxml import etree as et
import click

@click.command()
@click.option('--path','-p',default=None)
def parse_xml(path):
    print("Parsing xml {}".format(path))
    tree = et.parse(path)
    sentences = tree.findall(".//Sentence")
    print(len(sentences))
    for s in sentences:
        print(s.text)

if __name__ == "__main__":
    parse_xml()

