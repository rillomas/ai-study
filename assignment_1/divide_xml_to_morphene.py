#!/usr/bin/env python
from lxml import etree as et
import click
from subprocess import Popen, PIPE

@click.command()
@click.option('--path','-p',default=None)
def parse_xml(path):
    #print("Parsing xml {}".format(path))
    # merge all sentences into one string and parse it
    tree = et.parse(path)
    sentences = tree.findall(".//Sentence")
    lines = [l.text for l in sentences if l.text]
    #print(lines)
    input = "".join(lines)
    cmd = ["java", '-jar', 'bin/sudachi-0.1.1-SNAPSHOT.jar', '-r', 'sudachi.json', '-m', 'C']
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, universal_newlines=True)
    out, err = p.communicate(input)
    lines = out.splitlines()
    # extract only the lemmatized words and seperate it by comma
    words = []
    for l in lines:
        if l == "EOS":
            break
        outs = l.split('\t')
        words.append(outs[-1]) # lemmatized word comes last
    print(",".join(words))

if __name__ == "__main__":
    parse_xml()

