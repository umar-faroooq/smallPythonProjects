#!/usr/bin/env python
# coding: utf-8

#kanika.miglani

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import string
import fitz
import bs4 as bs
import urllib.request
import re

import warnings 
warnings.filterwarnings('ignore')

__name__ == "__main__"

#blockwise data extraction
def main(filename_given, filename_out):
    doc = fitz.open(filename_given)
    filename = filename_given
    ofile = filename_out + ".txt"
    doc = fitz.open(filename)
    fout = open(ofile, "wb")

    for page in doc:
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))
        for b in blocks:
            fout.write(b[4].encode("utf-8"))

    fout.close()

need = tuple('.?!:–()|')


def article(file_use):
    parsed_article = bs.BeautifulSoup(open(file_use).read())
    
def clean_data(file_use):
    text_data = []
    for i, line in enumerate(open(file_use)):
        line = re.sub("<(.*)>", "", line)
        line = re.sub("^(\d\.\d\d\.)", "", line)
        line = re.sub("(\d\.\d\.)", "", line)
        line = re.sub("^(\d\d\.)", "", line)
        line = re.sub("^(\d\.)", "", line)
        line = re.sub("^(\d\d(\r\n|\r|\n))", "", line)
        line = re.sub("^(\d(\r\n|\r|\n))", "", line)
        line = re.sub("^(\d\s+)", "", line)
        line = re.sub("^(\d\d\s+)", "", line)
        line = re.sub("\u2022", "", line)
        line = re.sub("(\.\d\d)", ".", line)
        line = re.sub(r'http\S+', '', line)
        line = re.sub("(economy-ni.gov.uk)", '', line)
        line = line.strip()
        text_data.append(line)
    return text_data

def del_empty(text_data):
    while("" in text_data):
        text_data.remove("")
    return text_data

def del_duplicates(text_data):
    res = []
    [res.append(x) for x in text_data if x not in res]
    return res

def sentence_groups(l, punctuation=need):
    group = []
    for w in l:
        group.append(w)
        if w.endswith(punctuation):
            yield group
            group = []
    if group:
        yield group
        
def join_text(res):
    text = [' '.join(group) for group in sentence_groups(res)]
    return text

def join_sent(text):
    formatted_text = (" ".join(text))
    return formatted_text


def complete(file_use, export_sent = False, export_text = False, export_name = 'extracted_text'):
    parsed_article = article(file_use)
    text_data = clean_data(file_use)
    text_data = del_empty(text_data)
    res = del_duplicates(text_data)
    text = join_text(res)
    formatted_text = join_sent(text)
    
    extracted_table = pd.DataFrame()
    extracted_table['text_sentences'] = text
    
    if export_sent:
        extracted_table.to_excel(export_name + '.xlsx', index = False)
        
    if export_text:
        with open(export_name + '.txt', 'a') as the_file:
            the_file.write(formatted_text)
    
    return text, formatted_text
