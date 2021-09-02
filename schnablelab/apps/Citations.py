# -*- coding: UTF-8 -*-

"""
estimate dependent citations
"""

import os.path as op
import sys
from schnablelab.apps.base import ActionDispatcher, OptionParser
import pandas as pd

def main():
    actions = (
        ('DependentCitations', 'calculate dependent citations'),
        ('DownloadCitations', 'download all related citations from google scholar searching results'),
            )
    p = ActionDispatcher(actions)
    p.dispatch(globals())

def DependentCitations(args):
    """
    %prog AuthorsFile(copyied from GS profile) AllBibTexFile(including all citations in bibtex format)
    estimate how many dependent citations
    """
    p = OptionParser(DependentCitations.__doc__)
    opts, args = p.parse_args(args)

    if len(args) == 0:
        sys.exit(not p.print_help())
    AuthorsFile, AllBibTexFile, = args

    f1 = open(AllBibTexFile)
    titles, authors, papernames = [], [], []
    for i in f1:
        if '@' in i:
            tit = i.split('{')[-1].split(',')[0]
            titles.append(tit)
        elif '  author={' in i:
            aut = i.split('author={')[-1].split('}')[0]
            authors.append(aut)
        elif '  title={' in i:
            papna = i.split('title={')[-1].split('}')[0]
            papernames.append(papna)
    df = pd.DataFrame(dict(zip(['title', 'authors', 'paper_name'], [titles, authors, papernames]))) 
    print('total %s citations in bibtex file'%df.shape[0])
    
    dependentPapers = []
    f2 = open(AuthorsFile)
    target_authors = f2.readline().split(', ')
    for i in target_authors:
        tar_a = set(i.split())
        for t,j in zip(df['title'], df['authors']):
            for k in j.split(' and '):
                obj_a = set(k.replace(',', ' ').split())
                if tar_a == obj_a:
                    dependentPapers.append(t)
    depPs = set(dependentPapers)
    print('%s dependent citations'%len(depPs))
    for i in depPs:
        print(i)

def DownloadCitations(args):
    from urllib import FancyURLopener
    class MyOpener(FancyURLopener):
        version = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.152 Safari/537.36'
    openurl = MyOpener().open
    openurl(url).read()
    from bs4 import SoupStrainer, BeautifulSoup
    page = BeautifulSoup(openurl(url).read(), parse_only=SoupStrainer('div', id='gs_ab_md'))
    


if __name__ == '__main__':
    main()
