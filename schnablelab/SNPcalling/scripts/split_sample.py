import gzip
from mimetypes import guess_type
from Bio import SeqIO
from functools import partial
import itertools
import re
import sys
import pandas as pd

left_barcode = pd.read_csv('/work/schnablelab/cmiao/TimeSeriesGWAS/Genotype_GBS/GBS_Raw_Data/sample_barcode_left.csv')
right_barcode = pd.read_csv('/work/schnablelab/cmiao/TimeSeriesGWAS/Genotype_GBS/GBS_Raw_Data/sample_barcode_right.csv')

Bsp = re.compile('^[A-Z]{6,9}[A,G,T]GC[A,C,T]C')

def write2f(sm, ftype, sfx, seq_header, seq, qual_header, qual):
    '''
    ftype: R1, R2, Unpaired
    '''
    with open('%s_%s_%s.fastq'%(sm, ftype, sfx), 'a') as f:
        f.write(seq_header)
        f.write(seq)
        f.write(qual_header)
        f.write(qual)

def splitsm(fn_r1, fn_r2, library):
    sufix = fn_r1.split('_')[-1].split('.')[0]
    df_bar1 = left_barcode[left_barcode['library']==library].set_index('barcode')
    df_bar2 = right_barcode[right_barcode['library']==library].set_index('barcode')
    f_r1 = open(fn_r1, 'r')
    f_r2 = open(fn_r2, 'r')
    n = 1
    for (seq_header_r1, seq_r1, qual_header_r1, qual_r1), (seq_header_r2, seq_r2, qual_header_r2, qual_r2) in zip(itertools.zip_longest(*[f_r1] * 4), itertools.zip_longest(*[f_r2] * 4)):
        print(n)
        n += 1
        sea_r1 = re.search(Bsp, seq_r1)
        sea_r2 = re.search(Bsp, seq_r2)
        if sea_r1 and sea_r2: # both have search results
            ed_r1 = sea_r1.span()[1]-5
            ed_r2 = sea_r2.span()[1]-5
            bar1 = sea_r1[0][:ed_r1]
            bar2 = sea_r2[0][:ed_r2]
            try:
                sm_r1 = df_bar1.loc[bar1, 'sample']
            except KeyError:
                sm_r1 = 'N1'
            try:
                sm_r2 = df_bar2.loc[bar2, 'sample']
            except KeyError:
                sm_r2 = 'N2'
            if sm_r1 == sm_r2:
                write2f(sm_r1, 'R1', sufix, seq_header_r1, seq_r1[ed_r1:], qual_header_r1, qual_r1[ed_r1:])
                write2f(sm_r2, 'R2', sufix, seq_header_r2, seq_r2[ed_r2:], qual_header_r2, qual_r2[ed_r2:])
            else:
                if sm_r1=='N1' and sm_r2!='N2':
                    write2f(sm_r2, 'R1', sufix, seq_header_r1, seq_r1[ed_r1:], qual_header_r1, qual_r1[ed_r1:])
                    write2f(sm_r2, 'R2', sufix, seq_header_r2, seq_r2[ed_r2:], qual_header_r2, qual_r2[ed_r2:])
                elif sm_r1 !='N1' and sm_r2=='N2':
                    write2f(sm_r1, 'R1', sufix, seq_header_r1, seq_r1[ed_r1:], qual_header_r1, qual_r1[ed_r1:])
                    write2f(sm_r1, 'R2', sufix, seq_header_r2, seq_r2[ed_r2:], qual_header_r2, qual_r2[ed_r2:])
                else:
                    continue

        elif sea_r1 and (not sea_r2): # sea_r1 is good
            ed_r1 = sea_r1.span()[1]-5
            bar1 = sea_r1[0][:ed_r1]
            try:
                sm_r1 = df_bar1.loc[bar1, 'sample']
                write2f(sm_r1, 'Unpaired', sufix, seq_header_r1, seq_r1[ed_r1:], qual_header_r1, qual_r1[ed_r1:])
            except KeyError:
                pass

        elif (not sea_r1) and sea_r2: # sea_r2 is good
            ed_r2 = sea_r2.span()[1]-5
            bar2 = sea_r2[0][:ed_r2]
            try:
                sm_r2 = df_bar2.loc[bar2, 'sample']
                write2f(sm_r2, 'Unpaired', sufix, seq_header_r2, seq_r2[ed_r2:], qual_header_r2, qual_r2[ed_r2:])
            except KeyError:
                pass

        else: # both sea_r1 and sea_r2 are bad
            continue

if __name__ == "__main__":
    if len(sys.argv)==4:
        splitsm(*sys.argv[1:])
    else:
        print('python split_sample.py fq_r1 fq_r2 bgpR1-8')

