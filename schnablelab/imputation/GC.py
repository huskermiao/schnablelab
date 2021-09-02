#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Correct Genotype Calls Using Sliding Window Method in biparental populations
"""
import re
import sys
import logging
import pandas as pd
import numpy as np
from scipy.stats import binom, chisquare
from pathlib import Path
from configparser import ConfigParser
from collections import defaultdict, Counter

from schnablelab import __version__ as version
from schnablelab.apps.Tools import eprint
from schnablelab.apps.base import (
    OptionParser, 
    OptionGroup, 
    ActionDispatcher, 
    SUPPRESS_HELP,
)
from schnablelab.imputation.base import (
    getChunk, 
    random_alternative, 
    get_blocks, 
    sort_merge_sort, 
    bin_markers,
)

homo_pattern1 = re.compile('9*09*09*|9*29*29*')
hete_pattern = re.compile('1|9*09*29*|9*29*09*')
homo_pattern2 = re.compile('9*29*29*2*9*2*9*2*9*2*9*2*9*|9*09*09*0*9*0*9*0*9*0*9*0*9*')

class ParseConfig(object):
    """
    Parse the configure file using configparser
    """
    def __init__(self, configfile):
        config = ConfigParser()
        assert config.read(configfile), 'config file does not exist!'
        self.po_type = config.get('Section1', 'Population_type')
        self.gt_a = config.get('Section2', 'Letter_for_homo1')
        self.gt_h = config.get('Section2', 'Letter_for_hete')
        self.gt_b = config.get('Section2', 'Letter_for_homo2')
        self.gt_miss = config.get('Section2', 'Letter_for_missing_data')
        self.error_a = config.getfloat('Section3', 'error_rate_for_homo1')
        self.error_b = config.getfloat('Section3', 'error_rate_for_homo2')
        self.error_h = abs(self.error_a - self.error_b)
        self.win_size = config.getint('Section4', 'Sliding_window_size')

def het2hom(seqnum):
    """
    temporarily convert heterozygous to homozygous as random as possible
    """
    seqnum_tmp = seqnum.copy()
    grouper = (seqnum_tmp.diff(1)!=0).astype('int').cumsum()
    for __, grp in seqnum_tmp.groupby(grouper):
        geno, lens = grp.unique()[0], grp.shape[0]
        if geno==1:
            if lens > 1:
                homos = random_alternative(lens)
                seqnum_tmp[grp.index] = homos
    if seqnum_tmp[0] == 1:
        seqnum_tmp[0] = np.random.choice([0,2])
    
    het_idx = seqnum_tmp[seqnum_tmp==1].index
    nearby_homo = seqnum_tmp[het_idx-1]
    rep_dict = {0:2, 2:0, 9:np.random.choice([0,2])}
    replace_genos = nearby_homo.apply(lambda x: rep_dict[x])
    seqnum_tmp[het_idx] = replace_genos
    return seqnum_tmp

def get_mid_geno(np_array, cargs_obj):
    """
    return the genotype with highest probability in the central.
    """
    a_count, b_count, miss_count = count_genos(np_array)
    ab_count = a_count + b_count
    if ab_count > cargs_obj.win_size//2:
        a_ex_prob = binom.pmf(b_count, ab_count, cargs_obj.error_a)
        h_ex_prob = binom.pmf(b_count, ab_count, 0.5+cargs_obj.error_h/2)
        b_ex_prob = binom.pmf(b_count, ab_count, 1-cargs_obj.error_b)
        d = {key: value for (key, value) in zip([0, 1, 2], [a_ex_prob, h_ex_prob, b_ex_prob])}
        return max(d, key=d.get)
    else:
        return np.nan

def count_genos(np_array):
    """
    count genotypes in a given seq
    """
    counts = Counter(np_array)
    a_count, b_count, miss_count = counts[0], counts[2], counts[9]
    return a_count, b_count, miss_count

def get_score(np_array, cargs_obj):
    """
    calculate the score for each sliding window in the seq_num_no1
    """
    a_count, b_count, __ = count_genos(np_array)
    if a_count+b_count > cargs_obj.win_size//2:
        return a_count/float(b_count) if b_count != 0 else a_count/(b_count+0.1)
    else:
        return np.nan

def judge_h_island(h_scores):
    """
    judge if the h island in the corrected seqs is real or fake.
    """
    length = h_scores.shape[0]
    if length >=3:
        trends = h_scores.diff().iloc[1:]
        ups, downs = (trends>0).sum(), (trends<0).sum() 
        if ups == 0 or downs == 0:
            return False # the score curve monotonically increases or decreases so it is the fake h island
        else:
            return True # real h island
    else: 
        return True # real h island

def callback(h_scores):
    """
    call back the h island to the origianl genotypes based on the socre
    """
    realgenos = []
    for val in h_scores:
        if val > 1: realgenos.append(0)
        elif val < 1: realgenos.append(2)
        else: realgenos.append(np.nan)
    return realgenos
    
def fix_sliding_case1(seqnum, initial_corrected_seq, h_island_idx):
    """
    original seq: hhhh,aaaa
    after initial correction: hhh,aaaaa
    """
    st = h_island_idx[-1]+1
    ed = st+6
    if ed <= seqnum.index[-1]:
        indent_genos = ''.join(map(str, seqnum.loc[st: ed].values))
        #print('case1 indent_geno: {}'.format(indent_genos))
        result = homo_pattern1.search(indent_genos)
        try:
            i = result.start()
            if i > 0:
                initial_corrected_seq.loc[st:st+i-1] = 1
                #print(pd.concat([seqnum, initial_corrected_seq], axis =1))
        except AttributeError:
            pass

def fix_sliding_case2(seqnum, initial_corrected_seq, h_island_idx):
    """
    original seq: hhhh,aaaa
    after initial correction: hhhhh,aaa
    """
    ed = h_island_idx[-1]
    st = ed - 6
    if st >= seqnum.index[0]:
        indent_genos = ''.join(map(str, seqnum.loc[st: ed].values))
        #print('case2 indent_geno: {}'.format(indent_genos))
        result = homo_pattern1.search(indent_genos)
        try:
            i = result.start()
            if i > 0:
                if hete_pattern.search(indent_genos, i) is None:
                    geno = 0 if '0' in result.group() else 2
                    initial_corrected_seq.loc[st+i:ed] = geno
                    #print(pd.concat([seqnum, initial_corrected_seq], axis =1))
        except AttributeError:
            pass
    
def fix_sliding_case3(seqnum, initial_corrected_seq, h_island_idx):
    """
    original seq: aaaa,hhhh
    after initial correction: aaa,hhhhh
    """
    st = h_island_idx[0]
    ed = st+6
    if ed <= h_island_idx[-1]:
        indent_genos = ''.join(map(str, seqnum.loc[st: ed].values))
        #print('case3 indent_geno: {}'.format(indent_genos))
        result = homo_pattern2.match(indent_genos)
        try:
            i = result.end()
            if i > 0:
                geno = 0 if '0' in result.group() else 2
                initial_corrected_seq.loc[st:st+i-1] = geno
                #print(pd.concat([seqnum, initial_corrected_seq], axis =1))
        except AttributeError:
            pass

def fix_sliding_case4(seqnum, initial_corrected_seq, h_island_idx):
    """
    original seq: aaaa,hhhh
    after initial correction: aaaaa,hhh
    """
    ed = h_island_idx[0]-1
    st = ed - 6
    if st >= seqnum.index[0]:
        indent_genos = ''.join(map(str, seqnum.loc[st: ed].values))
        #print('case4 indent_geno: {}'.format(indent_genos))
        result = homo_pattern2.search(indent_genos)
        try:
            i = result.end()
            if i < 7:
                initial_corrected_seq.loc[st+i:ed] = 1
                #print(pd.concat([seqnum, initial_corrected_seq], axis =1))
        except AttributeError:
            pass

def get_corrected_num(seqnum, corrected_seq):
    """
    count number of genotpes corrected
    """
    return (seqnum != corrected_seq).sum()

class CorrectOO(object):
    """
    This class contains the routine to correct the original seq per sample
    """
    def __init__(self, config_params, orig_seq_without_idx_num):
        self.cargs = config_params
        self.seq_num = orig_seq_without_idx_num
        self.seq_num_no1 = het2hom(self.seq_num)
        self.rolling_geno = self.seq_num_no1.rolling(self.cargs.win_size, center=True).apply(get_mid_geno, raw=True, args=(self.cargs,))
        self.rolling_score = self.seq_num_no1.rolling(self.cargs.win_size, center=True).apply(get_score, raw=True, args=(self.cargs,))

        # debug the h island
        grouper = (self.rolling_geno.diff(1)!=0).astype('int').cumsum()
        for __, grp in self.rolling_geno.groupby(grouper):
            geno = grp.unique()[0]
            if geno==1:
                h_score_island = self.rolling_score[grp.index]
                if judge_h_island(h_score_island): # adjust the slding problem of the real h island.
                    h_island_len = grp.shape[0]
                    fix_sliding_case1(self.seq_num, self.rolling_geno, grp.index)
                    fix_sliding_case2(self.seq_num, self.rolling_geno, grp.index)
                    fix_sliding_case3(self.seq_num, self.rolling_geno, grp.index)
                    fix_sliding_case4(self.seq_num, self.rolling_geno, grp.index)
            
                else: # call back the fake h island to the origianl genotypes
                    real_genos = callback(h_score_island)
                    self.rolling_geno[grp.index] = real_genos

        # substitute NaNs with original genotypes
        self.corrected = self.rolling_geno.where(~self.rolling_geno.isna(), other=self.seq_num).astype('int')

def correct(args):
    """
    %prog correct config.txt input.matrix 

    Correct wrong genotype calls and impute missing values in biparental populations
    """
    p = OptionParser(correct.__doc__)
    p.add_option("-c", "--configfile", help=SUPPRESS_HELP)
    p.add_option("-m", "--matrixfile", help=SUPPRESS_HELP)
    p.add_option('--itertimes', default=7, type='int', 
                help='maximum correction times to reach the stablized status')
    q = OptionGroup(p, "output options")
    p.add_option_group(q)
    q.add_option('--opp', default="'infer'",
                help='specify the prefix of the output file names')
    q.add_option("--logfile", default='GC.correct.log',
                help="specify the file saving running info")
    q.add_option('--debug', default=False, action="store_true",
                help='trun on the debug mode that will generate a tmp file containing both original and corrected genotypes for debug use')

    opts, args = p.parse_args(args)

    if len(args) != 2:
        sys.exit(not p.print_help())

    configfile, mapfile = args
    inputmatrix = opts.matrixfile or mapfile
    inputconfig = opts.configfile or configfile

    opf = inputmatrix.rsplit(".", 1)[0]+'.corrected.map' if opts.opp=="'infer'" else '{}.map'.format(opts.opp) # output file
    if Path(opf).exists():
        eprint("ERROR: Filename collision. The future output file `{}` exists".format(opf))
        sys.exit(1)

    logging.basicConfig(filename=opts.logfile, 
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(message)s")

    cargs = ParseConfig(inputconfig)
    if cargs.win_size % 2 == 0:
        eprint("ERROR: The slding window value cannot be even")
        sys.exit(1)
    logging.debug("Parameters in config file: {0}".format(cargs.__dict__))

    chr_order, chr_nums = getChunk(inputmatrix)
    map_reader = pd.read_csv(inputmatrix, delim_whitespace=True, index_col=[0, 1],  iterator=True)
    tmp_chr_list = []
    for chrom in chr_order:
        logging.debug('{}...'.format(chrom))
        print('{}...'.format(chrom))
        chunk = chr_nums[chrom]
        df_chr_tmp = map_reader.get_chunk(chunk)
        marker_num, sample_num = df_chr_tmp.shape
        logging.debug('{} contains {} markers and {} samples.'.format(chrom, marker_num, sample_num))
        tmp_sm_list = []
        for sm in df_chr_tmp:
            logging.debug('Start correcting {}...'.format(sm))
            orig_seq = df_chr_tmp[sm]
            orig_idx = orig_seq.index
            seq_no_idx = orig_seq.reset_index(drop=True)
            seq_no_idx_num = seq_no_idx.replace([cargs.gt_a, cargs.gt_b, cargs.gt_h, cargs.gt_miss], [0, 2, 1, 9])
            if seq_no_idx_num.shape[0] <= cargs.win_size:
                logging.debug('number of markers smaller than the window size, omit...')
                final_seq_no_idx = seq_no_idx
            else:
                logging.debug('correction round 1...')
                correct_obj = CorrectOO(cargs, seq_no_idx_num)
                corrected_n = get_corrected_num(seq_no_idx_num, correct_obj.corrected)
                round_n = 2
                while round_n <= opts.itertimes:
                    logging.debug('correction round %s...'%round_n)
                    corrected_obj = CorrectOO(cargs, correct_obj.corrected)
                    corrected_n_new = get_corrected_num(seq_no_idx_num, corrected_obj.corrected)
                    round_n += 1
                    if (corrected_n_new - corrected_n)/float(corrected_n+0.01) <= 0.01:
                        break
                    else:
                        corrected_n = corrected_n_new
                final_seq_no_idx = corrected_obj.corrected.replace([0, 2, 1, 9], [cargs.gt_a, cargs.gt_b, cargs.gt_h, cargs.gt_miss])
            final_seq_no_idx.index = orig_idx
            final_seq = final_seq_no_idx
            tmp_sm_list.append(final_seq)
        df_sm_tmp = pd.concat(tmp_sm_list, axis=1)
        tmp_chr_list.append(df_sm_tmp)
    df_corrected = pd.concat(tmp_chr_list)
    
    df_corrected.to_csv(opf, sep='\t', index=True)

    if opts.debug:
        logging.debug('generating the tmp file for debug use...')
        df_uncorrected = pd.read_csv(inputmatrix, delim_whitespace=True, index_col=[0, 1])
        df_debug = df_corrected.where(df_corrected==df_uncorrected, other=df_corrected+'('+df_uncorrected+')')
        df_debug.to_csv(opf+'.debug', sep='\t', index=True)
    print('Done!')

def qc_missing(args):
    """
    %prog filtermissing input.matrix output.matrix

    run quality control of the missing genotypes in the input.matrix before starting the correction.
    """
    p = OptionParser(qc_missing.__doc__)
    p.add_option("-i", "--input", help=SUPPRESS_HELP)
    p.add_option("-o", "--output", help=SUPPRESS_HELP)
    p.add_option('--cutoff_snp', default=0.5, type='float',
                help = "SNP with missing rate higher than this value will be removed")
    p.add_option('--rm_bad_samples', default=False, action="store_true",
                help='remove bad samples after controlling the SNPs with high missing rate')
    p.add_option('--cutoff_sample', type='float',
                help = "sample missing rate higher than this value will be removed after controlling the SNP missing rate")
    q = OptionGroup(p, "format options")
    p.add_option_group(q)
    q.add_option('--homo1', default="A",
                help='character for homozygous genotype')
    q.add_option("--homo2", default='B',
                help="character for alternative homozygous genotype")
    q.add_option('--hete', default='X',
                help='character for heterozygous genotype')
    q.add_option('--missing', default='-',
                help='character for missing value')
    opts, args = p.parse_args(args)

    if len(args) != 2:
        sys.exit(not p.print_help())

    if opts.rm_bad_samples and not opts.cutoff_sample:
        eprint('missing value cutoff for --cutoff_sample option must be specified when --rm_bad_samples added.')
        sys.exit(1)

    inmap, outmap = args
    inputmatrix = opts.input or inmap
    outputmatrix = opts.output or outmap

    chr_order, chr_nums = getChunk(inputmatrix)
    map_reader = pd.read_csv(inputmatrix, delim_whitespace=True, index_col=[0, 1],  iterator=True)
    Good_SNPs = []
    for chrom in chr_order:
        print('{}...'.format(chrom))
        chunk = chr_nums[chrom]
        df_chr_tmp = map_reader.get_chunk(chunk)
        df_chr_tmp_num = df_chr_tmp.replace([opts.homo1, opts.homo2, opts.hete, opts.missing], [0, 2, 1, 9])
        sample_num = df_chr_tmp_num.shape[1]
        good_rates = df_chr_tmp_num.apply(lambda x: (x==9).sum()/sample_num, axis=1) <= opts.cutoff_snp
        good_snp = df_chr_tmp.loc[good_rates, :]
        Good_SNPs.append(good_snp)
    df1 = pd.concat(Good_SNPs)
    before_snp_num = sum(chr_nums.values())
    after_snp_num, before_sm_num = df1.shape
    pct = after_snp_num/float(before_snp_num)*100
    print('{} SNP markers before quality control.'.format(before_snp_num))
    print('{}({:.1f}%) markers left after the quality control.'.format(after_snp_num, pct))

    if opts.rm_bad_samples:
        print('start quality control on samples')
        good_samples = df1.apply(lambda x: (x==opts.missing).sum()/after_snp_num, axis=0) <= opts.cutoff_sample
        df2 = df1.loc[:,good_samples]
        after_sm_num = df2.shape[1]
        pct_sm = after_sm_num/float(before_sm_num)*100
        print('{} samples before quality control.'.format(before_sm_num))
        print('{}({:.1f}%) markers left after the quality control.'.format(after_sm_num, pct_sm))
        df2.to_csv(outputmatrix, sep='\t', index=True)
    else:
        df1.to_csv(outputmatrix, sep='\t', index=True)

def qc_sd(args):
    """
    %prog sdtest input.matrix output.matrix

    run quality control on segregation distortions in each SNP.
    """
    p = OptionParser(qc_sd.__doc__)
    p.add_option("-i", "--input", help=SUPPRESS_HELP)
    p.add_option("-o", "--output", help=SUPPRESS_HELP)
    p.add_option('--population', default='RIL', choices=('RIL', 'F2', 'BCFn'),
                help = "population type")
    p.add_option('--sig_cutoff', default = 1e-2, type='float',
                help = "set the chi square test cutoff. 0(less strigent) to 1(more strigent)")
    q = OptionGroup(p, "format options")
    p.add_option_group(q)
    q.add_option('--homo1', default="A",
                help='character for homozygous genotype')
    q.add_option("--homo2", default='B',
                help="character for alternative homozygous genotype")
    q.add_option('--hete', default='X',
                help='character for heterozygous genotype')
    q.add_option('--missing', default='-',
                help='character for missing value')
    opts, args = p.parse_args(args)

    if len(args) != 2:
        sys.exit(not p.print_help())

    inmap, outmap = args
    inputmatrix = opts.input or inmap
    outputmatrix = opts.output or outmap

    if opts.sig_cutoff >=1 or opts.sig_cutoff <= 0:
        eprint('the cutoff chi square test should be smaller than 1 and larger than 0')
        sys.exit(1)

    chr_order, chr_nums = getChunk(inputmatrix)
    map_reader = pd.read_csv(inputmatrix, delim_whitespace=True, index_col=[0, 1], iterator=True)
    Good_SNPs = []
    for chrom in chr_order:
        print('{}...'.format(chrom))
        chunk = chr_nums[chrom]
        df_chr_tmp = map_reader.get_chunk(chunk)
        df_chr_tmp_num = df_chr_tmp.replace([opts.homo1, opts.homo2, opts.hete, opts.missing], [0, 2, 1, 9])
        ob0, ob2 = (df_chr_tmp_num==0).sum(axis=1), (df_chr_tmp_num==2).sum(axis=1)
        obsum = ob0 + ob2
        exp0, exp2 = (obsum*0.75, obsum*0.25) if opts.population == 'BCFn' else (obsum*0.5, obsum*0.5)
        df_chi = pd.DataFrame(dict(zip(['ob0', 'ob2', 'exp0', 'exp2'], [ob0, ob2, exp0, exp2])))
        min_cond = ((df_chi['ob0']>5) & (df_chi['ob2']>5)).values
        pval_cond = chisquare([df_chi['ob0'], df_chi['ob2']], [df_chi['exp0'], df_chi['exp2']]).pvalue >= opts.sig_cutoff
        good_snp = df_chr_tmp.loc[(min_cond & pval_cond), :]
        Good_SNPs.append(good_snp)
    df1 = pd.concat(Good_SNPs)
    before_snp_num = sum(chr_nums.values())
    after_snp_num = df1.shape[0]
    pct = after_snp_num/float(before_snp_num)*100
    print('{} SNP markers before quality control.'.format(before_snp_num))
    print('{}({:.1f}%) markers left after the quality control.'.format(after_snp_num, pct))
    df1.to_csv(outputmatrix, sep='\t', index=True)

def qc_hetero(args):
    """
    %prog qc_hetero input.matrix output.matrix

    run quality control on the continuous same homozygous in heterozygous region.
    """
    p = OptionParser(qc_hetero.__doc__)
    p.add_option("-i", "--input", help=SUPPRESS_HELP)
    p.add_option("-o", "--output", help=SUPPRESS_HELP)
    p.add_option("--read_len", default=100, type='int',
                help="read length for SNP calling")
    p.add_option("--logfile", default='GC.qc_hetero.info',
                help="specify the file saving binning info")
    q = OptionGroup(p, "format options")
    p.add_option_group(q)
    q.add_option('--homo1', default="A",
                help='character for homozygous genotype')
    q.add_option("--homo2", default='B',
                help="character for alternative homozygous genotype")
    q.add_option('--hete', default='X',
                help='character for heterozygous genotype')
    q.add_option('--missing', default='-',
                help='character for missing value')
    r = OptionGroup(p, 'advanced options')
    p.add_option_group(r)
    r.add_option('--nonhetero_lens', default=8, type='int',
                help='number of non heterozygous between two heterozygous in a heterozygous region')
    r.add_option('--min_homo', default=2, type='int',
                help='number of continuous homozygous within the read length in the heterozygous region')    
    opts, args = p.parse_args(args)

    if len(args) != 2:
        sys.exit(not p.print_help())

    inmap, outmap = args
    inputmatrix = opts.input or inmap
    outputmatrix = opts.output or outmap

    logging.basicConfig(filename=opts.logfile, 
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(message)s")

    chr_order, chr_nums = getChunk(inputmatrix)
    map_reader = pd.read_csv(inputmatrix, delim_whitespace=True, index_col=[0, 1], iterator=True)
    Good_SNPs = []
    for chrom in chr_order:
        print('{}...'.format(chrom))
        logging.debug(chrom)
        chunk = chr_nums[chrom]
        df_chr_tmp = map_reader.get_chunk(chunk)
        chr_idx = df_chr_tmp.index
        df_chr_tmp_num = df_chr_tmp.replace([opts.homo1, opts.homo2, opts.hete, opts.missing], [0, 2, 1, 9]).loc[chrom]
        
        chr_bin_ids = []
        for sm in df_chr_tmp_num:
            geno_grouper = (df_chr_tmp_num[sm].diff(1)!=0).astype('int').cumsum()
            idx, geno, lens = [], [], []
            for __, grp_geno in df_chr_tmp_num[sm].groupby(geno_grouper):
                idx.append(grp_geno.index)
                geno.append(grp_geno.unique()[0])
                lens.append(grp_geno.shape[0])
            df_grp_geno = pd.DataFrame(dict(zip(['idx', 'geno', 'lens'], [idx, geno, lens])))
            df_grp_geno['type'] = df_grp_geno['geno'].apply(lambda x: 1 if x==1 else 0) # 1: hetero genotype 0: others(homo1, homo2, missing)
            type_grouper = (df_grp_geno['type'].diff(1)!=0).astype('int').cumsum()
            for __, grp_type in df_grp_geno.groupby(type_grouper):
                if grp_type['type'].unique()[0] == 0:
                    nonhetero_lens = grp_type['lens'].sum()
                    if nonhetero_lens <= opts.nonhetero_lens:
                        for __, row in grp_type.iterrows():
                            if row.geno ==0 or row.geno ==2:
                                bin_ids = get_blocks(row['idx'].values, dist=opts.read_len, block_size=opts.min_homo)
                                if bin_ids:
                                    for bin_index in bin_ids:
                                        if bin_index not in chr_bin_ids:
                                            chr_bin_ids.append(bin_index)
        if chr_bin_ids:
            dropping_ids = []
            merged_bin_ids = sort_merge_sort(chr_bin_ids)
            for idx_block in merged_bin_ids:
                logging.debug('positions: {}'.format(idx_block))
                genos_block = df_chr_tmp_num.loc[idx_block, :]
                missings = genos_block.apply(lambda x: (x==9).sum(), axis=1)
                heteros = genos_block.apply(lambda x: (x==1).sum(), axis=1)
                dropping_index = list(pd.concat([missings, heteros], axis=1).sort_values([0, 1]).index[1:])
                dropping_ids.extend(dropping_index)
            df_chr_tmp = df_chr_tmp.drop(dropping_ids, level=1)
        Good_SNPs.append(df_chr_tmp)
    df1 = pd.concat(Good_SNPs)
    before_snp_num = sum(chr_nums.values())
    after_snp_num = df1.shape[0]
    pct = after_snp_num/float(before_snp_num)*100
    print('{} SNP markers before quality control.'.format(before_snp_num))
    print('{}({:.1f}%) markers left after the quality control.'.format(after_snp_num, pct))
    df1.to_csv(outputmatrix, sep='\t', index=True)
    print('Done! Check {} for running details.'.format(opts.logfile))

def cleanup(args):
    """
    %prog cleanup tmp.matrix out.matrix

    remove redundant info for Debug in the temporary matrix file
    """
    p = OptionParser(cleanup.__doc__)
    p.add_option("-i", "--input", help=SUPPRESS_HELP)
    p.add_option("-o", "--output", help=SUPPRESS_HELP)
    opts, args = p.parse_args(args)
    if len(args) != 2:
        sys.exit(not p.print_help())

    inmap, outmap = args
    inputmatrix = opts.input or inmap
    outputmatrix = opts.output or outmap

    df = pd.read_csv(inputmatrix, delim_whitespace=True, index_col=[0,1])
    df.applymap(lambda x: x.split('(')[0]).to_csv(outputmatrix, sep='\t', index=True)
    print('Done!')

mst_header = """population_type {}
population_name {}
distance_function {}
cut_off_p_value {}
no_map_dist {}
no_map_size {}
missing_threshold {}
estimation_before_clustering {}
detect_bad_data {}
objective_function {}
number_of_loci {}
number_of_individual {}

"""
mst_homos, mst_miss, mst_hete = ('a', 'A', 'b', 'B'), ('-', 'U'), 'X'

def format(args):
    """
    %prog format corrected.matrix 

    convert corrected genotype matrix file to other formats(mstmap, joinmap, r/qtl) for the genetic mapping software.
    Example:
    `python -m schnablelab.imputation.GC format test.map --mstmap --mstmap_pop_type RIL2`
    will generate `test.mstmap` for MSTmap use.
    """
    p = OptionParser(format.__doc__)
    p.add_option("-i", "--input", help=SUPPRESS_HELP)
    p.add_option("--mstmap",  default=False, action="store_true",
                help = 'convert to MSTmap format')
    p.add_option("--rqtl",  default=False, action="store_true",
                help = 'convert to R/qtl format')
    p.add_option("--joinmap",  default=False, action="store_true",
                help = 'convert to JoinMap format')

    q = OptionGroup(p, "format options for input matrix file")
    p.add_option_group(q)
    q.add_option('--homo1', default="A", choices=('a', 'A'),
                help='character for homozygous genotype')
    q.add_option("--homo2", default='B', choices=('b', 'B'),
                help="character for alternative homozygous genotype")
    q.add_option('--hete', default='X', choices=('h', 'H', 'X'),
                help='character for heterozygous genotype')
    q.add_option('--missing', default='-', choices=('-', 'U'),
                help='character for missing value')
    
    r = OptionGroup(p, "parameters for MSTmap")
    p.add_option_group(r)
    r.add_option('--mstmap_pop_type',
                help='Possible values are DH and RILd, where d is any natural number. \
                For example, RIL6 means a RIL population at generation 6. \
                You should use RIL2 for F2. Use DH for BC1, DH and Hap.')
    r.add_option("--population_name", default='LinkageGroup',
                help="ives a name for the mapping population. It can be any string of letters (a-z, A-Z) or digits (0-9)")
    r.add_option('--distance_function', default='kosambi', choices=('kosambi', 'haldane'),
                help="choose Kosambi's and Haldane's distance functions")
    r.add_option('--cut_off_p_value', default=0.000001,
                help='specifies the threshold to be used for clustering the markers into LGs')
    r.add_option('--no_map_dist', default=15,
                help='check mstmap manual for details')
    r.add_option('--no_map_size', default=5,
                help='check mstmap manual for details')
    r.add_option('--missing_threshold', default=0.4,
                help='any marker with more than this value will be removed completely without being mapped')
    r.add_option('--estimation_before_clustering', default='no', choices=('yes', 'no'),
                help='if yes, MSTmap will try to estimate missing data before clustering the markers into linkage groups')
    r.add_option('--detect_bad_data', default='yes', choices=('yes', 'no'),
                help='if yes turn on the error detection feature in MSTmap')
    r.add_option('--objective_function', default='COUNT', choices=('COUNT', 'ML'),
                help='specifies the objective function')
    
    s = OptionGroup(p, "parameters for JoinMap and R/qtl")
    p.add_option_group(s)
    s.add_option('--pop_type', default='RIL', choices=('RIL', 'F2'),
                help='specify mapping population type. Contact me if you need supports for other population types')
    
    opts, args = p.parse_args(args)
    if len(args) != 1:
        sys.exit(not p.print_help())

    inmap, = args
    inputmatrix = opts.input or inmap

    if (not opts.rqtl) and (not opts.joinmap) and (not opts.mstmap):
        eprint("ERROR: add at least one output format option.")
        sys.exit(1)

    if opts.mstmap:
        if not opts.mstmap_pop_type:
            eprint("ERROR: please choose population type for mstmap format.")
            sys.exit(1)
        if not (opts.mstmap_pop_type.startswith('RIL') or opts.mstmap_pop_type == 'DH'):
            eprint('ERROR: only RILd and DH supported in MSTmap')
            sys.exit(1)
        
        opf = inputmatrix.rsplit(".", 1)[0]+'.mstmap'  # output file
        if Path(opf).exists():
            eprint("ERROR: Filename collision. The future output file `{}` exists".format(opf))
            sys.exit(1)

        df = pd.read_csv(inputmatrix, delim_whitespace=True)
        cols = list(df.columns[2:])
        cols.insert(0, 'locus_name')
        df['locus_name'] = df.iloc[:,0].astype('str') + '_' +df.iloc[:,1].astype('str')
        df = df[cols]
        print(df.head())
        snp_num, sm_num = df.shape[0], df.shape[1]-1
        f1 = open(opf, 'w')
        f1.write(mst_header.format(opts.mstmap_pop_type, opts.population_name, opts.distance_function, opts.cut_off_p_value, \
            opts.no_map_dist, opts.no_map_size, opts.missing_threshold, opts.estimation_before_clustering, opts.detect_bad_data, \
            opts.objective_function, snp_num, sm_num))
        f1.close()

        df.to_csv(opf, sep='\t', index=False, mode='a')
        print('Done, check file {}!'.format(opf))
    
    if opts.joinmap:
        opf = inputmatrix.rsplit(".", 1)[0]+'.joinmap.xlsx'  # output file
        if Path(opf).exists():
            eprint("ERROR: Filename collision. The future output file `{}` exists".format(opf))
            sys.exit(1)

        df = pd.read_csv(inputmatrix, delim_whitespace=True)
        need_reps, reps = [], []
        if opts.homo1 != 'a': 
            need_reps.append(opts.homo1)
            reps.append('a')
        if opts.homo2 != 'b': 
            need_reps.append(opts.homo2)
            reps.append('b')
        if opts.hete != 'h': 
            need_reps.append(opts.hete)
            reps.append('h')
        if opts.missing != '-': 
            need_reps.append(opts.missing)
            reps.append('-')
        if need_reps:
            df = df.replace(need_reps, reps)

        cols = list(df.columns[2:])
        cols.insert(0, 'Classification')
        cols.insert(0, 'locus_name')
        df['locus_name'] = df.iloc[:,0].astype('str') + '_' +df.iloc[:,1].astype('str')
        df['Classification'] = '(a,h,b)'
        df = df[cols]
        df.to_excel(opf)
        print('Done! Now you can load the genotype data into the JoinMap project from the MS-Excel spreadsheet {} to a dataset node.'.format(opf))

    if opts.rqtl:
        opf = inputmatrix.rsplit(".", 1)[0]+'.rqtl.csv'  # output file
        if Path(opf).exists():
            eprint("ERROR: Filename collision. The future output file `{}` exists".format(opf))
            sys.exit(1)

        df = pd.read_csv(inputmatrix, delim_whitespace=True)
        need_reps, reps = [], []
        if opts.homo1 != 'A': 
            need_reps.append(opts.homo1)
            reps.append('A')
        if opts.homo2 != 'B': 
            need_reps.append(opts.homo2)
            reps.append('B')
        if opts.hete != 'H': 
            need_reps.append(opts.hete)
            reps.append('H')
        if opts.missing != '-': 
            need_reps.append(opts.missing)
            reps.append('-')
        if need_reps:
            df = df.replace(need_reps, reps)

        cols = list(df.columns[2:])
        cols.insert(0, 'id')
        cols.insert(1, 'group')
        df['id'] = df.iloc[:,0].astype('str') + '_' +df.iloc[:,1].astype('str')
        df['group'] = 1
        df = df[cols]
        
        df = df.set_index('id')
        df1 = df.transpose()
        df1 = df1.reset_index()
        columns = list(df1.columns)
        columns[0] = 'id'
        df1.columns = columns

        df1.iloc[0,0] = np.nan
        df1.to_csv(opf, index=False, na_rep='')
        print('Done, check file {}!'.format(opf))

def bin(args):
    """
    %prog bin corrected.matrix output.matrix

    compress markers byy merging consecutive markers with same genotypes
    """
    p = OptionParser(bin.__doc__)
    p.add_option("-i", "--input", help=SUPPRESS_HELP)
    p.add_option("-o", "--output", help=SUPPRESS_HELP)
    p.add_option('--diff_num', default=0, type='int',
        help='number of different genotypes between two consecutive markers less than or equal to this value will be merged. \
        missing values will not be counted.')
    p.add_option('--missing', default='-',
        help='character for missing value in genotype matrix file')
    p.add_option("--logfile", default='GC.bin.log',
        help="specify the file saving running info")
    opts, args = p.parse_args(args)
    if len(args) != 2:
        sys.exit(not p.print_help())

    inmap, outmap = args
    inputmatrix = opts.input or inmap
    outputmatrix = opts.output or outmap
    
    if Path(outputmatrix).exists():
        eprint("ERROR: Filename collision. The future output file `{}` exists".format(outputmatrix))
        sys.exit(1)
    
    chr_order, chr_nums = getChunk(inputmatrix)
    map_reader = pd.read_csv(inputmatrix, delim_whitespace=True, index_col=[0, 1],  iterator=True)
    Good_SNPs = []
    binning_info = []
    for chrom in chr_order:
        print('{}...'.format(chrom))
        chunk = chr_nums[chrom]
        df_chr_tmp = map_reader.get_chunk(chunk)
        if df_chr_tmp.shape[0] == 1:
            Good_SNPs.append(df_chr_tmp)
        else:
            represent_idx, block_idx, results = bin_markers(df_chr_tmp.loc[chrom], diff=opts.diff_num, missing_value=opts.missing)
            good_snp = df_chr_tmp.loc[(chrom, results), :]
            Good_SNPs.append(good_snp)
            if represent_idx:
                df_binning_info = pd.DataFrame(dict(zip(['chr', 'representative_marker', 'markers'], [chrom, represent_idx, block_idx])))
                binning_info.append(df_binning_info)
    df1 = pd.concat(Good_SNPs)
    df1.to_csv(outputmatrix, sep='\t', index=True)
    before_snp_num = sum(chr_nums.values())
    after_snp_num = df1.shape[0]
    pct = after_snp_num/float(before_snp_num)*100
    print('{} SNP markers before compression.'.format(before_snp_num))
    print('{}({:.1f}%) markers left after compression.'.format(after_snp_num, pct))

    if binning_info:
        df2 = pd.concat(binning_info)
        df2.to_csv(opts.logfile, sep='\t', index=False)
        print('Check {} for binning details.'.format(opts.logfile))

def vcf2map(args):
    """
    %prog vcf2map input.vcf output.matrix

    convert vcf format to genotype matrix format
    """
    p = OptionParser(vcf2map.__doc__)
    p.add_option("-i", "--input", help=SUPPRESS_HELP)
    p.add_option("-o", "--output", help=SUPPRESS_HELP)
    p.add_option('--homo1', default="A",
                help='character for homozygous genotype')
    p.add_option("--homo2", default='B',
                help="character for alternative homozygous genotype")
    p.add_option('--hete', default='X',
                help='character for heterozygous genotype')
    p.add_option('--missing', default='-',
                help='character for missing value')
    p.add_option("--logfile", default='GC.vcf2map.info',
                help="specify the log file")
    opts, args = p.parse_args(args)
    if len(args) != 2:
        sys.exit(not p.print_help())

    invcf, outmap = args
    inputvcf = opts.input or invcf
    outputmatrix = opts.output or outmap
    
    if Path(outputmatrix).exists():
        eprint("ERROR: Filename collision. The future output file `{}` exists".format(outputmatrix))
        sys.exit(1)

    logging.basicConfig(filename=opts.logfile, 
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(message)s")

    right_gt={'0|0':opts.homo1, '0/0':opts.homo1,
    '0|1':opts.hete,'1|0':opts.hete,
    '0/1':opts.hete,'1/0':opts.hete,
    '1|1':opts.homo2,'1/1':opts.homo2,
    '.|.':opts.missing,'./.':opts.missing, '.':opts.missing}
    useless_cols = ['ID', 'REF', 'ALT','QUAL', 'FILTER','INFO','FORMAT']
    index_cols = ['#CHROM', 'POS']
    vcffile = open(inputvcf)
    n = 0
    for i in vcffile:
        if i.startswith('##'):
            n += 1
        else:
            break
    vcffile.close()
    chr_order, chr_nums = getChunk(inputvcf, ignore=n+1)
    vcf_reader = pd.read_csv(inputvcf, header=n, delim_whitespace=True, usecols=lambda x: x not in useless_cols, iterator=True)
    tmp_chr_list = []
    for chrom in chr_order:
        logging.debug('{}...'.format(chrom))
        print('{}...'.format(chrom))
        chunk = chr_nums[chrom]
        df_chr_tmp = vcf_reader.get_chunk(chunk)
        df_chr_tmp = df_chr_tmp.set_index(index_cols)
        df_chr_tmp = df_chr_tmp.applymap(lambda x: x.split(':')[0])
        df_chr_tmp = df_chr_tmp.applymap(lambda x: right_gt[x] if x in right_gt else np.nan)
        df_chr_tmp.dropna(inplace=True)
        tmp_chr_list.append(df_chr_tmp)
    df1 = pd.concat(tmp_chr_list)
    df1.to_csv(outputmatrix, sep='\t', index=True)


    vcffile.close()
    print('Done!')

def sortPos(args):
    """
    %prog sortPos input.map output.sorted.map

    sort markers based on position 
    """
    p = OptionParser(sortPos.__doc__)
    opts, args = p.parse_args(args)
    if len(args) != 2:
        sys.exit(not p.print_help())

    inmap, outmap, = args
    
    if Path(outmap).exists():
        eprint("ERROR: Filename collision. The future output file `{}` exists".format(outputmatrix))
        sys.exit(1)
    df = pd.read_csv(inmap, delim_whitespace=True)
    idx_col = list(df.columns[0:2])
    df.sort_values(idx_col).to_csv(outmap, sep='\t', index=False)
    print('Done! Check %s file.'%outmap)

def main():
    actions = (
        ('qc_missing', 'quality control of the missing gneotypes'),
        ('qc_sd', 'quality control on segregation distortions'),
        ('qc_hetero', 'quality control on the continuous same homozygous in heterozygous region'),
        ('correct', 'correct wrong genotype calls'),
        ('bin', 'merge consecutive markers with same genotypes'),
        ('cleanup', 'clean redundant info in the tmp matirx file'),
        ('format', 'convert genotype matix file to other formats for the genetic map construction'),
        ('vcf2map', 'convert vcf to genotype matrix file'),
        ('sortPos', 'sort makers based on position'),
            )
    p = ActionDispatcher(actions)
    p.dispatch(globals())

if __name__ == '__main__':
    main()
