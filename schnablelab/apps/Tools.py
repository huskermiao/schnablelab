import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress

def print_json(data):
    print(json.dumps(data, indent=2))

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def GenDataFrameFromPath(path, pattern='*.png', fs=False):
    """
    generate a dataframe for all file in a dir with the specific pattern of file name.
    use: GenDataFrameFromPath(path, pattern='*.png')
    """
    fnpaths = list(path.glob(pattern))
    df = pd.DataFrame(dict(zip(['fnpath'], [fnpaths])))
    df['dir'] = df['fnpath'].apply(lambda x: x.parent)
    df['fn'] = df['fnpath'].apply(lambda x: x.name)
    if fs:
        df['size'] = df['fnpath'].apply(lambda x: os.path.getsize(x))
    return df

def ConciseVcf(fn):
    """
    concise the vcf file by remove the header, useless columns and simplfied genotype
    ConciseVcf(fn)
    """
    n = 0
    f = open(fn)
    for i in f:
        if i.startswith('##'):
            n += 1
        else:
            break
    df = pd.read_csv(fn, header=n, delim_whitespace=True)
    df = df.drop(['INFO', 'FORMAT', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER'], axis=1)
    for idx in df.columns[2:]:
        df[idx] = df[idx].map(lambda x: x.split(':')[0])
    df = df.replace(['0/0', '0/1', '1/0', '1/1', './.'], [0, 1, 1, 2, 9])
    return df

class SimpleStats(object):
    """
    This class will do the simple statistics on two series objecjts.
    a) linear regressoin: slope, intercept, r^2, p_value
    b) mean, std of the difference and absolute differnece
    c) MSE (mean squared error) and RMSE (root mean squared error)
    d) agreement
    e) plot the regreesion figure and the difference distribution figure
    """
    def __init__(self, series1, series2):
        self.s1 = series1
        self.s2 = series2
        self.length = series1.shape[0]
        self.diff = series1 - series2
        self.absdiff = (series1 - series2).abs()

    def regression(self):
        slope, intercept, r_value, p_value, __ = linregress(self.s1, self.s2)
        return slope, intercept, r_value**2, p_value

    def mean_std_diff(self):
        mean, std = self.diff.mean(), self.diff.std()
        return mean, std

    def mean_std_absdiff(self):
        abs_mean, abs_std = self.absdiff.mean(), self.absdiff.std()
        return abs_mean, abs_std

    def mse(self):
        mse = mean_squared_error(self.s1, self.s2)
        return mse

    def rmse(self):
        rmse = mean_squared_error(self.s1, self.s2)**0.5
        return rmse
    
    def agreement(self, cutoff):
        return (self.absdiff<=float(cutoff)).sum()/self.length

def simulate_gwas_results(num_chr=10, num_snp=50000, max_chr_size=60000000, 
                            num_gwas=25, sig_cutoff=6.5, num_sig_snps=5):
    '''
    num_chr: number of chromosomes
    num_snp: number of snps in each chr
    max_chr_size: the maximum chromosome size for each chr
    num_gwas: number of gwas analyses
    sig_cutoff: the -log transformed p value cutoff
    num_sig_snps: number of significant snps in each gwas analysis
    '''
    chrs = np.array([np.full(num_snp, i) for i in range(1, chrom+1)]).flatten()
    # simulate snp positions in each chromosomes
    snp_pos = np.array([np.sort(np.random.randint(0, max_chr_size, num_snp)) for i in range(1, 11)]).flatten()
    # simuate -log transformed pvalues in 25 GWAS analyses (assume all pvalues are not significant)
    df_pvalues = pd.DataFrame(np.random.uniform(0, sig_cutoff, (num_snp, num_gwas))) # simuate pvalues

    # simuate positions and pvalues for significant SNPs, assume only 5 significant snps in each GWAS
    sig_pos = pd.DataFrame(np.random.randint(0, num_snp, (num_sig_snps, num_gwas)))
    sig_pvalues = pd.DataFrame(np.random.uniform(sig_cutoff, sig_cutoff+5, (num_sig_snps, num_gwas)))
    # introduce the simuated snps
    for col in range(num_gwas): 
        for row in range(num_sig_snps):
            pos = sig_pos.iloc[row, col]
            pv = sig_pvalues.iloc[row, col]
            #print(pos, col)
            df_pvalues.iloc[pos, col] = pv
    df_pvalues.columns = np.arange(num_gwas)
    df_snp = pd.DataFrame(chrs, columns=['chr'])
    df_snp['pos'] = snp_pos
    df_gwas = pd.concat([df_snp, df_pvalues], axis=1)
    return df_gwas

def bin_gwas_snps(df, bin_size=16):
    '''
    df: the dataframe of your GWAS results
        Make sure the first column is 'chr' and 2nd column is 'pos'
        The rest columns are genus names.
    bin_size: how many bins for each chromosme (even integer)
    '''
    df = df.set_index(['chr', 'pos'])
    chros, sts, eds, ps = [], [], [], []
    for chrom, tmp in df.groupby(level=0):
        tmp = tmp.reset_index()
        bins = pd.qcut(tmp['pos'], bin_size)
        grps = tmp.groupby(bins)
        for _, grp in grps:
            idx = grp['pos'].values
            st, ed = idx[0], idx[-1]
            maxs = grp.iloc[:,2:].max(axis=0)
            chros.append(chrom)
            sts.append(st)
            eds.append(ed)
            ps.append(maxs)
    df_bins = pd.DataFrame(ps)
    df_bins.index = chros
    df_plot = df_bins.transpose().fillna(0)
    df_bins_info = pd.DataFrame(dict(zip(['chr', 'st', 'ed'], [chros, sts, eds])))    
    return df_plot, df_bins_info