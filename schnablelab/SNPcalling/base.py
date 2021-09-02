# -*- coding: UTF-8 -*-

"""
base class and functions to handle with vcf file
"""
import re
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from subprocess import call
from collections import Counter
from schnablelab.apps.base import ActionDispatcher, OptionParser

def main():
    actions = (
        ('FilterMissing', 'filter out SNPs with high missing rate'),
        ('FilterMAF', 'filter out SNP with extremely low minor allele frequency'),
        ('FilterHetero', 'filter out SNPs with high heterozygous rates'),
        ('SubsamplingSNPs', 'grep a subset of SNPs from a vcf file'),
        ('SubsamplingSMs', 'grep a subset of samples from a vcf file'),
        ('vcf2hmp', 'convert vcf to hmp foramt'),
        ('Info', 'get basic information of a vcf file'),
        ('reheader', 'edit sample names in header only'),
            )
    p = ActionDispatcher(actions)
    p.dispatch(globals())

def find_sm(target_str, re_pattern):
    sms = re_pattern.findall(target_str)
    if len(sms)==1:
        sm = sms[0][1:-1]
        return '-'.join(re.split('[_-]', sm))
    else:
        sys.exit(f"bad file name '{target_str}'!")

class ParseVCF():
    '''
    parse vcf file
    '''
    def __init__(self, filename):
        self.fn = filename
        with open(filename) as f:
            n = 0
            hash_chunk = []
            hash_chunk2 = []
            num_SNPs = 0
            for i in f:
                if i.startswith('##'):
                    n += 1
                    hash_chunk.append(i)
                    hash_chunk2.append(i)
                    continue
                if i.startswith('#'):
                    SMs_header = i.split()[:9]
                    SMs = i.split()[9:]
                    numSMs = len(SMs)
                    n += 1
                    hash_chunk.append(i)
                else:
                    num_SNPs += 1
        self.num_SNPs = num_SNPs
        self.SMs_header = SMs_header
        self.SMs = SMs
        self.numSMs = numSMs
        self.numHash = n
        self.HashChunk = hash_chunk
        self.HashChunk2 = hash_chunk2
        self.numHeaderLines = len(self.HashChunk)
        self.hmpfield11 = 'rs#\talleles\tchrom\tpos\tstrand\tassembly#\tcenter\tprotLSID\tassayLSID\tpanelLSID\tQCcode'
        self.hmpheader = self.hmpfield11 + '\t' + '\t'.join(self.SMs) + '\n'

    def AsDataframe(self):
        df = pd.read_csv(self.fn, skiprows=range(self.numHash-1), delim_whitespace=True)
        return df
    
    @property
    def ToHmp(self):
        '''
        yield line in hmp format
        '''
        cen_NA = '+\tNA\tNA\tNA\tNA\tNA\tNA'
        with open(self.fn) as f:
            for _ in range(self.numHash):
                next(f)
            for i in f:
                j = i.split()
                a1, a2 = j[3], j[4]
                if len(a1) == len(a2) ==1:
                    a1a2 = ''.join([a1, a2])
                    a2a1 = a1a2[::-1]
                    a1a1, a2a2 = a1*2, a2*2
                elif len(a1)==1 and len(a2)>1:
                    a1a2 = ''.join(['-', a2[-1]])
                    a2a1 = a1a2[::-1]
                    a1a1, a2a2 = '--', a2[-1]*2
                elif len(a1) >1 and len(a2)==1:
                    a1a2 = ''.join([a1[-1], '-'])
                    a2a1 = a1a2[::-1]
                    a1a1, a2a2 = a1[-1]*2, '--'
                else:
                    print('bad format line:\n  %s'%j[2])
                    continue
                geno_dict = {'0/0':a1a1, '0|0':a1a1, 
                            '0/1':a1a2, '0|1':a1a2, 
                            '1/0':a2a1, '1|0':a2a1,
                            '1/1':a2a2, '1|1':a2a2,
                            './.':'NN', '.|.':'NN'}
                genos = list(map(geno_dict.get, j[9:]))
                if None in genos:
                    print(i)
                    sys.exit('unknow genotype detected!')
                genos = '\t'.join(genos)
                rs, chr, pos = j[2], j[0], j[1]
                alleles = '/'.join([a1a2[0], a1a2[1]])
                new_line = '\t'.join([rs, alleles, chr, pos, cen_NA, genos])+'\n'
                yield new_line

    @property
    def Missings(self):
        '''
        yield missing rate for each line
        '''
        with open(self.fn) as f:
            for _ in range(self.numHash):
                next(f)
            for i in f:
                num_miss = i.count('./.')+i.count('.|.')
                yield i, num_miss/self.numSMs

    @staticmethod
    def CountGenos(geno_ls):
        c = Counter(geno_ls)
        num_a = c['0/0']+c['0|0']
        num_b = c['1/1']+ c['1|1']
        num_h = c['0/1']+ c['1/0'] + c['0|1']+c['1|0']
        return num_a, num_b, num_h

    @property
    def Heteros(self):
        '''
        yield (line, heterozygous rate) for each line
        '''
        with open(self.fn) as f:
            for _ in range(self.numHash):
                next(f)
            for i in f:
                num_a, num_b, num_h = ParseVCF.CountGenos(i.split()[9:])
                if num_h > max(num_a, num_b):
                    yield i, 1
                else:
                    yield i, num_h/float(num_a + num_b + num_h)

    @property
    def MAFs(self):
        '''
        yield minor allele frequence for each line
        '''
        with open(self.fn) as f:
            for _ in range(self.numHash):
                next(f)
            for i in f:
                num_a, num_b, num_h = ParseVCF.CountGenos(i.split()[9:])
                a1, a2 = num_a*2+num_h, num_b*2+num_h
                yield  i, min(a1,a2)/(a1+a2)

def reheader(args):
    """
    %prog reheader input_vcf names.csv

    substitute the sample names in vcf header using sed. 
    name.csv:
        comma separated without header line
        1st column is old name
        2nd column is the new name
    """
    p = OptionParser(reheader.__doc__)
    _, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    inputvcf, names_csv, = args
    outputvcf = Path(inputvcf).name.replace('.vcf', '_reheader.vcf')

    vcf = ParseVCF(inputvcf)

    cmd = 'sed '
    for _, row in pd.read_csv(names_csv, header=None).iterrows():
        old_nm, new_nm = row[0], row[1]
        if old_nm not in vcf.SMs:
            print('%s was not found in vcf...'%id)
        else:
            cmd += "-e '%ss/%s/%s/' "%(vcf.numHash, old_nm, new_nm)
    cmd += '%s > %s'%(inputvcf, outputvcf)
    print('command:\n%s'%cmd)
    choice = input("Run the above command? (yes/no) ")
    if choice == 'yes':
        call(cmd, shell=True)
        print('Done! check %s'%outputvcf)

def vcf2hmp(args):
    '''
    %prog vcf2hmp input_vcf
    convert file in vcf format to hmp format
    can also try tassel: 'run_pipeline.pl -Xms512m -Xmx10G -fork1 -vcf vcf_fn -export -exportType Hapmap'
    '''
    p = OptionParser(vcf2hmp.__doc__)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    inputvcf, = args
    outputhmp = Path(inputvcf).name.replace('.vcf', '.hmp')

    vcf = ParseVCF(inputvcf)
    with open(outputhmp, 'w') as f:
        f.write(vcf.hmpheader)
        pbar = tqdm(vcf.ToHmp, total=vcf.num_SNPs, desc='vcf 2 hmp', position=0)
        for i in pbar:
            f.write(i)
            pbar.set_description('converting chromosome %s'%i.split()[2])
            #pbar.update(1)
    print('Done! check output %s...'%outputhmp)    

def FilterMissing(args):
    """
    %prog FilterMissing input_vcf
    Remove SNPs with high missing rate
    """
    p = OptionParser(FilterMissing.__doc__)
    p.add_option('--missing_cutoff', default = 0.7, type='float', 
        help = 'specify the missing rate cutoff. SNPs higher than this cutoff will be removed.')
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    inputvcf, = args
    outputvcf = Path(inputvcf).name.replace('.vcf', '_mis%s.vcf'%opts.missing_cutoff)

    vcf = ParseVCF(inputvcf)
    n = 0
    with open(outputvcf, 'w') as f:
        f.writelines(vcf.HashChunk)
        pbar = tqdm(vcf.Missings, total=vcf.num_SNPs, desc='Filter Missing', position=0)
        for i, miss in pbar:
            if miss <= opts.missing_cutoff:
                f.write(i)
            else:
                n += 1
            pbar.set_description('processing chromosome %s'%i.split()[0])
    print('Done! %s SNPs removed! check output %s...'%(n, outputvcf))

def FilterMAF(args):
    """
    %prog FilterMAF input_vcf
    Remove rare MAF SNPs
    """
    p = OptionParser(FilterMAF.__doc__)
    p.add_option('--maf_cutoff', default = 0.01, type='float',
        help = 'specify the MAF rate cutoff, SNPs lower than this cutoff will be removed.')
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    inputvcf, = args
    outputvcf = Path(inputvcf).name.replace('.vcf', '_maf%s.vcf'%opts.maf_cutoff)

    vcf = ParseVCF(inputvcf)
    n = 0
    with open(outputvcf, 'w') as f:
        f.writelines(vcf.HashChunk)
        pbar = tqdm(vcf.MAFs, total=vcf.num_SNPs, desc='Filter MAF', position=0)
        for i, maf in pbar:
            if maf >= opts.maf_cutoff:
                f.write(i)
            else:
                n += 1
            pbar.set_description('processing chromosome %s'%i.split()[0])
    print('Done! %s SNPs removed! check output %s...'%(n, outputvcf))

def FilterHetero(args):
    """
    %prog FilterHetero input_vcf
    Remove bad and high heterizygous loci
    """
    p = OptionParser(FilterHetero.__doc__)
    p.add_option('--het_cutoff', default = 0.1, type='float',
        help = 'specify the heterozygous rate cutoff, SNPs higher than this cutoff will be removed.')
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    inputvcf, = args
    outputvcf = Path(inputvcf).name.replace('.vcf', '_het%s.vcf'%opts.het_cutoff)

    vcf = ParseVCF(inputvcf)
    n = 0
    with open(outputvcf, 'w') as f:
        f.writelines(vcf.HashChunk)
        pbar = tqdm(vcf.Heteros, total=vcf.num_SNPs, desc='Filter Heterozygous', position=0)
        for i, het in pbar:
            if het <= opts.het_cutoff:
                f.write(i)
            else:
                n += 1
            pbar.set_description('processing chromosome %s'%i.split()[0])
    print('Done! %s SNPs removed! check output %s...'%(n, outputvcf))
    
def SubsamplingSNPs(args):
    """
    %prog SubsamplingSNPs input_vcf SNPs.csv 
    grep a subset of SNPs defined in SNPs.csv (One ID per row without header) from the input_vcf
    """
    p = OptionParser(SubsamplingSNPs.__doc__)
    _, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    inputvcf, SNPcsv, = args
    outputvcf = Path(inputvcf).name.replace('.vcf', '_subSNPs.vcf')

    vcf = ParseVCF(inputvcf)
    df_vcf = vcf.AsDataframe()

    IDs = pd.read_csv(SNPcsv, header=None)[0].values
    num_IDs = IDs.shape[0]
    print('number of specified SNPs: %s'%num_IDs)
    df_vcf = df_vcf[df_vcf['ID'].isin(IDs)]
    print('%s out of %s found in VCF'%(df_vcf.shape[0], num_IDs))
    with open(outputvcf, 'w') as f:
        f.writelines(vcf.HashChunk2)
    df_vcf.to_csv(outputvcf, sep='\t', index=False, mode='a')
    print('Done! check output %s...'%outputvcf)

def SubsamplingSMs(args):
    """
    %prog SubsamplingSMs input_vcf SMs.csv 
    grep a subset of samples defined in SMs.csv (One sample name per row without header) from the input_vcf
    """
    p = OptionParser(SubsamplingSMs.__doc__)
    _, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    inputvcf, SMcsv, = args
    outputvcf = Path(inputvcf).name.replace('.vcf', '_subSMs.vcf')

    vcf = ParseVCF(inputvcf)
    df_vcf = vcf.AsDataframe()

    IDs = pd.read_csv(SMcsv, header=None)[0].values
    num_IDs = IDs.shape[0]
    print('number of specified Samples: %s'%num_IDs)

    subsm = vcf.SMs_header
    for id in IDs:
        if id not in vcf.SMs:
            print('%s not found in vcf...'%id)
        else:
            subsm.append(id)
    print('%s out of %s found in VCF'%(len(subsm)-9, num_IDs))

    df_vcf = df_vcf[subsm]
    with open(outputvcf, 'w') as f:
        f.writelines(vcf.HashChunk2)
    df_vcf.to_csv(outputvcf, sep='\t', index=False, mode='a')
    print('Done! check output %s...'%outputvcf)   

def Info(args):
    """
    %prog Info input_vcf

    get basic info including SMs, number of SMs, number of SNPs, 
    """
    p = OptionParser(Info.__doc__)
    _, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    inputvcf, = args
    vcf = ParseVCF(inputvcf)
    print('number of samples: {val:,}'.format(val=vcf.numSMs))
    print("number of hash ('#') lines: {val:,}".format(val=vcf.numHeaderLines))
    print('number of SNPs: {val:,}'.format(val=vcf.num_SNPs))
    print('Sample names: \n  %s'%'\n  '.join(vcf.SMs))

if __name__ == "__main__":
    main()
