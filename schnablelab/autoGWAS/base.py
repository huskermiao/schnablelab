# -*- coding: UTF-8 -*-

"""
base class and functions to handle with hmp file and GWAS results
"""
import re
import sys
import numpy as np
import pandas as pd
import os.path as op
from tqdm import tqdm
from pathlib import Path
from subprocess import call
from collections import Counter
from schnablelab.apps.base import ActionDispatcher, OptionParser, put2slurm

plink = op.abspath(op.dirname(__file__)) + '/../apps/plink'
GEC = op.abspath(op.dirname(__file__)) + '/../apps/gec.jar'

def main():
    actions = (
        ('FilterMissing', 'filter out SNPs with high missing rate'),
        ('FilterMAF', 'filter out SNP with extremely low minor allele frequency'),
        ('FilterHetero', 'filter out SNPs with high heterozygous rates'),
        ('SubsamplingSNPs', 'grep a subset of specified SNPs from a hmp file'),
        ('DownsamplingSNPs', 'pick up some SNPs from a huge hmp file using Linux sed command'),
        ('SubsamplingSMs', 'grep a subset of samples from a hmp file'),
        ('hmp2ped', 'convert hmp file to plink map and ped file'),
        ('ped2bed', 'convert plink ped format to binary bed format'),
        ('IndePvalue', 'estimate the number of independent SNPs using GEC'),
        ('hmpSingle2Double', 'convert single hmp to double type hmp'),
        ('Info', 'get basic info for a hmp file'),
        ('MAFs', 'calculate the MAF for all/specified SNPs in hmp'),
        ('sortHmp', 'Sort hmp file based on chromosome order and position'),
        ('reheader', 'edit sample names in header only'),

)
    p = ActionDispatcher(actions)
    p.dispatch(globals())

# N:missing, -:gap
geno_one2two = {
    'A':'AA', 'C':'CC', 'G':'GG', 'T':'TT',
    'R':'AG', 'Y':'CT', 'S':'GC', 'W':'AT', 'K':'GT', 'M':'AC',
    'N':'NN', '-':'--'} 
geno_two2one = {
    'AA': 'A', 'CC': 'C', 'GG': 'G', 'TT': 'T',
    'GA': 'R', 'AG': 'R', 'TC': 'Y', 'CT': 'Y',
    'CG': 'S', 'GC': 'S', 'TA': 'W', 'AT': 'W',
    'TG': 'K', 'GT': 'K', 'CA': 'M', 'AC': 'M',
    'NN': 'N', '--': '-'}

def sortchr(x):
    '''
    criteria to sort chromosome names
    '''
    x1 = re.findall(r'\d+$', x)
    if len(x1)==1:
        return int(x1[0])
    else:
        sys.exit('check chromosome name!')

class ParseHmp():
    '''
    parse hmp file
    '''
    def __init__(self, filename):
        '''
        args:
            filename: hmp file name
            type: hmp format. double or single
        '''
        self.fn = filename
        with open(filename) as f:
            headerline = f.readline()
            SMs_header = headerline.split()[:11]
            SMs = headerline.split()[11:]
            numSMs = len(SMs)
            firstgeno = f.readline().split()[11]
            type = 'single' if len(firstgeno)==1 else 'double'
            #print('guess hmp type: %s'%type)
            numSNPs = sum(1 for _ in f)
            dtype_dict = {i:'str' for i in headerline.split()}
            dtype_dict['pos'] = np.int64
        self.headerline = headerline
        self.SMs_header = SMs_header
        self.SMs = SMs
        self.numSMs = numSMs
        self.numSNPs = numSNPs+1
        self.type = type
        self.dtype_dict = dtype_dict
        
    def AsDataframe(self, needsort=False):
        '''
        args:
            needsort: if hmp need to be sorted
        '''
        df = pd.read_csv(self.fn, delim_whitespace=True, dtype=self.dtype_dict)
        if needsort:
            chrs = list(df['chrom'].unique())
            chrs_ordered = sorted(chrs, key=sortchr)
            df['chrom'] = pd.Categorical(df['chrom'], chrs_ordered, ordered=True)
            df = df.sort_values(['chrom', 'pos']).reset_index(drop=True)
        if self.type=='single':
            print('converting the single type to double type...')
            df_part2 = df.iloc[:, 11:].applymap(geno_one2two.get).fillna('NN')
            df = pd.concat([df.iloc[:, :11], df_part2], axis=1)
            self.type = 'double'
        return df
    
    def AsMapPed(self, missing=False):
        df_hmp = self.AsDataframe()
        df_map = df_hmp[['rs#', 'chrom', 'pos']]
        df_map['centi'] = 0
        map_cols = ['chrom', 'rs#', 'centi', 'pos']
        df_map = df_map[map_cols]

        if missing:
            df_hmp = df_hmp.replace('NN', '00')

        #part1_cols = ['fam_id', 'indi_id', 'indi_id_father', 'indi_id_mother', 'sex', 'pheno']
        df_ped = np.zeros((self.numSMs, 6+self.numSNPs*2), dtype='str')
        pbar = tqdm(self.SMs)
        for idx, col in enumerate(pbar):
            # this is much faster than df[col].apply(lambda x: pd.Series(list(x))).to_numpy().ravel()
            col_a1, col_a2 = df_hmp[col].str[0].to_numpy(), df_hmp[col].str[1].to_numpy()
            df_ped[idx, 6:] = np.column_stack((col_a1, col_a2)).ravel()
            pbar.set_description('converting %s'%col)
        df_ped = np.where(df_ped=='', 0, df_ped)
        df_ped = pd.DataFrame(df_ped, dtype='str')
        df_ped[1]=self.SMs
        return df_map, df_ped
    
    def BIMBAM(self):
        pass
    
    @staticmethod
    def ParseAllele_single(geno_list, a1, a2):
        '''
        geno_list: list of all genotypes
        a1: allele1, string, single character
        a2: allele2, string, single character
        '''
        h = geno_two2one[a1 + a2]
        c = Counter(geno_list)
        return  c[a1], c[a2], c[h]
     
    @staticmethod
    def ParseAllele_double(geno_list, a1, a2):
        '''
        same as ParseAllele_single parameters
        '''
        a, b = a1*2, a2*2
        h1, h2 = a1+a2, a2+a1
        c = Counter(geno_list)
        return c[a], c[b], c[h1]+ c[h2]

    @property
    def Missings(self):
        '''
        yield (line, missing rate) for each line
        '''
        with open(self.fn) as f:
            next(f)
            for i in f:
                c = Counter(i.split()[11:])
                num_miss = c['NN']+c['N']
                yield i, num_miss/self.numSMs

    @property
    def MAFs(self):
        '''
        yield (line, maf) for each line
        '''
        if self.type == 'double':
            with open(self.fn) as f:
                next(f)
                for i in f:
                    j = i.split()
                    alleles = j[1].split('/')
                    try:
                        allele1, allele2 = alleles
                    except ValueError:
                        yield i, 0
                    else:
                        num_a, num_b, num_h = ParseHmp.ParseAllele_double(j[11:], allele1, allele2)
                        a1, a2 = num_a*2+num_h, num_b*2+num_h
                        try:
                            maf = min(a1, a2)/(a1+a2)
                        except ZeroDivisionError:
                            yield i, 0
                        else:
                            yield i, maf 
        else:
            with open(self.fn) as f:
                next(f)
                for i in f:
                    j = i.split()
                    alleles = j[1].split('/')
                    try:
                        allele1, allele2 = alleles
                    except ValueError:
                        yield i, 0
                    else:
                        num_a, num_b, num_h = ParseHmp.ParseAllele_single(j[11:], allele1, allele2)
                        a1, a2 = num_a*2+num_h, num_b*2+num_h
                        try:
                            maf = min(a1, a2)/(a1+a2)
                        except ZeroDivisionError:
                            yield i, 0
                        else:
                            yield i, maf 

    @property
    def Heteros(self):
        '''
        yield (line, heterozgous rate) for each line
        '''
        if self.type == 'double':
            with open(self.fn) as f:
                next(f)
                for i in f:
                    j = i.split()
                    alleles = j[1].split('/')
                    try:
                        allele1, allele2 = alleles
                    except ValueError:
                        yield i, 1
                    else:
                        num_a, num_b, num_h = ParseHmp.ParseAllele_double(j[11:], allele1, allele2)
                        if num_h > max(num_a, num_b):
                            yield i, 1
                        else:
                            yield i, num_h/float(num_a + num_b + num_h)
        else:
            with open(self.fn) as f:
                next(f)
                for i in f:
                    j = i.split()
                    alleles = j[1].split('/')
                    try:
                        allele1, allele2 = alleles
                    except ValueError:
                        yield i, 1
                    else:
                        num_a, num_b, num_h = ParseHmp.ParseAllele_single(j[11:], allele1, allele2)
                        if num_h > max(num_a, num_b):
                            yield i, 1
                        else:
                            yield i, num_h/float(num_a + num_b + num_h)

class ReadGWASfile():
    
    def __init__(self, filename, software, needsort=False, usecols=None):
        '''
        Args:
            filename: gwas result filename
            software: gwas software (gemma, farmcpu, mvp, gapit)
            needsort: if the gwas file need to be sorted
            usecols: specify which pvlue column if multiple approaches used in MVP
        '''
        self.fn = filename
        self.software = software
        self.needsort = needsort
        self.usecols = usecols
        
        if self.software == 'gemma':
            dtype_dict = {'chr':'str', 'rs':'str', 'ps':np.int64, 'p_lrt':np.float64}
            df = pd.read_csv(self.fn, delim_whitespace=True, usecols=['chr', 'rs', 'ps', 'p_lrt'], dtype=dtype_dict)
            df = df[['rs', 'chr', 'ps', 'p_lrt']]
        elif self.software == 'farmcpu':
            dtype_dict = {'Chromosome':'str', 'SNP':'str', 'Position':np.int64, 'P.value':np.float64}
            df = pd.read_csv(self.fn, usecols=['SNP', 'Chromosome', 'Position', 'P.value'], dtype=dtype_dict)
        elif self.software == 'gapit':
            dtype_dict = {'Chromosome':'str', 'SNP':'str', 'Position':np.int64, 'P.value':np.float64}
            df = pd.read_csv(self.fn, usecols=['SNP', 'Chromosome', 'Position ', 'P.value'])
        elif self.software == 'other':
            if self.usecols is None:
                sys.exit('specify which columns for use if choosing other!')
            if (not isinstance(self.usecols, list)):
                sys.exit('usecols must be a list')
            if len(self.usecols) != 4:
                sys.exit('usecols must have the lenght of 4')
            with open(self.fn) as f:
                j = list(pd.read_csv(self.fn).columns)
                snp_idx, chr_idx, pos_idx, pv_idx = self.usecols
                snp, chr, pos, pv = j[snp_idx], j[chr_idx], j[pos_idx], j[pv_idx]
                dtype_dict = {snp:'str', chr:'str', pos:np.int64, pv:np.float64}
            df = pd.read_csv(self.fn, usecols=[snp, chr, pos, pv], dtype=dtype_dict)[[snp, chr, pos, pv]]
        else:
            sys.exit('only gemma, farmcpu, gapit, and other are supported!')
        df.columns = ['snp', 'chr', 'pos', 'pvalue']
        df['pvalue'] = -np.log10(df['pvalue'])
        df.columns = ['snp', 'chr', 'pos', '-log10Pvalue']
        if needsort:
            chrs = list(df['chr'].unique())
            chrs_ordered = sorted(chrs, key=sortchr)
            df['chr'] = pd.Categorical(df['chr'], chrs_ordered, ordered=True)
            df = df.sort_values(['chr', 'pos']).reset_index(drop=True)
        self.df = df
        self.numberofSNPs = df.shape[0]

    def SignificantSNPs(self, p_cutoff=0.05, MeRatio=1, sig_cutoff=None):
        '''
        extract Significant SNPs
        sig_cutoff: the log10 transformed p values cutoff 
        '''
        cutoff = -np.log10(p_cutoff/(MeRatio * self.numberofSNPs)) if sig_cutoff is None else sig_cutoff
        df_sigs = self.df[self.df['-log10Pvalue'] >= cutoff].reset_index(drop=True)
        return df_sigs

def reheader(args):
    """
    %prog reheader input_hmp names.csv

    substitute the sample names in hmp header using sed. 
    name.csv:
        comma separated without header line
        1st column is old name
        2nd column is the new name
    """
    p = OptionParser(reheader.__doc__)
    _, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    inputhmp, names_csv, = args
    outputhmp = Path(inputhmp).name.replace('.hmp', '_reheader.hmp')

    hmp = ParseHmp(inputhmp)

    cmd = 'sed '
    for _, row in pd.read_csv(names_csv, header=None).iterrows():
        old_nm, new_nm = row[0], row[1]
        if old_nm not in hmp.SMs:
            print('%s was not found in hmp...'%id)
        else:
            cmd += "-e '1s/%s/%s/' "%(old_nm, new_nm)
    cmd += '%s > %s'%(inputhmp, outputhmp)
    print('command:\n%s'%cmd)
    choice = input("Run the above command? (yes/no) ")
    if choice == 'yes':
        call(cmd, shell=True)
        print('Done! check %s'%outputhmp)

def FilterMissing(args):
    """
    %prog FilterMissing input_hmp
    Remove SNPs with high missing rate
    """
    p = OptionParser(FilterMissing.__doc__)
    p.add_option('--missing_cutoff', default = 0.7, type='float', 
        help = 'specify the missing rate cutoff. SNPs higher than this cutoff will be removed.')
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    inputhmp, = args
    outputhmp = Path(inputhmp).name.replace('.hmp', '_mis%s.hmp'%opts.missing_cutoff)

    hmp = ParseHmp(inputhmp)
    n = 0
    with open(outputhmp, 'w') as f:
        f.write(hmp.headerline)
        pbar = tqdm(hmp.Missings, total=hmp.numSNPs)
        for i, miss in pbar:
            if miss <= opts.missing_cutoff:
                f.write(i)
            else: 
                n +=1
            pbar.set_description('processing chromosome %s'%i.split()[2])
    print('Done! %s SNPs removed! check output %s...'%(n, outputhmp))

def FilterHetero(args):
    """
    %prog FilterHetero input_hmp
    Remove bad and high heterizygous loci (coducting Missing and MAF first)
    """
    p = OptionParser(FilterHetero.__doc__)
    p.add_option('--het_cutoff', default = 0.1, type='float',
        help = 'specify the heterozygous rate cutoff, SNPs higher than this cutoff will be removed.')
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    inputhmp, = args
    outputhmp = Path(inputhmp).name.replace('.hmp', '_het%s.hmp'%opts.het_cutoff)

    hmp = ParseHmp(inputhmp)
    n = 0
    with open(outputhmp, 'w') as f:
        f.write(hmp.headerline)
        pbar = tqdm(hmp.Heteros, total=hmp.numSNPs)
        for i, het in pbar:
            if het <= opts.het_cutoff:
                f.write(i)
            else:
                n += 1
            pbar.set_description('processing chromosome %s'%i.split()[2])
    print('Done! %s SNPs removed! check output %s...'%(n, outputhmp))

def FilterMAF(args):
    """
    %prog FilterMAF input_hmp
    Remove rare MAF SNPs (conducting Missing filter first)
    """
    p = OptionParser(FilterMAF.__doc__)
    p.add_option('--MAF_cutoff', default = 0.01, type='float',
        help = 'specify the MAF rate cutoff, SNPs lower than this cutoff will be removed.')
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    inputhmp, = args
    outputhmp = Path(inputhmp).name.replace('.hmp', '_maf%s.hmp'%opts.MAF_cutoff)

    hmp = ParseHmp(inputhmp)
    n = 0
    with open(outputhmp, 'w') as f:
        f.write(hmp.headerline)
        pbar = tqdm(hmp.MAFs, total=hmp.numSNPs)
        for i, maf in pbar:
            if maf >= opts.MAF_cutoff:
                f.write(i)
            else:
                n += 1
            pbar.set_description('processing chromosome %s'%i.split()[2])
    print('Done! %s SNPs removed! check output %s...'%(n, outputhmp))
    
def SubsamplingSNPs(args):
    """
    %prog SubsamplingSNPs input_hmp SNPs.csv 
    grep a subset of SNPs defined in SNPs.csv (One ID per row without header) from the input_hmp
    """
    p = OptionParser(SubsamplingSNPs.__doc__)
    _, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    inputhmp, SNPcsv, = args
    outputhmp = Path(inputhmp).name.replace('.hmp', '_subSNPs.hmp')

    hmp = ParseHmp(inputhmp)
    df_hmp = hmp.AsDataframe()

    IDs = pd.read_csv(SNPcsv, header=None)[0].values
    num_IDs = IDs.shape[0]
    print('number of specified SNPs: %s'%num_IDs)
    df_hmp = df_hmp[df_hmp['rs#'].isin(IDs)]
    print('%s out of %s found in Hmp'%(df_hmp.shape[0], num_IDs))
    df_hmp.to_csv(outputhmp, sep='\t', index=False, na_rep='NA')
    print('Done! check output %s...'%outputhmp)

def SubsamplingSMs(args):
    """
    %prog SubsamplingSMs input_hmp SMs.csv 
    grep a subset of samples defined in SMs.csv (One sample name per row without header) from the input_hmp
    """
    p = OptionParser(SubsamplingSMs.__doc__)
    _, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    inputhmp, SMcsv, = args
    outputhmp = Path(inputhmp).name.replace('.hmp', '_subSMs.hmp')

    hmp = ParseHmp(inputhmp)
    df_hmp = hmp.AsDataframe()

    IDs = pd.read_csv(SMcsv, header=None)[0].values
    num_IDs = IDs.shape[0]
    print('number of specified Samples: %s'%num_IDs)

    subsm = hmp.SMs_header
    for id in IDs:
        if id not in hmp.SMs:
            print('%s was not found in hmp...'%id)
        else:
            subsm.append(id)
    print('%s out of %s found in Hmp'%(len(subsm)-11, num_IDs))

    df_hmp = df_hmp[subsm]
    df_hmp.to_csv(outputhmp, sep='\t', index=False, na_rep='NA')
    print('Done! check output %s...'%outputhmp)

def DownsamplingSNPs(args):
    """
    %prog downsampling input_hmp

    Pick up some SNPs from a huge hmp file using Linux sed command
    """
    p = OptionParser(DownsamplingSNPs.__doc__)
    p.add_option('--downscale', default=10,
                 help='specify the downscale level')
    p.add_option('--disable_slurm', default=False, action="store_true",
                 help='do not convert commands to slurm jobs')
    p.add_slurm_opts(job_prefix=DownsamplingSNPs.__name__)
    opts, args = p.parse_args(args)

    if len(args) == 0:
        sys.exit(not p.print_help())

    inputhmp, = args
    outputhmp = Path(inputhmp).name.replace('.hmp', '_ds%s.hmp'% opts.downsize)
    cmd = "sed -n '1~%sp' %s > %s" % (opts.downsize, inputhmp, outputhmp)
    print('cmd:\n%s\n' % cmd)
    if not opts.disable_slurm:
        put2slurm_dict = vars(opts)
        put2slurm([cmd], put2slurm_dict)

def hmp2ped(args):
    """
    %prog input_hmp

    Convert hmp file to Plink map and ped files
    """
    p = OptionParser(hmp2ped.__doc__)
    _, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    inputhmp, = args
    output_prefix = Path(inputhmp).name.split('.hmp')[0]

    hmp = ParseHmp(inputhmp)
    df_map, df_ped = hmp.AsMapPed(missing=False)
    print('saving map file...')
    df_map.to_csv('%s.map'%output_prefix, sep='\t', index=False, header=None)
    print('saving ped file...')
    df_ped.to_csv('%s.ped'%output_prefix, sep='\t', index=False, header=None)

def ped2bed(args):
    """
    %prog ped_prefix

    Convert plink ped/map to binary bed/bim/fam format using Plink
    """
    p = OptionParser(ped2bed.__doc__)
    p.add_option('--disable_slurm', default=False, action="store_true",
                 help='add this option to disable converting commands to slurm jobs')
    p.add_slurm_opts(job_prefix=ped2bed.__name__)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    ped_prefix, = args
    cmd_header = 'ml plink'
    cmd = 'plink --noweb --file %s --make-bed --out %s' % (ped_prefix, ped_prefix)
    print('cmd on HCC:\n%s\n%s' % (cmd_header, cmd))

    cmd_local = '%s --noweb --file %s --make-bed --out %s' % (plink, ped_prefix, ped_prefix)
    print('cmd on local desktop:\n%s\n'%cmd_local)
    
    if not opts.disable_slurm:
        put2slurm_dict = vars(opts)
        put2slurm_dict['cmd_header'] = cmd_header
        put2slurm([cmd], put2slurm_dict)

def IndePvalue(args):
    """
    %prog IndePvalue bed_prefix output_fn

    Estimate number of idenpendent SNPs using GEC
    """
    p = OptionParser(IndePvalue.__doc__)
    p.add_option('--disable_slurm', default=False, action="store_true",
                 help='add this option to disable converting commands to slurm jobs')
    p.add_slurm_opts(job_prefix=IndePvalue.__name__)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    bed_prefix, output_fn = args
    cmd = 'java -Xmx18g -jar %s --noweb --effect-number --plink-binary %s --genome --out %s' % (GEC, bed_prefix, output_fn)
    print('cmd:\n%s\n' % cmd)

    if not opts.disable_slurm:
        put2slurm_dict = vars(opts)
        put2slurm_dict['memory'] = 20000
        put2slurm([cmd], put2slurm_dict)

def hmpSingle2Double(args):
    """
    %prog hmpSingle2Double input_single_hmp 
    convert single type hmp file to double type hmp file
    """
    p = OptionParser(hmpSingle2Double.__doc__)
    _, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    inputhmp, = args
    outputhmp = Path(inputhmp).name.replace('.hmp', '_db.hmp')

    hmp = ParseHmp(inputhmp)
    df_hmp = hmp.AsDataframe()
    df_hmp.to_csv(outputhmp, sep='\t', index=False, na_rep='NA')
    print('Done! check output %s...'%outputhmp)

def Info(args):
    """
    %prog Info input_hmp
    get basic info for a hmp file
    """
    p = OptionParser(Info.__doc__)
    _, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    inputhmp, = args
    hmp = ParseHmp(inputhmp)

    print('Genotype type: %s'%hmp.type)
    print('Number of samples: {val:,}'.format(val=hmp.numSMs))
    print('Number of SNPs: {val:,}'.format(val=hmp.numSNPs))
    print('Sample names: \n  %s'%'\n  '.join(hmp.SMs))

def MAFs(args):
    """
    %prog MAFs input_hmp 

    calculate MAF for all SNPs in hmp
    """
    p = OptionParser(MAFs.__doc__)
    _, args = p.parse_args(args)

    if len(args) == 0:
        sys.exit(not p.print_help())
    inputhmp, = args
    outputcsv = Path(inputhmp).name.replace('.hmp', '.maf.csv')
    hmp = ParseHmp(inputhmp)
    with open(outputcsv, 'w') as f:
        pbar = tqdm(hmp.MAFs, total=hmp.numSNPs, desc='get MAF', position=0)
        for i, maf in pbar:
            f.write('%s\n'%maf)
            pbar.set_description('calculating chromosome %s'%i.split()[2])
    print('Done! check output %s...'%(outputcsv))

def sortHmp(args):
    """
    %prog sortHmp input_hmp 
    Sort hmp based on chromosome order and position. Can also try tassel:
     'run_pipeline.pl -Xms16g -Xmx18g -SortGenotypeFilePlugin -inputFile in_fn -outputFile out_fn -fileType Hapmap'
    """
    p = OptionParser(sortHmp.__doc__)
    _, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    inputhmp, = args
    outputhmp = Path(inputhmp).name.replace('.hmp', '_sorted.hmp')

    hmp = ParseHmp(inputhmp)
    df_sorted_hmp = hmp.AsDataframe(needsort=True)
    df_sorted_hmp.to_csv(outputhmp, sep='\t', index=False, na_rep='NA')
    print('Done! check output %s...'%outputhmp)

if __name__ == "__main__":
    main()
