# -*- coding: UTF-8 -*-

"""
Convert GWAS dataset to particular formats for GEMMA, GAPIT, FarmCPU, and MVP.
"""
import sys
import numpy as np
import pandas as pd
import os.path as op
from pathlib import Path
from subprocess import call
from schnablelab.apps.base import ActionDispatcher, OptionParser, put2slurm

# the location of gemma executable file
gemma = op.abspath(op.dirname(__file__)) + '/../apps/gemma'
tassel = op.abspath(op.dirname(__file__)) + '/../apps/tassel-5-standalone/run_pipeline.pl'


def main():
    actions = (
        ('hmp2vcf', 'transform hapmap format to vcf format'),
        ('hmp2bimbam', 'transform hapmap format to BIMBAM format (GEMMA)'),
        ('hmp2numRow', 'transform hapmap format to numeric format in rows(gapit and farmcpu), more memory'),
        ('hmp2numCol', 'transform hapmap format to numeric format in columns(gapit and farmcpu), less memory'),
        ('hmp2MVP', 'transform hapmap format to MVP genotypic format'),
        ('genKinship', 'using gemma to generate centered kinship matrix'),
        ('genPCA', 'using tassel to generate the first N PCs'),
        ('reorgnzTasselPCA', 'reorganize PCA results from TASSEL so it can be used in other software'),
        ('reorgnzGemmaKinship', 'reorganize kinship results from GEMMA so it can be used in other software'),
        ('genGemmaPheno', 'reorganize normal phenotype format to GEMMA'),
        ('ResidualPheno', 'generate residual phenotype from two associated phenotypes'),
        ('combineHmp', 'combine split chromosome Hmps to a single large one'),
    )
    p = ActionDispatcher(actions)
    p.dispatch(globals())

def hmp2vcf(args):
    """
    %prog hmp2vcf input_hmp
    convert hmp to vcf format using tassel
    """
    p = OptionParser(hmp2vcf.__doc__)
    p.add_option('--disable_slurm', default=False, action="store_true",
                 help='add this option to disable converting commands to slurm jobs')
    p.add_slurm_opts(job_prefix=hmp2vcf.__name__)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    hmpfile, = args
    cmd_header = 'ml tassel/5.2'
    cmd = 'run_pipeline.pl -Xms512m -Xmx10G -fork1 -h %s -export -exportType VCF\n' % (hmpfile)
    print('cmd:\n%s\n%s' % (cmd_header, cmd))
    if not opts.disable_slurm:
        put2slurm_dict = vars(opts)
        put2slurm_dict['cmd_header'] = cmd_header
        put2slurm([cmd], put2slurm_dict)

def judge(ref, alt, genosList):
    newlist = []
    for k in genosList:
        if len(set(k)) == 1 and k[0] == ref:
            newlist.append('0')
        elif len(set(k)) == 1 and k[0] == alt:
            newlist.append('2')
        elif len(set(k)) == 2:
            newlist.append('1')
        else:
            sys.exit('genotype error !')
    return newlist

def hmp2bimbam(args):
    """
    %prog hmp2bimbam hmp bimbam_prefix
    Convert hmp genotypic data to GEMMA bimbam files (*.mean and *.annotation).
    """
    p = OptionParser(hmp2bimbam.__doc__)
    opts, args = p.parse_args(args)

    if len(args) == 0:
        sys.exit(not p.print_help())

    hmp, bim_pre = args
    f1 = open(hmp)
    f1.readline()
    f2 = open(bim_pre + '.mean', 'w')
    f3 = open(bim_pre + '.annotation', 'w')
    for i in f1:
        j = i.split()
        rs = j[0]
        try:
            ref, alt = j[1].split('/')
        except:
            print('omit rs...')
            continue
        newNUMs = judge(ref, alt, j[11:])
        newline = '%s,%s,%s,%s\n' % (rs, ref, alt, ','.join(newNUMs))
        f2.write(newline)
        pos = j[3]
        chro = j[2]
        f3.write('%s,%s,%s\n' % (rs, pos, chro))
    f1.close()
    f2.close()
    f3.close()

def hmp2numCol(args):
    """
    %prog hmp numeric_prefix

    Convert hmp genotypic data to numeric format in columns (*.GD and *.GM).
    Memory efficient than numeric in rows
    """
    p = OptionParser(hmp2numCol.__doc__)
    p.add_option('--mode', default='1', choices=('1', '2'),
                 help='specify the genotype mode 1: read genotypes, 2: only AA, AB, BB.')
    opts, args = p.parse_args(args)

    if len(args) == 0:
        sys.exit(not p.print_help())

    hmp, num_pre = args
    f1 = open(hmp)
    header = f1.readline()
    SMs = '\t'.join(header.split()[11:]) + '\n'
    f2 = open(num_pre + '.GD', 'w')
    f2.write(SMs)
    f3 = open(num_pre + '.GM', 'w')
    f3.write('SNP\tChromosome\tPosition\n')
    for i in f1:
        j = i.split()
        rs = j[0]
        try:
            ref, alt = j[1].split('/')
        except:
            print('omit rs...')
            continue
        newNUMs = judge(ref, alt, j[11:], opts.mode)
        newline = '\t'.join(newNUMs) + '\n'
        f2.write(newline)
        pos = j[3]
        chro = j[2]
        f3.write('%s\t%s\t%s\n' % (rs, chro, pos))
    f1.close()
    f2.close()
    f3.close()

def hmp2MVP(args):
    """
    %prog hmp2MVP hmp MVP_prefix

    Convert hmp genotypic data to bimnbam datasets (*.numeric and *.map).
    """
    p = OptionParser(hmp2MVP.__doc__)
    opts, args = p.parse_args(args)

    if len(args) == 0:
        sys.exit(not p.print_help())

    hmp, mvp_pre = args
    f1 = open(hmp)
    f1.readline()
    f2 = open(mvp_pre + '.numeric', 'w')
    f3 = open(mvp_pre + '.map', 'w')
    f3.write('SNP\tChrom\tBP\n')
    for i in f1:
        j = i.split()
        rs = j[0]
        ref, alt = j[1].split('/')[0], j[1].split('/')[1]
        newNUMs = judge(ref, alt, j[11:])
        newline = '\t'.join(newNUMs) + '\n'
        f2.write(newline)
        chro, pos = j[2], j[3]
        f3.write('%s\t%s\t%s\n' % (rs, chro, pos))
    f1.close()
    f2.close()
    f3.close()

def hmp2numRow(args):
    """
    %prog hmp numeric_prefix

    Convert hmp genotypic data to numeric datasets in rows(*.GD and *.GM).
    """
    p = OptionParser(hmp2numRow.__doc__)
    p.add_option('--mode', default='1', choices=('1', '2'),
                 help='specify the genotype mode 1: read genotypes, 2: only AA, AB, BB.')
    opts, args = p.parse_args(args)

    if len(args) == 0:
        sys.exit(not p.print_help())

    hmp, num_pre = args
    f1 = open(hmp)
    f2 = open(num_pre + '.GD', 'w')
    f3 = open(num_pre + '.GM', 'w')

    hmpheader = f1.readline()
    preConverted = []
    header = 'taxa\t%s' % ('\t'.join(hmpheader.split()[11:]))
    preConverted.append(header.split())

    f3.write('SNP\tChromosome\tPosition\n')
    for i in f1:
        j = i.split()
        try:
            ref, alt = j[1].split('/')
        except:
            print('omit rs...')
            continue
        taxa, chro, pos = j[0], j[2], j[3]
        f3.write('%s\t%s\t%s\n' % (taxa, chro, pos))
        newNUMs = judge(ref, alt, j[11:], opts.mode)
        newline = '%s\t%s' % (taxa, '\t'.join(newNUMs))
        preConverted.append(newline.split())
    rightOrder = map(list, zip(*preConverted))
    for i in rightOrder:
        newl = '\t'.join(i) + '\n'
        f2.write(newl)
    f1.close()
    f2.close()

def genKinship(args):
    """
    %prog genKinship genotype.mean

    Calculate kinship matrix file using gemma
    """
    p = OptionParser(genKinship.__doc__)
    p.add_option('--type', default='1', choices=('1', '2'),
                 help='specify the way to calculate the relateness, 1: centered; 2: standardized')
    p.add_option('--out_dir', default='.',
                 help='specify the output dir')
    p.add_option('--disable_slurm', default=False, action="store_true",
                 help='add this option to disable converting commands to slurm jobs')
    p.add_slurm_opts(job_prefix=genKinship.__name__)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    geno_mean, = args
    # generate a fake bimbam phenotype based on genotype
    with open(geno_mean) as f:
        num_SMs = len(f.readline().split(',')[3:])
    mean_prefix = geno_mean.replace('.mean', '')
    tmp_pheno = '%s.tmp.pheno' % mean_prefix
    with open(tmp_pheno, 'w') as f1:
        for i in range(num_SMs):
            f1.write('sm%s\t%s\n' % (i, 20))

    # the location of gemma executable file
    cmd = '%s -g %s -p %s -gk %s -outdir %s -o gemma.centered.%s' \
        % (gemma, geno_mean, tmp_pheno, opts.type, opts.out_dir, Path(mean_prefix).name)
    print('The kinship command:\n%s' % cmd)
    if not opts.disable_slurm:
        put2slurm_dict = vars(opts)
        put2slurm([cmd], put2slurm_dict)

def genPCA(args):
    """
    %prog genPCA input_hmp N

    Generate first N PCs using tassel
    """
    p = OptionParser(genPCA.__doc__)
    p.add_option('--disable_slurm', default=False, action="store_true",
                 help='add this option to disable converting commands to slurm jobs')
    p.add_slurm_opts(job_prefix=genPCA.__name__)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    hmpfile, N, = args
    out_prefix = Path(hmpfile).name.replace('.hmp', '')
    cmd_header = 'ml java/1.8\nml tassel/5.2'
    cmd = 'run_pipeline.pl -Xms28g -Xmx29g -fork1 -h %s -PrincipalComponentsPlugin -ncomponents %s -covariance true -endPlugin -export %s_%sPCA -runfork1\n' % (hmpfile, N, out_prefix, N)
    print('cmd:\n%s\n%s' % (cmd_header, cmd))
  
    if not opts.disable_slurm:
        put2slurm_dict = vars(opts)
        put2slurm_dict['memory'] = 30000
        put2slurm_dict['cmd_header'] = cmd_header
        put2slurm([cmd], put2slurm_dict)

def reorgnzTasselPCA(args):
    """
    %prog reorgnzTasselPCA tasselPCA1

    Reorganize PCA result from TASSEL so it can be used in other software.
    There are three different PC formats:
    gapit(header and 1st taxa column), farmcpu(only header), mvp/gemma(no header, no 1st taxa column)
    """
    p = OptionParser(reorgnzTasselPCA.__doc__)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    pc1, = args
    df = pd.read_csv(pc1, delim_whitespace=True, header=2)
    df1 = df[df.columns[1:]]
    gapit_pca, farm_pca, gemma_pca = pc1 + '.gapit', pc1 + '.farmcpu', pc1 + '.gemmaMVP'
    df.to_csv(gapit_pca, sep='\t', index=False)
    df1.to_csv(farm_pca, sep='\t', index=False)
    df1.to_csv(gemma_pca, sep='\t', index=False, header=False)
    print('finished! %s, %s, %s have been generated.' % (gapit_pca, farm_pca, gemma_pca))

def reorgnzGemmaKinship(args):
    """
    %prog reorgnzGemmaKinship GEMMAkinship hmp

    Reorganize kinship result from GEMMA so it can be used in other software, like GAPIT.
    The hmp file only provides the order of the smaple names.
    """
    p = OptionParser(reorgnzGemmaKinship.__doc__)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    gemmaKin, hmpfile, = args

    f = open(hmpfile)
    SMs = f.readline().split()[11:]
    f.close()
    f1 = open(gemmaKin)
    f2 = open('GAPIT.' + gemmaKin, 'w')
    for i, j in zip(SMs, f1):
        newline = i + '\t' + j
        f2.write(newline)
    f1.close()
    f2.close()
    print("Finished! Kinship matrix file for GEMMA 'GAPIT.%s' has been generated." % gemmaKin)

def genGemmaPheno(args):
    """
    %prog genGemmaPheno dir_in dir_out

    Change the phenotype format so that can be fed to GEMMA (missing value will be changed to NA)
    """
    p = OptionParser(genGemmaPheno.__doc__)
    p.add_option('--pattern', default='*.csv',
                 help='pattern of the normal phenotype files')
    p.add_option('--header', default=True,
                 help='whether a header exist in your normal phenotype file')
    p.add_option('--sep', default='\t', choices=('\t', ','),
                 help='specify the separator in your normal phenotype file')
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    in_dir, out_dir, = args
    out_path = Path(out_dir)
    if not out_path.exists():
        sys.exit('%s does not exist...')
    dir_path = Path(in_dir)
    old_pheno = dir_path.glob(opts.pattern)
    for ophe in old_pheno: 
        df = pd.read_csv(ophe, sep=opts.sep) \
            if opts.header == True \
            else pd.read_csv(ophe, sep=opts.sep, header=None)
        output = ophe.name+'.gemma'
        df.iloc[:, 1].to_csv(out_path/output, index=False, header=False, na_rep='NA')
        print('Finished! %s has been generated.' % output)

def combineHmp(args):
    """
    %prog combineHmp N pattern output
    combine split hmp (1-based) files to a single one. Pattern example: hmp321_agpv4_chr%s.hmp
    """

    p = OptionParser(combineHmp.__doc__)
    p.add_option('--header', default='yes', choices=('yes', 'no'),
                 help='choose whether add header or not')
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    N, hmp_pattern, new_f, = args
    N = int(N)

    f = open(new_f, 'w')

    fn1 = open(hmp_pattern % 1)
    print(1)
    if opts.header == 'yes':
        for i in fn1:
            f.write(i)
    else:
        fn1.readline()
        for i in fn1:
            f.write(i)
    fn1.close()
    for i in range(2, N + 1):
        print(i)
        fn = open(hmp_pattern % i)
        fn.readline()
        for j in fn:
            f.write(j)
        fn.close()
    f.close()

def ResidualPheno(args):
    from scipy.stats import linregress
    import matplotlib.pyplot as plt
    """
    %prog ResidualPheno OriginalPheno(header, sep, name, pheno1, pheno2)

    estimate the residual phenotypes from two origianl phenotypes
    """
    p = OptionParser(downsampling.__doc__)
    opts, args = p.parse_args(args)

    if len(args) == 0:
        sys.exit(not p.print_help())

    myfile, = args
    df = pd.read_csv(myfile)
    pheno1, pheno2 = df.iloc[:, 1], df.iloc[:, 2]

    # plotting
    fig, ax = plt.subplots()
    ax.scatter(pheno1, pheno2, color='lightblue', s=50, alpha=0.8, edgecolors='0.3', linewidths=1)
    slope, intercept, r_value, p_value, std_err = linregress(pheno1, pheno2)
    y_pred = intercept + slope * pheno1
    ax.plot(pheno1, y_pred, 'red', linewidth=1, label='Fitted line')
    text_x = max(pheno1) * 0.8
    text_y = max(y_pred)
    ax.text(text_x, text_y, r'${\mathrm{r^2}}$' + ': %.2f' % r_value**2, fontsize=15, color='red')
    xlabel, ylabel = df.columns[1], df.columns[2]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig('%s_%s_r2.png' % (xlabel, ylabel))

    # find residuals
    df['y_1'] = y_pred
    df['residuals'] = df[df.columns[2]] - df['y_1']
    residual_pheno = df[[df.columns[0], df.columns[-1]]]
    residual_pheno.to_csv('residual.csv', sep='\t', na_rep='NaN', index=False)

if __name__ == '__main__':
    main()
