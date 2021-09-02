    # -*- coding: UTF-8 -*-

"""
Prepare fastq files ready for SNP calling
"""
import re
import sys
import subprocess
import numpy as np
import pandas as pd
import os.path as op
from pathlib import Path
from subprocess import run
from .base import find_sm
from schnablelab.apps.natsort import natsorted
from schnablelab.apps.Tools import GenDataFrameFromPath
from schnablelab.apps.base import ActionDispatcher, OptionParser, put2slurm

def main():
    actions = (
        ('fastqc', 'check the reads quality'),
        ('trim_paired', 'quality control on paired reads'),
        ('trim_single', 'quality control on single reads'),
        ('combineFQ', 'combine splitted fastq files'),
        ('pre_ref', 'index the reference genome sequences'),
        ('pre_fqs', 'prepare fastq files read for mapping'),
        ('align_pe', 'paired-end alignment using bwa'),
        ('markdupBam', 'mark potential PRC duplicates in sorted bam files'),
        ('indexBam', 'index bam files'),
        ('pre_bams', 'parse preprocessed bam fils to get the sample names'),
        ('split_fa_region', 'genearte a list of genomic intervals'),
        ('sam2bam', 'convert sam format to bam format'),
        ('sortbam', 'sort bam files'),
        ('bam_list', 'genearte a list of bam files for freebayes -L use'),
    )
    p = ActionDispatcher(actions)
    p.dispatch(globals())

def pre_ref(args):
    """
    %prog pre_ref ref.fa

    index the reference genome sequences using bwa, samtools, and picard tools
    """
    p = OptionParser(pre_ref.__doc__)
    p.add_option('--disable_slurm', default=False, action="store_true",
                help='do not convert commands to slurm jobs')
    p.add_slurm_opts(job_prefix=pre_ref.__name__)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    ref_fn, = args
    ref_fn, ref_dir = Path(ref_fn), Path(ref_fn).parent
    if not ref_fn.exists():
        sys.exit(f'reference file {ref_fn} does not exist!')
    ref_prefix = re.split('.fa|.fasta', ref_fn.name)[0]
    bwa_idx_exs = ('.amb', '.ann', '.bwt', '.pac', '.sa')
    bwa_bool = sum([(ref_dir/(ref_prefix+bie)).exists() for bie in bwa_idx_exs])
    cmds = []
    if bwa_bool !=5:
        print('bwa index does not exist...')
        cmd = f'ml bwa\nbwa index -p {ref_dir/ref_prefix} {ref_fn}'
        cmds.append(cmd)

    if not (ref_dir/(ref_fn.name+'.fai')).exists():
        print('fai index does not exist...')
        cmd = f'ml samtools\nsamtools faidx {ref_fn}'
        cmds.append(cmd)
    
    dict_fn = ref_dir/(ref_prefix+'.dict')
    if not dict_fn.exists():
        print('dict index does not exist...')
        cmd = f'ml gatk4/4.1\ngatk CreateSequenceDictionary -R {ref_fn} -O {dict_fn}'
        cmds.append(cmd)

    if len(cmds)>0:
        if not opts.disable_slurm:
            put2slurm_dict = vars(opts)
            put2slurm(cmds, put2slurm_dict)
        else:
            print('commands running on local:\n%s'%('\n'.join(cmds)))
    else:
        print('All reference index files have already existed!')

def pre_fqs(args):
    """
    %prog pre_fqs dir1 ... output.csv

    parse RG and SM info of all fastq files and get them ready for mapping

    dir1: where fastq files are located
        add more directories if fastq files are located at different directories
    output.csv:
        output csv file containing all parsed fq files
    """
    p = OptionParser(pre_fqs.__doc__)
    p.add_option('--fq_fn_pattern', default='*.fastq.gz',
                help = 'file extension of fastq files')
    p.add_option('--sm_re_pattern', default=r"[^a-z0-9]P[0-9]{3}[_-]W[A-Z][0-9]{2}[^a-z0-9]", 
                help = 'the regular expression pattern to pull sample name from filename')
    opts, args = p.parse_args(args)
    if len(args)==0:
        sys.exit(not p.print_help())
    *fq_dirs, out_csv = args

    tmp_df_ls = []
    for fq_dir in fq_dirs:
        fq_dir = Path(fq_dir)
        if not fq_dir.exists():
            sys.exit(f'{fq_dir} does not exist!')
        tmp_df = GenDataFrameFromPath(fq_dir, pattern=opts.fq_fn_pattern)
        if tmp_df.shape[0]==0:
            sys.exit(f"no fastq files found under '{fq_dir}' directory!")
        print(f'{tmp_df.shape[0]} fastq files found under {fq_dir}!')
        tmp_df_ls.append(tmp_df)
    df = pd.concat(tmp_df_ls)

    prog = re.compile(opts.sm_re_pattern)
    df['sm'] = df['fn'].apply(lambda x: find_sm(x, prog))
    df = df.sort_values(['sm', 'fn']).reset_index(drop=True)
    print(f"Total {df['sm'].unique().shape[0]} samples found!")
    print(df['sm'].value_counts())
    df.to_csv(out_csv, index=False)
    print(f'{out_csv} has been generated')

def align_pe(args):
    """
    %prog align_pe ref_indx_base fq_fns.csv output_dir

    paire-end alignment using bwa.
    args:
        ref_index_base: the prefix of reference index files
        fq_fns.csv: the csv file including parsed fq files from pre_fqs function.
        output_dir: where the generated bam files save to
    """
    p = OptionParser(align_pe.__doc__)
    p.add_option('--disable_slurm', default=False, action="store_true",
                help='do not convert commands to slurm jobs')
    p.add_slurm_opts(job_prefix=align_pe.__name__)
    opts, args = p.parse_args(args)
    if len(args)==0:
        sys.exit(not p.print_help())
    ref_base, fq_csv, output_dir = args
    output_dir = Path(output_dir)
    if not output_dir.exists():
        sys.exit(f'output directory {output_dir} does not exist!')
    df = pd.read_csv(fq_csv)

    df_R1, df_R2 = df[::2], df[1::2]
    if df_R1.shape[0] != df_R2.shape[0]:
        sys.exit('number of R1 and R2 files are not consistent!')

    cmds = []
    for (_,r1), (_,r2) in zip(df_R1.iterrows(), df_R2.iterrows()):
        r1_fn, r2_fn, sm = Path(r1['fnpath']), Path(r2['fnpath']), r1['sm']
        r1_fn_arr, r2_fn_arr = np.array(list(r1_fn.name)), np.array(list(r2_fn.name))
        bools = (r1_fn_arr != r2_fn_arr)
        if bools.sum() != 1:
            print(r1_fn, r2_fn)
            sys.exit('check fq file names!')
        idx = np.argmax(bools)
        prefix = re.split('[-_]R', r1_fn.name[:idx])[0]
        RG = r"'@RG\tID:%s\tSM:%s'"%(sm, sm)
        bam_fn = f'{prefix}.pe.sorted.bam'
        cmd = f"bwa mem -t {opts.ncpus_per_node} -R {RG} {ref_base} {r1_fn} {r2_fn} | samtools sort -@{opts.ncpus_per_node} -o {output_dir/bam_fn} -"
        cmds.append(cmd)
    cmd_sh = '%s.cmds%s.sh'%(opts.job_prefix, len(cmds))
    pd.DataFrame(cmds).to_csv(cmd_sh, index=False, header=None)
    print(f'check {cmd_sh} for all the commands!')

    cmd_header = 'ml bwa\nml samtools'
    if not opts.disable_slurm:
        put2slurm_dict = vars(opts)
        put2slurm_dict['cmd_header'] = cmd_header
        put2slurm(cmds, put2slurm_dict)

def markdupBam(args):
    """
    %prog markdupBam input_dir output_dir

    mark potential PCR duplicates
    output bams will be indexed automatically
    args:
        input_dir: where sorted bam located
        output_dir: where the output rmduped bam shoud save to
    """
    p = OptionParser(markdupBam.__doc__)
    p.add_option('--bam_fn_pattern', default='*.sorted.bam',
                help = 'pattern of bam files')
    p.add_option('--disable_slurm', default=False, action="store_true",
                help='do not convert commands to slurm jobs')
    p.add_slurm_opts(job_prefix=markdupBam.__name__)
    opts, args = p.parse_args(args)
    if len(args)==0:
        sys.exit(not p.print_help())
    in_dir, out_dir = args
    in_dir_path, out_dir_path = Path(in_dir), Path(out_dir)
    if not out_dir_path.exists():
        sys.exit(f'output directory {out_dir_path} does not exist!')
    bams = in_dir_path.glob(opts.bam_fn_pattern)
    cmds = []
    for bam in bams:
        mdup_bam = bam.name.replace('.bam', '.mdup.bam')
        cmd = f'samtools markdup {bam} {out_dir_path/mdup_bam}\nsamtools index {out_dir_path/mdup_bam}'
        cmds.append(cmd)

    cmd_sh = '%s.cmds%s.sh'%(opts.job_prefix, len(cmds))
    pd.DataFrame(cmds).to_csv(cmd_sh, index=False, header=None)
    print(f'check {cmd_sh} for all the commands!')

    cmd_header = 'ml samtools'
    if not opts.disable_slurm:
        put2slurm_dict = vars(opts)
        put2slurm_dict['cmd_header'] = cmd_header
        put2slurm(cmds, put2slurm_dict)

def indexBam(args):
    """
    %prog indexBam dir1 ...

    index bam files using samtools index

    dir1: where bam files are located
        add more directories if bam files are located at different directories
    """
    p = OptionParser(indexBam.__doc__)
    p.add_option('--bam_fn_pattern', default='*.mdup.bam',
                help = 'file extension of preprocessed bam files')
    p.add_option('--disable_slurm', default=False, action="store_true",
                help='do not convert commands to slurm jobs')
    p.add_slurm_opts(job_prefix=indexBam.__name__)
    opts, args = p.parse_args(args)
    if len(args)==0:
        sys.exit(not p.print_help())
    for bam_dir in args:
        bam_dir = Path(bam_dir)
        if not bam_dir.exists():
            sys.exit(f'{bam_dir} does not exist!')
        bams = bam_dir.glob(opts.bam_fn_pattern)
        cmds = [f'samtools index {bam}' for bam in bams]
        cmd_sh = '%s.cmds%s.sh'%(opts.job_prefix, len(cmds))
        pd.DataFrame(cmds).to_csv(cmd_sh, index=False, header=None)
        print(f'check {cmd_sh} for all the commands!')
        
        cmd_header = 'ml samtools'
        if not opts.disable_slurm:
            put2slurm_dict = vars(opts)
            put2slurm_dict['cmd_header'] = cmd_header
            put2slurm(cmds, put2slurm_dict)

def pre_bams(args):
    """
    %prog pre_bams dir1 ... output.csv

    parse SM info of preprocessed bam files for GATK

    dir1: where bam files are located
        add more directories if bam files are located at different directories
    output.csv:
        output csv file containing sample names of bam files
    """
    p = OptionParser(pre_bams.__doc__)
    p.add_option('--bam_fn_pattern', default='*.mdup.bam',
                help = 'file extension of preprocessed bam files')
    p.add_option('--sm_re_pattern', default=r"[^a-z0-9]P[0-9]{3}[_-]W[A-Z][0-9]{2}[^a-z0-9]", 
                help = 'the regular expression pattern to pull sample name from filename')
    opts, args = p.parse_args(args)
    if len(args)==0:
        sys.exit(not p.print_help())
    *bam_dirs, out_csv = args

    tmp_df_ls = []
    for bam_dir in bam_dirs:
        bam_dir = Path(bam_dir)
        if not bam_dir.exists():
            sys.exit(f'{bam_dir} does not exist!')
        tmp_df = GenDataFrameFromPath(bam_dir, pattern=opts.bam_fn_pattern)
        if tmp_df.shape[0]==0:
            sys.exit(f"no bam files found under '{bam_dir}' directory!")
        print(f'{tmp_df.shape[0]} bam files found under {bam_dir}!')
        tmp_df_ls.append(tmp_df)
    df = pd.concat(tmp_df_ls)

    prog = re.compile(opts.sm_re_pattern)
    df['sm'] = df['fn'].apply(lambda x: find_sm(x, prog))
    df = df.sort_values(['sm', 'fn']).reset_index(drop=True)
    print(f"Total {df['sm'].unique().shape[0]} samples found!")
    print(df['sm'].value_counts())
    df.to_csv(out_csv, index=False)
    print(f'check out {out_csv}!')

def bam_list(args):
    """
    %prog bam_list bam_dir out_fn

    genearte a list of bam files for freebayes -L use
    """
    p = OptionParser(bam_list.__doc__)
    opts, args = p.parse_args(args)
    if len(args)==0:
        sys.exit(not p.print_help())
    bam_dir, fn_out, = args
    dir_path = Path(bam_dir)
    bams = sorted(dir_path.glob('*.bam'))
    f = open(fn_out, 'w')
    for bam in bams:
        f.write('%s\n'%bam)
    f.close()

def split_fa_region(args):
    """
    %prog split_fa_region fa.fai out_fn
        fa.fai: index file for the fa file
        out_fn: the output file

    split the whole genome to subset intervals to speed up SNP calling
    """
    p = OptionParser(split_fa_region.__doc__)
    p.add_option('--num_chunk', type='int', default=1,
                help = 'number of chunks in each chromosome')
    p.add_option('--include_contig', default=False, action='store_true',
                help = 'include both chr and contigs')
    opts, args = p.parse_args(args)
    if len(args)==0:
        sys.exit(not p.print_help())
    fai_fn, fn_out, = args
    df = pd.read_csv(fai_fn, header=None, delim_whitespace=True, usecols=[0,1])
    df.columns = ['chr', 'length']
    if not opts.include_contig:
        df = df[df['chr'].apply(lambda x: ('chr' in x) | ('Chr' in x))]

    with open(fn_out, 'w') as f:
        for _, row in df.iterrows():
            chr, length = row['chr'], row['length']
            print(f'spliting {chr}...')
            if opts.num_chunk == 1:
                f.write(chr+'\n')
            else:
                for interval in pd.qcut(np.arange(length), opts.num_chunk, precision=0).categories:
                    st, ed = int(interval.left+2), int(interval.right+1)
                    f.write(f'{chr}:{st}-{ed}\n')
    print(f'Done! check {fn_out}')

def sortbam(args):
    """
    %prog in_dir out_dir
        in_dir: bam files folder
        out_dir: sorted bam files folder

    sort bam files using samtools/0.1 sort function.
    """
    p = OptionParser(sortbam.__doc__)
    opts, args = p.parse_args(args)
    if len(args)==0:
        sys.exit(not p.print_help())
    in_dir, out_dir, = args

    out_path = Path(out_dir)
    if not out_path.exists():
        sys.exit('%s does not exist...')
    dir_path = Path(in_dir)
    bams = dir_path.glob('*.bam')
    for bam in bams:
        prf = bam.name.split('.bam')[0]
        sort_bam = prf+'.sorted'
        sort_bam_path = out_path/sort_bam
        cmd = 'samtools sort %s %s'%(bam, sort_bam_path)
        header = Slurm_header%(100, 15000, prf, prf, prf)
        header += 'ml samtools/0.1\n'
        header += cmd
        with open('%s.sortbam.slurm'%prf, 'w') as f:
            f.write(header)

def sam2bam(args):
    """
    %prog in_dir out_dir
        in_dir: sam files folder
        out_dir: bam files folder

    convert sam to bam using samtools/0.1.
    """
    p = OptionParser(sam2bam.__doc__)
    opts, args = p.parse_args(args)
    if len(args)==0:
        sys.exit(not p.print_help())
    in_dir, out_dir, = args

    out_path = Path(out_dir)
    if not out_path.exists():
        sys.exit('%s does not exist...')
    dir_path = Path(in_dir)
    sams = dir_path.glob('*.sam')
    for sam in sams:
        prf = sam.name.split('.sam')[0]
        bam = prf+'.bam'
        bam_path = out_path/bam
        cmd = 'samtools view -bS %s > %s'%(sam, bam_path)
        header = Slurm_header%(100, 15000, prf, prf, prf)
        header += 'ml samtools/0.1\n'
        header += cmd
        with open('%s.sam2bam.slurm'%prf, 'w') as f:
            f.write(header)

def fastqc(args):
    """
    %prog fastqc in_dir out_dir
        in_dir: the dir where fastq files are located
        out_dir: the dir saving fastqc reports

    generate slurm files for fastqc jobs
    """
    p = OptionParser(fastqc.__doc__)
    p.add_option("--pattern", default = '*.fastq', 
            help="the pattern of fastq files, qutation needed") 
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    in_dir, out_dir, = args

    out_path = Path(out_dir)
    if not out_path.exists():
        sys.exit('%s does not exist...')
    dir_path = Path(in_dir)
    fqs = dir_path.glob(opts.pattern)
    for fq in fqs:
        prf = '.'.join(fq.name.split('.')[0:-1])
        print(prf)
        cmd = 'fastqc %s -o %s'%(str(fq), out_dir)
        header = Slurm_header%(10, 10000, prf, prf, prf)
        header += 'ml fastqc\n'
        header += cmd
        with open('%s.fastqc.slurm'%(prf), 'w') as f:
            f.write(header)

def trim_paired(args):
    """
    %prog trim in_dir out_dir
    quality control on the paired reads
    """
    p = OptionParser(trim_paired.__doc__)
    p.add_option('--pattern_r1', default = '*_R1.fastq',
            help='filename pattern for forward reads')
    p.add_option('--pattern_r2', default = '*_R2.fastq',
            help='filename pattern for reverse reads')
    opts, args = p.parse_args(args)

    if len(args) == 0:
        sys.exit(not p.print_help())
    in_dir,out_dir, = args
    out_path = Path(out_dir)
    if not out_path.exists():
        sys.exit('output dir %s does not exist...'%out_dir)
    r1_fns = glob('%s/%s'%(in_dir, opts.pattern_r1))
    r2_fns = glob('%s/%s'%(in_dir, opts.pattern_r2))
    for r1_fn, r2_fn in zip(r1_fns, r2_fns):
        r1_path = Path(r1_fn)
        r2_path = Path(r2_fn)
        prf = '_'.join(r1_path.name.split('_')[0:-1])+'.PE'
        print(prf)
        r1_fn_out1 = r1_path.name.replace('R1.fastq', 'trim.R1.fastq')
        r1_fn_out2 = r1_path.name.replace('R1.fastq', 'unpaired.R1.fastq')
        r2_fn_out1 = r2_path.name.replace('R2.fastq', 'trim.R2.fastq')
        r2_fn_out2 = r2_path.name.replace('R2.fastq', 'unpaired.R2.fastq')
        cmd = 'java -jar $TM_HOME/trimmomatic.jar PE -phred33 %s %s %s %s %s %s TRAILING:20 SLIDINGWINDOW:4:20 MINLEN:40'%(r1_fn,r2_fn,str(out_path/r1_fn_out1),str(out_path/r1_fn_out2),str(out_path/r2_fn_out1),str(out_path/r2_fn_out2))
        header = Slurm_header%(10, 10000, prf, prf, prf)
        header += 'ml trimmomatic\n'
        header += cmd
        with open('%s.trim.slurm'%(prf), 'w') as f:
            f.write(header)

def trim_single(args):
    """
    %prog trim in_dir out_dir
    quality control on the single end reads
    """
    p = OptionParser(trim_paired.__doc__)
    p.add_option('--pattern', default = '*_Unpaired.fastq',
            help='filename pattern for all single end reads')
    opts, args = p.parse_args(args)

    if len(args) == 0:
        sys.exit(not p.print_help())
    in_dir,out_dir, = args
    out_path = Path(out_dir)
    if not out_path.exists():
        sys.exit('output dir %s does not exist...'%out_dir)
    fns = glob('%s/%s'%(in_dir, opts.pattern))
    for fn in fns:
        fn_path = Path(fn)
        prf = '_'.join(fn_path.name.split('_')[0:-1])+'.SE'
        print(prf)
        fn_out = fn_path.name.replace('Unpaired.fastq', 'trim.Unpaired.fastq')
        cmd = 'java -jar $TM_HOME/trimmomatic.jar SE -phred33 %s %s TRAILING:20 SLIDINGWINDOW:4:20 MINLEN:40'%(fn, str(out_path/fn_out))
        header = Slurm_header%(10, 10000, prf, prf, prf)
        header += 'ml trimmomatic\n'
        header += cmd
        with open('%s.trim.slurm'%(prf), 'w') as f:
            f.write(header)

def combineFQ(args):
    """
    %prog combineFQ pattern(with quotation) fn_out
    """

    p = OptionParser(combineFQ.__doc__)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    fq_pattern, fn_out, = args
    fns = glob(fq_pattern)
    cmd = 'cat %s > %s'%(' '.join(fns), fn_out)
    print(cmd)
    run(cmd, shell=True)

if __name__ == "__main__":
    main()
