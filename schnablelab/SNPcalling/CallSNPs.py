# -*- coding: UTF-8 -*-

"""
Call SNPs on HTS data using GATK, Freebayes.
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import os.path as op
from pathlib import Path
from subprocess import run
from .base import find_sm
from schnablelab.apps.Tools import GenDataFrameFromPath
from schnablelab.apps.base import ActionDispatcher, OptionParser, put2slurm

def main():
    actions = (
        ('genGVCFs', 'generate gvcf for each sample using GATK HaplotypeCaller'),
        ('aggGVCFs', 'aggregate GVCF files to a GenomicsDB datastore for each genomic interval'),
        ('genoGVCFs', 'create the raw VCFs from GenomicsDB datastores'),
        ('freebayes', 'call SNPs using freebayes'),
        ('samtools', 'call SNPs using samtools'),
        ('gatk', 'call SNPs using gatk'),
)
    p = ActionDispatcher(actions)
    p.dispatch(globals())

def genGVCFs(args):
    """
    %prog genGVCFs ref.fa bams.csv region.txt out_dir

    run GATK HaplotypeCaller in GVCF mode. 
    one g.vcf file for one smaple may contain multiple replicates
    args:
        ref.fa: reference sequence file
        bams.csv: csv file containing all bam files and their sample names
        region.txt: genomic intervals defined by each row to speed up GVCF calling. 
            example regions: Chr01, Chr01:1-100
        out_dir: where the gVCF files save to
    """
    p = OptionParser(genGVCFs.__doc__)
    p.add_option('--disable_slurm', default=False, action="store_true",
                help='do not convert commands to slurm jobs')
    p.add_slurm_opts(job_prefix=genGVCFs.__name__)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    ref, bams_csv, region_txt, out_dir, = args
    out_dir_path = Path(out_dir)
    if not out_dir_path.exists():
        print(f'output directory {out_dir_path} does not exist, creating...')
        out_dir_path.mkdir()
    
    regions = []
    with open(region_txt) as f:
        for i in f:
            regions.append(i.rstrip())
    
    mem = int(opts.memory)//1024

    df_bam = pd.read_csv(bams_csv)

    # check if bai files exist
    for bam in df_bam['fnpath']:
        if not Path(bam+'.bai').exists():
            print(f'no index file for {bam}...')
            sys.exit('Index your bam files first!')

    cmds = []
    for sm, grp in df_bam.groupby('sm'):
        print(f'{grp.shape[0]} bam files for sample {sm}')
        input_bam = '-I ' + ' -I '.join(grp['fnpath'].tolist())
        for region in regions:
            output_fn = f'{sm}_{region}.g.vcf'
            cmd = f"gatk --java-options '-Xmx{mem}g' HaplotypeCaller -R {ref} "\
		f"{input_bam} -O {out_dir_path/output_fn} --sample-name {sm} "\
		f"--emit-ref-confidence GVCF -L {region}"
            cmds.append(cmd)
    
    cmd_sh = '%s.cmds%s.sh'%(opts.job_prefix, len(cmds))
    pd.DataFrame(cmds).to_csv(cmd_sh, index=False, header=None)
    print(f'check {cmd_sh} for all the commands!')

    cmd_header = 'ml gatk4/4.1'
    if not opts.disable_slurm:
        put2slurm_dict = vars(opts)
        put2slurm_dict['cmd_header'] = cmd_header
        put2slurm(cmds, put2slurm_dict)

def aggGVCFs(args):
    """
    %prog aggGVCFs input_dir out_dir 

    aggregate GVCF files to a GenomicsDB datastore for each genomic interval
    args:
        intput_dir: the directory containing all gvcf files
        out_dir: the output directory. a subdir will be created for each genomic interval
    """
    p = OptionParser(aggGVCFs.__doc__)
    p.add_option('--gvcf_fn_pattern', default='*.g.vcf',
                help = 'file extension of gvcf files')
    p.add_option('--sm_re_pattern', default=r"^P[0-9]{3}[_-]W[A-Z][0-9]{2}[^a-z0-9]", 
                help = 'the regular expression pattern to pull sample name from filename')
    p.add_option('--gatk_tmp_dir', default='./gatk_tmp',
                help = 'temporary directory for genomicsDBImport')
    p.add_option('--disable_slurm', default=False, action="store_true",
                help='do not convert commands to slurm jobs')
    p.add_slurm_opts(job_prefix=aggGVCFs.__name__)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    in_dir, out_dir, = args
    in_dir_path = Path(in_dir)
    out_dir_path = Path(out_dir)
    if not in_dir_path.exists():
        sys.exit(f'input directory {in_dir_path} does not exist!')
    if not out_dir_path.exists():
        print(f'output directory {out_dir_path} does not exist, creating...')
        out_dir_path.mkdir()
    tmp_dir = Path(opts.gatk_tmp_dir)
    if not tmp_dir.exists():
        print('tmp directory does not exist, creating...')
        tmp_dir.mkdir()
    
    # The -Xmx value the tool is run with should be less than the total amount of physical memory available by at least a few GB
    mem = int(opts.memory)//1024-2

    # set the environment variable TILEDB_DISABLE_FILE_LOCKING=1
    try:
        os.environ['TILEDB_DISABLE_FILE_LOCKING']
    except KeyError:
        sys.exit('Set the environment variable TILEDB_DISABLE_FILE_LOCKING=1 before running gatk!')

    df = GenDataFrameFromPath(in_dir_path, pattern=opts.gvcf_fn_pattern)
    df['interval'] = df['fn'].apply(lambda x: x.split('.')[0].split('_')[1])
    prog = re.compile(opts.sm_re_pattern)
    df['sm'] = df['fn'].apply(lambda x: find_sm(x, prog))

    cmds = []
    for interval, grp in df.groupby('interval'):
        interval_dir = out_dir_path/(interval.replace(':','_'))
        # The --genomicsdb-workspace-path must point to a non-existent or empty directory
        if interval_dir.exists():
            if len(interval_dir.glob('*')) != 0:
                sys.exit(f'{interval_dir} is not an empty directory!')
        gvcf_map = str(interval) + '.map'
        print(f'{grp.shape[0]} gvcf files found for interval {interval}, generating the corresponding map file {gvcf_map}...')
        grp[['sm', 'fnpath']].to_csv(gvcf_map, header=None, index=False, sep='\t')

        cmd = f"gatk --java-options '-Xmx{mem}g -Xms{mem}g' GenomicsDBImport "\
	f"--sample-name-map {gvcf_map} --genomicsdb-workspace-path {interval_dir} "\
	f"--batch-size 50 --intervals {interval} "\
        f"--reader-threads {opts.ncpus_per_node} --tmp-dir {tmp_dir}"
        cmds.append(cmd)

    cmd_sh = '%s.cmds%s.sh'%(opts.job_prefix, len(cmds))
    pd.DataFrame(cmds).to_csv(cmd_sh, index=False, header=None)
    print(f'check {cmd_sh} for all the commands!')

    cmd_header = 'ml gatk4/4.1'
    if not opts.disable_slurm:
        put2slurm_dict = vars(opts)
        put2slurm_dict['cmd_header'] = cmd_header
        put2slurm(cmds, put2slurm_dict)

def genoGVCFs(args):
    """
    %prog genoGVCFs ref.fa genomicDB_dir out_dir 

    create the raw VCFs from GenomicsDB datastores
    args:
        ref.fa: the reference sequence fasta file
        genomicDB_dir: the root directory of genomicDB workspace
        out_dir: where the vcf files will be saved
    """
    p = OptionParser(genoGVCFs.__doc__)
    p.add_option('--gatk_tmp_dir', default='./gatk_tmp',
                help = 'temporary directory to use')
    p.add_option('--disable_slurm', default=False, action="store_true",
                help='do not convert commands to slurm jobs')
    p.add_slurm_opts(job_prefix=genoGVCFs.__name__)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    ref, db_dir, out_dir, = args
    out_dir_path = Path(out_dir)
    if not out_dir_path.exists():
        print(f'output directory {out_dir_path} does not exist, creating...')
        out_dir_path.mkdir()
    mem = int(opts.memory)//1024-1

    cmds = []
    for db in Path(db_dir).glob('*'):
        if db.is_dir():
            region = db.name
            vcf_fn = f"{region}.vcf.gz"
            cmd = f"gatk --java-options '-Xmx{mem}g' GenotypeGVCFs "\
            f"-R {ref} -V gendb://{db} -O {out_dir_path/vcf_fn} --tmp-dir={opts.gatk_tmp_dir}"
            cmds.append(cmd)
    cmd_sh = '%s.cmds%s.sh'%(opts.job_prefix, len(cmds))
    pd.DataFrame(cmds).to_csv(cmd_sh, index=False, header=None)
    print(f'check {cmd_sh} for all the commands!')

    cmd_header = 'ml gatk4/4.1'
    if not opts.disable_slurm:
        put2slurm_dict = vars(opts)
        put2slurm_dict['cmd_header'] = cmd_header
        put2slurm(cmds, put2slurm_dict)
    
def gatk(args):
    """
    %prog gatk ref.fa bam_list.txt region.txt out_dir

    run GATK HaplotypeCaller
    """
    p = OptionParser(gatk.__doc__)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    ref, bams, regions, out_dir, = args
    out_path = Path(out_dir)
    if not out_path.exists():
        sys.exit('%s does not exist...')
    with open(bams) as f:
        inputs = ''.join(['-I %s \\\n'%(i.rstrip()) for i in f])
    with open(regions) as f:
        for reg in f:
            reg = reg.strip()
            if ':0-' in reg:
                reg = reg.replace(':0-', ':1-')
            reg_fn = reg.replace(':','_')
            reg_fn_vcf = '%s.gatk.vcf'%reg_fn
            reg_fn_vcf_path = out_path/reg_fn_vcf
            cmd = "gatk --java-options '-Xmx13G' HaplotypeCaller \\\n-R %s -L %s \\\n%s-O %s"%(ref, reg, inputs, reg_fn_vcf_path)
            header = Slurm_header%(165, 15000, reg_fn, reg_fn, reg_fn)
            header += 'ml gatk4/4.1\n'
            header += cmd
            with open('%s.gatk.slurm'%reg_fn, 'w') as f1:
                f1.write(header)

def freebayes(args):
    """
    %prog freebayes region.txt ref.fa bam_list.txt out_dir

    create freebayes slurm jobs for each splitted region defined in region.txt file
    """
    p = OptionParser(freebayes.__doc__)
    p.add_option('--max_depth', default=10000,
            help = 'cites where the mapping depth higher than this value will be ignored')
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    region, ref, bams,out_dir, = args
    out_path = Path(out_dir)
    if not out_path.exists():
        sys.exit('%s does not exist...')

    with open(region) as f:
        for reg in f:
            reg = reg.strip()
            reg_fn = reg.replace(':','_')
            reg_fn_vcf = '%s.fb.vcf'%reg_fn
            reg_fn_vcf_path = out_path/reg_fn_vcf
            cmd = 'freebayes -r %s -f %s -C 1 -F 0.05 -L %s -u -n 2 -g %s > %s\n'%(reg, ref, bams,opts.max_depth, reg_fn_vcf_path)
            header = Slurm_header%(165, 50000, reg_fn, reg_fn, reg_fn)
            header += 'ml freebayes/1.3\n'
            header += cmd
            with open('%s.fb.slurm'%reg_fn, 'w') as f1:
                f1.write(header)
            print('slurm files %s.fb.slurm has been created'%reg_fn)

if __name__ == "__main__":
    main()