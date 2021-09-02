# -*- coding: UTF-8 -*-

"""
Generate the R script file and the slurm job file for performing FarmCPU. Find more details in FarmCPU manual at <http://www.zzlab.net/FarmCPU/FarmCPU_help_document.pdf>
"""

import os.path as op
import sys
from pathlib import Path
from schnablelab.apps.base import ActionDispatcher, OptionParser
from schnablelab.apps.headers import Slurm_header, FarmCPU_header
from schnablelab.apps.natsort import natsorted

def main():
    actions = (
        ('farmcpu', 'Perform GWAS using FarmCPU (muti-loci mixed model)'),
        ('pdf2png', 'convert pdf image to png format'),
            )
    p = ActionDispatcher(actions)
    p.dispatch(globals())

def pdf2png(args):
    """
    %prog pdf2png dir_in dir_out

    Run imagemagick to convert pdf to png
    """
    p = OptionParser(pdf2png.__doc__)
    p.set_slurm_opts(jn=True)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())

    in_dir, out_dir, = args
    out_path = Path(out_dir)
    if not out_path.exists():
        sys.exit('%s does not exist...')
    dir_path = Path(in_dir)
    pdfs = dir_path.glob('*.pdf')
    for pdf in pdfs:
        print(pdf)
        prf = pdf.name.replace('.pdf', '')
        png = pdf.name.replace('.pdf', '.png')
        header = Slurm_header%(100, 15000, prf, prf, prf)
        header += 'ml imagemagick\n'
        cmd = 'convert -density 300 {} -resize 25% {}/{}\n'.format(pdf, out_path, png)
        header += cmd
        with open('pdf2png.%s.slurm'%prf, 'w') as f:
            f.write(header)

def farmcpu(args):
    """
    %prog farmcpu pheno(with header, tab delimited) geno_prefix(GM(chr must be nums) and GD prefix) PCA

    Run automated FarmCPU
    """
    p = OptionParser(farmcpu.__doc__)
    p.set_slurm_opts(jn=True)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())

    pheno, geno_prefix, PCA = args
    mem = '.'.join(pheno.split('/')[-1].split('.')[0:-1])
    f1 = open('%s.FarmCPU.R'%mem, 'w')
    farmcpu_cmd = FarmCPU_header%(pheno,geno_prefix,geno_prefix,PCA,mem)
    f1.write(farmcpu_cmd)

    f2 = open('%s.FarmCPU.slurm'%mem, 'w')
    h = Slurm_header
    h += 'module load R/3.3\n'
    header = h%(opts.time, opts.memory, opts.prefix, opts.prefix, opts.prefix)
    f2.write(header)
    cmd = 'R CMD BATCH %s.FarmCPU.R'%mem
    f2.write(cmd)
    f1.close()
    f2.close()
    print('R script %s.FarmCPU.R and slurm file %s.FarmCPU.slurm has been created, you can sbatch your job file.'%(mem, mem))

if __name__ == "__main__":
    main()
