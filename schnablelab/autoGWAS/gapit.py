# -*- coding: UTF-8 -*-

"""
Generate the R script file and the slurm job file for performing GAPIT. Find more details in GAPIT manual at <http://www.zzlab.net/GAPIT/gapit_help_document.pdf>
"""

import os.path as op
import sys
from schnablelab.apps.base import ActionDispatcher, OptionParser
from schnablelab.apps.header import Slurm_header
from schnablelab.apps.header import Gapit_header
from schnablelab.apps.natsort import natsorted

def main():
    actions = (
        ('cMLM', 'Perform GWAS using compressed mixed linear model'),
        ('SUPER', 'Perform GWAS using SUPER'),
            )
    p = ActionDispatcher(actions)
    p.dispatch(globals())

def cMLM(args):
    """
    %prog cMLM pheno(with header, tab delimited) geno_prefix(GM and GD prefix) PCA Kinship
    
    Run automated GAPIT compressed mixed linear model
    """
    p = OptionParser(cMLM.__doc__)
    p.set_slurm_opts(array=False)
    opts, args = p.parse_args(args)

    if len(args) == 0:
        sys.exit(not p.print_help())
    
    pheno, geno_prefix, PCA, Kinship = args
    mem = '.'.join(pheno.split('.')[0:-1])
    f1 = open('%s.cMLM.R'%mem, 'w')
    #print(Gapit_header)
    gapit_cmd = Gapit_header%(pheno,geno_prefix,geno_prefix,PCA,Kinship,mem)
    f1.write(gapit_cmd)
    
    f2 = open('%s.cMLM.slurm'%mem, 'w')
    h = Slurm_header
    h += 'module load R/3.3\n'
    header = h%(opts.time, opts.memory, opts.prefix, opts.prefix, opts.prefix)
    f2.write(header)
    cmd = 'R CMD BATCH %s.cMLM.R\n'%mem
    f2.write(cmd)
    f1.close()
    f2.close()
    print('R script %s.cMLM.R and slurm file %s.cMLM.slurm has been created, you can sbatch your job file.'%(mem, mem))

def SUPER(args):
    """
    %prog ...
    Run automated GAPIT using SUPER
    """
    pass

if __name__ == "__main__":
    main()
