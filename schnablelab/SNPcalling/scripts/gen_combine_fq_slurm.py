header='''#!/bin/sh
#SBATCH --time=167:00:00          # Run time in hh:mm:ss
#SBATCH --mem-per-cpu=5000       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=%s
#SBATCH --error=./%s.err
#SBATCH --output=./%s.out

%s
'''

from glob import glob
from pathlib import Path

pfx_r1 = glob('*_R1_1.fastq')
pfx_r2 = glob('*_R2_1.fastq')
pfx_un = glob('*_Unpaired_1.fastq')

for i in [pfx_r1, pfx_r2, pfx_un]:
    for j in i:
        pat = '_'.join(j.split('_')[0:2])
        fn_out = '%s.fastq'%pat
        if Path(fn_out).exists():
            print('%s exists, skip...'%fn_out)
        else:
            cmd = "python -m schnablelab.SNPcalling.Postprocess combineFQ '%s_*.fastq' %s"%(pat, fn_out)
            print(cmd)
            with open('combine_%s.slurm'%(pat), 'w') as f:
                f.write(header%(pat, pat, pat, cmd))
