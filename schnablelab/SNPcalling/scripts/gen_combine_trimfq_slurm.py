header='''#!/bin/sh
#SBATCH --time=10:00:00          # Run time in hh:mm:ss
#SBATCH --mem-per-cpu=5000       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=%s
#SBATCH --error=./%s.err
#SBATCH --output=./%s.out

%s
'''

from glob import glob
from pathlib import Path

fns = glob('*_trimed.Unpaired.fastq')
for fn in fns:
    sm = '_'.join(fn.split('_')[0:-1])
    fn_out = '%s_trim.Unpaired.fastq'%sm
    if Path(fn_out).exists():
        print('%s exists, skip...'%fn_out)
    else:
        cmd = "python -m schnablelab.SNPcalling.Postprocess combineFQ '%s_*npaired*.fastq' %s"%(sm, fn_out)
        print(cmd)
        with open('combine_%s.slurm'%(sm), 'w') as f:
            f.write(header%(sm, sm, sm, cmd))
