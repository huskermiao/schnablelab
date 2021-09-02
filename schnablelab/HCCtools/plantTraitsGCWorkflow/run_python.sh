#!/bin/bash

. /util/opt/lmod/lmod/init/profile
export -f module
module use /util/opt/hcc-modules/Common/

module load anaconda
conda activate /home/csesd/pnnguyen/.conda/envs/sd

export PYTHONPATH=/work/csesd/pnnguyen:/work/csesd/pnnguyen/run/greenhouseEI:$PYTHONPATH
python3 "$@"

conda deactivate
