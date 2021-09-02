#!/usr/bin/env python

# data needs to be in this structure workflow to work with hyperspectral images:
# - data (does not have to be called data)
#   - [plant folder name].zip
#   - [plant folder name].zip
#   ....

# Example of data value: 'data': '/work/csesd/pnnguyen/schnablelab/HCCtools/plantTraitsGCWorkflow/test_data/*'

file_paths = {'data': '/work/csesd/pnnguyen/schnablelab/HCCtools/plantTraitsGCWorkflow/test_data/*',
              'model': 'fold3_model_4_300_0.0005.h5'
}

use_anonymous = True
