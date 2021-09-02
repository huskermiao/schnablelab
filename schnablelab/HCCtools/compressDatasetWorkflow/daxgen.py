#!/usr/bin/env python

import os
import pwd
import sys
import time
import file_paths_config as paths
from glob import glob
from Pegasus.DAX3 import *

# The name of the DAX file is the first argument
if len(sys.argv) != 2:
        sys.stderr.write("Usage: %s DAXFILE\n" % (sys.argv[0]))
        sys.exit(1)
daxfile = sys.argv[1]

USER = pwd.getpwuid(os.getuid())[0]

# Create a abstract dag
dax = ADAG("process")

# Add some workflow-level metadata
dax.metadata("creator", "%s@%s" % (USER, os.uname()[1]))
dax.metadata("created", time.ctime())

#path_to_data = paths.file_paths['data'].replace('*','')
path_list = glob(paths.file_paths['data'])
path_list_index = paths.file_paths['data'].split('/').index('*')

for path in sorted(path_list):
	plant_folder_name = path.split('/')[path_list_index]
	preprocess = Job("zip")
	zip_file_name = plant_folder_name + ".zip"
	preprocess.addArguments(path, plant_folder_name)
	zip_file = File(zip_file_name)
	preprocess.uses(zip_file, link=Link.OUTPUT, transfer=True, register=False)
	dax.addJob(preprocess)

f = open(daxfile, "w")
dax.writeXML(f)
f.close()
