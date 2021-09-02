#!/bin/bash

. /util/opt/lmod/lmod/init/profile
export -f module
module use /util/opt/hcc-modules/Common/

CURRENT=${PWD}
cd "$1"
zip -r "$2" *
mv "$2.zip" $CURRENT/"$2".zip
cd $CURRENT