#!/bin/sh

export LD_LIBRARY_PATH=$(pwd)/venv/lib/python3.8/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
echo "$LD_LIBRARY_PATH"