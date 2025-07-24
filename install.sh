#!/bin/bash
set -e

CURRENT_DIR=$(pwd)
PYTHON_DIR="$CURRENT_DIR"/python

# Remove old dist file, build, and install
bash ./build.sh

cd $PYTHON_DIR
rm -rf dist
python setup.py bdist_wheel
pip uninstall dist/*.whl
pip install dist/*.whl
cd -
