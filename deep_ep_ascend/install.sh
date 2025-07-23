#!/bin/bash
set -e

# Remove old dist file, build, and install
rm -rf dist

bash ./build.sh
python setup.py bdist_wheel
pip uninstall dist/*.whl
pip install dist/*.whl

