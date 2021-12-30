#!/usr/bin/env bash

# Copy relevant files from root to docs
cp ./CONTRIBUTING.md docs/
cp ./LICENSE.md docs/
#cp ./README.md docs/welcome.md
# TODO: switch all 'docs/assets' with 'assets/' in welcome.md to support images, etc. using 'sed' utility

# Copying the assets to the docs
cd docs
mkdir -p ./build/html/assets && cp -r ./assets ./build/html/

# Compiling
make html

# Starting an http server
cd build/html && python -m http.server 8080