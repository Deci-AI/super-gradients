#!/usr/bin/env bash

# Copy relevant files from root to docs
cp ./CONTRIBUTING.md docs/source
cp ./LICENSE.md docs/source
#cp ./README.md docs/welcome.md

# Copying the assets to the docs
cd docs && echo "Copying assets..." && cp -r ./assets ./build/html/assets

# Compiling
make clean html && \
 echo "Copying assets..." && \
 cp -r ./assets ./build/html/assets  && \
 echo && \
 echo "The docs are ready at $(pwd)/build/html"

# Starting an http server
cd build/html && python -m http.server 8080