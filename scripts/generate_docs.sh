#!/usr/bin/env bash

# Copy relevant files from root to documentation
cp ./CONTRIBUTING.md documentation/source
cp ./LICENSE.md documentation/source
#cp ./README.md documentation/welcome.md

# Copying the assets to the documentation
cd documentation && echo "Copying assets..." && cp -r ./assets ./build/html/assets

# Compiling
# Copying to documentation/ to docs/ for github pages
cd .. && sphinx-build -b html documentation/source docs && \
 echo "Copying assets..." && \
 cp -r documentation/assets docs/assets  && \
 echo && \
 echo "The docs are ready at $(pwd)/build/html"

# Starting an http server
cd docs && python -m http.server 8080