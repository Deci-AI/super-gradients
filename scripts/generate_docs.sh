#!/usr/bin/env bash

# Copy relevant files from root to documentation
cp ./CONTRIBUTING.md documentation/source
cp ./LICENSE.md documentation/source
#cp ./README.md documentation/welcome.md

# Copying the assets to the documentation
cd documentation && echo "Copying assets..."

# Compiling
# Copying to documentation/ to docs/ for github pages
cd .. && sphinx-build -b html documentation/source docs && \
 echo "Copying assets..." && \
 cp -r documentation/assets docs/ && \
 echo "Successfully generated docs"

# Starting an http server, mimicking github pages root directory (static HTML serving from docs/)
#cd docs && python -m http.server 8080