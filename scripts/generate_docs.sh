#!/usr/bin/env bash

# Copy relevant files from root to documentation
cp ../CONTRIBUTING.md ../documentation/source/CONTRIBUTING.md
cp ../LICENSE.md ../documentation/source/LICENSE.md
cp ../README.md ../documentation/source/welcome.md

# Editing the links to contain assets/ instead of docs/assets in the urls
sed -i 's/docs\/assets/assets/g' ../documentation/source/welcome.md

# Copying the assets to the documentation
#cd ../documentation && echo "Copying assets..."

# Compiling
# Copying to documentation/ to docs/ for github pages
cd .. && sphinx-build -b html documentation/source docs && \
 echo "Copying assets..." && \
 cp -r documentation/assets docs/ && \
 echo "Successfully generated docs"

touch docs/.nojekyll

# Starting an http server, mimicking github pages root directory (static HTML serving from docs/)
#cd docs && python -m http.server 8080
