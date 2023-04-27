# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# Imports for auto documentation
import super_gradients
import super_gradients.common as common
import super_gradients.training as training

__all__ = ["super_gradients", "common", "training"]

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------

project = "SuperGradients"
copyright = "2021, SuperGradients team"
author = "SuperGradients team"

# The full version, including alpha/beta/rc tags
release = "3.0.9"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc']
extensions = []

autodoc_default_options = {
    "member-order": "bysource",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

pygments_style = "sphinx"
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

extensions.append("sphinx.ext.todo")
extensions.append("sphinx.ext.autodoc")
extensions.append("sphinx.ext.autosummary")
extensions.append("sphinx.ext.intersphinx")
extensions.append("sphinx.ext.mathjax")
extensions.append("sphinx.ext.viewcode")
extensions.append("sphinx.ext.graphviz")
extensions.append("sphinxcontrib.napoleon")
extensions.append("myst_parser")

autosummary_generate = True

# html_theme = 'default'
# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    "custom.css",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
