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
from typing import List
from recommonmark.transform import AutoStructify

# Set python project root
sys.path.insert(0, os.path.abspath("../../python/"))

# The master toctree document
master_doc = "index"

# -- Project information -----------------------------------------------------

project = "ml4ir"
copyright = "2020, Search Relevance (Salesforce.com, Inc.)"
author = "Search Relevance (Salesforce.com, Inc.)"

# The full version, including alpha/beta/rc tags
release = "0.2.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions: List = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "recommonmark",
]  # noqa: E501

# Add any paths that contain templates here, relative to this directory.
templates_path: List = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: List = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme: str = "default"

# Title appended to <title> tag of individual pages
html_title: str = "ml4ir"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path: List = ["_static"]

# Overriding default theme with custom CSS for text wrapping bug on tables
html_context = {
    "css_files": ["_static/theme_overrides.css"],
}


def setup(app):
    app.add_config_value(
        "recommonmark_config", {"enable_eval_rst": True}, True
    )  # noqa E501
    app.add_transform(AutoStructify)


# Use both class definition doc and constructor doc for
# generating sphinx docs for python classes
autoclass_content = "both"
autodoc_member_order = "bysource"
