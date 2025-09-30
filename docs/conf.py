# Sphinx configuration for profinder documentation
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "profinder"
author = "Jesse Cusack"
release = "0.2.4"
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/oceancascades/profinder",
    "use_repository_button": True,
    # Uncomment to add issues and edit page buttons:
    # "use_issues_button": True,
    # "use_edit_page_button": True,
}
nbsphinx_execute = "always"
