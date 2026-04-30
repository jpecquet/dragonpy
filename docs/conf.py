from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

project = "Dragonfly Flight Control"
author = "Jean Pecquet"
copyright = f"{date.today().year}, {author}"

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinxcontrib.bibtex",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

html_theme = "furo"
html_title = "Dragonfly Flight Control Docs"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["force_dark.js"]
html_show_sphinx = False
html_show_sourcelink = False

mathjax3_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"], ["$", "$"]],
        "displayMath": [["\\[", "\\]"], ["$$", "$$"]],
    },
    "options": {
        # Furo sets mathjax_ignore on the content section, which blocks
        # MathJax from processing $...$ in raw HTML (e.g. figure captions).
        # Disable the ignore class so MathJax processes the full page.
        # Code blocks are safe: MathJax skips <pre>, <code>, <script>, etc.
        "ignoreHtmlClass": "mathjax_ignore_disabled",
    },
}

bibtex_bibfiles = ["references.bib"]

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "colon_fence",
    "deflist",
    "substitution",
]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_mock_imports = [
    "numpy",
    "matplotlib",
    "h5py",
    "scipy",
    "mpl_toolkits",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
if os.environ.get("DOCS_ENABLE_INTERSPHINX", "0") != "1":
    intersphinx_mapping = {}

todo_include_todos = True
suppress_warnings = [
    "bibtex.duplicate_citation",
]
