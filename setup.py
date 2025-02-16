from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

from setuptools import setup, find_packages

install_requires = [
    "tqdm>=4.48.2"
]

setup_requires = []

extras_require = {"hyperopt": ["hyperopt==0.2.5"]}

classifiers = ["License :: OSI Approved :: MIT License"]

long_description = (
    "RecBole-GNN is a library built upon PyTorch and RecBole for reproducing and developing "
    "recommendation algorithms based on graph neural networks (GNNs). The library provides "
    "implementations of various GNN-based models, facilitating research and experimentation in "
    "graph-based recommendation systems.\n\n"

    "This library supports algorithms across three major categories:\n"
    "- General Recommendation: Utilizing user-item interaction graphs to model user preferences.\n"
    "- Sequential Recommendation: Leveraging session and sequence graphs for personalized content recommendations.\n"
    "- Social Recommendation: Incorporating social network structures to enhance recommendation accuracy.\n\n"

    "For the investigations conducted in this thesis, UltraGCN has been implemented, offering an efficient, "
    "scalable solution for graph-based collaborative filtering. Additionally, a GraphDatasetEvaluator has been "
    "developed to streamline the evaluation process, enabling in-depth analysis of GNN-based recommendation models.\n\n"

    "These enhancements allow for more comprehensive benchmarking, better insight into graph-based recommendation "
    "dynamics, and improved evaluation methodologies.\n\n"

    "For more information, visit the RecBole-GNN GitHub repository: "
    "https://github.com/mkhe93/RecBole-GNN/tree/mkhe/thesis"
)

# Readthedocs requires Sphinx extensions to be specified as part of
# install_requires in order to build properly.
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if on_rtd:
    install_requires.extend(setup_requires)

setup(
    name="recbole_gnn",
    version="2025.17.2",  # please remember to edit recbole_gnn/__init__.py in response, once updating the version
    description="The RecBole-GNN version used for conducting investigations in my thesis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mkhe93/RecBole-GNN/tree/mkhe/thesis",
    author="Markus Hoefling",
    author_email="markus.hoefling01@gmail.com",
    packages=[package for package in find_packages() if package.startswith("recbole_gnn")],
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False,
    classifiers=classifiers,
)
