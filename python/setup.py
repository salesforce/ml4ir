from setuptools import find_namespace_packages, setup
import yaml
from itertools import chain


optional_requirements_spec = yaml.safe_load(open("optional_requirements.yaml"))


def load_required_dependencies():
    """Fetch required depedencies as specified in requirements.txt"""
    with open("requirements.txt") as f:
        required_deps = f.read().splitlines()

    return required_deps


def getReadMe():
    """Fetch readme for the project"""
    with open("README.md", "r") as f:
        long_description = f.read()
    return long_description


setup(
    name="ml4ir",
    packages=find_namespace_packages(include=["ml4ir.*"]),
    version="0.1.16",
    description="Machine Learning libraries for Information Retrieval",
    long_description=getReadMe(),
    long_description_content_type="text/markdown",
    author="Search Relevance, Salesforce",
    author_email="searchrelevancyscrumteam@salesforce.com",
    url="https://www.salesforce.com/",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
    ],
    include_package_data=True,
    license="ASL 2.0",
    python_requires=">=3.9",
    install_requires=load_required_dependencies(),
    extras_require=optional_requirements_spec
)
