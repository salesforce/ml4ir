from setuptools import find_namespace_packages, setup
import yaml
from itertools import chain


optional_requirements_spec = yaml.safe_load(open("optional_requirements.yaml"))


def load_required_dependencies():
    with open("requirements.txt") as f:
        required = f.read().splitlines()

    # Remove optional requirements from required dependencies
    optional_requirements = set(chain(*optional_requirements_spec.values()))
    return [package for package in required if package not in optional_requirements]


def getReadMe():
    with open("README.md", "r") as f:
        long_description = f.read()
    return long_description


setup(
    name="ml4ir",
    packages=find_namespace_packages(include=["ml4ir.*"]),
    version="0.1.10",
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
    python_requires=">=3.7",
    install_requires=load_required_dependencies(),
    extras_require=optional_requirements_spec
)
