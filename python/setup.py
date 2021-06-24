from setuptools import find_namespace_packages, setup


def load_dependencies():
    with open("requirements.txt") as f:
        required = f.read().splitlines()
    return required


def getReadMe():
    with open("README.md", "r") as f:
        long_description = f.read()
    return long_description


setup(
    name="ml4ir",
    packages=find_namespace_packages(include=["ml4ir.*"]),
    version="0.1.3",
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
    install_requires=load_dependencies(),
    extras_require={
        "all": ["pyspark==3.0.1"]  # Used by ml4ir.base.io.spark_io
    }
)
