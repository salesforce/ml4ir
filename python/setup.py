from setuptools import find_namespace_packages, setup

def load_dependencies():
    with open('requirements.txt') as f:
        required = f.read().splitlines()
    return required

setup(
    name="ml4ir",
    packages=find_namespace_packages(include=["ml4ir.*"]),
    version="0.0.1",
    description="Machine Learning libraries for Information Retrieval",
    author="Search Relevance, Salesforce",
    author_email="searchrelevancyscrumteam@salesforce.com ",
    data_files=[("ml4ir/build", ["ml4ir/build/Dockerfile"])],
    license="ASL 2.0",
    install_requires=load_dependencies()
)
