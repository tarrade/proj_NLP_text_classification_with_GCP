from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['apache-beam[gcp]==2.16.0',
                     'beautifulsoup4==4.8.1',
                     'spacy==2.1.8',
                     'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz'
                     ]

setup(
    name='preprocessing_v1',
    version='0.1',
    author = 'F. Tarrade',
    author_email = 'fabien.tarrade@gmail.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Classification of Stackoverflow post using NLP on GCP'
)