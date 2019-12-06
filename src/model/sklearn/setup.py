from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['google-cloud-bigquery==1.19.0',
                     'cloudml-hypertune',
                     'psutil']

#setup(name='trainer',
#      version='1.0',
#      description='Stackoverflow with scikit-learn',
#      author='Google',
#      author_email='nobody@google.com',
#      license='Apache2',
#      packages=['trainer'],
#      ## WARNING! Do not upload this package to PyPI
#      ## BECAUSE it contains a private key
#      package_data={'': ['privatekey.json']},
#      install_requires=[
#          'pandas-gbq==0.3.0',
#          'urllib3',
#          'google-cloud-bigquery==0.29.0',
#          'cloudml-hypertune'
#      ],
#      zip_safe=False)

setup(
    name='trainer',
    version='0.1',
    author = 'F. Tarrade',
    author_email = 'fabien.tarrade@gmail.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Classification of Stackoverflow post using scikit-learn on GCP'
)