# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst', encoding='utf-8') as f:
    readme = f.read()

with open('LICENSE', encoding='utf-8') as f:
    license = f.read()

setup(
    name='nodeclass',
    version='0.0.3',
    description='Node classification package',
    long_description=readme,
    long_description_content_type='text/x-rst',
    author='Miguel Romero',
    author_email='romeromiguelin@gmail.com',
    url='https://github.com/migueleci/node_classification',
    license='MIT',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
      'matplotlib==3.4.1',
      'pandas==1.2.4',
      'xgboost==1.4.1',
      'numpy==1.20.2',
      'tqdm==4.60.0',
      'networkx==2.5.1',
      'seaborn==0.11.1',
      'imbalanced_learn==0.8.0',
      'scipy==1.6.3',
      'node2vec==0.4.3',
      'imblearn==0.0',
      'python_igraph==0.9.6',
      'scikit_learn==0.24.2'
    ],
    classifiers=[
      'Development Status :: 1 - Planning',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: BSD License',
      'Operating System :: POSIX :: Linux',
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 3.6',
    ],
    python_requires=">=3.6",
)
