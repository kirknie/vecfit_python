# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 16:12:51 2018

@author: kirknie
"""

from setuptools import setup

setup(name='vecfit',
      version='0.1',
      description='Generate csv file from OTA raw file, also plot near and far field pattern.',
      url='https://github.com/kirknie/vecfit_python',
      author='Ding Nie',
      author_email='kirknie@gmail.com',
      license='',
      packages=['vecfit'],
      install_requires=[
            'numpy',
            'matplotlib',
            'scikit-rf',
      ],
      zip_safe=False)
