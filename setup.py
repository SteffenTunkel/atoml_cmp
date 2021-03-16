from setuptools import setup

setup(
    name='atoml_cmp',
    version='0.0.1',
    packages=['atoml_cmp'],
    description='Comparative testing of machine learning frameworks based on atoml',
    install_requires=[
	   'pandas',
	   'sklearn',
	   'scipy',
	   'matplotlib',
	   'seaborn',
	   'numpy'
	]
)