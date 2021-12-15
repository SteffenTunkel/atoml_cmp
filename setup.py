from setuptools import setup
import atoml_cmp

setup(
    name='atoml_cmp',
    version=atoml_cmp.__version__,
    packages=['atoml_cmp'],
    description='Comparative testing of machine learning frameworks based on atoml',
    install_requires=[
	   'pandas',
	   'scikit-learn',
	   'scipy',
	   'matplotlib',
	   'numpy'
	]
)