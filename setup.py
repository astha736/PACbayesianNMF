from distutils.core import setup

setup(
    name='PACbayesianNMF',
    version='0.1.1',
    author='Astha Gupta',
    author_email='astha736@gmail.com',
    maintainer = "Benjamin Guedj",
    maintainer_email = "benjamin.guedj@inria.fr",
    packages=['pacbayesiannmf'],
    keywords = "PAC-Bayesian Non-Negative Matrix Factorization Quasi-Bayesian Block Gradient Descent",
    scripts=['bin/packageTest1.py','bin/packageTest2.py','bin/packageTest3.py','bin/train.txt','LICENSE.txt','CHANGES.txt'],
    license='GPLv3',
    description='Implementing NMF with a PAC-Bayesian approach relying upon block gradient descent',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy >= 1.11.0",
    ],
    classifiers = [ 'Programming Language :: Python :: 2.7']
)