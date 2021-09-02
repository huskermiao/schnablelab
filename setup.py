from setuptools import setup, find_packages

with open('README.md') as f:
    long_des = f.read()

NAME = 'schnablelab'
packages = [NAME] + [
    '.'.join((NAME, x)) for x in find_packages(NAME, exclude=['test*.py'])
]

setup(
    name = 'schnablelab',
    version = '0.1.0',
    description = 'python library for projects in schnablelab',
    long_description = long_des,
    author = 'Chenyong Miao',
    author_email = 'cmiao@huskers.unl.edu',
    packages = packages,
    include_package_data = True,
    package_data={'schnablelab.imputation.data':['*.*']},
    url = 'https://github.com/huskermiao/schnablelab',
    install_requires = [
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'PyVCF'],
    extra_require = {'DL':['tensorflow', 'torch']}
)