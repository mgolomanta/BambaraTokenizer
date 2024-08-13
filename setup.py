from setuptools import setup, find_packages


setup(
    name='BambaraTokenizer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'nltk',
        'numpy',
    ],
    description='A package for tokenizing and processing Bambara text',
    author='mgolomanta',
    author_email='pelengana1@example.com',
    url='https://github.com/mgolomanta/BambaraTokenizer',
)
