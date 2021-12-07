from setuptools import setup, find_packages

setup(
    name="bertVectorizer",
    packages=find_packages(exclude=["notebooks", "docs"]),
    version='beta',
    author='Ivan José dos Reis Filho, Luiz Henrique Dutra Martins',
    author_email='ivanfilhoreis@gmail.com, luizmartins.uemg@gmail.com',
    description='convert a collection of raw documents to a matrix extracted from BERT resources',

    long_description_content_type='text/markdown',
    url='https://github.com/ivanfilhoreis/bertVectorizer',
    keywords='BERT vectorizer',
    classifiers=[
        'Programming Language :: Python',
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires='>=3.6',
)
