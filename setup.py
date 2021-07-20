import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='lbl2vec',
    version='1.0',
    url='https://github.com/sebischair/Lbl2Vec',
    license='BSD 3-Clause "New" or "Revised" License',
    author='Tim Schopf',
    author_email='tim.schopf@tum.de',
    description='Lbl2Vec learns jointly embedded label, document and word vectors to retrieve documents with predefined topics from an unlabeled document corpus.',
    classifiers=[
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Information Analysis",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
        ],
    install_requires=[
            'pandas >= 1.3.0',
            'numpy >= 1.21.0',
            'swifter >= 1.0.9',
            'gensim >= 4.0.1',
            'scikit-learn >= 0.24.2',
            'psutil >= 5.8.0'
        ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires='>=3.8',
)
