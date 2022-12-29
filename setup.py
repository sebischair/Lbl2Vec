from distutils.util import convert_path

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

main_ns = {}
ver_path = convert_path('lbl2vec/_version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

ver_path = convert_path('requirements.txt')
with open(ver_path) as ver_file:
    base_packages = ver_file.read().splitlines()

setuptools.setup(
    name='lbl2vec',
    version=main_ns['__version__'],
    url='https://github.com/sebischair/Lbl2Vec',
    license='BSD 3-Clause "New" or "Revised" License',
    author='Tim Schopf',
    author_email='tim.schopf@tum.de',
    description='Lbl2Vec learns jointly embedded label, document and word vectors to retrieve documents with predefined topics from an unlabeled document corpus.',
    long_description=long_description,
    long_description_content_type='text/markdown',
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
    install_requires=base_packages,
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires='>=3.8',
    data_files=[('requirements', ['requirements.txt'])],
)
