#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ["torch==1.3.1", "transformers==2.4.1", "pyyaml==5.1.2"]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Jiang Chaodi",
    author_email='929760274@qq.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="train nlp models much more easily",
    entry_points={
        'console_scripts': [
            'fst=fst2.main:entry',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=history,
    include_package_data=True,
    keywords='fst2',
    name='fst2',
    packages=find_packages(include=['fst2', 'fst2.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/superjcd/fst2',
    version='0.1.4',
    zip_safe=False,
)
