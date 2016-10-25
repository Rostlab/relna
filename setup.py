from setuptools import setup
from setuptools import find_packages


def readme():
    with open('README.md', encoding='utf-8') as file:
        return file.read()

setup(
    name='relna',
    version='0.2.0',
    description='Relation Extraction Pipeline for Transcription Factor and Gene or Gene Product relations',
    long_description=readme(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Topic :: Text Processing :: Linguistic'
    ],
    keywords='svm relation extraction nlp natural language ner transcription factor gene product',
    url='https://github.com/Rostlab/relna',
    author='Ashish Baghudana, Juan Miguel Cejuela',
    author_email='i@juanmi.rocks',

    include_package_data=True,
    packages=find_packages(exclude=['tests']),
    zip_safe=False,

    test_suite='pytest-runner',
    setup_requires=['pytest'],

    dependency_links=[
        'https://github.com/Rostlab/nalaf/tree/feature/Experimental#egg=nalaf'
    ],

    install_requires=[
        # 'nalaf',
        'spacy',
        'ujson'  # It should be included with spacy, AFAIK
    ]
)
