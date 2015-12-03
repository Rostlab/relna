from setuptools import setup
from setuptools import find_packages


def readme():
    with open('README.md') as file:
        return file.read()

setup(
    name='relna',
    version='0.1.0',
    description='Relation Extraction Pipeline for Transcription Factor and Gene or Gene Product relations',
    long_description=readme(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.4',
        'Topic :: Text Processing :: Linguistic'
    ],
    keywords='svm relation extraction nlp natural language ner transcription factor gene product',
    url='https://github.com/ashishbaghudana/relna',
    author='Ashish Baghudana',
    author_email='abaghudana@rostlab.org',
    license='MIT',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'nala',
        'nltk',
        'beautifulsoup4',
        'requests',
        'spacy',
        'progress'
        ],
    include_package_data=True,
    zip_safe=False,
    test_suite='nose.collector',
    setup_requires=['nose>=1.0'],
)
