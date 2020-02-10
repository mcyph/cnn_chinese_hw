from setuptools import setup, find_packages
from codecs import open
from os import path
from os.path import join

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
# TODO: Write a specific reStructedText version!
with open(join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='cnn_chinese_hw',
    version='0.1.0',
    description='A convolutional neural network using Keras for recognising '
                'Chinese (Simplified/Traditional) and Japanese Kanji',
    long_description=long_description,
    url='https://github.com/mcyph/cnn_chinese_hw',
    author='Dave Morrissey',
    author_email='20507948+mcyph@users.noreply.github.com',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: '
            'GNU Lesser General Public License v2 or later (LGPLv2+)',

        'Natural Language :: Chinese (Simplified)',
        'Natural Language :: Chinese (Traditional)',
        'Natural Language :: Japanese',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    keywords='input ime handwriting cnn keras kanji hanzi chinese',
    packages=find_packages(),
    install_requires=[
        #'tensorflow',
        ''
    ],
    zip_safe=False
)
