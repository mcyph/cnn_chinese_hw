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
    version='0.2.0',
    description='A PyTorch convolutional neural network for recognising '
                'online (stroke-based) Chinese (Simplified/Traditional) and '
                'Japanese Kanji handwriting',
    long_description=long_description,
    long_description_content_type='text/markdown',
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
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],

    keywords='input ime handwriting cnn pytorch kanji hanzi chinese',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'svg.path',
        'numpy',
        'torch>=2.1',
    ],
    extras_require={
        # Only needed to export/run the optional ONNX inference graph.
        'onnx': ['onnx', 'onnxruntime'],
        # Only needed for the wxPython demo GUIs.
        'gui': ['wxPython', 'matplotlib', 'pillow'],
    },
    package_data={
        '': ['*.xml', '*.txt', '*.json', '*.pt'],
    },
    include_package_data=True,
    zip_safe=False
)
