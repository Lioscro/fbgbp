import platform

import numpy as np
from Cython.Build import cythonize
from setuptools import find_packages, Extension, setup


def read(path):
    with open(path, 'r') as f:
        return f.read()


long_description = read('README.md')

include_dirs = ['fbgbp/src', np.get_include()]
extra_compile_args = [
    '-Wno-unused-variable',
    '-Wno-unused-function',
    '-O3',
    '-std=c++11',
]
if platform.system().lower() == 'darwin':
    extra_compile_args.append('-stdlib=libc++')
to_cythonize = [
    Extension(
        'fbgbp.binary_belief_propagation', [
            'fbgbp/binary_belief_propagation.pyx',
            'fbgbp/src/binary_belief_propagation.cpp',
        ],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language='c++',
    ),
    Extension(
        'fbgbp.binary_grid_belief_propagation', [
            'fbgbp/binary_grid_belief_propagation.pyx',
            'fbgbp/src/binary_belief_propagation.cpp',
        ],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language='c++',
    ),
]

requirements = read('requirements.txt').strip().split('\n')
setup(
    name='fbgbp',
    version='0.2.0',
    url='https://github.com/Lioscro/fbgbp',
    author='Kyung Hoi (Joseph) Min',
    author_email='phoenixter96@gmail.com',
    description='Optimized belief propagation on a grid MRF with binary states.',  # noqa
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='belief-propagation markov-random-field',
    python_requires='>=3.6',
    license='MIT',
    packages=find_packages(exclude=('tests', 'docs')),
    zip_safe=False,
    include_package_data=True,
    install_requires=requirements,
    ext_modules=cythonize(
        to_cythonize,
        language_level='3',
        compiler_directives={'embedsignature': True}
    ),
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Cython',
    ],
)
