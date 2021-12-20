import numpy as np
from setuptools import find_packages, Extension, setup

from Cython.Build import cythonize


def read(path):
    with open(path, 'r') as f:
        return f.read()


long_description = read('README.md')

include_dirs = ['fbgbp/seq/src', np.get_include()]
extra_compile_args = ['-Wno-unused-variable', '-Wno-unused-function', '-O3']
to_cythonize = [
    Extension(
        'fbgbp.grid_belief_propagation', [
            'fbgbp/grid_belief_propagation.pyx',
            'fbgbp/src/grid_belief_propagation.cpp',
        ],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language='c++',
    ),
]

requirements = read('requirements.txt').strip().split('\n')
setup(
    name='fbgbp',
    version='0.0.3',
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
    setup_requires=requirements,
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
