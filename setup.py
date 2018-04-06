from setuptools import setup, find_packages

setup(
    name='rtPGM',
    version='0.0.1',
    author='Ruben Van Parys',
    author_email='ruben.vanparys@kuleuven.be',
    description=('rtPGM - real-time Proximal Gradient Method for fast embedded linear MPC'),
    license='LGPLv3',
    keywords='MPC PGM first-order methods',
    url='https://github.com/meco-group/rtPGM',
    packages=find_packages(),
    test_suite='nose.collector',
    tests_require=['nose>=1.0'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
)
