from setuptools import setup, find_packages

setup(
    name='rl-drl-safeguard',
    version='1.0.dev0',
    description=' ',
    url='',
    author=' ',
    author_email='examle@gmail.com',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ],

    keywords='deep reinforcement learning',
    packages=find_packages(exclude=['docs', 'scripts', 'tests*']),
    install_requires=['gymnasium>=0.28.1', 'PyYAML~=6.0.2', 'docopt~=0.6.2', 'tqdm~=4.66.4', 'setuptools~=75.3.0',
                      'highway-env>=1.8.2', 'stable_baselines3>=2.3.2'],
    tests_require=['pytest'],
    extras_require={
        'dev': ['scipy'],
    },
    entry_points={
        'console_scripts': [],
    },
)
