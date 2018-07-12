from setuptools import setup, find_packages

REQUIRES = [
    "numpy"
]
TEST_REQUIRES = [
    "numpy", "pytest-cov", "coverage"
]
setup(
        name='sympyle',
        version='0.0.1dev1',
        packages=find_packages(),
        url='https://github.com/harveyslash/sympyle',
        license='GNU GENERAL PUBLIC LICENSE',
        author='Harshvardhan Gupta',
        author_email='theharshvardhangupta@gmail.com',
        description='Simple Automatic Differentiation in Python ',

        install_requires=REQUIRES,
        test_requires=TEST_REQUIRES,
)
