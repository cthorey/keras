from setuptools import setup
from setuptools import find_packages


setup(name='Keras',
      version='1.2.0',
      description='Deep Learning for Python',
      author='Clement Thorey',
      author_email='clement.thorey@gmail.com',
      url='https://github.com/xihelm/keras',
      license='MIT',
      install_requires=['theano', 'pyyaml', 'six'],
      extras_require={
          'h5py': ['h5py'],
          'visualize': ['pydot-ng'],
      },
      packages=find_packages())
