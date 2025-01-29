from os.path import join, dirname

from setuptools import setup, find_packages


def get_version():
    fname = join(dirname(__file__), "src/brain_twt/__version__.py")
    with open(fname) as f:
        ldict = {}
        code = f.read()
        exec(code, globals(), ldict)  # version defined here
        return ldict['version']


package_name = "brain_twt"

setup(name=package_name,
      version=get_version(),
      description='',
      long_description=open('README.md').read().strip(),
      author='',
      author_email='',
      url='',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      py_modules=[package_name],
      install_requires=[
        'networkx==2.8.7',
        'numpy==1.22.2',
        'torch>=1.13.0',
        'pandas==1.4.2',
        'gensim==4.3.1',
        'node2vec==0.4.6',
        'matplotlib>=3.5.2',
        'scikit-learn>=1.1.2',
        'scipy==1.8.1',
        'python-dateutil>=2.8.2',
        'datasets==2.3.2',
        'tokenizers>=0.13.2',
        'transformers>=4.25.1',
        'stellargraph==1.2.1',
        'chardet==5.2.0',
      ],
      extras_require={
          'dev': [
              'ipykernel',
              'mypy',
              'autopep8',
              'pytest',
              'pytest-cov'
          ],
          'test': [
              'pytest',
              'pytest-cov'
          ]
      },
      license='Private',
      zip_safe=False,
      keywords='',
      classifiers=[''],
      package_data={
          package_name: ['py.typed'],
      }
      )
