from setuptools import find_packages ,setup
from typing import List

## to ignore -e .
HYPEN_E_DOT = '-e .'

## Ip: file path, Op: list of strings(i.e packages)
def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj: #open that file
        requirements=file_obj.readlines()
        ## remove /n present in each package name
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

        return requirements



setup(
    name = 'DiamondPricePrediction',
    version='0.0.1',
    author='Dax',
    author_email='dakshpatel731@gmail.com',
    install_requires = get_requirements('requirements.txt'), ## to install all packages mention in requirements.txt file
    packages=find_packages() ## to find sub modules






)