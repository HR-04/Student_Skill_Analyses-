from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirement(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
        
    return requirements


setup(
    name='FirstMLProject',
    version='0.1dev',
    author='Harini',
    author_email='harinirengarajan203@gmail.com',
    packages=find_packages(),
    install_requires=get_requirement('Requirements.txt'))