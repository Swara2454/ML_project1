from setuptools import find_packages, setup
from typing import List

Hypen_e_dot='-e .'
def get_requirnments(file_path:str)->List[str]:
    requirnments=[]
    with open(file_path) as file_obj:
        requirnments=file_obj.readlines()
        requirnments=[req.replace("\n"," ") for req in requirnments]

        if Hypen_e_dot in requirnments:
            requirnments.remove(Hypen_e_dot)

    return requirnments

setup(
    name='ML_project',
    version='0.0.1',
    author='Swarangi',
    author_email='swarangigurav1@gmail.com',
    packages=find_packages(),
    install_requires=get_requirnments('requirnments.txt')
)