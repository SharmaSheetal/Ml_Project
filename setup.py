from setuptools import find_packages,setup

HYPHEN_DOT_E="e ."
def get_requirements(filepath):
    with open(filepath) as fileobj:
        requirements=fileobj.readlines()
    requirements=[req.replace("\n","") for req in requirements]
    if HYPHEN_DOT_E in requirements:
        requirements.remove(HYPHEN_DOT_E)
    return requirements


setup(
    name="Practice Ml Project",
    author="Sheetal",
    author_email="sharmasheetal9798@gmail.com",
    password=find_packages(),
    install_require=get_requirements("requirements.txt")
)