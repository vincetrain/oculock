from setuptools import setup

setup(
   name="oculock",
   version="1.0",
   description="Package for Python 3.9 used for security purposes via eye recognition",
   authors="Vincent Tran, An Ha, Joshua Triffo, Ronniel Ghande",
   packages=["oculock"],
   install_requires=["opencv-python", "tensorflow", "numpy", "Pillow"],
)