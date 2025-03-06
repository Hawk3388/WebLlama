from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name='webllama',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    entry_points={
    'console_scripts': [
        'webllama=WebLlama.webllama:main',
        ],
    },
    author='Hawk3388',
    license='MIT',
    python_requires='>=3.10',
)