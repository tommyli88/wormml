from setuptools import setup, find_packages

setup(
    name='wormml',
    version='1.0.0',
    description='Cross-camera C. elegans worm counting pipeline',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'ultralytics>=8.0.0',
        'opencv-python-headless>=4.8.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'matplotlib>=3.7.0',
        'tqdm>=4.65.0',
        'PyYAML>=6.0',
        'Pillow>=9.5.0',
    ],
)
