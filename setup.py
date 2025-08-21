from setuptools import setup, find_packages

setup(
    name='sllama-cli',  # The name of your package
    version='0.1.0',    # The current version of your package
    author='Sammy Lord', # Replace with your name
    author_email='python.projects@sllord.info', # Replace with your email
    description='A simple frontend for llama.cpp CLI, similar to Ollama.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    py_modules=['sllama'], # This tells setuptools that sllama.py is a module
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7', # Specify your Python version
        'License :: OSI Approved :: MIT License', # Or another appropriate license
        'Operating System :: OS Independent',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    # This is crucial for creating a command-line entry point.
    # 'sllama' is the command name that will be created.
    # 'sllama:main' means it will call the 'main' function inside 'sllama.py'.
    entry_points={
        'console_scripts': [
            'sllama = sllama:main',
        ],
    },
    python_requires='>=3.7', # Minimum Python version required
    # No external dependencies are needed for sllama.py itself,
    # so install_requires is empty or omitted.
    install_requires=['requests'],
)

