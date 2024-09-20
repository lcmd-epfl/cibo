from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="cibo",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            # Add any console scripts here if needed
        ],
    },
    author="Jan Weinreich, Alexandre Schoepfer, Ruben Laplaza, Clémence Corminboeuf, LCMD, EPFL",
    author_email="jan.weinreich@epfl.ch",
    description="Cost-Informed Bayesian Reaction Optimization (CIBO). Bayesian optimization (BO) of reactions becomes increasingly important for advancing chemical discovery. Although effective in guiding experimental design, BO does not account for experimentation costs. For example, it may be more cost-effective to measure a reaction with the same ligand multiple times at different temperatures than buying a new one. We present Cost-Informed BO (CIBO), a policy tailored for chemical experimentation to prioritize experiments with lower costs. In contrast to BO, CIBO finds a cost-effective sequence of experiments towards the global optimum, the “mountain peak”. We envision use cases for efficient resource allocation in experimentation planning for traditional or self-driving laboratories.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lcmd-epfl/cibo",  # Replace with your project URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
