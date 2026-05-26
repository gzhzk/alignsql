"""AlignSQL: NL2SQL post-training pipeline from SFT to RL."""

from setuptools import find_packages, setup

setup(
    name="alignsql",
    version="0.2.0",
    description="NL2SQL post-training: SFT → DPO → RL",
    author="gzhzk",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pyarrow",
        "tqdm",
    ],
    extras_require={
        "dev": ["pytest", "ruff"],
    },
)
