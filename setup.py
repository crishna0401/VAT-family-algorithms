from setuptools import setup, find_packages

setup(
    name="VATfamily",
    version="1.0.0",
    license='MIT',
    author="Krishna Murthy",
    author_email="crishna0401@gmail.com",
    description="VAT family algorithms implemented in python",
    url = 'https://github.com/crishna0401/VAT-family-algorithms',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "statistics","scikit-learn","random","scipy"],
    entry_points={"console_scripts": ["cloudquicklabs1 = src.main:main"]},
)