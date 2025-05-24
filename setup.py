from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="local-rag",
    version="0.1.0",
    author="Alexander Woodward",
    author_email="",  # Add your email if desired
    description="A local RAG (Retrieval Augmented Generation) system using Mistral-7B and ChromaDB",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/neurodyn/local-rag-llm-chat.git",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.1.1",
            "pytest-asyncio>=0.23.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "local-rag=app.main:main",  # Assuming you have a main() function in app.main
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.txt", "*.md"],
    },
) 