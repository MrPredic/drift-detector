from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="drift-detector",
    version="2.0.0",
    author="Mr. Predic",
    description="Behavioral drift detection for LLM agents - detect when your agent starts acting unpredictably",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MrPredic/drift-detector",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Quality Assurance",
    ],
    python_requires=">=3.8",
    keywords=["drift-detection", "llm", "agents", "monitoring", "behavioral-analysis", "langchain", "crewai", "agent-health", "model-monitoring"],
    install_requires=[
        "python-dotenv>=1.0.0",
        # requests<2.33 has 3 known CVEs (CVE-2024-35195, CVE-2024-47081, CVE-2026-25645)
        "requests>=2.33.0",
    ],
    extras_require={
        "ui": [
            "fastapi>=0.80.0",
            "uvicorn>=0.17.4",
        ],
        "langchain": [
            "langchain>=0.1.0,<0.4.0",
            "langchain-openai>=0.0.0",
            "langchain-google-genai>=0.1.0",
        ],
        "crewai": [
            "crewai>=0.3.0",
        ],
        "all": [
            "fastapi>=0.80.0",
            "uvicorn>=0.17.4",
            "langchain>=0.1.0,<0.4.0",
            "langchain-openai>=0.0.0",
            "langchain-google-genai>=0.1.0",
            "crewai>=0.3.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ]
    },
)
