from setuptools import setup, find_packages

setup(
    name="eco-civilization-mdp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.1",
        "numpy>=1.24.3",
        "pettingzoo>=1.24.1",
        "pygame>=2.5.2",
        "langchain>=0.1.0",
        "langchain-community>=0.0.16",
        "langgraph>=0.3.1",
        "openai>=1.3.0",
        "pydantic>=2.5.2",
        "pymongo>=4.6.1",
        "python-dotenv>=1.0.0",
        "aiohttp>=3.9.1",
        "asyncio>=3.4.3",
        "wandb>=0.16.1",
    ],
    extras_require={
        'dev': [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.23.2",
            "black>=23.11.0",
            "isort>=5.12.0",
            "mypy>=1.7.1",
        ]
    },
    python_requires='>=3.9',
    description="A multi-agent environment for studying sustainable growth",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)