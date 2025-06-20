from setuptools import setup, find_packages

# It's a good practice to read long descriptions outside the setup function
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='GraphReasoning',
    version='0.2.0',
    author='Markus J. Buehler',
    author_email='mbuehler@mit.edu',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'networkx',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'scipy',
        'transformers>=4.39',
        'torch',
        'torchvision',
        'torchaudio',
        'huggingface_hub',
        'bitsandbytes',
        'peft',
        'accelerate',
        'langchain',
        'langchain-community',
        'llama-index',
        'guidance',
        'markdown2',
        'pdfkit',
        'pdfminer.six',
        'pytesseract',
        'pyvis',
        'yachalk',
        'ipython',
        'tqdm',
        'powerlaw',
        'python-louvain',
        'community',
        # Note: uuid is part of Python standard library
        # Note: wkhtmltopdf requires system-level installation
    ],
    description='GraphReasoning: Use LLM to reason over graphs, combined with multi-agent modeling.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/lamm-mit/GraphReasoning',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11'
    ],
    python_requires='>=3.10',
)
