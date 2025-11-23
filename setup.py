from setuptools import setup, find_packages  

with open("README.md", "r", encoding="utf-8") as fh:  
    long_description = fh.read()  

setup(  
    name='scfocus',  
    version='0.0.4',  
    description='single cell reinforcement learning for focusing',  
    long_description=long_description,  
    long_description_content_type="text/markdown",  
    url='https://github.com/PeterPonyu/scfocus',  
    author='Zeyu Fu',  
    author_email='fuzeyu99@126.com',  
    license='MIT',  
    packages=find_packages(),  
    install_requires=[  
        'scanpy>=1.10.4',
        'torch>=1.13.1',    
        'joblib>=1.2.0',  
        'tqdm>=4.64.1', 
        'streamlit>=1.24.0' 
    ],  
    entry_points={  
        'console_scripts': [  
            'scfocus=scfocus.cli:main',  
        ],  
    },  
    package_data={  
        'scfocus': ['*.py', 'pages/*.py', 'graphic_abstract.png'],  
    },  
    classifiers=[  
        'License :: OSI Approved :: MIT License',  
        'Development Status :: 4 - Beta',  
        'Programming Language :: Python :: 3',  
        'Intended Audience :: Science/Research',  
        'Topic :: Scientific/Engineering :: Bio-Informatics'  
    ],  
    python_requires='>=3.9',  
)
