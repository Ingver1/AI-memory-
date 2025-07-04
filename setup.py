#!/usr/bin/env python3
"""
Unified Memory System v0.8-alpha – setup configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enterprise-grade memory system with vector search, FastAPI and monitoring.
"""

import os
import sys
from pathlib import Path

from setuptools import setup, find_packages

# Ensure we can import from the package for version info
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "memory_system"))

ROOT = Path(__file__).parent
README = ROOT / "README.md"

# Read version from package
try:
    from memory_system import __version__
    version = __version__
except ImportError:
    version = "0.8.0a0"

# Read long description
long_description = ""
if README.exists():
    try:
        long_description = README.read_text(encoding="utf-8")
    except Exception:
        long_description = "Enterprise-grade memory system with vector search, FastAPI and monitoring."

# --------------------------------------------------------------------------- #
# Production dependencies
# --------------------------------------------------------------------------- #
INSTALL_REQUIRES = [
    # Framework
    "fastapi>=0.105.0,<0.106.0",
    "uvicorn[standard]>=0.25.0,<0.26.0",
    "pydantic>=2.6.0,<3.0.0",
    "pydantic-settings>=2.1.0,<3.0.0",
    
    # Networking
    "httpx>=0.26.0,<0.27.0",
    "anyio>=3.7.1,<4.0.0",
    "requests>=2.32.0,<3.0.0",
    
    # ML / Vector search
    "faiss-cpu>=1.7.4,<2.0.0",
    "sentence-transformers>=2.2.2,<3.0.0",
    "torch>=2.1.0,<3.0.0",
    "numpy>=1.24.0,<2.0.0",
    
    # Storage / security
    "aiosqlite>=0.19.0,<1.0.0",
    "cryptography>=42.0.0,<43.0.0",
    "PyJWT>=2.8.0,<3.0.0",
    "passlib[bcrypt]>=1.7.4,<2.0.0",
    "python-multipart>=0.0.6,<1.0.0",
    
    # Monitoring
    "prometheus-client>=0.19.0,<1.0.0",
    "psutil>=5.9.0,<6.0.0",
    
    # Utilities
    "python-dotenv>=1.0.0,<2.0.0",
    "rich>=13.6.0,<14.0.0",
    "typer>=0.9.0,<1.0.0",
    "python-dateutil>=2.8.2,<3.0.0",
    "orjson>=3.9.0,<4.0.0",
]

# GPU optional
GPU_REQUIRES = ["faiss-gpu>=1.7.4,<2.0.0"]

# Development dependencies
DEV_REQUIRES = [
    # Testing
    "pytest>=7.4.0,<8.0.0",
    "pytest-asyncio>=0.21.0,<1.0.0",
    "pytest-cov>=4.1.0,<5.0.0",
    "pytest-mock>=3.12.0,<4.0.0",
    "pytest-benchmark>=4.0.0,<5.0.0",
    "pytest-xdist>=3.3.0,<4.0.0",
    "httpx>=0.26.0,<0.27.0",  # For async testing
    
    # Code Quality
    "ruff>=0.4.0,<1.0.0",
    "black>=24.3.0,<25.0.0",
    "isort>=5.13.0,<6.0.0",
    "mypy>=1.10.0,<2.0.0",
    "types-requests>=2.31.0",
    "types-python-dateutil>=2.8.0",
    "types-orjson>=3.6.0",
    
    # Development Tools
    "pre-commit>=3.7.0,<4.0.0",
    "ipython>=8.23.0,<9.0.0",
    "watchdog>=3.0.0,<4.0.0",
    
    # Documentation
    "mkdocs>=1.5.3,<2.0.0",
    "mkdocs-material>=9.5.10,<10.0.0",
    "mkdocstrings[python]>=0.24.0,<1.0.0",
    
    # Profiling
    "memory-profiler>=0.61.0,<1.0.0",
    "py-spy>=0.3.14,<1.0.0",
]

# --------------------------------------------------------------------------- #
# Setup configuration
# --------------------------------------------------------------------------- #
setup(
    name="unified-memory-system",
    version=version,
    description="Enterprise-grade memory system with vector search, FastAPI and monitoring",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Enhanced Memory Team",
    author_email="team@memory-system.dev",
    url="https://github.com/your-org/unified-memory-system",
    project_urls={
        "Documentation": "https://unified-memory-system.readthedocs.io/",
        "Source": "https://github.com/your-org/unified-memory-system",
        "Tracker": "https://github.com/your-org/unified-memory-system/issues",
        "Changelog": "https://github.com/your-org/unified-memory-system/blob/main/CHANGELOG.md",
    },
    
    # Package configuration
    packages=find_packages(
        include=["memory_system", "memory_system.*"],
        exclude=["tests*", "docs*", "examples*", "scripts*"]
    ),
    include_package_data=True,
    package_data={
        "memory_system": ["py.typed"],
    },
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
        "gpu": GPU_REQUIRES,
        "full": INSTALL_REQUIRES + GPU_REQUIRES + DEV_REQUIRES,
        "test": [
            "pytest>=7.4.0,<8.0.0",
            "pytest-asyncio>=0.21.0,<1.0.0",
            "pytest-cov>=4.1.0,<5.0.0",
            "httpx>=0.26.0,<0.27.0",
        ],
    },
    
    # Console scripts
    entry_points={
        "console_scripts": [
            "unified-memory=memory_system.cli:main",
            "umem=memory_system.cli:main",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Framework :: FastAPI",
        "Framework :: Pydantic",
        "Framework :: Pytest",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Database :: Database Engines/Servers",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Topic :: System :: Monitoring",
        "Typing :: Typed",
    ],
    
    # Keywords
    keywords=[
        "memory", "vector-search", "embeddings", "faiss", "fastapi", 
        "machine-learning", "nlp", "semantic-search", "enterprise", 
        "production", "monitoring", "api", "async", "performance"
    ],
    
    # License
    license="Apache-2.0",
    
    # Additional metadata
    zip_safe=False,
    platforms=["any"],
    
    # Minimum requirements check
    cmdclass={},
)

# --------------------------------------------------------------------------- #
# Post-install validation
# --------------------------------------------------------------------------- #
def validate_installation():
    """Validate that the installation was successful."""
    try:
        import memory_system
        print(f"✅ Successfully installed Unified Memory System v{memory_system.__version__}")
        
        # Check core dependencies
        required_modules = [
            "fastapi", "uvicorn", "pydantic", "numpy", 
            "sentence_transformers", "faiss", "cryptography"
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            print(f"⚠️  Warning: Missing optional dependencies: {', '.join(missing_modules)}")
        else:
            print("✅ All core dependencies are available")
            
    except ImportError as e:
        print(f"❌ Installation validation failed: {e}")
        sys.exit(1)

# Run validation if this script is executed directly
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "install":
    # Add post-install hook
    import atexit
    atexit.register(validate_installation)
