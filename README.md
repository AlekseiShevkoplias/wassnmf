# WassNMF

A Python wrapper for Wasserstein NMF implementation in Julia.

## Prerequisites

1. **Julia Installation**
   - Install Julia 1.6 or later from [https://julialang.org/downloads/](https://julialang.org/downloads/)
   - Add Julia to your system PATH
   - Verify installation by running `julia --version` in your terminal

2. **Python Requirements**
   - Python 3.8 or later
   - pip package manager

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd wassnmf
   ```

2. **Set up Python environment**
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install Python dependencies
   pip install -r requirements.txt
   ```

3. **Set up Julia environment**
   ```bash
   # Navigate to Julia package directory
   cd JuWassNMF
   
   # Start Julia REPL
   julia
   
   # In Julia REPL:
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   # Exit Julia REPL with Ctrl+D
   ```

## Verify Installation

Run the micro test to verify everything is working:
```bash
pytest tests/python/test_micro.py -v
```

## Usage

See `notebooks/example_wrapped.ipynb` for usage examples.

## Project Structure
```
wassnmf/
├── JuWassNMF/              # Julia package
│   ├── Project.toml        # Julia dependencies
│   ├── Manifest.toml       # Julia lockfile
│   └── src/
│       └── JuWassNMF.jl    # Julia implementation
├── src/
│   └── wassnmf/           # Python package
│       ├── __init__.py
│       └── wassnmf.py     # Python wrapper
├── tests/                 # Tests
│   ├── julia/            # Julia tests
│   └── python/           # Python tests
└── notebooks/            # Example notebooks
```

## Common Issues

1. **Julia not found**
   - Make sure Julia is in your system PATH
   - Try running `julia --version` to verify

2. **JuliaCall import errors**
   - Ensure you've activated the Python virtual environment
   - Verify Julia installation is accessible from command line

3. **Julia package errors**
   - Make sure you're in the correct directory when activating Julia project
   - Run `Pkg.status()` in Julia REPL to verify package installation

## Development

To run all tests:
```bash
pytest tests/python -v
```

To run Julia tests:
```bash
cd JuWassNMF
julia --project=. test/runtests.jl
```