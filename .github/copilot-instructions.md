# Copilot Instructions for hazen

## Repository Overview

hazen is a command-line tool (Python package) for performing automated analysis of Magnetic Resonance Imaging (MRI) quality assurance (QA) data. It provides quantitative processing and analysis of MRI phantom data, supporting the ACR Large MRI Phantom and the MagNET Test Objects collection of phantoms.

### Key Features
- Signal-to-noise ratio (SNR) measurements
- Spatial resolution analysis
- Slice position and width measurements
- Uniformity analysis
- Ghosting detection
- MR relaxometry (T1/T2)

## Project Structure

```
hazen-wales/
├── hazenlib/           # Main Python package
│   ├── __init__.py    # CLI entry point with main() function
│   ├── tasks/         # Individual QA task implementations
│   ├── ACRObject.py   # ACR phantom handling
│   ├── HazenTask.py   # Base task class
│   ├── utils.py       # Utility functions
│   ├── types.py       # Type definitions
│   └── constants.py   # Constants
├── tests/             # Unit tests and test data
│   ├── data/          # Test DICOM files for various tasks
│   └── test_*.py      # pytest test files
├── docs/              # Sphinx documentation
├── .github/           # GitHub workflows and configurations
└── pyproject.toml     # Project dependencies and configuration
```

## Development Setup

### Requirements
- Python 3.11, 3.12, or 3.13
- `uv` package manager (recommended) or pip
- Virtual environment recommended

### Installation Commands
```bash
# Clone the repository
git clone https://github.com/sbu-physics-mri/hazen-wales.git
cd hazen-wales

# Create and activate virtual environment
python3 -m venv hazen-venv
source hazen-venv/bin/activate

# Install with uv (preferred)
uv sync --group dev

# Or install with pip
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

### Running hazen
```bash
# CLI usage
hazen <task> <folder> [options]

# Example
hazen snr tests/data/snr/Siemens --report

# Run directly without installation
python hazenlib/__init__.py snr tests/data/snr/GE
```

## Testing

### Test Framework
- Use **pytest** for all testing
- Tests are located in `tests/` directory
- Test data is in `tests/data/` organized by task type

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_snr.py

# Run with coverage
pytest --cov=hazenlib tests/

# Run tests with uv
uv run pytest tests/
```

### Writing Tests
- Follow existing test patterns in `tests/test_*.py` files
- Use DICOM test data from `tests/data/` directory
- Include both unit tests and integration tests for tasks
- Test multiple scanner vendors (Siemens, GE, Philips) where applicable

## Code Quality and Linting

### Linting with flake8
```bash
# Check for syntax errors and undefined names
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Check for style warnings (max line length 127)
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Run with uv
uv run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

### Code Style Guidelines
- Maximum line length: 127 characters
- Maximum complexity: 10
- Follow PEP 8 style guide
- Use type hints where appropriate (see `types.py` for common types)
- Add docstrings to functions and classes

## Domain-Specific Knowledge

### MRI QA Concepts
- **DICOM**: Medical imaging file format; hazen uses pydicom for parsing
- **Phantoms**: Physical test objects used for QA (ACR phantom, MagNET objects, Caliber phantom)
- **SNR**: Signal-to-noise ratio, key quality metric
- **ROI**: Region of interest for measurements
- **Slice**: Individual 2D image from MRI acquisition

### Task Implementation
- Each task inherits from `HazenTask` base class
- Tasks process DICOM files and return numerical results
- Optional `--report` flag generates visualization images
- Results can be output in JSON, CSV, or TSV format

### Key Dependencies
- `pydicom`: DICOM file reading and manipulation
- `numpy`: Numerical computations
- `scipy`: Scientific computing (e.g., curve fitting)
- `scikit-image`: Image processing
- `opencv-python`: Computer vision operations
- `matplotlib`: Visualization for reports

## Continuous Integration

### GitHub Actions Workflows
- **tests_development.yml**: Runs on PRs, executes pytest with coverage
- **test_cli.yml**: Tests all CLI commands with example data
- **tests_release.yml**: Runs on releases
- **publish_pypi.yml**: Publishes to PyPI on release
- **publish_docker.yml**: Builds and publishes Docker image

### CI Test Matrix
- Operating Systems: Ubuntu (latest)
- Python Versions: 3.11, 3.12, 3.13
- All tests must pass before merging

## Contributing Guidelines

### Branch and PR Process
1. Create issue describing the change
2. Create branch named `<issue-number>-short-description`
3. Make changes following code quality guidelines
4. Run tests locally: `pytest tests/`
5. Create PR with detailed description
6. Address review feedback
7. Ensure all CI checks pass

### Commit Messages
- Use descriptive commit messages
- Make small, granular commits
- Reference issue numbers where relevant

### Areas for Contribution
1. **Enhancements**: General functionality and performance
2. **Bugfixes**: Issues with existing code
3. **MRI**: New image processing methods or phantom support
4. **DICOM**: File and metadata handling improvements
5. **Documentation**: User guidance and API docs

## Building Documentation

```bash
# Generate API documentation
sphinx-apidoc -o docs/source hazenlib

# Build HTML documentation
cd docs/
make html -f Makefile
```

Documentation is built with Sphinx and hosted on ReadTheDocs.

## Release Process

### Version Management
- Version is set in `hazenlib/_version.py`
- Update version in `CITATION.cff` for releases
- Update `docs/source/contributors.rst` with new contributors

### Docker
- Docker image: `gsttmriphysics/hazen`
- CLI wrapper script: `hazen-app`
- Published automatically on GitHub release

## Important Notes

### File Exclusions (.gitignore)
- Virtual environments (`hazen-venv`, `.venv/`)
- Test artifacts (`pytest.xml`, `.coverage`)
- Build outputs (`dist/`, `build/`, `*.egg-info`)
- Generated reports (`*.png`, `report*`)
- Cache directories (`__pycache__`, `.pytest_cache`)
- Documentation builds (`docs/_build`)

### Scanner Vendor Support
- hazen supports multiple MRI scanner vendors: Siemens, GE, Philips
- Test data includes examples from different vendors
- DICOM metadata can vary by vendor; handle vendor-specific cases

### Output Formats
- Default: Dictionary printed to console
- JSON: `--format json`
- CSV: `--format csv`
- TSV: `--format tsv`
- Report images: `--report [output_directory]`

## Common Patterns

### Adding a New Task
1. Create new file in `hazenlib/tasks/`
2. Implement task function following existing patterns
3. Register task in `hazenlib/__init__.py` CLI
4. Add test data to `tests/data/<task_name>/`
5. Create test file `tests/test_<task_name>.py`
6. Add CLI test to `.github/workflows/test_cli.yml`
7. Update documentation

### Working with DICOM Files
```python
import pydicom

# Read DICOM file
dcm = pydicom.dcmread(filepath)

# Access pixel data
pixel_array = dcm.pixel_array

# Access metadata
manufacturer = dcm.Manufacturer
slice_thickness = dcm.SliceThickness
```

### Task Output Format
Tasks should return a dictionary with descriptive keys:
```python
{
    'snr_measured_value': 173.97,
    'snr_normalised_value': 1698.21
}
```

## Support and Resources

- **Documentation**: https://hazen.readthedocs.io/en/latest/
- **Issues**: https://github.com/sbu-physics-mri/hazen-wales/issues
- **Contributing Guide**: See CONTRIBUTING.md
- **Main Repository**: https://github.com/GSTT-CSC/hazen

When making changes, always:
- Run linting and tests before committing
- Use the `uv` package manager for consistency
- Test with multiple scanner vendor data when applicable
- Generate reports to visually verify changes to image processing
- Keep changes focused and minimal
- Update documentation if changing public APIs or adding features
