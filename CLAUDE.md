# IDTAP Python API Development Guide

## Overview
The Python API (`idtap`) is a sophisticated client library for interacting with the IDTAP (Interactive Digital Transcription and Analysis Platform) server, specifically designed for transcribing, analyzing, and managing Hindustani music recordings.

## Key Development Points

### Dependencies Management
- **Keep `Pipfile` and `pyproject.toml` in sync** - this is critical!
- Add new packages: `pipenv install package-name`
- Then manually add to `pyproject.toml` dependencies array
- Core deps: requests, pyhumps, keyring, cryptography, PyJWT, pymongo, google-auth-oauthlib

### Testing
- **Unit tests**: `pytest python/idtap/tests/` (uses `responses` for HTTP mocking)
- **Integration tests**: `python python/api_testing/api_test.py` (requires live server auth)
- Test structure: Complete coverage of data models, client functionality, and authentication

**⚠️ IMPORTANT FOR CLAUDE: Before running the full test suite (`pytest idtap/tests/`), ALWAYS warn Jon first!**
- Some tests may require browser authorization for OAuth authentication
- Running tests without warning can waste time waiting for authorization that Jon doesn't realize is needed
- Best practice: Ask "Ready to run the full test suite? (May require browser authorization)" before executing

### Build/Package/Publish - AUTOMATED via GitHub Actions
**⚠️ IMPORTANT: Manual publishing is now automated. See "Automated Version Management" section below.**

Manual publishing (for local testing only):
```bash
python -m build
python -m twine upload dist/*  # or --repository testpypi for testing
```

## Architecture

### Main Components
- **`SwaraClient`** (`client.py`) - Main HTTP client with OAuth authentication
- **Data Models** (`/classes/`) - Rich musical transcription classes (Piece, Phrase, Trajectory, etc.)
- **Authentication** (`auth.py` + `secure_storage.py`) - OAuth flow with secure token storage
- **Utils** (`utils.py`) - camelCase ↔ snake_case conversion

### Key Classes
- **`Piece`**: Central transcription container with multi-track support, sections, audio association
- **`SwaraClient`**: API interface with methods for transcription CRUD, audio download, permissions
- **Musical Elements**: Phrase, Trajectory, Pitch, Raga, Section, Meter, Articulation

### Security/Authentication
- **OAuth Flow**: Server-based OAuth (not direct Google) → local HTTP server → secure storage
- **Storage Layers**: OS Keyring (primary) → AES-256 encrypted file (fallback) → plaintext (legacy)
- **CSRF Protection**: State parameter validation
- **Permissions**: User-based access control with public/private visibility

## Development Patterns

### Code Conventions
- **snake_case** for Python code
- **camelCase ↔ snake_case** conversion via `pyhumps` for API communication
- **Type hints** throughout
- **Backwards compatibility** maintained (especially token storage migration)

### Serialization Pattern
```python
class DataModel:
    def to_json(self) -> Dict[str, Any]:
        # Convert to dict with camelCase keys for API
        
    @staticmethod 
    def from_json(obj: Dict[str, Any]) -> 'DataModel':
        # Parse from API response with snake_case conversion
```

### Package Structure
```
python/idtap/
├── __init__.py           # Public API exports
├── client.py             # HTTP client (SwaraClient)
├── auth.py               # OAuth authentication
├── secure_storage.py     # Token security layers
├── enums.py              # Instrument types, etc.
├── utils.py              # camelCase conversion utilities
├── classes/              # Musical data models
└── tests/                # Unit tests
```

## API Endpoints (via SwaraClient)
- **Transcriptions**: GET/POST `/api/transcription/{id}`, GET `/api/transcriptions`
- **Data Export**: GET `/api/transcription/{id}/json`, `/api/transcription/{id}/excel`
- **Audio**: GET `/audio/{format}/{id}.{format}`
- **Permissions**: POST `/api/visibility`
- **OAuth**: GET `/oauth/authorize`, POST `/oauth/token`

## Musical Domain Knowledge
- **Hindustani Music Focus**: Transcription system for Indian classical music
- **Complex Data Models**: Supports microtonal pitches, ragas, articulations, meter cycles
- **Multi-instrument**: Sitar, Vocal (Male/Female) with instrument-specific features
- **Analytical Tools**: Trajectory categorization, phrase grouping, temporal analysis

## Automated Version Management & Publishing

### 🤖 How It Works (python-semantic-release + GitHub Actions)

**PATCH-ONLY MODE**: All commits automatically increment patch version (0.1.14 → 0.1.15) regardless of commit message.

#### Automatic Workflow
1. **Make changes** → Commit with any message
2. **Create PR** → GitHub Actions test + upload to TestPyPI automatically
3. **Merge to main** → Version bumps + PyPI published + GitHub release created
4. **Zero manual intervention** required!

#### Version Locations (Auto-Updated)
- `idtap/__init__.py:3` - `__version__ = "0.1.14"`  
- `pyproject.toml:7` - `version = "0.1.14"`
- `docs/conf.py:16-17` - `release = '0.1.14'` and `version = '0.1.14'`

### 🎛️ Manual Version Control Commands

**For normal development (patch bumps)**:
```bash
# Any commit message works - always bumps patch
git commit -m "fix something"           # 0.1.14 → 0.1.15
git commit -m "add new feature"         # 0.1.14 → 0.1.15 
git commit -m "update docs"             # 0.1.14 → 0.1.15
```

**When Jon wants to control version bumps manually**:
```bash
# Force minor version bump (0.1.15 → 0.2.0)
semantic-release version --increment minor

# Force major version bump (0.2.0 → 1.0.0)  
semantic-release version --increment major

# Dry run to see what would happen
semantic-release version --print --no-commit --no-tag --no-push --no-vcs-release
```

### 🔧 Configuration Details

**Location**: `pyproject.toml` `[tool.semantic_release]` section
- **patch_tags**: ALL commit types → patch version increment
- **minor_tags**: `[]` (empty - no automatic minor bumps)  
- **major_tags**: `[]` (empty - no automatic major bumps)
- **Files updated**: `__init__.py`, `pyproject.toml`, `docs/conf.py`

### 🚀 GitHub Actions Workflows

**`.github/workflows/test-pr.yml`**: 
- Runs on every PR
- Tests + builds package + uploads to TestPyPI
- Comments PR with TestPyPI install link

**`.github/workflows/release.yml`**:
- Runs on merge to main
- Tests → version bump → build → PyPI publish → GitHub release

### 📋 Required Setup (One-Time)

**GitHub Secrets**:
- `TESTPYPI_API_TOKEN` - For PR testing uploads

**PyPI Trusted Publisher** (configured):
- Owner: `HAI-Res`  
- Repository: `idtap-client`
- Workflow: `release.yml`
- Uses OIDC - no API tokens needed for production PyPI

**⚠️ CLAUDE: When Jon asks to update version in special way, use the manual commands above!**

## Development Workflow (Updated for Automation)
1. **Data Model Development**: Create/modify classes in `/classes/` with proper serialization
2. **Client Method Development**: Add HTTP methods in `client.py` with authentication  
3. **Testing**: Write unit tests (mocked) + integration tests (live API)
4. **PR Creation**: GitHub Actions automatically test + upload to TestPyPI
5. **Merge to main**: Automatic version bump + PyPI publish + GitHub release
6. **No manual version management needed!**

## Installation Commands
```bash
# Development
pip install -e python/
pipenv install --dev

# Testing  
pytest python/idtap/tests/
python python/api_testing/api_test.py

# Package management
pipenv install package-name  # then manually add to pyproject.toml
```

This API provides a production-ready foundation for complex musical transcription analysis with modern security practices and comprehensive testing coverage.

# PyPI Publishing Guide

## Quick Reference Checklist

**Before ANY publish, run these commands:**
```bash
# 1. Fix critical bugs first (see CRITICAL FIXES section below)
# 2. Update versions to match in BOTH files
# 3. Run tests
pytest idtap/tests/
# 4. Clean build
rm -rf dist/ build/ *.egg-info/
# 5. Build
python -m build
# 6. Test upload (optional but recommended)
python -m twine upload --repository testpypi dist/*
# 7. Production upload
python -m twine upload dist/*
# 8. Tag and push
git tag vX.X.X && git push origin vX.X.X
```

## Complete Steps to Publish New Version to PyPI

### Prerequisites
- PyPI account with API token configured
- `build` and `twine` packages installed (`pipenv install build twine`)
- All changes committed and pushed to GitHub

### Step-by-Step Publishing Process

#### 1. Update Version Numbers (CRITICAL - Must be done in both places)

**A. Update `idtap/__init__.py`:**
```python
__version__ = "0.1.13"  # Increment from current "0.1.12"
```

**B. Update `pyproject.toml`:**
```toml
[project]
version = "0.1.13"  # Must match __init__.py exactly
```

**Version Increment Rules:**
- **Patch** (0.1.7 → 0.1.8): Bug fixes, small improvements, query system enhancements
- **Minor** (0.1.8 → 0.2.0): New features, API additions (backwards compatible)
- **Major** (0.2.0 → 1.0.0): Breaking changes, API modifications

#### 2. Update Dependencies (if needed)

**A. Sync `Pipfile` and `pyproject.toml`:**
```bash
# If adding new packages:
pipenv install new-package-name
# Then manually add to pyproject.toml dependencies array
```

**B. Check dependency versions in `pyproject.toml`:**
```toml
dependencies = [
    "requests>=2.31.0",
    "pyhumps>=3.8.0",
    # ... ensure all deps are current
]
```

#### 3. Pre-Publishing Checks

**A. Run Full Test Suite:**
```bash
pytest idtap/tests/  # Must pass all tests
```

**B. Test Package Installation Locally:**
```bash
pip install -e .  # Test editable install
python -c "import idtap; print(idtap.__version__)"  # Verify version
```

**C. Check Package Structure:**
```bash
python -m build --sdist --wheel .  # Build without uploading
# Check dist/ directory for .whl and .tar.gz files
```

#### 4. Build Package

**A. Clean Previous Builds:**
```bash
rm -rf dist/ build/ *.egg-info/
```

**B. Build Distribution Files:**
```bash
python -m build
```
- Creates `dist/idtap-X.X.X-py3-none-any.whl` (wheel)
- Creates `dist/idtap-X.X.X.tar.gz` (source distribution)

#### 5. Test on TestPyPI (Recommended)

**A. Authentication Setup:**
TestPyPI and production PyPI require separate API tokens. Configure in `.envrc` (gitignored):
```bash
# TestPyPI token (get from https://test.pypi.org/)
export TWINE_TESTPYPI_PASSWORD="pypi-[YOUR_TESTPYPI_TOKEN]"

# Production PyPI token (get from https://pypi.org/)
export TWINE_PASSWORD="pypi-[YOUR_PRODUCTION_TOKEN]"

# Twine username for both
export TWINE_USERNAME="__token__"
```

**B. Upload to TestPyPI:**
```bash
TWINE_PASSWORD="$TWINE_TESTPYPI_PASSWORD" python -m twine upload --repository testpypi dist/*
```

**C. Test Installation from TestPyPI:**
```bash
pip install --index-url https://test.pypi.org/simple/ idtap
```

**D. Verify TestPyPI Installation:**
```bash
python -c "import idtap; print(idtap.__version__)"
```

#### 6. Publish to Production PyPI

**A. Upload to PyPI:**
```bash
python -m twine upload dist/*
```

**B. Verify Upload:**
- Check https://pypi.org/project/idtap/
- Verify new version is listed
- Check that description/metadata displays correctly

#### 7. Post-Publishing Verification

**A. Test Production Installation:**
```bash
pip install --upgrade idtap
python -c "import idtap; print(idtap.__version__)"
```

**B. Test Key Functionality:**
```python
from idtap.classes.raga import Raga
from idtap import SwaraClient

# Test that new validation works
try:
    Raga({'rules': {}})  # Should raise helpful error
except ValueError as e:
    print(f"✅ Validation working: {e}")
```

#### 8. Git Tagging and Cleanup

**A. Create Git Tag:**
```bash
git tag v0.1.8  # Match the version number
git push origin v0.1.8
```

**B. Clean Build Artifacts:**
```bash
rm -rf dist/ build/ *.egg-info/
```

### Troubleshooting Common Issues

#### Version Conflicts
- **Error**: "File already exists" on PyPI upload
- **Fix**: Ensure version number is incremented in BOTH `__init__.py` AND `pyproject.toml`

#### Package Structure Issues
- **Error**: Import errors after installation
- **Fix**: Check `pyproject.toml` `[tool.setuptools.packages.find]` section
- **Historical Issue (FIXED)**: Config used to look for `idtap*` but package is `idtap*`
- **Current Status**: ✅ Fixed - now correctly configured to include `idtap*`

#### Dependency Conflicts
- **Error**: Dependencies not installing correctly
- **Fix**: Ensure `Pipfile` and `pyproject.toml` dependencies are exactly synchronized

#### Authentication Issues
- **Error**: 403 Forbidden on upload
- **Fix**: Configure PyPI API token: `python -m twine configure`

### PyPI Configuration Files

**API Token Setup** (one-time):
```bash
# Create ~/.pypirc with API token
[distutils]
index-servers = pypi testpypi

[pypi]
username = __token__
password = pypi-YOUR_API_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TEST_API_TOKEN_HERE
```

### Current Package Status
- **Package Name**: `idtap` (changed from `idtap-api`)
- **Current Version**: `0.1.13` (synced in both pyproject.toml and __init__.py) ✅
- **Package Structure**: Fixed - now correctly includes `idtap*` instead of `idtap*` ✅
- **Package Data**: Fixed - now correctly references `idtap` directory ✅
- **Python Support**: >= 3.10
- **Key Dependencies**: requests, pyhumps, keyring, cryptography, PyJWT

**✅ All critical packaging bugs have been resolved!**

### Post-Validation Release Notes Template

For the current comprehensive validation release:
```
## v0.1.8 - Query System Improvements

### 🚀 New Features
- Enhanced query system functionality
- Improved authentication token handling investigation
- Better development workflow for query branch integration

### 🐛 Bug Fixes  
- Resolved query system integration issues
- Improved error handling in authentication flow

### 🧪 Testing
- All existing tests continue to pass
- Query system thoroughly tested and validated

### ⚠️ Breaking Changes
None - fully backwards compatible with existing usage

## v0.1.6 - Enhanced Parameter Validation

### 🚀 New Features
- Comprehensive parameter validation for Raga constructor
- Helpful error messages for common parameter mistakes
- Type validation for all constructor parameters
- Structure validation for rule_set and tuning dictionaries

### 🐛 Bug Fixes  
- Prevents silent parameter failures that led to incorrect musical analysis
- Fixed unused variable warnings in existing code

### 🧪 Testing
- Added 29 comprehensive validation test cases
- All 254 tests in full suite passing
- Tests cover every possible invalid input scenario

### ⚠️ Breaking Changes
None - fully backwards compatible with existing valid usage
```
