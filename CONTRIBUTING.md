# Contributing to ComfyUI-WhisperCPP

Thank you for your interest in contributing!

## ⚠️ IMPORTANT: Please update to the latest version first!

**Before contributing, please make sure you're using the latest version:**
```bash
cd ComfyUI/custom_nodes/ComfyUI-WhisperCPP
git pull
pip install -r requirements.txt
```

**If you're using an outdated version, DO NOT submit PRs. Update first!**

## How to Contribute

### 1. Fork the Repository

```bash
git clone https://github.com/your-username/ComfyUI-WhisperCPP.git
cd ComfyUI-WhisperCPP
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Your Changes

- Follow the existing code style
- Add comments for complex logic
- Test your changes

### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: add new feature"
```

Use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` for new features
- `fix:` for bug fixes
- `chore:` for maintenance
- `docs:` for documentation

### 5. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 6. Create a Pull Request

- Go to the original repository
- Click "New Pull Request"
- Select your branch
- Add a clear description
- Submit!

## Development Setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/obvirm/ComfyUI-WhisperCPP.git

# Install dependencies
pip install -r requirements.txt

# Build DLLs (optional)
python build_whisper_cpp.py
python build_bs_roformer.py
python build_cpp_annote.py
```

## Code Style

- Follow PEP 8 for Python
- Use meaningful variable names
- Add docstrings for functions
- Keep functions short and focused

## Questions?

Feel free to open an issue for any questions!
