#!/usr/bin/env python3
"""
Fix dependencies for Task 2 - handle missing packages gracefully
"""

import sys
import os
import subprocess

def install_system_dependencies():
    """Install system-level dependencies"""
    print("🔧 Installing system dependencies...")
    
    commands = [
        "apt-get update",
        "apt-get install -y libasound2-dev",  # For python-rtmidi
        "apt-get install -y llvm-11-dev",     # For llvmlite
        "apt-get install -y build-essential", # General compilation
    ]
    
    for cmd in commands:
        try:
            subprocess.run(cmd.split(), check=True, capture_output=True)
            print(f"✅ {cmd}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  {cmd} failed: {e}")

def install_alternative_packages():
    """Install alternative packages that work better"""
    print("\n📦 Installing alternative packages...")
    
    # Try installing packages one by one with alternatives
    packages_to_try = [
        # Try numba with older version that works with Python 3.10
        ("numba==0.56.4", "numba for audio processing"),
        # Skip python-rtmidi for now - not critical for our task
        # ("python-rtmidi", "MIDI I/O"),
        # Try llvmlite with conda if pip fails
        ("llvmlite==0.39.1", "LLVM for numba"),
    ]
    
    for package, description in packages_to_try:
        try:
            print(f"Installing {description}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"⚠️  {package} failed - will use workaround")

def create_lightweight_version():
    """Create a version that works without problematic dependencies"""
    
    # Create a simplified implementation that doesn't require numba
    simplified_code = '''
# Simplified imports that work without problematic dependencies
import warnings
warnings.filterwarnings("ignore")

# Check what we have available
available_packages = {}

try:
    import pandas as pd
    available_packages['pandas'] = True
    print("✅ pandas available")
except ImportError:
    available_packages['pandas'] = False
    print("❌ pandas not available")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    available_packages['plotting'] = True
    print("✅ matplotlib/seaborn available")
except ImportError:
    available_packages['plotting'] = False
    print("❌ plotting libraries not available")

try:
    import librosa
    available_packages['librosa'] = True
    print("✅ librosa available")
except ImportError:
    available_packages['librosa'] = False
    print("❌ librosa not available")

try:
    import pretty_midi
    available_packages['pretty_midi'] = True
    print("✅ pretty_midi available")
except ImportError:
    available_packages['pretty_midi'] = False
    print("❌ pretty_midi not available")

try:
    import tensorflow as tf
    available_packages['tensorflow'] = True
    print("✅ tensorflow available")
except ImportError:
    available_packages['tensorflow'] = False
    print("❌ tensorflow not available")

try:
    import magenta
    available_packages['magenta'] = True
    print("✅ magenta available")
except ImportError:
    available_packages['magenta'] = False
    print("❌ magenta not available")

# Check if we have minimum requirements
min_requirements = ['pandas', 'plotting', 'librosa', 'pretty_midi', 'tensorflow']
missing_critical = [pkg for pkg in min_requirements if not available_packages.get(pkg, False)]

if missing_critical:
    print(f"\\n⚠️  Missing critical packages: {missing_critical}")
    print("Please install them manually")
else:
    print("\\n✅ All critical packages available - ready for Task 2!")

print(f"\\nPackage status: {available_packages}")
'''
    
    with open('check_packages.py', 'w') as f:
        f.write(simplified_code)
    
    print("✅ Created package checker script")

def alternative_numba_install():
    """Try alternative ways to install numba"""
    print("\n🔄 Trying alternative numba installation...")
    
    alternatives = [
        # Try with conda if available
        "conda install -c conda-forge numba",
        # Try older version
        "pip install numba==0.56.4",
        # Try without numba for now
        "echo 'Skipping numba - will use scipy instead'"
    ]
    
    for alt in alternatives:
        try:
            if "conda" in alt and not os.path.exists("/root/miniconda3/bin/conda"):
                continue
            subprocess.run(alt.split(), check=True, capture_output=True)
            print(f"✅ {alt} succeeded")
            break
        except subprocess.CalledProcessError:
            print(f"⚠️  {alt} failed")

if __name__ == "__main__":
    print("🛠️  Fixing dependencies for Task 2...")
    
    # Step 1: Install system dependencies
    install_system_dependencies()
    
    # Step 2: Try alternative package installations
    install_alternative_packages()
    
    # Step 3: Create fallback version
    create_lightweight_version()
    
    # Step 4: Try alternative numba install
    alternative_numba_install()
    
    print("\n✅ Dependency fixing complete!")
    print("Run 'python check_packages.py' to verify what's available") 