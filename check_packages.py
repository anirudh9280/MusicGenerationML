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
    print(f"   TensorFlow version: {tf.__version__}")
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   GPU available: {gpus[0]}")
    else:
        print("   No GPU detected")
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

try:
    import numpy as np
    available_packages['numpy'] = True
    print("✅ numpy available")
except ImportError:
    available_packages['numpy'] = False
    print("❌ numpy not available")

try:
    import scipy
    available_packages['scipy'] = True
    print("✅ scipy available")
except ImportError:
    available_packages['scipy'] = False
    print("❌ scipy not available")

# Check if we have minimum requirements
min_requirements = ['pandas', 'plotting', 'tensorflow', 'numpy']
missing_critical = [pkg for pkg in min_requirements if not available_packages.get(pkg, False)]

print("\n" + "="*50)
if missing_critical:
    print(f"⚠️  Missing critical packages: {missing_critical}")
    print("Please install them manually")
else:
    print("✅ All critical packages available - ready for Task 2!")

print(f"\nPackage status: {available_packages}")

# Additional suggestions
print("\n📋 Recommendations:")
if not available_packages.get('librosa', False):
    print("- librosa: Optional for advanced audio processing")
if not available_packages.get('pretty_midi', False):
    print("- pretty_midi: Optional for proper MIDI file creation")
if not available_packages.get('magenta', False):
    print("- magenta: Optional for full Onsets and Frames model")

print("\n🚀 Ready to proceed with Task 2 using available packages!") 