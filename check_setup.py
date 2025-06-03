#!/usr/bin/env python3
"""
Setup verification script for Music Generation ML Assignment
Checks that environments and kernels are properly configured.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, capture_output=True):
    """Run a command and return success status and output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_environment(env_name):
    """Check if a virtual environment exists and has required packages"""
    env_path = Path.home() / env_name
    if not env_path.exists():
        return False, f"Environment {env_name} not found at {env_path}"
    
    # Check if it's activated
    activate_script = env_path / "bin" / "activate"
    if not activate_script.exists():
        return False, f"Activation script not found in {env_name}"
    
    return True, f"Environment {env_name} exists"

def check_jupyter_kernels():
    """Check if Jupyter kernels are installed"""
    success, output, error = run_command("jupyter kernelspec list")
    if not success:
        return False, f"Failed to list kernels: {error}"
    
    kernels = output.lower()
    has_task2 = "task2_env" in kernels
    has_task4 = "task4_env" in kernels
    
    return has_task2 and has_task4, f"Task2 kernel: {has_task2}, Task4 kernel: {has_task4}"

def check_git_submodules():
    """Check if Git submodules are properly initialized"""
    magenta_path = Path("libs/magenta")
    musika_path = Path("libs/musika")
    
    magenta_exists = magenta_path.exists() and any(magenta_path.iterdir())
    musika_exists = musika_path.exists() and any(musika_path.iterdir())
    
    return magenta_exists and musika_exists, f"Magenta: {magenta_exists}, Musika: {musika_exists}"

def main():
    """Run all checks and display results"""
    print("🎵 Music Generation ML Assignment - Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Virtual Environment - Task 2", lambda: check_environment("env_task2")),
        ("Virtual Environment - Task 4", lambda: check_environment("env_task4")),
        ("Jupyter Kernels", check_jupyter_kernels),
        ("Git Submodules", check_git_submodules),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            success, message = check_func()
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{check_name:.<30} {status}")
            if not success:
                print(f"   └─ {message}")
                all_passed = False
            elif message:
                print(f"   └─ {message}")
        except Exception as e:
            print(f"{check_name:.<30} ❌ ERROR")
            print(f"   └─ {str(e)}")
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 All checks passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Start Jupyter Lab: source ~/env_task4/bin/activate && jupyter lab")
        print("2. Open notebooks and select appropriate kernels")
        print("3. Download datasets as needed")
    else:
        print("⚠️  Some checks failed. Please review the setup instructions.")
        print("\nCommon fixes:")
        print("- Recreate virtual environments if missing")
        print("- Run 'git submodule update --init --recursive' for submodules")
        print("- Reinstall Jupyter kernels with 'python -m ipykernel install'")

if __name__ == "__main__":
    main() 