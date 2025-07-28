#!/usr/bin/env python3
"""
Packaging script for creating a standalone executable of the document analysis system.
This script uses PyInstaller to create an executable that includes all dependencies.
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and print output in real time."""
    print(f"Running command: {cmd}")
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd
    )
    
    # Print output in real time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    return process.poll()


def install_requirements():
    """Install required packages."""
    print("Installing requirements...")
    cmd = "pip install -r requirements.txt"
    run_command(cmd)


def create_pyinstaller_spec():
    """Create a PyInstaller spec file."""
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'sentence_transformers',
        'transformers',
        'numpy',
        'torch',
        'tqdm',
        'faiss',
        'onnxruntime',
        'pydantic'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Add data files
a.datas += [('models/*.onnx', '.', 'DATA')]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='document_analyst',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
"""
    
    with open("document_analyst.spec", "w") as f:
        f.write(spec_content)
    
    print("Created PyInstaller spec file")


def package_application():
    """Package the application using PyInstaller."""
    # Install PyInstaller if not already installed
    run_command("pip install pyinstaller")
    
    # Create spec file
    create_pyinstaller_spec()
    
    # Run PyInstaller
    print("Packaging application with PyInstaller...")
    run_command("pyinstaller document_analyst.spec")
    
    print("Packaging completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Package the document analysis system")
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip installing requirements"
    )
    
    args = parser.parse_args()
    
    # Install requirements if not skipped
    if not args.skip_install:
        install_requirements()
    
    # Package application
    package_application()
    
    print("Done!")


if __name__ == "__main__":
    main() 