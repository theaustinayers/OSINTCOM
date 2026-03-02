#!/usr/bin/env python3
"""
Build script for OSINTCOM v1.0
Generates a standalone Windows .exe using PyInstaller

Usage: python build_exe.py
Output: dist/OSINTCOM.exe
"""

import PyInstaller.__main__
import os
import sys

def build():
    """Build OSINTCOM executable"""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(project_dir, "icon.ico")
    
    print("=" * 60)
    print("OSINTCOM v1.0 - PyInstaller Build")
    print("=" * 60)
    
    # PyInstaller arguments
    args = [
        os.path.join(project_dir, "osintcom_qt.py"),
        "--name=OSINTCOM",
        f"--icon={icon_path}",
        "--onefile",
        "--windowed",
        "--add-data=icon.ico:.",
        "--hidden-import=sounddevice",
        "--hidden-import=numpy",
        "--hidden-import=scipy",
        "--hidden-import=scipy.io.wavfile",
        "--hidden-import=requests",
        "--hidden-import=PyQt5",
        "--hidden-import=PyQt5.QtWidgets",
        "--hidden-import=PyQt5.QtCore",
        "--hidden-import=PyQt5.QtGui",
        "--collect-all=sounddevice",
        "--collect-all=scipy",
        "--collect-all=numpy",
        "--distpath=dist",
    ]
    
    # Optional: Add noisereduce if available
    try:
        import noisereduce
        args.append("--hidden-import=noisereduce")
        args.append("--collect-all=noisereduce")
        print("[OK] noisereduce found - including in build")
    except ImportError:
        print("[WARN] noisereduce not found - build will work without it")
    
    print("\n[*] Building standalone executable...")
    print(f"[*] This may take 1-3 minutes...")
    print()
    
    try:
        PyInstaller.__main__.run(args)
        exe_path = os.path.join(project_dir, "dist", "OSINTCOM.exe")
        if os.path.exists(exe_path):
            size_mb = os.path.getsize(exe_path) / (1024 * 1024)
            print("\n" + "=" * 60)
            print("[SUCCESS] Build complete!")
            print(f"Executable: {exe_path}")
            print(f"Size: {size_mb:.1f} MB")
            print("=" * 60)
            return 0
        else:
            print("[ERROR] Executable not found in dist directory")
            return 1
    except Exception as e:
        print(f"\n[ERROR] Build failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(build())
