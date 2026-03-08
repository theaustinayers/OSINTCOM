#!/usr/bin/env python3
"""
cx_Freeze build script for OSINTCOM
Usage: python setup_cxfreeze.py build_exe
"""

from cx_Freeze import setup, Executable
from PyQt5.QtCore import QLibraryInfo
import os

# Get PyQt5 plugins directory
qt_plugin_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)

executables = [
    Executable(
        script="osintcom_qt.py",
        base="Win32GUI",
        target_name="OSINTCOM.exe",
        icon="icon.ico"
    )
]

build_exe_options = {
    "packages": [
        "sounddevice",
        "numpy",
        "scipy",
        "requests",
        "PyQt5",
        "noisereduce"
    ],
    "include_files": [
        (qt_plugin_path, "PyQt5/plugins"),
    ],
}

setup(
    name="OSINTCOM",
    version="1.13",
    description="Open Source Intelligence Communications Monitor",
    executables=executables,
    options={"build_exe": build_exe_options}
)
