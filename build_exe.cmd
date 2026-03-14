@echo off
cd /d C:\OSINTCOM
C:\Users\theau\AppData\Local\Programs\Python\Python312\python.exe -m PyInstaller osintcom_qt.spec --noconfirm > build_latest.log 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> build_latest.log
