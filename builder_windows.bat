@echo off
REM START
echo Building Shared Library
REM Build
gcc -c test2.c
gcc -shared -o libtest2.dll test2.o -Wl,--out-implib,libtest2_lib.a
REM Cleaning Folder
echo Cleaning
del libtest2_lib.a
del test2.o
echo Done
REM Exit
