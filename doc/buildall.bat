@echo off
rem change directory to the location of this batch file
rem (but remember original directory)
pushd %~dp0

rem call c64.bat
rem call activate larray

call make html

call make htmlhelp
pushd build\htmlhelp
hhc.exe larray.hhp > hhc.log
popd

call make latex
pushd build\latex
rem we can also use --run-viewer to open the pdf
texify.exe --clean --pdf --tex-option=-synctex=1 larray.tex > texify.log
popd

echo.
echo Build finished: all documentation built
popd