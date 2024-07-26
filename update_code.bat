@echo off
SET /P commitMessage=Please enter commit message: 
git add .
git commit -m "%commitMessage%"
git push
pause