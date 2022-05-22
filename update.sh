#!/bin/sh
# find . -name '*.py' -exec autopep8 --max-line-length 200 --ignore=E402,E101,E131,E126 --in-place '{}' \;
git add *
git add -u
git status
echo Message: 
read message
git commit -m "$message"
