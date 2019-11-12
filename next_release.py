#!/usr/bin/python
# script to start a new release cycle
# Licence: GPLv3
from os.path import abspath, dirname
from make_release import PACKAGE_NAME, SRC_CODE, SRC_DOC
from releaser import add_release


if __name__ == '__main__':
    import sys

    argv = sys.argv
    if len(argv) < 2:
        print(f"Usage: {argv[0]} release_name")
        sys.exit()

    local_repository = abspath(dirname(__file__))
    add_release(local_repository, PACKAGE_NAME, SRC_CODE, release_name=argv[1], src_documentation=SRC_DOC)
