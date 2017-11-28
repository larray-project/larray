#!/usr/bin/python
# encoding: utf-8
# script to start a new release cycle
# Licence: GPLv3
from os.path import abspath, dirname
from make_release import PACKAGE_NAME, SRC_CODE, SRC_DOC
from releaser import add_release


if __name__ == '__main__':
    import sys

    argv = sys.argv
    if len(argv) < 2:
        print("Usage: {} release_name [branch]".format(argv[0]))
        sys.exit()

    local_repository = abspath(dirname(__file__))
    add_release(local_repository, PACKAGE_NAME, SRC_CODE, *argv[1:], src_documentation=SRC_DOC)
