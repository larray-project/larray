#!/usr/bin/python
# coding=utf-8
# Release script for LArray
# Licence: GPLv3
# Requires:
# * git
from __future__ import print_function, unicode_literals

import sys
from os.path import abspath, dirname, join
from subprocess import check_call

from releaser import make_release, update_feedstock, short, no, chdir, set_config, insert_step_func
from releaser.make_release import steps_funcs as make_release_steps
from releaser.update_feedstock import steps_funcs as update_feedstock_steps

TMP_PATH = r"c:\tmp\larray_new_release"
TMP_PATH_CONDA = r"c:\tmp\larray_conda_new_release"
PACKAGE_NAME = "larray"
SRC_CODE = "larray"
SRC_DOC = join('doc', 'source')
GITHUB_REP = "https://github.com/larray-project/larray"
CONDA_FEEDSTOCK_REP = "https://github.com/larray-project/larray-feedstock.git"

LARRAY_READTHEDOCS = "http://larray.readthedocs.io/en/stable/"
LARRAY_ANNOUNCE_MAILING_LIST = "larray-announce@googlegroups.com"
LARRAY_USERS_GROUP = "larray-users@googlegroups.com"


def update_metapackage(context):
    if not context['public_release']:
        return

    chdir(context['repository'])
    version = short(context['release_name'])

    def fill(s):
        return s.format(version=version)

    # TODO: this should be echocall(redirect_stdout=False)
    print(fill('Updating larrayenv metapackage to version {version}'))
    # - excluded versions 5.0 and 5.1 of ipykernel because these versions make the console useless after any exception
    #   https://github.com/larray-project/larray-editor/issues/166
    check_call(['conda', 'metapackage', 'larrayenv', version, '--dependencies', fill('larray =={version}'),
                fill('larray-editor =={version}'), fill('larray_eurostat =={version}'),
                "qtconsole", "matplotlib", "pyqt", "qtpy", "pytables",
                "xlsxwriter", "xlrd", "openpyxl", "xlwings", "ipykernel !=5.0,!=5.1.0",
                '--user', 'larray-project',
                '--home', 'http://github.com/larray-project/larray',
                '--license', 'GPL-3.0',
                '--summary', "'Package installing larray and all sub-projects and optional dependencies'"])


def merge_changelogs(config):
    if config['src_documentation'] is not None:
        chdir(join(config['build_dir'], config['src_documentation']))

        if not config['public_release']:
            return

        check_call(['python', 'merge_changelogs.py', config['release_name']])


insert_step_func(merge_changelogs, msg='append changelogs from larray-editor project', before='update_changelog')


# TODO : move to larrayenv project
def announce_new_release(release_name):
    import win32com.client as win32
    version = short(release_name)
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.To = LARRAY_ANNOUNCE_MAILING_LIST
    mail.Subject = "LArray {} released".format(version)
    mail.Body = """\
Hello all, 

We are pleased to announce that version {version} of LArray is now available. 
The complete description of changes including examples can be found at:

{readthedocs}changes.html#version-{version_doc}



As always, *any* feedback is very welcome, preferably on the larray-users 
mailing list: {larray_users_group} (you need to register to be able to post).
""".format(version=version, readthedocs=LARRAY_READTHEDOCS, version_doc=version.replace('.', '-'),
           larray_users_group=LARRAY_USERS_GROUP)
    mail.Display(True)


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) < 2:
        print("Usage: {} [-c|--conda] release_name|dev [step|startstep:stopstep] [branch]".format(argv[0]))
        print("make release steps:", ', '.join(f.__name__ for f, _ in make_release_steps))
        print("update conda-forge feedstock steps:", ', '.join(f.__name__ for f, _ in update_feedstock_steps))
        print("or")
        print("Usage: {} -a|--announce release_name".format(argv[0]))
        print("Usage: {} -m|--meta release_name".format(argv[0]))
        sys.exit()

    local_repository = abspath(dirname(__file__))
    if argv[1] == '-m' or argv[1] == '--meta':
        local_config = set_config(local_repository, PACKAGE_NAME, SRC_CODE, argv[2], branch='master',
                                  src_documentation=SRC_DOC, tmp_dir=TMP_PATH)
        update_metapackage(local_config)
    elif argv[1] == '-a' or argv[1] == '--announce':
        no("Is metapackage larrayenv updated?")
        announce_new_release(argv[2])
    elif argv[1] == '-c' or argv[1] == '--conda':
        update_feedstock(GITHUB_REP, CONDA_FEEDSTOCK_REP, SRC_CODE, *argv[2:], tmp_dir=TMP_PATH_CONDA)
    else:
        make_release(local_repository, PACKAGE_NAME, SRC_CODE, *argv[1:], src_documentation=SRC_DOC, tmp_dir=TMP_PATH)
