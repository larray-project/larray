#!/usr/bin/python
# Release script for LArray
# Licence: GPLv3
# Requires:
# * git with a Personal Access Token to access the Github repositories
# * releaser
# * conda-build
# * anaconda-client
# * twine (to upload packages to pypi)
import sys
from os.path import abspath, dirname, join
from subprocess import check_call

from releaser import make_release, update_feedstock, short, chdir, insert_step_func, yes
from releaser.utils import prepare_outlook_email
from releaser.make_release import steps_funcs as make_release_steps
from releaser.update_feedstock import steps_funcs as update_feedstock_steps

TMP_PATH = r"c:\tmp\larray_new_release"
TMP_PATH_CONDA = r"c:\tmp\larray_conda_new_release"
PACKAGE_NAME = "larray"
SRC_CODE = "larray"
SRC_DOC = join('doc', 'source')
GITHUB_REP = "https://github.com/larray-project/larray"
UPSTREAM_CONDAFORGE_FEEDSTOCK_REP = "https://github.com/conda-forge/larray-feedstock.git"
ORIGIN_CONDAFORGE_FEEDSTOCK_REP = "https://github.com/larray-project/larray-feedstock.git"
ANACONDA_UPLOAD_ARGS = {'--user': 'larray-project'}

LARRAY_READTHEDOCS = "http://larray.readthedocs.io/en/stable/"
LARRAY_ANNOUNCE_MAILING_LIST = "larray-announce@googlegroups.com"
LARRAY_USERS_GROUP = "larray-users@googlegroups.com"


def update_metapackage(local_repository, release_name, public_release=True, **extra_kwargs):
    if not public_release:
        return

    chdir(local_repository)
    version = short(release_name)

    # TODO: this should be echocall(redirect_stdout=False)
    print(f'Updating larrayenv metapackage to version {version}')
    dependencies = [
        f'larray =={version}', f'larray-editor =={version}', f'larray_eurostat =={version}', 
        'qtconsole', 'matplotlib', 'pyqt', 'qtpy', 'pytables', 'pydantic',
        'xlsxwriter', 'xlrd', 'openpyxl', 'xlwings',
    ]
    check_call([
        'conda', 'metapackage', 'larrayenv', version,
        '--dependencies'] + dependencies + [
        '--user', 'larray-project',
        '--home', 'http://github.com/larray-project/larray',
        '--license', 'GPL-3.0',
        '--summary', "'Package installing larray and all sub-projects and optional dependencies'",
    ])


def merge_changelogs(build_dir, src_documentation, release_name, public_release, branch='master', **extra_kwargs):
    chdir(join(build_dir, src_documentation))

    if not public_release:
        return

    check_call(['python', 'fetch_changelogs.py', release_name, branch])


insert_step_func(merge_changelogs, msg='append changelogs from larray-editor project', before='update_changelog')


# * the goal is to be able to fetch the editor changelog several times during development (manually) and automatically
#   during release
# * we need to commit either a merged changelog or changes.rst which includes both and a copy of the editor changelog
#   because the doc is built on rtd
# * the documentation must be buildable at any point (before a merge), so I need next_release to add an empty file

# def merge_changelogs(release_name):
#     include_changelog('CORE', release_name, LARRAY_GITHUB_REP, reset=True)
#     include_changelog('EDITOR', release_name, EDITOR_GITHUB_REP)


# TODO : move to larrayenv project
def announce_new_release(release_name):
    version=short(release_name)
    prepare_outlook_email(
        to=LARRAY_ANNOUNCE_MAILING_LIST,
        subject=f"LArray {version} released",
        body = f"""\
Hello all, 

We are pleased to announce that version {version} of LArray is now available. 
The complete description of changes including examples can be found at:

{LARRAY_READTHEDOCS}changes.html#version-{version.replace('.', '-')}



As always, *any* feedback is very welcome, preferably on the larray-users 
mailing list: {LARRAY_USERS_GROUP} (you need to register to be able to post).
""")


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) < 2:
        print(f"""
Usage:
 * {argv[0]} release_name|dev [step|startstep:stopstep] [branch]
   main release script

   steps: {', '.join(f.__name__ for f, _ in make_release_steps)}

 * {argv[0]} -c|--conda release_name
   update conda-forge feedstock

   steps: {', '.join(f.__name__ for f, _ in update_feedstock_steps)}

 * {argv[0]} -a|--announce release_name
   announce release

 * {argv[0]} -m|--meta release_name
   update metapackage
""")
        sys.exit()

    local_repository = abspath(dirname(__file__))
    if argv[1] == '-m' or argv[1] == '--meta':
        update_metapackage(local_repository, release_name=argv[2])
    elif argv[1] == '-a' or argv[1] == '--announce':
        if yes("Is metapackage larrayenv updated?", default='n'):
            announce_new_release(argv[2])
    elif argv[1] == '-c' or argv[1] == '--conda':
        update_feedstock(GITHUB_REP, UPSTREAM_CONDAFORGE_FEEDSTOCK_REP, ORIGIN_CONDAFORGE_FEEDSTOCK_REP,
                         SRC_CODE, *argv[2:], tmp_dir=TMP_PATH_CONDA)
    else:
        make_release(local_repository, PACKAGE_NAME, SRC_CODE, *argv[1:],
                     src_documentation=SRC_DOC, tmp_dir=TMP_PATH,
                     anaconda_upload_args=ANACONDA_UPLOAD_ARGS)
