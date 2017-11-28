#!/usr/bin/python
# coding=utf-8
# Release script for LArray
# Licence: GPLv3
# Requires:
# * git
from __future__ import print_function, unicode_literals

import errno
import fnmatch
import os
import re
import stat
import subprocess
import sys
import zipfile
import hashlib
import urllib.request as request

from datetime import date
from os import chdir, makedirs
from os.path import exists, abspath, dirname
from shutil import copytree, copy2, rmtree as _rmtree
from subprocess import check_output, STDOUT, CalledProcessError


PY2 = sys.version_info[0] < 3
TMP_PATH = r"c:\tmp\larray_new_release"
TMP_PATH_CONDA = r"c:\tmp\larray_conda_new_release"
LARRAY_REP = 'https://github.com/liam2/larray'
LARRAY_READTHEDOCS = "http://larray.readthedocs.io/en/stable/"
CONDA_LARRAY_FEEDSTOCK_REP = 'https://github.com/larray-project/larray-feedstock.git'
LARRAY_ANNOUNCE_MAILING_LIST = "larray-announce@googlegroups.com"
LARRAY_USERS_GROUP = "larray-users@googlegroups.com"

try:
    input = raw_input
except NameError:
    pass

if PY2:
    import io
    # add support for encoding. Slow on Python2, but that is not a problem given what we do with it.
    open = io.open


# ------------- #
# generic tools #
# ------------- #

def size2str(value):
    unit = "bytes"
    if value > 1024.0:
        value /= 1024.0
        unit = "Kb"
        if value > 1024.0:
            value /= 1024.0
            unit = "Mb"
        return "{:.2f} {}".format(value, unit)
    else:
        return "{:d} {}".format(value, unit)


def generate(fname, **kwargs):
    with open('{}.tmpl'.format(fname)) as in_f, open(fname, 'w') as out_f:
        out_f.write(in_f.read().format(**kwargs))


def _remove_readonly(function, path, excinfo):
    if function in {os.rmdir, os.remove, os.unlink} and excinfo[1].errno == errno.EACCES:
        # add write permission to owner
        os.chmod(path, stat.S_IWUSR)
        # retry removing
        function(path)
    else:
        raise Exception("Cannot remove {}".format(path))


def rmtree(path):
    _rmtree(path, onerror=_remove_readonly)


def call(*args, **kwargs):
    try:
        res = check_output(*args, stderr=STDOUT, **kwargs)
        if not PY2 and 'universal_newlines' not in kwargs:
            res = res.decode('utf8')
        return res
    except CalledProcessError as e:
        print(e.output)
        raise e


def echocall(*args, **kwargs):
    print(' '.join(args))
    return call(*args, **kwargs)


def git_remote_last_rev(url, branch=None):
    """
    :param url: url of the remote repository
    :param branch: an optional branch (defaults to 'refs/heads/master')
    :return: name/hash of the last revision
    """
    if branch is None:
        branch = 'refs/heads/master'
    output = call('git ls-remote {} {}'.format(url, branch))
    for line in output.splitlines():
        if line.endswith(branch):
            return line.split()[0]
    raise Exception("Could not determine revision number")


def branchname(statusline):
    """
    computes the branch name from a "git status -b -s" line
    ## master...origin/master
    """
    statusline = statusline.replace('#', '').strip()
    pos = statusline.find('...')
    return statusline[:pos] if pos != -1 else statusline


def yes(msg, default='y'):
    choices = ' ({}/{}) '.format(*tuple(c.capitalize() if c == default else c
                                  for c in ('y', 'n')))
    answer = None
    while answer not in ('', 'y', 'n'):
        if answer is not None:
            print("answer should be 'y', 'n', or <return>")
        answer = input(msg + choices).lower()
    return (default if answer == '' else answer) == 'y'


def no(msg, default='n'):
    return not yes(msg, default)


def do(description, func, *args, **kwargs):
    print(description + '...', end=' ')
    func(*args, **kwargs)
    print("done.")


def allfiles(pattern, path='.'):
    """
    like glob.glob(pattern) but also include files in subdirectories
    """
    return (os.path.join(dirpath, f)
            for dirpath, dirnames, files in os.walk(path)
            for f in fnmatch.filter(files, pattern))


def zip_pack(archivefname, filepattern):
    with zipfile.ZipFile(archivefname, 'w', zipfile.ZIP_DEFLATED) as f:
        for fname in allfiles(filepattern):
            f.write(fname)


def zip_unpack(archivefname, dest=None):
    with zipfile.ZipFile(archivefname) as f:
        f.extractall(dest)


def short(rel_name):
    return rel_name[:-2] if rel_name.endswith('.0') else rel_name


def long_release_name(release_name):
    """
    transforms a short release name such as 0.8 to a long one such as 0.8.0
    >>> long_release_name('0.8')
    '0.8.0'
    >>> long_release_name('0.8.0')
    '0.8.0'
    >>> long_release_name('0.8rc1')
    '0.8.0rc1'
    >>> long_release_name('0.8.0rc1')
    '0.8.0rc1'
    """
    dotcount = release_name.count('.')
    if dotcount >= 2:
        return release_name
    assert dotcount == 1, "{} contains {} dots".format(release_name, dotcount)
    pos = pretag_pos(release_name)
    if pos is not None:
        return release_name[:pos] + '.0' + release_name[pos:]
    return release_name + '.0'


def pretag_pos(release_name):
    """
    gives the position of any pre-release tag
    >>> pretag_pos('0.8')
    >>> pretag_pos('0.8alpha25')
    3
    >>> pretag_pos('0.8.1rc1')
    5
    """
    # 'a' needs to be searched for after 'beta'
    for tag in ('rc', 'c', 'beta', 'b', 'alpha', 'a'):
        match = re.search(tag + '\d+', release_name)
        if match is not None:
            return match.start()
    return None


def strip_pretags(release_name):
    """
    removes pre-release tags from a version string
    >>> strip_pretags('0.8')
    '0.8'
    >>> strip_pretags('0.8alpha25')
    '0.8'
    >>> strip_pretags('0.8.1rc1')
    '0.8.1'
    """
    pos = pretag_pos(release_name)
    return release_name[:pos] if pos is not None else release_name


def isprerelease(release_name):
    """
    tests whether the release name contains any pre-release tag
    >>> isprerelease('0.8')
    False
    >>> isprerelease('0.8alpha25')
    True
    >>> isprerelease('0.8.1rc1')
    True
    """
    return pretag_pos(release_name) is not None


# -------------------- #
# end of generic tools #
# -------------------- #

# ------------------------- #
# specific helper functions #
# ------------------------- #


def relname2fname(release_name):
    short_version = short(strip_pretags(release_name))
    return r"version_{}.rst.inc".format(short_version.replace('.', '_'))


def release_changes(context):
    directory = r"doc\source\changes"
    fname = relname2fname(context['release_name'])
    with open(os.path.join(context['build_dir'], directory, fname), encoding='utf-8-sig') as f:
        return f.read()


def replace_lines(fpath, changes, end="\n"):
    """
    Parameters
    ----------
    changes : list of pairs
        List of pairs (substring_to_find, new_line).
    """
    with open(fpath) as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:]):
            for substring_to_find, new_line in changes:
                if substring_to_find in line and not line.strip().startswith('#'):
                    lines[i] = new_line + end
    with open(fpath, 'w') as f:
        f.writelines(lines)


def create_source_archive(release_name, rev):
    call(r'git archive --format zip --output ..\larray-{}-src.zip {}'.format(release_name, rev))


def copy_release(release_name):
    pass


def create_bundle_archives(release_name):
    pass


def check_bundle_archives(release_name):
    """
    checks the bundles unpack correctly
    """
    makedirs('test')
    zip_unpack('larray-{}-src.zip'.format(release_name), r'test\src')
    rmtree('test')

# -------------------------------- #
# end of specific helper functions #
# -------------------------------- #

# ----- #
# steps #
# ----- #

def check_local_repo(context):
    # releasing from the local clone has the advantage we can prepare the
    # release offline and only push and upload it when we get back online
    s = "Using local repository at: {repository} !".format(**context)
    print("\n", s, "\n", "=" * len(s), "\n", sep='')

    status = call('git status -s -b')
    lines = status.splitlines()
    statusline, lines = lines[0], lines[1:]
    curbranch = branchname(statusline)
    if curbranch != context['branch']:
        print("{branch} is not the current branch ({curbranch}). "
              "Please use 'git checkout {branch}'.".format(**context, curbranch=curbranch))
        exit(1)

    if lines:
        uncommited = sum(1 for line in lines if line[1] in 'MDAU')
        untracked = sum(1 for line in lines if line.startswith('??'))
        print('Warning: there are {:d} files with uncommitted changes and '
              '{:d} untracked files:'.format(uncommited, untracked))
        print('\n'.join(lines))
        if no('Do you want to continue?'):
            exit(1)

    ahead = call('git log --format=format:%%H origin/{branch}..{branch}'.format(**context))
    num_ahead = len(ahead.splitlines())
    print("Branch '{branch}' is {num_ahead:d} commits ahead of 'origin/{branch}'"
          .format(**context, num_ahead=num_ahead), end='')
    if num_ahead:
        if yes(', do you want to push?'):
            do('Pushing changes', call, 'git push')
    else:
        print()

    if no('Release version {release_name} ({rev})?'.format(**context)):
        exit(1)


def create_tmp_directory(context):
    tmp_dir = context['tmp_dir']
    if exists(tmp_dir):
        rmtree(tmp_dir)
    makedirs(tmp_dir)


def clone_repository(context):
    chdir(context['tmp_dir'])

    # make a temporary clone in /tmp. The goal is to make sure we do not include extra/unversioned files. For the -src
    # archive, I don't think there is a risk given that we do it via git, but the risk is there for the bundles
    # (src/build is not always clean, examples, editor, ...)

    # Since this script updates files (update_changelog), we need to get those changes propagated to GitHub. I do that
    # by updating the temporary clone then push twice: first from the temporary clone to the "working copy clone" (eg
    # ~/devel/project) then to GitHub from there. The alternative to modify the "working copy clone" directly is worse
    # because it needs more complicated path handling that the 2 push approach.
    do('Cloning repository', call, 'git clone -b {branch} {repository} build'.format(**context))


def check_clone(context):
    chdir(context['build_dir'])

    # check last commit
    print()
    print(call('git log -1'))
    print()

    if no('Does that last commit look right?'):
        exit(1)

    if context['public_release']:
        # check release changes
        print(release_changes(context))
        if no('Does the release changelog look right?'):
            exit(1)


def build_exe(context):
    pass


def test_executables(context):
    pass


def create_archives(context):
    chdir(context['build_dir'])

    release_name = context['release_name']
    create_source_archive(release_name, context['rev'])

    chdir(context['tmp_dir'])

    # copy_release(release_name)
    # create_bundle_archives(release_name)
    # check_bundle_archives(release_name)


def run_tests():
    """
    assumes to be in build
    """
    echocall('pytest')


def update_version(context):
    chdir(context['build_dir'])
    version = short(context['release_name'])

    # meta.yaml
    meta_file = r'condarecipe\larray\meta.yaml'
    changes = [('version: ', "  version: {}".format(version)),
               ('git_tag: ', "  git_tag: {}".format(version))]
    replace_lines(meta_file, changes)

    # __init__.py
    init_file = r'larray\__init__.py'
    changes = [('__version__ =', "__version__ = '{}'".format(version))]
    replace_lines(init_file, changes)

    # setup.py
    setup_file = r'setup.py'
    changes = [('VERSION =', "VERSION = '{}'".format(version))]
    replace_lines(setup_file, changes)

    # check, commit and push
    print(call('git status -s'))
    print(call('git diff {} {} {}'.format(meta_file, init_file, setup_file)))
    if no('Does that last changes look right?'):
        exit(1)
    do('Adding', call, 'git add {} {} {}'.format(meta_file, init_file, setup_file))
    do('Commiting', call, 'git commit -m "bump to version {}"'.format(version))
    print(call('git log -1'))
    do('Pushing to GitHub', call, 'git push origin {branch}'.format(**context))


def update_changelog(context):
    """
    Update release date in changes.rst
    """
    chdir(context['build_dir'])

    if not context['public_release']:
        return

    release_name = context['release_name']
    fpath = r'doc\source\changes.rst'
    with open(fpath) as f:
        lines = f.readlines()
        title = "Version {}".format(short(release_name))
        if lines[5] != title + '\n':
            print("changes.rst not modified (the last release is not {})".format(title))
            return
        release_date = lines[8]
        if release_date != "In development.\n":
            print('changes.rst not modified (the last release date is "{}" '
                  'instead of "In development.", was it already released?)'.format(release_date))
            return
        lines[8] = "Released on {}.\n".format(date.today().isoformat())
    with open(fpath, 'w') as f:
        f.writelines(lines)
    with open(fpath, encoding='utf-8-sig') as f:
        print('\n'.join(f.read().splitlines()[:20]))
    if no('Does the full changelog look right?'):
        exit(1)
    call('git commit -m "update release date in changes.rst" {}'.format(fpath))


def update_version_conda_forge_package(context):
    if not context['public_release']:
        return

    chdir(context['build_dir'])

    # compute sha256 of archive of current release
    version = short(context['release_name'])
    url = LARRAY_REP + '/archive/{version}.tar.gz'.format(version=version)
    print('Computing SHA256 from archive {url}'.format(url=url), end=' ')
    with request.urlopen(url) as response:
        sha256 = hashlib.sha256(response.read()).hexdigest()
        print('done.')
        print('SHA256: ', sha256)

    # set version and sha256 in meta.yml file
    meta_file = r'recipe\meta.yaml'
    changes = [('set version', '{{% set version = "{version}" %}}'.format(version=version)),
               ('set sha256', '{{% set sha256 = "{sha256}" %}}'.format(sha256=sha256))]
    replace_lines(meta_file, changes)

    # add, commit and push
    print(call('git status -s'))
    print(call('git diff {meta_file}'.format(meta_file=meta_file)))
    if no('Does that last changes look right?'):
        exit(1)
    do('Adding', call, 'git add {meta_file}'.format(meta_file=meta_file))
    do('Commiting', call, 'git commit -m "bump to version {version}"'.format(version=version))


def update_metapackage(context):
    if not context['public_release']:
        return

    chdir(context['repository'])
    version = short(context['release_name'])
    do('Updating larrayenv metapackage to version {version}'.format(version=version), call,
       'conda metapackage larrayenv {version} --dependencies "larray =={version}" "larray-editor =={version}" '
       '"larray_eurostat =={version}" qtconsole matplotlib pyqt qtpy pytables xlsxwriter xlrd openpyxl xlwings'
        .format(version=version))


def build_doc(context):
    chdir(context['build_dir'])
    chdir('doc')
    call('buildall.bat')


def final_confirmation(context):
    if not context['public_release']:
        return

    msg = """Is the release looking good? If so, the tag will be created and pushed, everything will be uploaded to 
the production server. Stuff to watch out for:
* version numbers (executable & changelog)
* changelog
* doc on readthedocs
"""
    if no(msg):
        exit(1)


def tag_release(context):
    chdir(context['build_dir'])

    if not context['public_release']:
        return

    call('git tag -a {release_name} -m "tag release {release_name}"'.format(**context))


def push_on_pypi(context):
    chdir(context['build_dir'])

    if not context['public_release']:
        return

    msg = """Ready to push on pypi? If so, command line 
'python setup.py clean register sdist bdist_wheel --universal upload -r pypi' 
will now be executed.
"""
    if no(msg):
        exit(1)
    call('python setup.py clean register sdist bdist_wheel --universal upload -r pypi')


def pull(context):
    if not context['public_release']:
        return

    # pull the changelog commits to the branch (usually master)
    # and the release tag (which refers to the last commit)
    chdir(context['repository'])
    do('Pulling changes in {repository}'.format(**context),
       call, 'git pull --ff-only --tags {build_dir} {branch}'.format(**context))

def push(context):
    if not context['public_release']:
        return

    chdir(context['repository'])
    do('Pushing main repository changes to GitHub',
       call, 'git push origin {branch} --follow-tags'.format(**context))


def pull_conda_forge(context):
    if not context['public_release']:
        return

    chdir(context['build_dir'])
    branch = context['branch']
    repository = context['repository'].replace('larray-project', 'conda-forge')
    do('Rebasing from upstream {branch}'.format(**context),
       call, "git pull --rebase {repository} {branch}".format(repository=repository, branch=branch))


def push_conda_forge(context):
    if not context['public_release']:
        return

    chdir(context['build_dir'])
    do('Pushing changes to GitHub', call, 'git push origin {branch}'.format(**context))


def cleanup(context):
    chdir(context['tmp_dir'])
    rmtree('build')


def set_context_for_conda_forge(context):
    context['repository'] = CONDA_LARRAY_FEEDSTOCK_REP
    context['tmp_dir'] = TMP_PATH_CONDA
    context['build_dir'] = os.path.join(TMP_PATH_CONDA, 'build')


def announce_new_release(context):
    import win32com.client as win32
    version = short(context['release_name'])
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.To = LARRAY_ANNOUNCE_MAILING_LIST
    mail.Subject = "LArray {} released".format(version)
    mail.Body  = """\
Hello all, 

We are pleased to announce that version {version} of LArray is now available. 
The complete description of changes including examples can be found at:

{readthedocs}changes.html#version-{version_doc}



As always, *any* feedback is very welcome, preferably on the larray-users 
mailing list: {larray_users_group} (you need to register to be able to post).
""".format(version=version, readthedocs=LARRAY_READTHEDOCS, version_doc=version.replace('.', '-'),
           larray_users_group=LARRAY_USERS_GROUP)
    mail.Display(True)

# ------------ #
# end of steps #
# ------------ #

steps_funcs = [
    #########################
    # CREATE LARRAY PACKAGE #
    #########################
    (check_local_repo, ''),
    (create_tmp_directory, ''),
    (clone_repository, ''),
    (check_clone, ''),
    (update_version, ''),
    (build_exe, 'Building executables'),
    (test_executables, 'Testing executables'),
    (update_changelog, 'Updating changelog'),
    (create_archives, 'Creating archives'),
    (final_confirmation, ''),
    (tag_release, 'Tagging release'),
    # We used to push from /tmp to the local repository but you cannot push
    # to the currently checked out branch of a repository, so we need to
    # pull changes instead. However pull (or merge) add changes to the
    # current branch, hence we make sure at the beginning of the script
    # that the current git branch is the branch to release. It would be
    # possible to do so without a checkout by using:
    # git fetch {tmp_path} {branch}:{branch}
    # instead but then it only works for fast-forward and non-conflicting
    # changes. So if the working copy is dirty, you are out of luck.
    (pull, ''),
    # >>> need internet from here
    (push, ''),
    (push_on_pypi, 'Pushing on Pypi'),
    # assume the tar archive for the new release exists
    (cleanup, 'Cleaning up'),
    #############################
    # CREATE LARRAY METAPACKAGE #
    #############################
    # assume the tar archive for the new release exists
    (update_metapackage, ''),
    ########################################
    # UPDATE LARRAY PACKAGE ON CONDA-FORGE #
    ########################################
    (set_context_for_conda_forge, 'Setting context in order to update packages on conda-forge'),
    (create_tmp_directory, ''),
    (clone_repository, ''),
    (update_version_conda_forge_package, ''),
    (pull_conda_forge, ''),
    (push_conda_forge, ''),
    (cleanup, 'Cleaning up'),
    ############################################
    # UPDATE LARRAY METAPACKAGE ON CONDA-FORGE #
    ############################################
    # assume larray package on conda-forge has been updated

    ####################
    # ANNOUNCE RELEASE #
    ####################
    (announce_new_release, 'Sending release announce email')
]


def make_release(release_name='dev', steps=':', branch='master'):
    func_names = [f.__name__ for f, desc in steps_funcs]
    if ':' in steps:
        start, stop = steps.split(':')
        start = func_names.index(start) if start else 0
        # + 1 so that stop bound is inclusive
        stop = func_names.index(stop) + 1 if stop else len(func_names)
    else:
        # assuming a single step
        start = func_names.index(steps)
        stop = start + 1

    if release_name != 'dev':
        if 'pre' in release_name:
            raise ValueError("'pre' is not supported anymore, use 'alpha' or 'beta' instead")
        if '-' in release_name:
            raise ValueError("- is not supported anymore")

        release_name = long_release_name(release_name)

    repository = abspath(dirname(__file__))
    rev = git_remote_last_rev(repository, 'refs/heads/{}'.format(branch))
    public_release = release_name != 'dev'
    if not public_release:
        # take first 7 digits of commit hash
        release_name = rev[:7]

    context = {'branch': branch,
               'release_name': release_name,
               'rev': rev,
               'repository': repository,
               'tmp_dir': TMP_PATH,
               'build_dir': os.path.join(TMP_PATH, 'build'),
               'public_release': public_release}
    for step_func, step_desc in steps_funcs[start:stop]:
        if step_desc:
            do(step_desc, step_func, context)
        else:
            step_func(context)


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) < 2:
        print("Usage: {} release_name|dev [step|startstep:stopstep] [branch]".format(argv[0]))
        print("steps:", ', '.join(f.__name__ for f, _ in steps_funcs))
        sys.exit()

    make_release(*argv[1:])
