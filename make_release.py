#!/usr/bin/python
# coding=utf-8
# Licence: GPLv3
from __future__ import print_function, unicode_literals

import errno
import fnmatch
import os
import re
import stat
import subprocess
import sys
import zipfile

from datetime import date
from os import chdir, makedirs
from os.path import exists, getsize, abspath, dirname
from shutil import copytree, copy2, rmtree as _rmtree
from subprocess import check_output, STDOUT, CalledProcessError


try:
    input = raw_input
except NameError:
    pass

if sys.version < '3':
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
        return "%.2f %s" % (value, unit)
    else:
        return "%d %s" % (value, unit)


def generate(fname, **kwargs):
    with open('%s.tmpl' % fname) as in_f, open(fname, 'w') as out_f:
        out_f.write(in_f.read().format(**kwargs))


def _remove_readonly(function, path, excinfo):
    if function in (os.rmdir, os.remove) and excinfo[1].errno == errno.EACCES:
        # add write permission to owner
        os.chmod(path, stat.S_IWUSR)
        # retry removing
        function(path)
    else:
        raise


def rmtree(path):
    _rmtree(path, onerror=_remove_readonly)


def call(*args, **kwargs):
    try:
        res = check_output(*args, stderr=STDOUT, **kwargs)
        if sys.version >= '3' and 'universal_newlines' not in kwargs:
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
    output = call('git ls-remote %s %s' % (url, branch))
    for line in output.splitlines():
        if line.endswith(branch):
            return line.split()[0]
    raise Exception("Could not determine revision number")


def yes(msg, default='y'):
    choices = ' (%s/%s) ' % tuple(c.capitalize() if c == default else c
                                  for c in ('y', 'n'))
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


def strip_pretags(release_name):
    """
    removes pre-release tags from a version string

    >>> str(strip_pretags('0.8'))
    '0.8'
    >>> str(strip_pretags('0.8alpha25'))
    '0.8'
    >>> str(strip_pretags('0.8.1rc1'))
    '0.8.1'
    """
    # 'a' needs to be searched for after 'beta'
    for tag in ('rc', 'c', 'beta', 'b', 'alpha', 'a'):
        release_name = re.sub(tag + '\d+', '', release_name)
    return release_name


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
    return any(tag in release_name for tag in ('rc', 'c', 'beta', 'b', 'alpha', 'a'))


# -------------------- #
# end of generic tools #
# -------------------- #

def relname2fname(release_name):
    short_version = short(strip_pretags(release_name))
    return r"version_%s.rst.inc" % short_version.replace('.', '_')


def release_changes(release_name):
    fpath = r"doc\source\changes\\" + relname2fname(release_name)
    with open(fpath, encoding='utf-8-sig') as f:
        return f.read()


def build_doc():
    chdir('doc')
    call('buildall.bat')
    chdir('..')


def update_changelog(release_name):
    fname = relname2fname(release_name)

    # include version changelog in changes.rst
    fpath = r'doc\source\changes.rst'
    changelog_template = """{title}
{underline}

Released on {date}.

.. include:: {fpath}


"""

    with open(fpath) as f:
        lines = f.readlines()
        title = "Version %s" % short(release_name)
        if lines[5] == title + '\n':
            print("changes.rst not modified (it already contains %s)" % title)
            return
        variables = dict(title=title,
                         underline="=" * len(title),
                         date=date.today().isoformat(),
                         fpath='changes/' + fname)
        this_version = changelog_template.format(**variables)
        lines[5:5] = this_version.splitlines(True)
    with open(fpath, 'w') as f:
        f.writelines(lines)
    with open(fpath, encoding='utf-8-sig') as f:
        print()
        print('\n'.join(f.read().splitlines()[:20]))
    if no('Does the full changelog look right?'):
        exit(1)
    call('git commit -m "include release changes (%s) in changes.rst" %s' % (fname, fpath))


def run_tests():
    """
    assumes to be in build
    """
    call('nosetests -v --with-doctest')


def make_release(release_name=None, branch='master'):
    if release_name is not None:
        if 'pre' in release_name:
            raise ValueError("'pre' is not supported anymore, use 'alpha' or 'beta' instead")
        if '-' in release_name:
            raise ValueError("- is not supported anymore")

    # releasing from the local clone has the advantage I can prepare the
    # release offline and only push and upload it when I get back online
    repository = abspath(dirname(__file__))
    s = "Using local repository at: %s !" % repository
    print("\n", s, "\n", "=" * len(s), "\n", sep='')

    status = call('git status -s')
    lines = status.splitlines()
    if lines:
        uncommited = sum(1 for line in lines if line.startswith(' M'))
        untracked = sum(1 for line in lines if line.startswith('??'))
        print('Warning: there are %d files with uncommitted changes and %d untracked files:' % (uncommited, untracked))
        print(status)
        if no('Do you want to continue?'):
            exit(1)

    ahead = call('git log --format=format:%%H origin/%s..%s' % (branch, branch))
    num_ahead = len(ahead.splitlines())
    print("Branch '%s' is %d commits ahead of 'origin/%s'" % (branch, num_ahead, branch), end='')
    if num_ahead:
        if yes(', do you want to push?'):
            do('Pushing changes', call, 'git push')
    else:
        print()

    rev = git_remote_last_rev(repository, 'refs/heads/%s' % branch)

    public_release = release_name is not None
    if release_name is None:
        # take first 7 digits of commit hash
        release_name = rev[:7]

    if no('Release version %s (%s)?' % (release_name, rev)):
        exit(1)

    chdir(r'c:\tmp')
    if exists('larray_new_release'):
        rmtree('larray_new_release')
    makedirs('larray_new_release')
    chdir('larray_new_release')

    # make a temporary clone in /tmp. The goal is to make sure we do not include extra/unversioned files. For the -src
    # archive, I don't think there is a risk given that we do it via git, but the risk is there for the bundles
    # (src/build is not always clean, examples, editor, ...)

    # Since this script updates files (update_changelog), we need to get those changes propagated to GitHub. I do that
    # by updating the temporary clone then push twice: first from the temporary clone to the "working copy clone" (eg
    # ~/devel/project) then to GitHub from there. The alternative to modify the "working copy clone" directly is worse
    # because it needs more complicated path handling that the 2 push approach.
    do('Cloning', call, 'git clone -b %s %s build' % (branch, repository))

    # ---------- #
    chdir('build')
    # ---------- #

    print()
    print(call('git log -1'))
    print()

    if no('Does that last commit look right?'):
        exit(1)

    if public_release:
        print(release_changes(release_name))
        if no('Does the release changelog look right?'):
            exit(1)

    if public_release:
        test_release = True
    else:
        test_release = yes('Do you want to test the executables after they are created?')

    if test_release:
        do('Testing release', run_tests)

    if public_release:
        do('Updating changelog', update_changelog, release_name)

    do('Building doc', build_doc)

    do('Creating source archive', call,
       r'git archive --format zip --output ..\LARRAY-%s-src.zip %s' % (release_name, rev))

    # ------- #
    chdir('..')
    # ------- #

    if public_release:
        if no('Is the release looking good? If so, the tag will be created and pushed.'):
            exit(1)

        # ---------- #
        chdir('build')
        # ---------- #

        do('Tagging release', call, 'git tag -a v%(name)s -m "tag release %(name)s"' % {'name': release_name})
        # push the changelog commits to the branch (usually master)
        # and the release tag (which refers to the last commit)
        do('Pushing to %s' % repository, call, 'git push origin %s --follow-tags' % branch)

        # ------- #
        chdir('..')
        # ------- #

    if public_release:
        chdir(repository)
        do('Pushing to GitHub', call, 'git push origin %s --follow-tags' % branch)


if __name__ == '__main__':
    from sys import argv

    # chdir(r'c:\tmp')
    # chdir('larray_new_release')
    make_release(*argv[1:])
