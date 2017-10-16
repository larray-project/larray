#!/usr/bin/python
# encoding: utf-8
# script to start a new release cycle
# Licence: GPLv3
from os.path import join, abspath, dirname
from make_release import relname2fname, short, call, do, no, push, update_version
from shutil import copy


def update_changelog(release_name):
    fname = relname2fname(release_name)

    # create "empty" changelog for that release
    changes_dir = r'doc\source\changes'
    copy(join(changes_dir, 'template.rst.inc'),
         join(changes_dir, fname))

    # include release changelog in changes.rst
    fpath = r'doc\source\changes.rst'
    changelog_index_template = """{title}
{underline}

In development.

.. include:: {fpath}


"""

    with open(fpath) as f:
        lines = f.readlines()
        title = "Version {}".format(short(release_name))
        if lines[3] == title + '\n':
            print("changes.rst not modified (it already contains {})".format(title))
            return
        this_version = changelog_index_template.format(title=title,
                                                       underline="=" * len(title),
                                                       fpath='./changes/' + fname)
        lines[3:3] = this_version.splitlines(True)
    with open(fpath, 'w') as f:
        f.writelines(lines)
    with open(fpath, encoding='utf-8-sig') as f:
        print('\n'.join(f.read().splitlines()[:20]))
    if no('Does the full changelog look right?'):
        exit(1)
    call('git add {}'.format(fpath))


def add_release(release_name, branch='master'):
    update_changelog(release_name)
    context = {'branch': branch,
               'release_name': release_name+'-dev',
               'repository': abspath(dirname(__file__)),
               'build_dir': abspath(dirname(__file__)),
               'public_release': True}
    update_version(context)
    push(context)


if __name__ == '__main__':
    from sys import argv

    add_release(*argv[1:])
