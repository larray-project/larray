#!/usr/bin/python
# encoding: utf-8
# script to start a new release cycle
# Licence: GPLv3
from os.path import join
from make_release import relname2fname, short
from shutil import copy


def add_release(release_name):
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
        title = "Version %s" % short(release_name)
        if lines[3] == title + '\n':
            print("changes.rst not modified (it already contains %s)" % title)
            return
        this_version = changelog_index_template.format(title=title,
                                                       underline="=" * len(title),
                                                       fpath='./changes/' + fname)
        lines[3:3] = this_version.splitlines(True)
    with open(fpath, 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    from sys import argv

    add_release(*argv[1:])
