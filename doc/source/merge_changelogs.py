#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Append release notes from other projects of the larray-project constellation to the current changelog
"""

import os
import requests

from releaser import relname2fname


# TODO: add Eurostat project
LARRAY_GITHUB_REP = "https://github.com/larray-project/larray"
EDITOR_GITHUB_REP = "https://github.com/larray-project/larray-editor"


def include_changelogs(section_name, release_name, github_rep, rel_changes_dir='/doc/source/changes',
                       branch='master', reset=False):
    changelog_file = relname2fname(release_name)

    # get changelog file content from github repository
    url = github_rep.replace('github.com', 'raw.githubusercontent.com') \
          + '/{}/{}/'.format(branch, rel_changes_dir) + changelog_file
    req = requests.get(url)
    if req.status_code != requests.codes.ok:
        raise ValueError("Content at URL {} could not been found.".format(url))
    github_changelog = req.text

    # append to local changelog file
    changelog_file = os.path.abspath(os.path.join('.', 'changes', changelog_file))
    with open(changelog_file) as f:
        content = f.read() if not reset else ''
    with open(changelog_file, 'w', encoding="utf-8") as f:
        new_section = """

{section_name}
{underline}

""".format(section_name=section_name.upper(), underline='-' * len(section_name))
        content += new_section
        content += github_changelog
        f.write(content)


def merge_changelogs(release_name):
    include_changelogs('CORE', release_name, LARRAY_GITHUB_REP, reset=True)
    include_changelogs('EDITOR', release_name, EDITOR_GITHUB_REP)


if __name__ == '__main__':
    import sys

    argv = sys.argv
    if len(argv) < 2:
        print("Usage: {} release_name".format(argv[0]))
        sys.exit()

    release_name = argv[1]
    merge_changelogs(release_name)
