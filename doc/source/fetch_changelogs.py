#!/usr/bin/env python3
"""
Fetch release notes from satellite projects of the larray-project constellation to the main project.
"""

from pathlib import Path

import requests

from releaser import relname2fname, doechocall, echocall, short, yes

# TODO: add Eurostat project
EDITOR_GITHUB_REP = "https://github.com/larray-project/larray-editor"


def fetch_changelog(section_name, release_name, github_rep, rel_changes_dir='/doc/source/changes', branch='master'):
    fname = relname2fname(release_name)

    # get changelog file content from github repository
    url = github_rep.replace('github.com', 'raw.githubusercontent.com') + f'/{branch}/{rel_changes_dir}/{fname}'
    req = requests.get(url)
    if req.status_code != requests.codes.ok:
        raise ValueError(f"Content at URL {url} could not be found.")
    # save in local directory
    fpath = Path('.') / 'changes' / section_name / fname
    fpath.write_text(req.text, encoding="utf-8")

    doechocall('Adding', ['git', 'add', str(fpath)])
    doechocall('Committing',
               ['git', 'commit', '-m', f'fetched {section_name} changelog for {short(release_name)}', str(fpath)])


def fetch_changelogs(release_name, branch='master'):
    fetch_changelog('editor', release_name, EDITOR_GITHUB_REP, branch=branch)

    print(echocall(['git', 'log', f'origin/{branch}..HEAD']))
    if yes('Are the above commits ready to be pushed?', default='n'):
        doechocall('Pushing changes to GitHub',
                   ['git', 'push', 'origin', branch, '--follow-tags'])


if __name__ == '__main__':
    import sys

    argv = sys.argv
    if len(argv) < 2:
        print(f"Usage: {argv[0]} release_name [branch]")
        sys.exit()

    fetch_changelogs(*argv[1:])
