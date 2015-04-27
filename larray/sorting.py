# coding: utf-8

from collections import defaultdict

from larray.oset import OrderedSet as oset
from larray.utils import multi_index_from_arrays


def _get_deps(idx_columns):
    nb_index = len(idx_columns)
    combseen = [set() for i in range(nb_index)]
    curcomb = [None for i in range(nb_index)]
    curvalue = [None for i in range(nb_index)]
    deps = [defaultdict(set) for i in range(nb_index)]

    for ndvalue in zip(*idx_columns):
        for level, v in enumerate(ndvalue):
            level_combseen = combseen[level]
            subcomb = ndvalue[:level]
            if subcomb != curcomb[level]:
                if subcomb in level_combseen:
                    raise ValueError("bad order: %s" % str(subcomb))
                else:
                    curvalue[level] = None
                    level_combseen.add(subcomb)
                    curcomb[level] = subcomb
            level_curvalue = curvalue[level]
            if v != level_curvalue:
                if level_curvalue is not None:
                    deps[level][v].add(level_curvalue)
                curvalue[level] = v
    return deps


# adapted from SQLAlchemy/util/topological.py
def topological_sort(allvalues, dependencies):
    out = []
    todo = oset(allvalues)
    while todo:
        step_out = []
        for value in todo:
            if todo.isdisjoint(dependencies[value]):
                step_out.append(value)
        if not step_out:
            raise ValueError("Circular dependency detected")
        todo.difference_update(step_out)
        out.extend(step_out)
    return out


def get_topological_index(df, index_col):
    idx_columns = [df.iloc[:, i] for i in index_col]
    deps = _get_deps(idx_columns)
    categories = [topological_sort(level_values, level_deps)
                  for level_values, level_deps
                  in zip(idx_columns, deps)]
    return multi_index_from_arrays(idx_columns, len(idx_columns),
                                   names=df.columns[index_col],
                                   categories=categories)


def set_topological_index(df, index_col, drop=True, inplace=False):
    if not inplace:
        df = df.copy()

    df.index = get_topological_index(df, index_col)
    if drop:
        colnames = df.columns[index_col]
        for name in colnames:
            del df[name]