from __future__ import absolute_import, division, print_function


def view(obj=None, title='', depth=0):
    r"""
    Opens a new viewer window. Arrays are loaded in readonly mode and their content cannot be modified.

    Parameters
    ----------
    obj : np.ndarray, LArray, Session, dict or str, optional
        Object to visualize. If string, array(s) will be loaded from the file given as argument.
        Defaults to the collection of all local variables where the function was called.
    title : str, optional
        Title for the current object. Defaults to the name of the first object found in the caller namespace which
        corresponds to `obj` (it will use a combination of the 3 first names if several names correspond to the same
        object).
    depth : int, optional
        Stack depth where to look for variables. Defaults to 0 (where this function was called).

    Examples
    --------
    >>> a1 = ndtest(3)                                                                                 # doctest: +SKIP
    >>> a2 = ndtest(3) + 1                                                                             # doctest: +SKIP
    >>> # will open a viewer showing all the arrays available at this point
    >>> # (a1 and a2 in this case)
    >>> view()                                                                                         # doctest: +SKIP
    >>> # will open a viewer showing only a1
    >>> view(a1)                                                                                       # doctest: +SKIP
    """
    try:
        from larray_editor import view
        view(obj, title, depth + 1)
    except ImportError:
        raise Exception('view() is not available because the larray_editor package is not installed')


def edit(obj=None, title='', minvalue=None, maxvalue=None, readonly=False, depth=0):
    r"""
    Opens a new editor window.

    Parameters
    ----------
    obj : np.ndarray, LArray, Session, dict, str or REOPEN_LAST_FILE, optional
        Object to visualize. If string, array(s) will be loaded from the file given as argument.
        Passing the constant REOPEN_LAST_FILE loads the last opened file.
        Defaults to the collection of all local variables where the function was called.
    title : str, optional
        Title for the current object. Defaults to the name of the first object found in the caller namespace which
        corresponds to `obj` (it will use a combination of the 3 first names if several names correspond to the same
        object).
    minvalue : scalar, optional
        Minimum value allowed.
    maxvalue : scalar, optional
        Maximum value allowed.
    readonly : bool, optional
        Whether or not editing array values is forbidden. Defaults to False.
    depth : int, optional
        Stack depth where to look for variables. Defaults to 0 (where this function was called).

    Examples
    --------
    >>> a1 = ndtest(3)                                                                                 # doctest: +SKIP
    >>> a2 = ndtest(3) + 1                                                                             # doctest: +SKIP
    >>> # will open an editor with all the arrays available at this point
    >>> # (a1 and a2 in this case)
    >>> edit()                                                                                         # doctest: +SKIP
    >>> # will open an editor for a1 only
    >>> edit(a1)                                                                                       # doctest: +SKIP
    """
    try:
        from larray_editor import edit

        edit(obj, title, minvalue, maxvalue, readonly, depth + 1)
    except ImportError:
        raise Exception('edit() is not available because the larray_editor package is not installed')


def compare(*args, **kwargs):
    r"""
    Opens a new comparator window, comparing arrays or sessions.

    Parameters
    ----------
    *args : LArrays or Sessions
        Arrays or sessions to compare.
    title : str, optional
        Title for the window. Defaults to ''.
    names : list of str, optional
        Names for arrays or sessions being compared. Defaults to the name of the first objects found in the caller
        namespace which correspond to the passed objects.
    depth : int, optional
        Stack depth where to look for variables. Defaults to 0 (where this function was called).

    Examples
    --------
    >>> a1 = ndtest(3)                                                                                 # doctest: +SKIP
    >>> a2 = ndtest(3) + 1                                                                             # doctest: +SKIP
    >>> compare(a1, a2, title='first comparison')                                                      # doctest: +SKIP
    >>> compare(a1 + 1, a2, title='second comparison', names=['a1+1', 'a2'])                           # doctest: +SKIP
    """
    try:
        from larray_editor import compare

        depth = kwargs.pop('depth', 0)
        compare(*args, depth=depth + 1, **kwargs)
    except ImportError:
        raise Exception('compare() is not available because the larray_editor package is not installed')


def run_editor_on_exception(root_path=None, usercode_traceback=True):
    r"""
    Runs the editor when an unhandled exception (a fatal error) happens.

    Parameters
    ----------
    root_path : str, optional
        Defaults to None (the directory of the main script).
    usercode_traceback : bool, optional
        Whether or not to show only the part of the traceback (error log) which corresponds to the user code.
        Otherwise, it will show the complete traceback, including code inside libraries. Defaults to True.

    Notes
    -----
    sets sys.excepthook
    """
    try:
        from larray_editor import run_editor_on_exception

        run_editor_on_exception(root_path=root_path, usercode_traceback=usercode_traceback)
    except ImportError:
        raise Exception('run_editor_on_exception() is not available because the larray_editor package is not installed')
