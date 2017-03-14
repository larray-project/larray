from __future__ import absolute_import, division, print_function

import os
import sys
import traceback
from collections import OrderedDict
import numpy as np
import larray as la

from qtpy.QtWidgets import QApplication, QMainWindow
from larray.viewer.view import MappingEditor, ArrayEditor, SessionComparator, ArrayComparator, REOPEN_LAST_FILE
from larray.viewer.view import Figure, FigureCanvas, PlotDialog

__all__ = ['view', 'edit', 'compare', 'REOPEN_LAST_FILE', 'animate', 'animate_barh']


def qapplication():
    return QApplication(sys.argv)


def find_names(obj, depth=0):
    """Return all names an object is bound to.

    Parameters
    ----------
    obj : object
        the object to find names for.
    depth : int
        depth of call frame to inspect. 0 is where find_names was called,
        1 the caller of find_names, etc.

    Returns
    -------
    list of str
        all names obj is bound to, sorted alphabetically. Can be [] if we
        computed an array just to view it.
    """
    # noinspection PyProtectedMember
    l = sys._getframe(depth + 1).f_locals
    return sorted(k for k, v in l.items() if v is obj)


def get_title(obj, depth=0, maxnames=3):
    """Return a title for an object (a combination of the names it is bound to).

    Parameters
    ----------
    obj : object
        the object to find a title for.
    depth : int
        depth of call frame to inspect. 0 is where get_title was called,
        1 the caller of get_title, etc.

    Returns
    -------
    str
        title for obj. This can be '' if we computed an array just to view it.
    """
    names = find_names(obj, depth=depth + 1)
    # names can be == []
    # eg. view(arr['M'])
    if len(names) > maxnames:
        names = names[:maxnames] + ['...']
    return ', '.join(names)


def edit(obj=None, title='', minvalue=None, maxvalue=None, readonly=False, depth=0):
    """
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
    _app = QApplication.instance()
    if _app is None:
        install_except_hook()
        _app = qapplication()
        _app.setOrganizationName("LArray")
        _app.setApplicationName("Viewer")
        parent = None
    else:
        parent = _app.activeWindow()

    if obj is None:
        local_vars = sys._getframe(depth + 1).f_locals
        obj = OrderedDict([(k, local_vars[k]) for k in sorted(local_vars.keys())])

    if not isinstance(obj, la.Session) and hasattr(obj, 'keys'):
        obj = la.Session(obj)

    if not title and obj is not REOPEN_LAST_FILE:
        title = get_title(obj, depth=depth + 1)

    dlg = MappingEditor(parent) if obj is REOPEN_LAST_FILE or isinstance(obj, (str, la.Session)) else ArrayEditor(parent)
    if dlg.setup_and_check(obj, title=title, minvalue=minvalue, maxvalue=maxvalue, readonly=readonly):
        if parent or isinstance(dlg, QMainWindow):
            dlg.show()
            _app.exec_()
        else:
            dlg.exec_()
    if parent is None:
        restore_except_hook()


def view(obj=None, title='', depth=0):
    """
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
    edit(obj, title=title, readonly=True, depth=depth + 1)


def compare(*args, **kwargs):
    """
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

    Examples
    --------
    >>> a1 = ndtest(3)                                                                                 # doctest: +SKIP
    >>> a2 = ndtest(3) + 1                                                                             # doctest: +SKIP
    >>> compare(a1, a2, title='first comparison')                                                      # doctest: +SKIP
    >>> compare(a1 + 1, a2, title='second comparison', names=['a1+1', 'a2'])                           # doctest: +SKIP
    """
    title = kwargs.pop('title', '')
    names = kwargs.pop('names', None)
    _app = QApplication.instance()
    if _app is None:
        install_except_hook()
        _app = qapplication()
        parent = None
    else:
        parent = _app.activeWindow()

    if any(isinstance(a, la.Session) for a in args):
        dlg = SessionComparator(parent)
        default_name = 'session'
    else:
        dlg = ArrayComparator(parent)
        default_name = 'array'

    def get_name(i, obj, depth=0):
        obj_names = find_names(obj, depth=depth + 1)
        return obj_names[0] if obj_names else '%s %d' % (default_name, i)

    if names is None:
        # depth=2 because of the list comprehension
        names = [get_name(i, a, depth=2) for i, a in enumerate(args)]
    else:
        assert isinstance(names, list) and len(names) == len(args)

    if dlg.setup_and_check(args, names=names, title=title):
        if parent:
            dlg.show()
        else:
            dlg.exec_()
    if parent is None:
        restore_except_hook()


def animate(arr, x_axis=-2, time_axis=-1, repeat=False, interval=200, repeat_delay=None, filepath=None,
            writer='ffmpeg', fps=5, metadata=None, bitrate=None):
    import matplotlib.animation as animation

    if arr.ndim < 2:
        raise ValueError('array should have at least 2 dimensions')

    _app = QApplication.instance()
    if _app is None:
        _app = qapplication()
        parent = None
    else:
        parent = _app.activeWindow()

    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    x_axis = arr.axes[x_axis]
    time_axis = arr.axes[time_axis]
    xdata = np.arange(len(x_axis))
    if arr.ndim == 2:
        arr = arr.expand(la.Axis('dummy', 1))
    arr = arr.transpose(x_axis)
    arr = arr.combine_axes(arr.axes - [time_axis, x_axis], sep=' ')
    initial_data = arr[time_axis.i[0]]
    lines = [ax.plot(xdata, initial_data[row].data, lw=2, label=str(row))[0]
             for row in initial_data.axes[1]]
    # TODO: to stack bars, we need to compute bottom value for each bar (use cumsum(stack_dim))
    # xdata = xdata + 0.5
    # bars = ax.barh(xdata, initial_data[initial_data.axes[1].i[0]].data)

    ax.grid()
    ax.set_ylim(arr.min(), arr.max() * 1.05)
    # set x axis
    ax.set_xlabel(x_axis.name)
    ax.set_xlim(0, len(x_axis) - 1)
    # we need to do that because matplotlib is smart enough to
    # not show all ticks but a selection. However, that selection
    # may include ticks outside the range of x axis
    xticks = [t for t in ax.get_xticks().astype(int) if t <= len(x_axis.labels) - 1]
    xticklabels = [x_axis.labels[j] for j in xticks]
    ax.set_xticklabels(xticklabels)
    ax.legend()
    ax.set_title(str(time_axis.i[0]))

    def run(y):
        data = arr[y].data
        for line, line_data in zip(lines, data.T):
            line.set_data(xdata, line_data)
        ax.set_title(str(y))
        return lines
        # data = arr[y, initial_data.axes[1].i[0]].data
        # for bar, height in zip(bars, data):
        #     bar.set_height(height)
        # ax.set_title(str(y))
        # return bars

    ani = animation.FuncAnimation(fig, run, arr.axes[time_axis], blit=False, interval=interval, repeat=repeat,
                                  repeat_delay=repeat_delay)
    if filepath is None:
        dlg = PlotDialog(canvas, parent)
        if parent:
            dlg.show()
        else:
            dlg.exec_()
    else:
        print("Writing animation to", filepath, '...', end=' ')
        sys.stdout.flush()
        if '.htm' in os.path.splitext(filepath)[1]:
            content = '<html>{}</html>'.format(ani.to_html5_video())
            with open(filepath, mode='w', encoding='utf8') as f:
                f.write(content)
        else:
            Writer = animation.writers[writer]
            writer = Writer(fps=fps, metadata=metadata, bitrate=bitrate)
            ani.save(filepath, writer=writer)
        print("done.")
        return ani


def animate_barh(arr, x_axis=-2, time_axis=-1, title=None, repeat=False, interval=200, repeat_delay=None, filepath=None,
                 writer='ffmpeg', fps=5, metadata=None, bitrate=None):
    import matplotlib.animation as animation

    if arr.ndim < 2:
        raise ValueError('array should have at least 2 dimensions')

    _app = QApplication.instance()
    if _app is None:
        _app = qapplication()
        parent = None
    else:
        parent = _app.activeWindow()

    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    if title is not None:
        # fig.suptitle(title, y=1.05, fontsize=18)
        ax.set_title(title, fontsize=16)

    x_axis = arr.axes[x_axis]
    time_axis = arr.axes[time_axis]
    xdata = np.arange(len(x_axis)) + 0.5
    assert 2 <= arr.ndim <= 3
    if arr.ndim == 2:
        arr = arr.expand(la.Axis('dummy', 1))
    gender_axis = (arr.axes - [time_axis, x_axis])[0]
    arr = arr.transpose(gender_axis, x_axis, time_axis)
    initial_data = arr[time_axis.i[0]]
    nd_bars = []
    for g in gender_axis:
        data = initial_data[g]
        if any(data < 0):
            left = data
        else:
            left = 0
        nd_bars.append(ax.barh(xdata, data.data, left=left, label=str(g)))

    # ax.grid()
    amax = abs(arr).max() * 1.05
    # set x axis
    ax.set_xlim(-amax, amax)
    ax.set_xticklabels([abs(v) for v in ax.get_xticks()])

    # set y axis
    ax.set_ylabel(x_axis.name)
    ax.set_ylim(0, len(x_axis))
    # we need to do that because matplotlib is smart enough to
    # not show all ticks but a selection. However, that selection
    # may include ticks outside the range of x axis
    yticks = [t for t in ax.get_yticks().astype(int) if t <= len(x_axis.labels) - 1]
    yticklabels = [x_axis.labels[j] for j in yticks]
    ax.set_yticks([yt + 0.5 for yt in yticks])
    ax.set_yticklabels(yticklabels)

    ax.legend()
    # ax.set_title(str(time_axis.i[0]))

    def run(y):
        artists = []
        for nd_bar, c in zip(nd_bars, arr.axes[0]):
            data = arr[y, c]
            for bar, width in zip(nd_bar, data):
                if width < 0:
                    bar.set_width(-width)
                    bar.set_x(width)
                else:
                    bar.set_width(width)
            artists.extend(nd_bar)
        if filepath is None:
            artists.append(ax.annotate(str(y), (0.03, 0.92), xycoords='axes fraction', fontsize=16, color=(.2, .2, .2)))
        else:
            ax.set_title('{} ({})'.format(title, str(y)))
        return artists

    def init():
        return run(time_axis.i[0])

    ani = animation.FuncAnimation(fig, run, arr.axes[time_axis], init_func=init, blit=filepath is None,
                                  interval=interval, repeat=repeat, repeat_delay=repeat_delay)
    if filepath is None:
        dlg = PlotDialog(canvas, parent)
        if parent:
            dlg.show()
        else:
            dlg.exec_()
    else:
        print("Writing animation to", filepath, '...', end=' ')
        sys.stdout.flush()
        if '.htm' in os.path.splitext(filepath)[1]:
            content = '<html>{}</html>'.format(ani.to_html5_video())
            with open(filepath, mode='w', encoding='utf8') as f:
                f.write(content)
        else:
            Writer = animation.writers[writer]
            writer = Writer(fps=fps, metadata=metadata, bitrate=bitrate)
            ani.save(filepath, writer=writer)
        print("done.")
        return ani


_orig_except_hook = sys.excepthook


def _qt_except_hook(type, value, tback):
    # only print the exception and do *not* exit the program
    traceback.print_exception(type, value, tback)


def install_except_hook():
    sys.excepthook = _qt_except_hook


def restore_except_hook():
    sys.excepthook = _orig_except_hook


_orig_display_hook = sys.displayhook


def _qt_display_hook(value):
    if isinstance(value, la.LArray):
        view(value)
    else:
        _orig_display_hook(value)


def install_display_hook():
    sys.displayhook = _qt_display_hook


def restore_display_hook():
    sys.displayhook = _orig_display_hook


if __name__ == "__main__":
    """Array editor test"""

    lipro = la.Axis(['P%02d' % i for i in range(1, 16)], 'lipro')
    age = la.Axis('age=0..115')
    sex = la.Axis('sex=M,F')

    vla = 'A11,A12,A13,A23,A24,A31,A32,A33,A34,A35,A36,A37,A38,A41,A42,' \
          'A43,A44,A45,A46,A71,A72,A73'
    wal = 'A25,A51,A52,A53,A54,A55,A56,A57,A61,A62,A63,A64,A65,A81,A82,' \
          'A83,A84,A85,A91,A92,A93'
    bru = 'A21'
    # list of strings
    belgium = la.union(vla, wal, bru)

    geo = la.Axis(belgium, 'geo')

    # data1 = np.arange(30).reshape(2, 15)
    # arr1 = la.LArray(data1, axes=(sex, lipro))
    # edit(arr1)

    # data2 = np.arange(116 * 44 * 2 * 15).reshape(116, 44, 2, 15) \
    #           .astype(float)
    # data2 = np.random.random(116 * 44 * 2 * 15).reshape(116, 44, 2, 15) \
    #           .astype(float)
    # data2 = (np.random.randint(10, size=(116, 44, 2, 15)) - 5) / 17
    # data2 = np.random.randint(10, size=(116, 44, 2, 15)) / 100 + 1567
    # data2 = np.random.normal(51000000, 10000000, size=(116, 44, 2, 15))
    data2 = np.random.normal(0, 1, size=(116, 44, 2, 15))
    arr2 = la.LArray(data2, axes=(age, geo, sex, lipro))
    # arr2 = la.ndrange([100, 100, 100, 100, 5])
    # arr2 = arr2['F', 'A11', 1]

    # view(arr2[0, 'A11', 'F', 'P01'])
    # view(arr1)
    # view(arr2[0, 'A11'])
    # edit(arr1)
    # print(arr2[0, 'A11', :, 'P01'])
    # edit(arr2.astype(int), minvalue=-99, maxvalue=55.123456)
    # edit(arr2.astype(int), minvalue=-99)
    # arr2.i[0, 0, 0, 0] = np.inf
    # arr2.i[0, 0, 1, 1] = -np.inf
    # arr2 = [0.0000111, 0.0000222]
    # arr2 = [0.00001, 0.00002]
    # edit(arr2, minvalue=-99, maxvalue=25.123456)
    # print(arr2[0, 'A11', :, 'P01'])

    # data2 = np.random.normal(0, 10.0, size=(5000, 20))
    # arr2 = la.LArray(data2,
    #                  axes=(la.Axis(list(range(5000)), 'd0'),
    #                        la.Axis(list(range(20)), 'd1')))
    # edit(arr2)

    # view(['a', 'bb', 5599])
    # view(np.arange(12).reshape(2, 3, 2))
    # view([])

    data3 = np.random.normal(0, 1, size=(2, 15))
    arr3 = la.ndrange((30, sex))
    # data4 = np.random.normal(0, 1, size=(2, 15))
    # arr4 = la.LArray(data4, axes=(sex, lipro))

    # arr4 = arr3.copy()
    # arr4['F'] /= 2
    arr4 = arr3.min(la.x.sex)
    arr5 = arr3.max(la.x.sex)
    arr6 = arr3.mean(la.x.sex)

    # test isssue #35
    arr7 = la.from_lists([['a',                   1,                    2,                    3],
                          [ '', 1664780726569649730, -9196963249083393206, -7664327348053294350]])

    # compare(arr3, arr4, arr5, arr6)

    # view(la.stack((arr3, arr4), la.Axis('arrays=arr3,arr4')))
    ses = la.Session(arr2=arr2, arr3=arr3, arr4=arr4, arr5=arr5, arr6=arr6, arr7=arr7,
                     data2=data2, data3=data3)

    # from larray.tests.common import abspath
    # file = abspath('test_session.xlsx')
    # ses.save(file)

    edit(ses)
    # edit(file)
    # edit('fake_path')
    # edit(REOPEN_LAST_FILE)

    # s = la.local_arrays()
    # view(s)
    # print('HDF')
    # s.save('x.h5')
    # print('\nEXCEL')
    # s.save('x.xlsx')
    # print('\nCSV')
    # s.save('x_csv')
    # print('\n open HDF')
    # edit('x.h5')
    # print('\n open EXCEL')
    # edit('x.xlsx')
    # print('\n open CSV')
    # edit('x_csv')

    # compare(arr3, arr4, arr5, arr6)

    # arr3 = la.ndrange((1000, 1000, 500))
    # print(arr3.nbytes * 1e-9 + 'Gb')
    # edit(arr3, minvalue=-99, maxvalue=25.123456)
