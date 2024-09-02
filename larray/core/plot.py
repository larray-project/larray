from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from larray.core.abstractbases import ABCArray
from larray.core.axis import Axis, AxisCollection
from larray.core.group import Group, IGroup
from larray.util.misc import deprecate_kwarg


def _use_pandas_plot_docstring(f):
    f.__doc__ = getattr(pd.DataFrame.plot, f.__name__).__doc__
    return f


class PlotObject:
    __slots__ = ('array',)

    def __init__(self, array):
        self.array = array

    @staticmethod
    def _handle_x_y_axes(axes, animate, subplots, x, y):
        label_axis = None

        if np.isscalar(x) and x not in axes:
            x_label_axis, x_indices = axes._translate_axis_key(x)
            x = IGroup(x_indices, axis=x_label_axis)
            label_axis = x_label_axis

        if np.isscalar(y) and y not in axes:
            y_label_axis, y_indices = axes._translate_axis_key(y)
            y = IGroup(y_indices, axis=y_label_axis)
            if label_axis is not None and y_label_axis is not label_axis:
                raise ValueError(f'{x} and {y} are labels from different axes')
            label_axis = y_label_axis

        def handle_axes_arg(avail_axes, arg):
            if arg is not None:
                arg = avail_axes[arg]
                avail_axes = avail_axes - arg
                if isinstance(arg, Axis):
                    arg = AxisCollection([arg])
            return avail_axes, arg

        available_axes = axes
        if animate:
            available_axes, animate_axes = handle_axes_arg(available_axes, animate)
        else:
            animate_axes = AxisCollection()

        if label_axis is not None:
            available_axes = available_axes - label_axis
        else:
            available_axes, x = handle_axes_arg(available_axes, x)
            available_axes, y = handle_axes_arg(available_axes, y)

        if subplots is True:
            # use last available axis by default
            subplots = [-1]

        if subplots:
            available_axes, subplot_axes = handle_axes_arg(available_axes, subplots)
        else:
            subplot_axes = AxisCollection()

        if label_axis is not None:
            series_axes = available_axes[:-1]
            if y is None:
                # create a Group with the labels of label_axis not used for x
                # the weird construction is to get a Group (and not an Axis) but avoid getting an LSet
                # which would evaluate to an OrderedSet which is not supported by later code
                y = label_axis.i[label_axis[:].difference(x).translate()]
        else:
            if x is None and y is None:
                # use last available axis by default
                x = available_axes[[-1]]
                series_axes = available_axes - x
            elif x is None:
                x = available_axes
                series_axes = y
                y = None
            elif y is None:
                series_axes = available_axes
            else:
                if available_axes:
                    raise ValueError(f"some axes are not used: {available_axes}")
                series_axes = y
                y = None
            assert isinstance(x, AxisCollection)
            assert isinstance(series_axes, AxisCollection)
            assert isinstance(subplot_axes, AxisCollection)
            assert y is None

        return animate_axes, subplot_axes, x, y, series_axes

    @staticmethod
    def _to_pd_obj(array):
        if array.ndim == 1:
            return array.to_series()
        else:
            return array.to_frame()

    @staticmethod
    def _plot_heat_map(array, x=None, y=None, numhaxes=1, axes_names=True, maxticks=10, ax=None,
                       # TODO: we *might* want to default to False for wildcard axes (for label axes, even
                       #       numeric ones, an inverted axis is more natural)
                       # TODO: rename to topdown_yaxis or zero_top_yaxis or y0_top or whatever where
                       #       the name actually helps knowing the direction
                       invert_yaxis=True,
                       x_ticks_top=True, colorbar=False, **kwargs):
        from larray.util.plot import MaxNMultipleWithOffsetLocator

        assert ax is not None

        # TODO: check if we should handle those here???
        kwargs.pop('kind')
        kwargs.pop('legend')
        # This is needed to support plotting using imshow (see below)
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'auto'
        if 'origin' not in kwargs:
            kwargs['origin'] = 'lower'
        title = kwargs.pop('title', None)
        if title is not None:
            ax.set_title(title)
        if array.ndim < 2:
            array = array.expand(Axis([''], ''))

        # TODO: see how much of this is already handled in _plot_array
        axes = array.axes
        if x is None and y is None:
            x = axes[:-numhaxes]

        if y is None:
            y = array.axes - x
        else:
            if isinstance(y, str):
                y = [y]
            y = array.axes[y]

        if x is None:
            x = array.axes - y
        else:
            if isinstance(x, str):
                x = [x]
            x = array.axes[x]

        array = array.transpose(y + x).combine_axes([y, x])

        # block size is the size of the other (non-first) combined axes
        x_block_size = int(x[1:].size)
        y_block_size = int(y[1:].size)
        c = ax.imshow(array.data, **kwargs)

        # place major ticks in the middle of blocks so that labels are centered
        xlabels = x[0].labels
        ylabels = y[0].labels

        def format_x_tick(tick_val, tick_pos):
            label_index = int(tick_val) // x_block_size
            return xlabels[label_index] if label_index < len(xlabels) else '<bad tick>'

        def format_y_tick(tick_val, tick_pos):
            label_index = int(tick_val) // y_block_size
            return ylabels[label_index] if label_index < len(ylabels) else '<bad tick>'

        # A FuncFormatter is created automatically.
        ax.xaxis.set_major_formatter(format_x_tick)
        ax.yaxis.set_major_formatter(format_y_tick)

        if invert_yaxis:
            ax.invert_yaxis()

        # offset=0 because imshow has some kind of builtin offset
        x_locator = MaxNMultipleWithOffsetLocator(min(maxticks, len(xlabels)), offset=0)
        y_locator = MaxNMultipleWithOffsetLocator(min(maxticks, len(ylabels)), offset=0)
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)

        if x_ticks_top:
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')

        # enable grid lines for minor ticks on axes when we have several "levels" for that axis
        if len(x) > 1:
            # place minor ticks for grid lines between each block on the main axis
            ax.set_xticks(np.arange(x_block_size, x.size, x_block_size), minor=True)
            ax.grid(True, axis='x', which='minor')
            # hide all ticks on x axis
            ax.tick_params(axis='x', which='both', bottom=False, top=False)

        if len(y) > 1:
            ax.set_yticks(np.arange(y_block_size, y.size, y_block_size), minor=True)
            ax.grid(True, axis='y', which='minor')
            # hide all ticks on y axis
            ax.tick_params(axis='y', which='both', left=False, right=False)

        # set axes names
        if axes_names:
            ax.set_xlabel('\n'.join(x.names))
            ax.set_ylabel('\n'.join(y.names))

        if colorbar:
            ax.figure.colorbar(c)
        return ax

    @staticmethod
    def _plot_array(array, *args, x=None, y=None, series=None, _x_axes_last=False, **kwargs):
        kind = kwargs.get('kind', 'line')
        if kind is None:
            kind = 'line'
        # heatmaps are special because they do not go via Pandas
        if kind == 'heatmap':
            return PlotObject._plot_heat_map(array, x=x, y=y, **kwargs)

        label_axis = None
        if array.ndim == 1:
            pass
        elif isinstance(x, AxisCollection):
            # FIXME: arr.plot(x='b', y='d1', subplots='a') does not work
            # (and since arr.plot(y='d1', subplots='a') defaults to x='c' we can't get this easily
            # XXX: arr.plot(y='d', subplots=True) => x=(a, b), subplots='c'
            #      I wonder if x='a', subplots=(b, c) wouldn't be better?
            assert y is None
            # move x_axes first
            array = array.transpose(x)
            array = array.combine_axes(x, sep=' ') if len(x) >= 2 else array
            if _x_axes_last:
                # move combined axis last
                array = array.transpose(..., array.axes[0])
            x = None
        else:
            assert (x is None or isinstance(x, Group)) and (y is None or isinstance(y, Group))
            if isinstance(x, Group):
                label_axis = x.axis
                x = x.eval()
            if isinstance(y, Group):
                label_axis = y.axis
                y = y.eval()

            if label_axis is not None:
                # move label_axis last (it must be a dataframe column)
                array = array.transpose(..., label_axis)

        lineplot = kind == 'line'
        # TODO: why don't we handle all line plots this way?
        if lineplot and label_axis is not None and series is not None and len(series) > 0:
            # the problem with this approach (n calls to pandas.plot) is that the color
            # cycling and "stacked" bar/area of pandas break for all kinds of plots except "line"
            # when we have more than one dimension involved
            for series_key, series_data in array.items(series):
                # workaround for issue #1076 (see below for more details)
                # matplotlib default behavior when there are few ticks is to interpolate
                # them, which looks pretty silly for integer types
                x_labels = series_data.axes[0].labels
                if len(x_labels) < 6 and np.issubdtype(x_labels.dtype, np.integer) and 'xticks' not in kwargs:
                    kwargs['xticks'] = x_labels
                series_name = ' '.join(str(k) for k in series_key)
                # support for list-like y
                if isinstance(y, (list, np.ndarray)):
                    label = [f'{series_name} {y_label}' for y_label in y]
                else:
                    label = f'{series_name} {y}'
                PlotObject._to_pd_obj(series_data).plot(*args, x=x, y=y, label=label, **kwargs)
            return kwargs['ax']
        else:
            # this version works fine for all kinds of plots as long as we only use axes and not labels
            if series is not None and len(series) >= 1:
                # move series axes first and combine them
                array = array.transpose(series).combine_axes(series, sep=' ')
                # move it last (as columns) unless we need x axes or label axis last
                if not _x_axes_last and label_axis is None:
                    array = array.transpose(..., array.axes[0])

            # workaround for issue #1076 (see above)
            # We only do it for line and area charts because specifying xticks
            # breaks bar plots (and bar plots do not suffer from the issue anyway)
            # (see https://github.com/pandas-dev/pandas/issues/55508)
            if kind in {'line', 'area'}:
                x_labels = array.axes[0].labels
                if len(x_labels) < 6 and np.issubdtype(x_labels.dtype, np.integer) and 'xticks' not in kwargs:
                    kwargs['xticks'] = x_labels

            return PlotObject._to_pd_obj(array).plot(*args, x=x, y=y, **kwargs)

    @deprecate_kwarg('stacked', 'stack')
    def __call__(self, x=None, y=None, ax=None, subplots=False, layout=None, figsize=None,
                 sharex=None, sharey=False, tight_layout=None, constrained_layout=None, title=None, legend=None,
                 animate=None, filepath=None, show=None, **kwargs):
        from matplotlib import pyplot as plt

        array = self.array
        legend_kwargs = legend if isinstance(legend, dict) else {}

        if 'stack' in kwargs:
            if y is not None:
                raise ValueError("in Array.plot(), cannot use both the 'y' argument and "
                                 "give axes for the 'stack' argument")
            # checking that 'stacked' is not also given is done in deprecate_kwarg
            y = kwargs.pop('stack')
            if y is True:
                y = None
                warnings.warn("in Array.plot(), using stack=True is deprecated, please use "
                              "stack=axis_name instead", FutureWarning)
            kwargs['stacked'] = True

        animate_axes, subplot_axes, x, y, series_axes = PlotObject._handle_x_y_axes(array.axes, animate, subplots, x, y)
        if show is None:
            show = filepath is None and ax is None
        if constrained_layout is None and tight_layout is None:
            constrained_layout = True

        if ax is None:
            fig = plt.figure(figsize=figsize, tight_layout=tight_layout, constrained_layout=constrained_layout)
            if subplots or layout is not None:
                if layout is None:
                    subplots_shape = subplot_axes.shape
                    if len(subplots_shape) > 2:
                        # default to last axis horizontal, other axes combined vertically
                        layout = np.prod(subplots_shape[:-1]), subplots_shape[-1]
                    else:
                        layout = subplot_axes.shape
                if sharex is None:
                    sharex = True
                ax_to_return = fig.subplots(*layout, sharex=sharex, sharey=sharey)
                ax = ax_to_return if subplots else ax_to_return.flat[0]
            else:
                ax = fig.add_subplot()
                ax_to_return = ax
        else:
            fig = ax.figure
            ax_to_return = ax

        anim_kwargs = kwargs.pop('anim_params', {})
        if animate:
            from matplotlib.animation import FuncAnimation

            if subplots:
                def run(t):
                    for subplot_ax in ax.flat:
                        subplot_ax.clear()
                    self._plot_many(array[t], ax, kwargs, series_axes, subplot_axes, title, x, y)
            else:
                def run(t):
                    ax.clear()
                    self._plot_many(array[t], ax, kwargs, series_axes, subplot_axes, title, x, y)
            # TODO: add support for interpolation between frames/labels. Would be best to implement this via
            #       a generic interpolation API in larray though.
            #  see https://github.com/julkaar9/pynimate for inspiration
            ani = FuncAnimation(fig, run, frames=animate_axes.iter_labels())
        else:
            ani = None
            self._plot_many(array, ax, kwargs, series_axes, subplot_axes, title, x, y)

        if legend or legend is None:
            first_ax = ax.flat[0] if subplots else ax
            handles, labels = first_ax.get_legend_handles_labels()
            if legend is None:
                # if there is a single series (per plot), a legend is useless
                legend = len(handles) > 1 or legend_kwargs

            if legend:
                if 'title' not in legend_kwargs:
                    axes_names = series_axes.names
                    # if y is a label (not an axis), this counts as an extra axis as far as the legend is concerned
                    if isinstance(y, Group):
                        axes_names += y.axis.name
                    legend_kwargs['title'] = ' '.join(axes_names)
                # use figure to place legend to add a single legend for all subplots
                legend_parent = first_ax.figure if subplots else ax
                legend_parent.legend(handles, labels, **legend_kwargs)

        if filepath is not None:
            if ani is None:
                fig.savefig(filepath)
            else:
                if not isinstance(filepath, Path):
                    filepath = Path(filepath)
                if filepath.suffix in {'.htm', '.html'}:
                    # TODO: we should offer the option to use to_jshtml instead of to_html5_video. Even if it makes the
                    #       files (much) bigger (because they are stored as individual frames) it also adds some useful
                    #       play/pause/next frame/... buttons.
                    filepath.write_text(f'<html>{ani.to_html5_video()}</html>', encoding='utf8')
                else:
                    writer = anim_kwargs.pop('writer', None)
                    fps = anim_kwargs.pop('fps', 5)
                    metadata = anim_kwargs.pop('metadata', None)
                    bitrate = anim_kwargs.pop('bitrate', None)
                    if writer is None:
                        # pillow only supports .gif, .png and .tiff ffmpeg supports .avi, .mov, .mp4 (but needs the
                        # ffmpeg package installed)
                        writer = 'pillow' if filepath.suffix in {'.gif', '.png', '.tiff'} else 'ffmpeg'
                    from matplotlib.animation import writers
                    if not writers.is_available(writer):
                        raise Exception(f"Cannot write animation using '{filepath.suffix}' extension "
                                        f"because '{writer}' writer is not available.\n"
                                        "Installing an optional package is probably necessary.")

                    ani.save(filepath, writer=writer, fps=fps, metadata=metadata, bitrate=bitrate)

        if show:
            # The following line displays the plot window. Note however that
            # it is only blocking when no Qt loop is already running (i.e. we are
            # not running inside the editor).
            # When using the Qt backend this boils down to:
            #     manager = fig.canvas.manager
            #     manager.show()
            #     if block:
            #         manager.start_main_loop()
            # the last line just gets the current Qt QApplication instance (created during
            # the first canvas creation) and .exec() it
            plt.show(block=True)
            # It is important to return ani, because otherwise in the non-blocking case
            # (i.e. when run in the editor), the animation is garbage-collected before
            # it is drawn, and we get a blank animation.
            return (ax_to_return, ani) if ani is not None else ax_to_return
        elif filepath is not None:  # filepath and not show
            plt.close(fig)
            return None
        else:                       # no filepath and not show
            return (ax_to_return, ani) if ani is not None else ax_to_return

    def _plot_many(self, array, ax, kwargs, series_axes, subplot_axes, title, x, y):
        if len(subplot_axes):
            num_subplots = subplot_axes.size
            if not isinstance(ax, (np.ndarray, ABCArray)) or ax.size < num_subplots:
                raise ValueError(f"ax argument value is not compatible with subplot axes ({subplot_axes})")
            # it is easier to always work with a flat array
            flat_ax = ax.flat
            if title is not None:
                fig = flat_ax[0].figure
                fig.suptitle(title)
            # remove blank plot(s) at the end, if any
            if len(flat_ax) > num_subplots:
                for plot_ax in flat_ax[num_subplots:]:
                    plot_ax.remove()
                # this not strictly necessary but is cleaner in case we reuse flat_ax
                flat_ax = flat_ax[:num_subplots]
            if kwargs.get('kind') == 'heatmap' and 'x_ticks_top' not in kwargs:
                kwargs['x_ticks_top'] = False
            for i, (ndkey, subarr) in enumerate(array.items(subplot_axes)):
                subplot_title = ' '.join(str(ak) for ak in ndkey)
                self._plot_array(subarr, x=x, y=y, series=series_axes, ax=flat_ax[i], legend=False, title=subplot_title,
                                 **kwargs)
        else:
            self._plot_array(array, x=x, y=y, series=series_axes, ax=ax, legend=False, title=title, **kwargs)

    @deprecate_kwarg('stacked', 'stack')
    @_use_pandas_plot_docstring
    def line(self, x=None, y=None, **kwds):
        return self(kind='line', x=x, y=y, **kwds)

    @deprecate_kwarg('stacked', 'stack')
    @_use_pandas_plot_docstring
    def bar(self, x=None, y=None, **kwds):
        return self(kind='bar', x=x, y=y, **kwds)

    @deprecate_kwarg('stacked', 'stack')
    @_use_pandas_plot_docstring
    def barh(self, x=None, y=None, **kwds):
        return self(kind='barh', x=x, y=y, **kwds)

    @_use_pandas_plot_docstring
    def box(self, by=None, x=None, **kwds):
        if x is None:
            x = by if by is not None else ()
        ax = self(kind='box', x=x, _x_axes_last=True, **kwds)
        if 'ax' not in kwds and by is None:
            # avoid having a single None tick
            ax.get_xaxis().set_visible(False)
        return ax

    def heatmap(self, x=None, y=None, **kwds):
        """plot an ND array as a heatmap.

        By default, it uses the last array axis as the X axis and other array axes as Y axis (like the viewer table).
        Only the first axis in each "direction" will have its name and labels shown.

        Parameters
        ----------
        arr : Array
            data to display.
        y_axes : int, str, Axis, tuple or AxisCollection, optional
            axis or axes to use on the Y axis. Defaults to all array axes except the last `numhaxes` ones.
        x_axes : int, str, Axis, tuple or AxisCollection, optional
            axis or axes to use on the X axis. Defaults to all array axes except `y_axes`.
        numhaxes : int, optional
            if x_axes and y_axes are not specified, use the last numhaxes as X axes. Defaults to 1.
        axes_names : bool, optional
            whether to show axes names. Defaults to True
        ax : matplotlib axes object, optional
        **kwargs
            any extra keyword argument is passed to Matplotlib imshow.
            Likely of interest are cmap, vmin, vmax or norm.

        Returns
        -------
        matplotlib.AxesSubplot
        """
        return self(kind='heatmap', x=x, y=y, **kwds)

    @_use_pandas_plot_docstring
    def hist(self, by=None, bins=10, y=None, **kwds):
        if y is None:
            if by is None:
                y = self.array.axes
                if 'legend' not in kwds:
                    kwds['legend'] = False
            else:
                y = by
        return self(kind='hist', y=y, bins=bins, **kwds)

    @_use_pandas_plot_docstring
    def kde(self, by=None, bw_method=None, ind=None, y=None, **kwds):
        if y is None:
            if by is None:
                y = self.array.axes
                if 'legend' not in kwds:
                    kwds['legend'] = False
            else:
                y = by
        return self(kind='kde', bw_method=bw_method, ind=ind, y=y, **kwds)

    @deprecate_kwarg('stacked', 'stack')
    @_use_pandas_plot_docstring
    def area(self, x=None, y=None, **kwds):
        return self(kind='area', x=x, y=y, **kwds)

    @_use_pandas_plot_docstring
    def pie(self, y=None, legend=False, **kwds):
        if y is None:
            # add a dummy axis with blank name and a 'value' label and plot that label to avoid 'None' labels for
            # each subplot (when used) if we had used y = () instead
            self = self.array.expand(' =__dummy_value').plot
            y = '__dummy_value'
            if 'ylabel' not in kwds:
                # avoid showing '__dummy_value' as ylabel
                kwds['ylabel'] = ''

        # avoid a deprecation warning issued by matplotlib 3.3+ (and not fixed in Pandas as of Pandas 1.3.0)
        if 'normalize' not in kwds:
            kwds['normalize'] = True

        ax = self(kind='pie', y=y, legend=legend, **kwds)

        # if we created the Axes and have subplots, hide all x axes because as of now
        # (pandas 1.3.0 and matplotlib 3.3.4) there are some ugly and useless x axes
        # with a few ticks when we have subplots in a vertical layout
        if 'ax' not in kwds and isinstance(ax, np.ndarray):
            for axes in ax.flat:
                axes.get_xaxis().set_visible(False)
        return ax

    @_use_pandas_plot_docstring
    def scatter(self, x, y, s=None, c=None, **kwds):
        # TODO: add support for 'c' and 's' even when x and y are not specified
        return self(kind='scatter', x=x, y=y, c=c, s=s, **kwds)

    @_use_pandas_plot_docstring
    def hexbin(self, x, y, C=None, reduce_C_function=None, gridsize=None, **kwds):
        if reduce_C_function is not None:
            kwds['reduce_C_function'] = reduce_C_function
        if gridsize is not None:
            kwds['gridsize'] = gridsize
        return self(kind='hexbin', x=x, y=y, C=C, **kwds)
