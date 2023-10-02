import numpy as np
import pandas as pd

from larray import IGroup, Axis, AxisCollection, Group


def _use_pandas_plot_docstring(f):
    f.__doc__ = getattr(pd.DataFrame.plot, f.__name__).__doc__
    return f


class PlotObject:
    __slots__ = ('array',)

    def __init__(self, array):
        self.array = array

    @staticmethod
    def _handle_x_y_axes(axes, x, y, subplots):
        label_axis = None

        if np.isscalar(x) and x not in axes:
            label_axis, x_indices = axes._translate_axis_key(x)
            x = IGroup(x_indices, axis=label_axis)

        if np.isscalar(y) and y not in axes:
            y_label_axis, y_indices = axes._translate_axis_key(y)
            y = IGroup(y_indices, axis=y_label_axis)
            if label_axis is not None and y_label_axis is not label_axis:
                raise ValueError(f'{x} and {y} are labels from different axes')
            label_axis = y_label_axis

        def handle_axes_arg(avail_axes, arg):
            if arg is not None:
                arg = avail_axes[arg]
                if isinstance(arg, Axis):
                    arg = AxisCollection([arg])
                avail_axes = avail_axes - arg
            return avail_axes, arg

        if label_axis is not None:
            available_axes = axes - label_axis
        else:
            available_axes, x = handle_axes_arg(axes, x)
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

        return subplot_axes, series_axes, x, y

    @staticmethod
    def _to_pd_obj(array):
        if array.ndim == 1:
            return array.to_series()
        else:
            return array.to_frame()

    @staticmethod
    def _plot_array(array, *args, x=None, y=None, series=None, _x_axes_last=False, **kwargs):
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

        lineplot = 'kind' not in kwargs or kwargs['kind'] == 'line'
        if lineplot and label_axis is not None and series is not None and len(series) > 0:
            # the problem with this approach (n calls to pandas.plot) is that the color
            # cycling and "stacked" bar/area of pandas break for all kinds of plots except "line"
            # when we have more than one dimension involved
            for series_key, series_data in array.items(series):
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

            return PlotObject._to_pd_obj(array).plot(*args, x=x, y=y, **kwargs)

    def __call__(self, x=None, y=None, ax=None, subplots=False, layout=None, figsize=None,
                 sharex=None, sharey=False, tight_layout=None, constrained_layout=None, title=None, legend=None,
                 **kwargs):
        from matplotlib import pyplot as plt

        array = self.array
        legend_kwargs = legend if isinstance(legend, dict) else {}

        subplot_axes, series_axes, x, y = PlotObject._handle_x_y_axes(array.axes, x, y, subplots)

        if constrained_layout is None:
            constrained_layout = True

        if subplots:
            if ax is not None:
                raise ValueError("ax cannot be used in combination with subplots argument")
            fig = plt.figure(figsize=figsize, tight_layout=tight_layout, constrained_layout=constrained_layout)

            num_subplots = subplot_axes.size
            if layout is None:
                subplots_shape = subplot_axes.shape
                if len(subplots_shape) > 2:
                    # default to last axis horizontal, other axes combined vertically
                    layout = np.prod(subplots_shape[:-1]), subplots_shape[-1]
                else:
                    layout = subplot_axes.shape

            if sharex is None:
                sharex = True
            ax = fig.subplots(*layout, sharex=sharex, sharey=sharey)
            # it is easier to always work with a flat array
            flat_ax = ax.flat
            # remove blank plot(s) at the end, if any
            if len(flat_ax) > num_subplots:
                for plot_ax in flat_ax[num_subplots:]:
                    plot_ax.remove()
                # this not strictly necessary but is cleaner in case we reuse flax_ax
                flat_ax = flat_ax[:num_subplots]
            if title is not None:
                fig.suptitle(title)
            for i, (ndkey, subarr) in enumerate(array.items(subplot_axes)):
                title = ' '.join(str(ak) for ak in ndkey)
                self._plot_array(subarr, x=x, y=y, series=series_axes, ax=flat_ax[i], legend=False, title=title,
                                 **kwargs)
        else:
            if ax is None:
                fig = plt.figure(figsize=figsize, tight_layout=tight_layout, constrained_layout=constrained_layout)
                ax = fig.subplots(1, 1)
            self._plot_array(array, x=x, y=y, series=series_axes, ax=ax, legend=False, title=title, **kwargs)

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
        return ax

    @_use_pandas_plot_docstring
    def line(self, x=None, y=None, **kwds):
        return self(kind='line', x=x, y=y, **kwds)

    @_use_pandas_plot_docstring
    def bar(self, x=None, y=None, **kwds):
        return self(kind='bar', x=x, y=y, **kwds)

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

        # if we created the Axes and we have subplots, hide all x axis because as of now
        # (pandas 1.3.0 and matplotlib 3.3.4) there are some ugly and useless x axes
        # with a few ticks when have subplots in a vertical layout
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
