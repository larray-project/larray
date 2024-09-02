import numpy as np
from matplotlib.ticker import MaxNLocator


class MaxNMultipleWithOffsetLocator(MaxNLocator):
    def __init__(self, nbins=None, offset=0.5, **kwargs):
        super().__init__(nbins, **kwargs)
        self.offset = offset

    def tick_values(self, vmin, vmax):
        # matplotlib calls them vmin and vmax but they are actually the limits and vmin can be > vmax
        invert = vmin > vmax
        if invert:
            vmin, vmax = vmax, vmin

        max_desired_ticks = self._nbins
        # not + 1 because we place ticks in the middle
        num_ticks = vmax - vmin
        desired_numticks = min(num_ticks, max_desired_ticks)
        if desired_numticks < num_ticks:
            step = np.ceil(num_ticks / desired_numticks)
        else:
            step = 1
        vmin = int(vmin)
        vmax = int(vmax)
        # when we have an offset, we do not add 1 to vmax because we place ticks in the middle
        # (by adding the offset), and would result in the last "tick" being outside the limits
        stop = vmax + 1 if self.offset == 0 else vmax
        new_ticks = np.arange(vmin, stop, step)
        if invert:
            new_ticks = new_ticks[::-1]
        return new_ticks + self.offset

    def __call__(self):
        """Return the locations of the ticks."""
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)
