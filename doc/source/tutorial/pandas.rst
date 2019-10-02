Compatibility with pandas
=========================

To convert an Array object into a pandas DataFrame, the method :py:meth:`~Array.to_frame` can be used:

.. ipython:: python

    df = pop.to_frame()
    df

Inversely, to convert a DataFrame into an Array object, use the function :py:func:`aslarray`:

.. ipython:: python

    pop = aslarray(df)
    pop
