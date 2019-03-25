Compatibility with pandas
=========================

To convert a LArray object into a pandas DataFrame, the method :py:meth:`~LArray.to_frame` can be used:

.. ipython:: python

    df = pop.to_frame()
    df

Inversely, to convert a DataFrame into a LArray object, use the function :py:func:`aslarray`:

.. ipython:: python

    pop = aslarray(df)
    pop
