Viewer
------

* implemented min/maxvalue arguments for edit()
* do not close the window when pressing Enter
* allow to start editing cells by pressing Enter
* fixed copy of changed cells (copy the changed value)
* fixed pasted values to not be accepted directly (they go to "changes" like for manual edits)
* fixed color updates on paste
* disabled experimental tooltips on headers
* better error message when entering invalid values

Core
----

* implemented indexing by position on several dimensions at once (like numpy)

    >>> # takes the first item in the first and third dimensions, leave the second dimension intact
    >>> arr.i[0, :,  0]
    <some result>
    >>> # sets all the cells corresponding to the first item in the first dimension and the second item in the fourth
    >>> # dimension
    >>> arr.i[0, :,  :, 1] = 42

* added optional 'readonly' argument to expand() to produce a readonly view (much faster since no copying is done)
