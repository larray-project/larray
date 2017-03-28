Core
----

* implemented dropna argument for to_csv, to_frame and to_series to avoid writing lines with either 'all' or 'any'
  NA values.
* implemented read_sas. Needs pandas >= 0.18 (though it seems still buggy on some files).
* implemented experimental support for __getattr__ and __setattr__ on LArray. One can use arr.H instead of arr['M'].
  It only works for single string labels though (not for slices or list of labels nor integer labels). Not sure it is a
  good idea :).

* implemented Session +-`*`/
   Eg. sess1 - sess2 will compute the difference on each array present in either session. If an array is present in one
   session and not in the other, it is replaced by "NaN".
* added .nbytes property to LArray objects (to know how many bytes of memory the array uses)

* made sort_axis accept a tuple of axes
* raises an error on a.i[tuple_with_len_greater_than_array_ndim]
* slightly better support for axes with no name (no, still no complete support yet ;-))
* improved AxisCollection: implemented __delitem__(slice), __setitem__(list), __setitem__(slice)
* fixed exception on AxisCollection.index(invalid_index)
* better docstrings for a few functions
* misc code cleanups, refactoring & improved tests

Editor
------

* added .dirty property on ArrayEditorWidget
* fixed viewing arrays with "inf" (infinite)
* fixed a few edge cases for the ndigit detection code
* fixed colors in some cases in edit()
* made copy-paste of large regions faster in some cases
