Viewer
------

* renamed "Ok" button in array/session viewer to "Close".
* added apply and discard buttons in session editor, which permanently apply or discard changes to the current array.

Core
----

* fixed array[sequence, scalar] = value
* fixed array.to_excel() which was broken in 0.16 (by the upgrade to xlwings 0.9+).
* improved a few tests
