# spl
_spl_ (statistics python library) is a Python library for statistics, data analysis, plotting data developed by Giorgio Cividini.

## Notes from the developer
The current version of _spl_ does not have a complete _try-and-catch_ function because of the complexity of the algorithm.
In the module _generate_ there is a function called _tac_box_ which implements the algorithm for a function in a defined square.
The user is, therefore, asked to insert the limits of this "box" and an incorrect input can lead to malfunctions of the function.
The same problem recurs in the _hit-or-miss_ method for defined integrals.

The implementation of _iMinuit_ is shown with three examples: least squares, binned extended likelihood (with composite pdf), unbinned likelihood.
