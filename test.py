# Only using doctests
import scoring, analyse, exmin, gibbs
import doctest

doctest.testmod(scoring)
doctest.testmod(analyse)
doctest.testmod(exmin)
doctest.testmod(gibbs)