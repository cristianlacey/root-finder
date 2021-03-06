# N-Dimensional Root-Finder
## APC 524: HW3

## Description

The provided code initially included newton.py, which defined the
class Newton(), and functions.py, which defined the
approximateJacobian function and polynomial class. Tests were
added to testNewton.py and testFunctions.py and the code was updated
so that the new tests passed.

## Features added:
1. Bound the root
   User can specify a bound for the Newton.solve() method. If at any
   point during the calculation the root leaves the bounded region,
   raises ValueError.
2. Support analytical Jacobians
   Allows user to input a function for the Jacobian. If no function
   is input, approximateJacobian() defaults to numerical
   approximation.
3. Symmetric difference quotient
   Changed method of finding numerical approximation of Jacobian to
   symmetric difference quotient.

## Tests added to testNewton.py:
1. testQuadratic()
   Tests that Newton() can find the roots of a parabola for different
   initial conditions.
2. testPolynomial()
   Tests that Newton() can find the roots of a thrid order polynomial
   for different intial conditions.
3. testLocalExtrema()
   If a guess for the root occurs on a local extrema, then the slope
   is locally zero, resulting in a singular Jacobian matrix. Code was
   modified to try a new guess (previous guess plus some epsilon) to
   move away from local extrema and continue. Works for 1D function
   so long as the local extrema isn't a root.
4. testRootlessFunction()
   If given a function without any real roots, Newton.solve() should
   raise the exception RuntimeError when the maximum number of
   iterations is reached.
5. test2DFunction()
   Tests that Newton.solve() works with 2D functions.
6. testRootBound()
   Tests functionality of root bounding feature.
7. testAnalyticalJacobian1D()
   Tests functionality of analytical Jacobian with 1D function.
8. testAnalyticalJacobian2D()
   Tests functionality of analytical Jacobian with 2D function.

## Tests added to testFunctions.py:
1. test_ApproxJacobian2DHigherOrder()
   Tests that numerical approximation of Jacobian is accurate for
   higher order 2D functions.
