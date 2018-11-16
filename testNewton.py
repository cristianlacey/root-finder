#!/usr/bin/env python3

import unittest
import numpy as np

import newton

class TestNewton(unittest.TestCase):
    def testLinear(self):
        # Just so you see it at least once, this is the lambda keyword
        # in Python, which allows you to create anonymous functions
        # "on the fly". As I commented in testFunctions.py, you can
        # define regular functions inside other
        # functions/methods. lambda expressions are just syntactic
        # sugar for that.  In other words, the line below is
        # *completely equivalent* under the hood to:
        #
        # def f(x):
        #     return 3.0*x + 6.0
        #
        # No difference.
        f = lambda x : 3.0*x + 6.0

        # Setting maxiter to 2 b/c we're guessing the actual root
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(-2.0)
        # Equality should be exact if we supply *the* root, ergo
        # assertEqual rather than assertAlmostEqual
        self.assertEqual(x, -2.0)

    def testQuadratic(self):
        # Testing parabola of the form y = x^2 -1, where we'd
        # expect roots at x = +-1. Tests starting points
        # on either side of each of the two roots to ensure
        # the closest root is found.

        f = lambda x : x**2 - 1
        solver = newton.Newton(f, tol=1.e-8, maxiter=100)

        # Taking root1 to be x = -1, iterate over several
        # initial conditions to the left of zero, where
        # the closest root is x = -1.
        root1_init_cond = np.linspace(-1000.0,-0.1,100)
        for init in root1_init_cond:
            x1 = solver.solve(init)
            self.assertAlmostEqual(x1, -1.0)

        # Taking root2 to be x = 1, iterate over several
        # initial conditions to the right of zero, where
        # the closest root is x = 1.
        root2_init_cond = np.linspace(0.1,1000.0,100)
        for init in root2_init_cond:
            x2 = solver.solve(init)
            self.assertAlmostEqual(x2, 1.0)
        
if __name__ == "__main__":
    unittest.main()

    
