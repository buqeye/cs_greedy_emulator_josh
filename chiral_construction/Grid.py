import numpy as np


class Grid:
    def __init__(self, start, end, numIntervals=1, numPointsPerInterval=10, type="Gauss", test=True):
        self.type = type
        self.numIntervals = numIntervals
        self.numPointsPerInterval = numPointsPerInterval
        self.start = start
        self.end = end
        self.points = None
        self.weights = None

        if (np.array([numIntervals, numPointsPerInterval]) <= 0).any():
            raise ValueError("Number of sampling must be > 0.")

        self.setup(test)

    @property
    def weightMat(self):
        return np.diag(self.weights)

    @property
    def getNumPointsTotal(self):
        return self.numIntervals*self.numPointsPerInterval

    def _rescaleGlPoints(self, start, end, weights, points):
        tmp = (end - start)/2.
        return tmp * weights, tmp * points + (end + start)/2.

    def setup(self, test):
        if (np.array([self.numPointsPerInterval, self.numIntervals]) <= 0).any():
            raise ValueError("Invalid input for Grid")

        if self.type == "Gauss":
            # Gauss-Legendre Quadrature
            # since the first point (origin) is need (boundary condition u(r=0)=0
            # we add the origin with zero weight, and apply the quadrature rule with
            # one point fewer to the last integration interval
            glPoints, glWeights = np.polynomial.legendre.leggauss(self.numPointsPerInterval)
            boundaries = np.linspace(self.start, self.end, self.numIntervals+1)

            gridWeights = np.array([])
            gridPoints = np.array([])
            for s, e in zip(boundaries[:-2], boundaries[1:-1]):
                #print(s, e)
                wts, pts = self._rescaleGlPoints(s, e, glWeights, glPoints)
                gridWeights = np.append(gridWeights, wts)
                gridPoints = np.append(gridPoints, pts)

            glPointsLast, glWeightsLast = np.polynomial.legendre.leggauss(self.numPointsPerInterval-1)
            wts, pts = self._rescaleGlPoints(boundaries[-2], boundaries[-1], glWeightsLast, glPointsLast)
            gridWeights = np.append(gridWeights, wts)
            gridPoints = np.append(gridPoints, pts)

            gridWeights = np.concatenate(([0], gridWeights))
            gridPoints = np.concatenate(([self.start], gridPoints))

            self.weights = gridWeights
            self.points = gridPoints
        elif self.type == "linear*":
            # improved linear grid
            # since the wave function vanishes at the origin exclude "start" from grid
            pts, step = np.linspace(self.end, self.start, self.getNumPointsTotal, endpoint=False, retstep=True)
            self.points = np.flip(pts)
            self.weights = np.append(np.diff(self.points), [np.abs(step)])
        else:
            # standard linear grid
            # ignoring the first (last) point results in upper (lower) Riemann sum
            # use the upper sum here because the wave function vanishes at the first sampling anyway
            self.points = np.linspace(self.start, self.end, self.getNumPointsTotal)
            self.weights = np.append([0.], np.diff(self.points))

        if test:
            self.testQuadrature()

    def testQuadrature(self, eps=1e-9):
        # Sin(), Cos() functions are in particular relevant here (free space solutions)
        testFuncs = [lambda x: x**0, lambda x: np.exp(-x/2.), lambda x: np.cos(2*x)]
        antiDervs = [lambda x: x, lambda x: -2.*np.exp(-x/2.), lambda x: np.sin(2*x)/2.]

        for ifunc, (f, F) in enumerate(zip(testFuncs, antiDervs)):
            est = self.weights @ f(self.points)
            exact = F(self.end) - F(self.start)
            print(est, exact)
            if np.abs(est - exact) > eps:
                raise ValueError(f"Quadrature test {ifunc} failed: {exact} (exact) vs {est} (est.)")

    def print(self):
        print(self)
        for i in range(self.getNumPointsTotal):
            print(f"{self.points[i]:.8f} | {self.weights[i]:.8f}")
        print(f"sum of weights: {np.sum(self.weights)}")

    def __str__(self):
        return f"Grid '{self.type}' in range ({self.start}..{self.end})"