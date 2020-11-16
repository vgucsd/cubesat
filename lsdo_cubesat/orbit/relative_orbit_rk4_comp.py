"""
RK4 component for orbit compute
"""
import os
from six.moves import range

import numpy as np
import scipy.sparse

from openmdao.api import ExplicitComponent
from lsdo_cubesat.utils.rk4_comp import RK4Comp

# Constants
mu = 398600.44
Re = 6378.137
J2 = 1.08264e-3
J3 = -2.51e-6
J4 = -1.60e-6

C1 = -mu
C2 = -1.5 * mu * J2 * Re**2
C3 = -2.5 * mu * J3 * Re**3
C4 = 1.875 * mu * J4 * Re**4

# rho = 3.89e-12 # kg/m**3 atmoshperic density at altitude = 400 km with mean solar activity
# C_D = 2.2 # Drag coefficient for cube
# area = 0.1 * 0.1 # m**2 cross sectional area
drag = 1.e-6


class RelativeOrbitRK4Comp(RK4Comp):
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('step_size', types=float)

        self.options['state_var'] = 'relative_orbit_state'
        self.options['init_state_var'] = 'initial_orbit_state'
        self.options['external_vars'] = ['force_3xn', 'mass', 'radius']

    def setup(self):
        n = self.options['num_times']
        h = self.options['step_size']

        self.add_input('force_3xn', shape=(3, n), desc='Thrust on the cubesat')

        self.add_input('mass', shape=(1, n), desc='mass of Cubesat')

        self.add_input('radius', shape=(1, n))

        self.add_input(
            'initial_orbit_state',
            shape=6,  # fd_step=1e-2,
            desc='Initial position and velocity vectors from earth to '
            'satellite in Earth-centered inertial frame')

        self.add_output(
            'relative_orbit_state',
            shape=(6, n),
            desc='Position and velocity vectors from earth to satellite '
            'in Earth-centered inertial frame over time')

        # self.dfdx = np.zeros((6, 5))
        # self.dfdy = np.zeros((6, 6))

    def f_dot(self, external, state):
        n = self.options['num_times']
        h = self.options['step_size']
        # Px = external[0]
        # Py = external[1]
        # Pz = external[2]
        #
        # a_Tx = Tx/m
        # a_Ty = Ty/m
        # a_Tz = Tz/m

        x = state[0]
        y = state[1]
        z = state[2] if abs(state[2]) > 1e-15 else 1e-5

        z2 = z * z
        z3 = z2 * z
        z4 = z3 * z

        r = external[4]

        r2 = r * r
        r3 = r2 * r
        r4 = r3 * r
        r5 = r4 * r
        r7 = r5 * r * r

        T2 = 1 - 5 * z2 / r2
        T3 = 3 * z - 7 * z3 / r2
        T4 = 1 - 14 * z2 / r2 + 21 * z4 / r4
        T3z = 3 * z - 0.6 * r2 / z
        T4z = 4 - 28.0 / 3.0 * z2 / r2

        f_dot = np.zeros((6, ))
        f_dot[0:3] = state[3:]
        f_dot[3:] = state[0:3] * (
            C1 / r3 + C2 / r5 * T2 + C3 / r7 * T3 + C4 / r7 *
            T4) + external[0:3] / external[3] - drag * state[3:] / external[3]
        f_dot[5] += z * (2.0 * C2 / r5 + C3 / r7 * T3z + C4 / r7 * T4z)

        return f_dot

    def df_dy(self, external, state):

        x = state[0]
        y = state[1]
        z = state[2] if abs(state[2]) > 1e-15 else 1e-5

        z2 = z * z
        z3 = z2 * z
        z4 = z3 * z

        r = external[4]

        r2 = r * r
        r3 = r2 * r
        r4 = r3 * r
        r5 = r4 * r
        r6 = r5 * r
        r7 = r6 * r
        r8 = r7 * r

        dr = np.zeros(3)

        T2 = 1 - 5 * z2 / r2
        T3 = 3 * z - 7 * z3 / r2
        T4 = 1 - 14 * z2 / r2 + 21 * z4 / r4
        T3z = 3 * z - 0.6 * r2 / z
        T4z = 4 - 28.0 / 3.0 * z2 / r2

        dT2 = (10 * z2) / (r3) * dr
        dT2[2] -= 10. * z / r2

        dT3 = 14 * z3 / r3 * dr
        dT3[2] -= 21. * z2 / r2 - 3

        dT4 = (28 * z2 / r3 - 84. * z4 / r5) * dr
        dT4[2] -= 28 * z / r2 - 84 * z3 / r4

        dT3z = -1.2 * r / z * dr
        dT3z[2] += 0.6 * r2 / z2 + 3

        dT4z = 56.0 / 3.0 * z2 / r3 * dr
        dT4z[2] -= 56.0 / 3.0 * z / r2

        # f_dot = np.zeros((6, ))
        # f_dot[0:3] = state[3:]
        # f_dot[3:] = state[0:3] * (
        #     C1 / r3 + C2 / r5 * T2 + C3 / r7 * T3 +
        #     C4 / r7 * T4) + external[0:3] / external[3] - drag * state[3:] / external[3]
        # f_dot[5] += z * (2.0 * C2 / r5 + C3 / r7 * T3z + C4 / r7 * T4z)

        eye = np.identity(3)

        # dfdy = self.dfdy
        # dfdy[:, :] = 0.
        dfdy = np.zeros((6, 6))

        dfdy[0:3, 3:] += eye

        dfdy[3:, :3] += eye * (C1 / r3 + C2 / r5 * T2 + C3 / r7 * T3 +
                               C4 / r7 * T4)
        # fact = (-3 * C1 / r4 - 5 * C2 / r6 * T2 - 7 * C3 / r8 * T3 -
        #         7 * C4 / r8 * T4)
        # dfdy[3:, 0] += dr[0] * state[:3] * fact
        # dfdy[3:, 1] += dr[1] * state[:3] * fact
        # dfdy[3:, 2] += dr[2] * state[:3] * fact
        dfdy[3:, 0] += state[:3] * (C2 / r5 * dT2[0] + C3 / r7 * dT3[0] +
                                    C4 / r7 * dT4[0])
        dfdy[3:, 1] += state[:3] * (C2 / r5 * dT2[1] + C3 / r7 * dT3[1] +
                                    C4 / r7 * dT4[1])
        dfdy[3:, 2] += state[:3] * (C2 / r5 * dT2[2] + C3 / r7 * dT3[2] +
                                    C4 / r7 * dT4[2])
        dfdy[3:, 3:] += np.eye(3) * -drag / external[3]
        # dfdy[5, :3] += dr * z * (-5 * C2 / r6 * 2 - 7 * C3 / r8 * T3z -
        #                          7 * C4 / r8 * T4z)
        dfdy[5, :3] += z * (C3 / r7 * dT3z + C4 / r7 * dT4z)
        dfdy[5, 2] += (C2 / r5 * 2 + C3 / r7 * T3z + C4 / r7 * T4z)

        return dfdy

    def df_dx(self, external, state):

        x = state[0]
        y = state[1]
        z = state[2] if abs(state[2]) > 1e-15 else 1e-5

        z2 = z * z
        z3 = z2 * z
        z4 = z3 * z

        r = external[4]

        r2 = r * r
        r3 = r2 * r
        r4 = r3 * r
        r5 = r4 * r
        r6 = r5 * r
        r7 = r6 * r
        r8 = r7 * r

        dr = 1.

        T2 = 1 - 5 * z2 / r2
        T3 = 3 * z - 7 * z3 / r2
        T4 = 1 - 14 * z2 / r2 + 21 * z4 / r4
        T3z = 3 * z - 0.6 * r2 / z
        T4z = 4 - 28.0 / 3.0 * z2 / r2

        dT2 = (10 * z2) / (r3) * dr

        dT3 = 14 * z3 / r3 * dr

        dT4 = (28 * z2 / r3 - 84. * z4 / r5) * dr

        dT3z = -1.2 * r / z * dr

        dT4z = 56.0 / 3.0 * z2 / r3 * dr

        # f_dot = np.zeros((6, ))
        # f_dot[0:3] = state[3:]
        # f_dot[3:] = state[0:3] * (
        #     C1 / r3 + C2 / r5 * T2 + C3 / r7 * T3 +
        #     C4 / r7 * T4) + external[0:3] / external[3] - drag * state[3:] / external[3]
        # f_dot[5] += z * (2.0 * C2 / r5 + C3 / r7 * T3z + C4 / r7 * T4z)

        eye = np.identity(3)

        # dfdx = self.dfdx
        # dfdx[:, :] = 0.
        dfdx = np.zeros((6, 5))

        fact = (-3 * C1 / r4 - 5 * C2 / r6 * T2 - 7 * C3 / r8 * T3 -
                7 * C4 / r8 * T4)
        dfdx[3:, 4] += dr * state[:3] * fact
        dfdx[5, 4] += z * (-5 * C2 / r6 * 2 - 7 * C3 / r8 * T3z -
                           7 * C4 / r8 * T4z)

        # x = external[0]
        # y = external[1]
        # z = external[2]

        # eye = np.identity(3)

        dfdx[3, 0] = 1 / external[3]
        dfdx[4, 1] = 1 / external[3]
        dfdx[5, 2] = 1 / external[3]

        dfdx[3, 3] = -external[0] / external[3]**2 + drag * state[
            3] / external[3]**2
        dfdx[4, 3] = -external[1] / external[3]**2 + drag * state[
            4] / external[3]**2
        dfdx[5, 3] = -external[2] / external[3]**2 + drag * state[
            5] / external[3]**2

        # dfdx[3, :4] = [
        #     1 / external[3] / 1.e3, 0, 0,
        #     -external[0] / external[3]**2 / 1.e3 +
        #     drag * state[3] / external[3]**2 / 1.e3
        # ]
        # dfdx[4, :4] = [
        #     0, 1 / external[3] / 1.e3, 0,
        #     -external[1] / external[3]**2 / 1.e3 +
        #     drag * state[4] / external[3]**2 / 1.e3
        # ]
        # dfdx[5, :4] = [
        #     0, 0, 1 / external[3] / 1.e3,
        #     -external[2] / external[3]**2 / 1.e3 +
        #     drag * state[5] / external[3]**2 / 1.e3
        # ]

        # dfdx[3:,:3] += eye/mass

        # dfdx[0, :] = [0., -external[2], external[1]]
        # dfdx[1, :] = [external[2], 0., -external[0]]
        # dfdx[2, :] = [-external[1], external[0], 0.]
        return dfdx


if __name__ == '__main__':

    from openmdao.api import Problem, Group
    from openmdao.api import IndepVarComp
    import matplotlib.pyplot as plt

    np.random.seed(0)

    group = Group()

    n = 30
    m = 1
    npts = 1
    h = 1.5e-4

    r_e2b_I0 = np.empty(6)
    r_e2b_I0[:3] = 1. * np.random.rand(3)
    r_e2b_I0[3:] = 1. * np.random.rand(3)

    comp = IndepVarComp()
    comp.add_output('force_3xn', val=np.random.rand(3, n))
    comp.add_output('initial_orbit_state', val=r_e2b_I0)
    comp.add_output('radius', val=6400e3 + np.random.rand(1, n))
    comp.add_output('mass', val=1.e-2, shape=(1, n))
    group.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = RelativeOrbitRK4Comp(num_times=n, step_size=h)
    group.add_subsystem('comp', comp, promotes=['*'])

    prob = Problem()
    prob.model = group
    prob.setup(check=True)
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)

    orbit_X = prob['relative_orbit_state_km'][0, :]
    orbit_Y = prob['relative_orbit_state_km'][1, :]
    plt.plot(orbit_X, orbit_Y)
    prob.check_partials()
