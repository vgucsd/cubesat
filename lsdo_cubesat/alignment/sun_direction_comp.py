import numpy as np
import scipy.sparse

from openmdao.api import ExplicitComponent


class SunDirectionComp(ExplicitComponent):

    # constants
    d2r = np.pi / 180.

    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('launch_date', types=float)

    def setup(self):
        num_times = self.options['num_times']
        launch_date = self.options['launch_date']

        self.add_input('times', shape=num_times)

        self.add_output('sun_unit_vec', shape=(3, num_times))

        self.Ja = np.zeros(3 * num_times)
        self.Ji = np.zeros(3 * num_times)
        self.Jj = np.zeros(3 * num_times)

    def compute(self, inputs, outputs):
        num_times = self.options['num_times']
        launch_date = self.options['launch_date']

        sun_unit_vec = outputs['sun_unit_vec']

        T = launch_date + inputs['times'][:] / 3600. / 24.
        for i in range(0, num_times):
            L = self.d2r * 280.460 + self.d2r * 0.9856474 * T[i]
            g = self.d2r * 357.528 + self.d2r * 0.9856003 * T[i]
            Lambda = L + self.d2r * 1.914666 * np.sin(
                g) + self.d2r * 0.01999464 * np.sin(2 * g)
            eps = self.d2r * 23.439 - self.d2r * 3.56e-7 * T[i]
            sun_unit_vec[0, i] = np.cos(Lambda)
            sun_unit_vec[1, i] = np.sin(Lambda) * np.cos(eps)
            sun_unit_vec[2, i] = np.sin(Lambda) * np.sin(eps)

    def compute_partials(self, inputs, partials):
        num_times = self.options['num_times']
        launch_date = self.options['launch_date']

        T = launch_date + inputs['times'][:] / 3600. / 24.
        dr_dt = np.empty(3)
        for i in range(0, num_times):
            L = self.d2r * 280.460 + self.d2r * 0.9856474 * T[i]
            g = self.d2r * 357.528 + self.d2r * 0.9856003 * T[i]
            Lambda = L + self.d2r * 1.914666 * np.sin(
                g) + self.d2r * 0.01999464 * np.sin(2 * g)
            eps = self.d2r * 23.439 - self.d2r * 3.56e-7 * T[i]

            dL_dt = self.d2r * 0.9856474
            dg_dt = self.d2r * 0.9856003
            dlambda_dt = (dL_dt + self.d2r * 1.914666 * np.cos(g) * dg_dt +
                          self.d2r * 0.01999464 * np.cos(2 * g) * 2 * dg_dt)
            deps_dt = -self.d2r * 3.56e-7

            dr_dt[0] = -np.sin(Lambda) * dlambda_dt
            dr_dt[1] = np.cos(Lambda) * np.cos(eps) * dlambda_dt - np.sin(
                Lambda) * np.sin(eps) * deps_dt
            dr_dt[2] = np.cos(Lambda) * np.sin(eps) * dlambda_dt + np.sin(
                Lambda) * np.cos(eps) * deps_dt

            for k in range(0, 3):
                iJ = i * 3 + k
                self.Ja[iJ] = dr_dt[k]
                self.Ji[iJ] = iJ
                self.Jj[iJ] = i

        self.J = scipy.sparse.csc_matrix((self.Ja, (self.Ji, self.Jj)),
                                         shape=(3 * num_times, num_times))
        self.JT = self.J.transpose()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        num_times = self.options['num_times']
        launch_date = self.options['launch_date']

        dsun_unit_vec = d_outputs['sun_unit_vec']

        if mode == 'fwd':
            if 'times' in d_inputs:
                # TODO - Should split this up so we can hook one up but not the other.
                dsun_unit_vec[:] += (self.J.dot(d_inputs['times'] / 3600. /
                                                24.).reshape((3, num_times),
                                                             order='F'))
        else:
            sun_unit_vec = dsun_unit_vec[:].reshape((3 * num_times), order='F')
            if 'times' in d_inputs:
                d_inputs['times'] += self.JT.dot(sun_unit_vec) / 3600.0 / 24.0


if __name__ == '__main__':

    from openmdao.api import Problem, Group
    from openmdao.api import IndepVarComp

    np.random.seed(0)

    num_times = 30

    group = Group()

    comp = IndepVarComp()
    comp.add_output('times', val=np.random.random(num_times))

    group.add_subsystem('Inputcomp', comp, promotes=['*'])

    group.add_subsystem('Statecomp_Implicit',
                        SunDirectionComp(num_times=num_times, launch_date=0.),
                        promotes=['*'])

    prob = Problem()
    prob.model = group
    prob.setup(check=True)
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials()