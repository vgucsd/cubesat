import numpy as np

from lsdo_utils.api import OptionsDictionary


class Ground_station(OptionsDictionary):
    def initialize(self):
        self.declare('name', types=str)
        self.declare('lon', types=float)
        self.declare('lat', types=float)
        self.declare('alt', types=float)
        self.declare('antAngle', default=2., types=float)
