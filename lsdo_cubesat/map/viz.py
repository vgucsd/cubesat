from lsdo_viz.api import BaseViz, Frame

import seaborn as sns

sns.set()


class Viz(BaseViz):
    def setup(self):

        self.frame_name_format = 'output_{}'

        self.add_frame(
            Frame(
                height_in=8.,
                width_in=12.,
                nrows=2,
                ncols=3,
                wspace=0.4,
                hspace=0.4,
            ), 1)
