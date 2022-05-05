from StateGenerator import StateGenerator
import auxiliary as aux
import Distributions as dist
import pathlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import auxiliary as aux
import numpy as np
import Distributions as dist
from Ring import Ring, _MRing
from Plotter import Plotter, AlignPlotter
from Propagator import Propagator
from Kicker import Kicker
from StateGenerator import StateGenerator
import ROOT

# Column 4 - radial displacement  (x)
# Column 5-  angle
# Column 6,7 are vertical displacement and angle
# column 9 - fractional momentum offset (with respect to magic momentum)
# column 10 - time
propagator = Propagator(
    # output = None,
    output = 'test2',
    plot = False,
    display_options_s = [], # ["r", "phi", "dr", "dphi", "p", "animation"]
    display_options_m = [], #  ["p", "f", "hist2d", "delta_p", "spatial"]
    # p_acceptance_dir = "two_kick",
    p_acceptance_dir = None,
    plot_option = "presentation",
    # plot_option = "maximize_graph",
    animate = False,
    store_root = True,
    suppress_print = False,
    # multiplex_plots = ['mt1']
    multiplex_plots = []
)

ring = Ring(
    # r_max = 7.157, # radius of outer ring edge (m)
    # r_min = 7.067, # radius of inner ring edge (m)
    r_max = aux.r_magic + 0.043,
    r_min = aux.r_magic - 0.043,
    b_nom = aux.B_nom, # Nominal b-field strength (T)
    b_k = Kicker("file", "traces-Ir", b_norm = 204, kick_max = 100, kicker_num = 3),
    # b_k = Kicker("uniform", 100),
    # b_k = Kicker("uniform", 0),
    # b_k = Kicker("one turn", 250),
    quad_model = "linear", # "linear" "full"
    # quad_num = 1, # 1 or 4
    quad_num = 4, # 1 or 4
    # n = 0,
    n = 0.108,
    collimators = "continuous",
    # collimators = "discrete",
    # collimators = "none",
    fringe = False # turn fringe effect on or off
)

generator = StateGenerator(
    mode = "mc",
    seed = None,
    muons = 1E6,
    initial_offset = dist.Gaussian(mean = aux.r_inj_offset - 2.6, std = 4.2),
    # initial_offset = aux.r_inj_offset,
    # momentum = dist.Single([aux.p_magic, aux.p_magic * 0.0001]),
    # momentum = dist.Single([aux.p_magic * 1.1, aux.p_magic]),
    momentum = dist.Gaussian(mean = aux.p_magic, std = 0.008 * aux.p_magic),
    # momentum = dist.Gaussian(mean = [aux.p_magic, aux.p_magic * 1.001, aux.p_magic * 1.002, aux.p_magic * 1.003, aux.p_magic * 1.004], std = 0.008 * aux.p_magic),
    # offset = 34,
    # alpha = dist.Single(value = [0, 2.5]),
    # alpha = 2.5,
    # alpha = 0,
    alpha = dist.Gaussian(mean = -0.9, std = 3.1),
    # alpha = dist.Gaussian(mean = -0.9, std = [2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4]),
    # alpha = d1st.Gaussian(mean = [-1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6], std = 3.1),
    phi_0 = 0,
    # t = 100
    # t = dist.Gaussian(mean = 0, std = 25)
    t = dist.Custom(dir = "pulseshape_1_12543_mod", zero = 24827.5)
)

propagator.align(state_generator = generator, ring = ring, integration_method = "rk4",
    dt = 0.1, # the time-step in nanoseconds
    t_f = 1000, # ending time in ns
    t_0 = 24827.5,
    coarseRange = 200,
    coarseStep = 10,
    fineRange = 10,
    fineStep = 1
)

# plotter = AlignPlotter(dir = "a-test")
# plotter.plot()