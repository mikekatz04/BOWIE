#########################################################
BOWIE: Binary Observability With Illustrative Exploration
#########################################################

.. image:: logo/Bowie_logo.png
   :width: 100px
   :height: 100px
   :scale: 50 %
   :alt: alternate text
   :align: center

BOWIE is a tool designed for graphical analysis of binary signals from gravitational waves. It takes gridded data sets and produces different types of plots in customized arrangements for detailed analysis of gravitational wave sensitivity curves and/or binary signals. The paper detailing this tool and examples of its usage can be found at `arXiv:1807.02511`_ (Evaluating Black Hole Detectability with LISA). There are three main portions of the code: a gridded data generator (``bowie_gencondata.generate_contour_data.py``), a plotting tool (``bowie_makeplot.make_plot.py``), and waveform generator for general use (``pyphenomd.pyphenomd.py``). The waveform generator creates PhenomD waveforms for binary black hole inspiral, merger, and ringdown. PhenomD is from Husa et al 2016 (`arXiv:1508.07250`_) and Khan et al 2016 (`arXiv:1508.07253`_). Gridded data sets are created using the PhenomD generator for signal-to-noise (SNR) analysis. Using the gridded data sets, customized configurations of plots are created with the plotting package.

.. _arXiv:1807.02511: https://arxiv.org/abs/1807.02511
.. _arXiv:1508.07250: https://arxiv.org/abs/1508.07250
.. _arXiv:1508.07253: https://arxiv.org/abs/1508.07253

The three plots to choose from are Waterfall, Ratio, and Horizon. A Waterfall plot is a filled contour plot similar to figure 3 in the LISA Mission Proposal (arxiv:1702.00786). Ratio plots show the ratio of SNRs between two different binary and sensitivity configurations. Horizon plots show line contours of multiple configurations for a given SNR value. See BOWIE documentation, paper, and examples for more information.

Available via pip and on github: https://github.com/mikekatz04/BOWIE


