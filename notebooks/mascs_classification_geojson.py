# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Title 
#
# **Automated surface mapping via unsupervised learning and classification of Mercury Visible--Near-Infrared reflectance spectra**
#

# %% [markdown]
# ## Abstract
#
# In this work we apply unsupervised learning techniques for  dimensionality reduction and clustering to remote sensing  hyperspectral Visible-Near Infrared (VNIR) reflectance spectra  datasets of the planet Mercury obtained by the MErcury Surface, Space  ENvironment, GEochemistry, and Ranging (MESSENGER) mission.
# This  approach produces cluster maps, which group different regions of the  surface based on the properties of their spectra as inferred during  the learning process.
# While results depend on the choice of model  parameters and available data, comparison to expert-generated geologic  maps shows that some clusters correspond to expert-mapped classes such  as smooth plains on Mercury.
# These automatically generated maps can  serve as a starting point or comparison for traditional methods of  creating geologic maps based on spectral patterns.
# The code and data  used in this work is available as python jupyter notebook on the  github public repository  [MESSENGER-Mercury-Surface-Cassification-Unsupervised_DLR](https://github.
# com/epn-ml/MESSENGER-Mercury-Surface-Cassification-Unsupervised_DLR)[^1] funded by the European Union's Horizon 2020 grant No 871149.
#
# Authors:
# - Mario D'Amore$^1$
# - Sebastiano Padovan$^{1,2,3}$
#
# Affiliations : 
#
# -  $^1$German Aerospace Center (DLR), Rutherfordstraße 2, 12489 Berlin,Germany
# -  $^2$EUMETSAT, Eumetsat Allee 1, 64295 Darmstadt, Germany
# -  $^3$WGS, Berliner Allee 47, 64295 Darmstadt, Germany
#
#

# %% [markdown]
# ## [Introduction](#sec:4b.intro)
#
# The sheer amount of data returned by scientific missions aimed at
# exploring the solar system and observing exoplanets in recent decades
# overwhelms classical methods to explore and discover important
# scientific aspects of the target body. As an example, the Mercury data
# return for Mariner 10 was less than 100 MB, while MESSENGER delivered
# about 23 TB. Future missions are expected to exceed this limit. In
# addition, there is a trend of increasing complexity in the data itself,
# e.g., going from the of Mariner-10 to the hyperspectral datasets
# expected from BepiColombo. This situation clearly indicates that some
# form of automated analysis would be beneficial, provided it is able to
# save time without a loss of the information content of the data.
#
# Keeping the focus on hyperspectral remote sensing data, the typical
# approach for analysing this kind of data is to model the observed
# radiation with a forward radiative model[like Hapke, as in
# @Hamilton2005] or attempt to reproduce the observed radiation by setting
# up relevant samples in a laboratory setting using chemical and/or
# geomorphological context information.[e.g., @Helbert2013] Complex
# forward models that are able to take into account the relevant physics
# are typically computationally intensive and difficult to use to
# investigate the very large parameter space covered by hyperspectral
# data. This consideration is even more relevant for laboratory
# investigations : physical simulation needs the target to be physically
# fabricated, hence more and and more parameters means more experiments
# and more time. Models need computational power to be calculated in a
# reasonable amount of time, but could be distributed on several machines
# to overcome this limitation. This workaround is not effective for
# laboratory experiment, because most only few places meets of the
# environment needed for space sample simulation, like high-vacuum,
# -temperature, -radiation and so on.
#
# Without a way to efficiently and rapidly explore large amounts of
# complex data, it is likely that valuable information will be missed in
# large hyperspectral data sets.
#
# Geological maps are the gold standard for remote planetary surface
# studies, but producing them is an extremely time-consuming task. This
# process can suffer from user bias and typically only uses a few data
# points (e.g., 3-channel images) to describe different units. For
# example,[@Denevi2009] mapped the distribution and extent of major
# terrain types of Mercury using MESSENGER Mercury Dual Imaging System
# (MDIS) camera observations of Mercury. While the camera has 11 spectral
# bands, the maps typically used for the terrain differentiation are RGB,
# where 3 representative spectral bands are mapped onto the three image
# color channels.
#
# Geomorphological maps take in account additional features like surface
# roughness and crater density as a proxy for the age, where the
# correlation between age and crater density are derived from
# models.[e.g., @blandCraterCounting2003; @kerrWhoCanRead2006] Automated
# techniques are becoming more common in planetary science applications,
# as this books testifies, and the aim of this chapter is to illustrate
# how to apply unsupervised learning techniques to remote sensing data.
# This approach requires minimal user interaction and yields
# scientifically interesting products like classification maps that can be
# directly compared with geomorphological maps and models. We present an
# analysis of spectral reflectance data of Mercury's surface collected by
# the Mercury Atmospheric and Surface Composition Spectrometer (MASCS)
# instrument during orbital observations of the NASA MESSENGER mission
# between 2011 and 2015.[@McClintock2007] MASCS is a three sensor point
# spectrometer with a spectral coverage from 200 nm to 1450 nm. After a
# brief overview of the instrument and its significance for the
# investigation of Mercury (section
# [2](#sec:4b.mercury_mascs){reference-type="ref"
# reference="sec:4b.mercury_mascs"}), we will illustrate how we extract
# and resample the data to a format useful for our ML application (section
# [3](#sec:4b.dataprep){reference-type="ref"
# reference="sec:4b.dataprep"}). Then we show how to compress the data
# (section
# [4.1](#sec:4b.dimensionality_reduction_ica){reference-type="ref"
# reference="sec:4b.dimensionality_reduction_ica"}), how to project them
# to a lower number of dimensions (section
# [4.2](#sec:4b.manifold_learning){reference-type="ref"
# reference="sec:4b.manifold_learning"}), and finally, how to group
# "similar" data points together to discover salient spectral classes and
# their distribution on the surface. We conclude in section
# [4.4](#sec:4b.conclusion){reference-type="ref"
# reference="sec:4b.conclusion"} by providing a basic comparison of the
# result of the discovered spectral class distribution with maps of the
# surface of Mercury obtained using classical methods, in order to provide
# a first assessment of the machine learning techniques presented here.

# %% [markdown]
# the case, i.e. [Bandfield2000](#ref-Bandfield2000) did found spectral classes on Mars and
#

# %% [markdown]
# ## [Mercury and the MASCS instrument](#sec:4b.mercury_mascs)
#
# Surface mineralogy and composition are important indicators of the past
# of a planetary body, since they provide hints about the processes that
# formed and altered the crust, which is largely the result of the
# interior evolution. For example, the possibility of identifying specific
# mineral assemblage like metamorphic rocks, which are known to form in
# specific pressure and temperature conditions, would provide indications
# on the physical processes occurring in the subsurface that produced
# those rocks and later transported the rocks to the surface.[e.g.,
# @namurSilicateMineralogySurface2017] Similarly, observations of hydrated
# minerals can be interpreted as indicating the possible past presence of
# water, as in the case of Mars.[@meslinSoilDiversityHydration2013]
#
# While some investigations have been published on Mercury's surface
# mineralogy,[e.g.,
# @e.vanderkaadenGeochemistryMineralogyPetrology2017; @namurSilicateMineralogySurface2017; @Vilas2016a; @Sprague2009]
# its link to the endogenous (e.g., mantle convection) and exogenous
# (e.g., impacts) processes that operated during the history of the planet
# is still difficult to elucidate.[e.g.,
# @padovanImpactinducedChangesSource2017] A relevant example is the
# geological features known as hollows, discovered on the surface of
# Mercury in MESSENGER data. Hollows are rimless depressions with flat
# floors, surrounded by halos of high-albedo material, and typically found
# in clusters.[@blewett2011hollows] Given this evidence, their formation
# mechanism likely includes the loss of volatile material through one or
# more processes such as sublimation, space weathering, outgassing, or
# pyroclastic flow. Hollows are associated with a particular spectral
# signature in MESSENGER's MDIS camera,[@Vilas2016a] but a specific
# spectral signature in spectrometer data could not be identified due to
# the coarse spatial resolution of the spectrometer. Overall, the only
# clear inference based on VNIR spectra obtained by the MASCS instrument
# is that Mercury's surface shows little variation, displaying no distinct
# spectral features except for the possible indication of sulfide
# mineralogy within the hollows.[@Vilas2016a]
#
# MASCS consists of a small Cassegrain telescope with an effective focal
# length of 257 mm and a 50-mm aperture that simultaneously feeds an
# UltraViolet and Visible Spectrometer (UVVS) and a Visible and InfraRed
# Spectrograph (VIRS) channel. VIRS is a fixed concave grating
# spectrograph with a focal length of 210 mm, equipped with a beam
# splitter that simultaneously disperses the light onto a 512-element
# sensor (VIS, 300--1050 nm) and a 256-element infrared sensor array (NIR,
# 850--1450 nm). Data obtained by MASCS covers almost the entire surface
# of Mercury. The spatial resolution is highly latitude dependent due to
# the very elliptical orbit of the spacecraft, but a reference value
# $\sim5$ km. This low spatial resolution is a trade-off for higher
# spectral resolution and more spectral channels compared to the imaging
# instruments (i.e., the MDIS).
#
# The NIR sensor is characterized by 3 -- 5 times lower signal-to-noise
# ratios (SNRs) than the VIS detector and does not add significant
# information to the VIS sensor in our tests. NIR measurement cann be
# linked and corrected to match corresponding VIS measurements
# following.[However, see @Besse2015 for a successful VIS/NIR cross
# correction.] The biggest obstacle is that the the most accurate
# photometric corrections is only available for the VIS channel.[see.
# @domingueAnalysisMESSENGERMASCS2019; @domingueAnalysisMESSENGERMASCS2019a]
# We then analysed only data from VIS channel, that is enough for the sake
# of illustrating unsupervised learning techniques.

# %% [markdown]
# ## Bibliography(#sec:bibliography)
#
# <a target="_self" href="#ref-Bandfield2000">¶</a>
# Bandfield, JL, VE Hamilton, and PR Christensen. "A Global View of
# Martian Surface Compositions from MGS-TES." *Science* 287, no. March
# (2000): 1626--1630. doi:[fjh6x2](https://doi.org/fjh6x2).
# :::

# %% [markdown]
# ::: {#ref-Besse2015 .csl-entry role="doc-biblioentry"}
# Besse, S., A. Doressoundiram, and J. Benkhoff. "Spectroscopic Properties
# of Explosive Volcanism Within the Caloris Basin with MESSENGER
# Observations." *Journal of Geophysical Research: Planets* 120, no. 12
# (December 2015): 2102--2117.
# doi:[10.1002/2015JE004819](https://doi.org/10.1002/2015JE004819).
# :::
#
# ::: {#ref-blandCraterCounting2003 .csl-entry role="doc-biblioentry"}
# Bland, Phil. "Crater Counting." *Astronomy & Geophysics* 44, no. 4
# (August 2003): 4.21--4.21. doi:[dsw66x](https://doi.org/dsw66x).
# :::
#
# ::: {#ref-blewett2011hollows .csl-entry role="doc-biblioentry"}
# Blewett, D. T., N. L. Chabot, B. W. Denevi, C. M. Ernst, J. W. Head, N.
# R. Izenberg, S. L. Murchie, et al. "Hollows on Mercury: MESSENGER
# Evidence for Geologically Recent Volatile-Related Activity." *Science*
# 333, no. 6051 (September 2011): 1856--1859.
# doi:[d8hhvw](https://doi.org/d8hhvw).
# :::
#
# ::: {#ref-coenenUnderstandingUMAP2019a .csl-entry role="doc-biblioentry"}
# Coenen, Andy, and Adam Pearce. "Understanding UMAP."
# https://pair-code.github.io/understanding-umap/, 2019.
# :::
#
# ::: {#ref-Denevi2013 .csl-entry role="doc-biblioentry"}
# Denevi, Brett W., Carolyn M. Ernst, Heather M. Meyer, Mark S. Robinson,
# Scott L. Murchie, Jennifer L. Whitten, James W. Head, et al. "The
# Distribution and Origin of Smooth Plains on Mercury." *Journal of
# Geophysical Research: Planets* 118, no. 5 (May 2013): 891--907.
# doi:[10.1002/jgre.20075](https://doi.org/10.1002/jgre.20075).
# :::
#
# ::: {#ref-Denevi2009 .csl-entry role="doc-biblioentry"}
# Denevi, Brett W., Mark S. Robinson, David T. Blewett, Deborah L.
# Domingue, James W. Head III, Timothy J. McCoy, Ralph L. McNutt Jr.,
# Scott L. Murchie, and Sean C. Solomon. "MESSENGER Global Color
# Observations: Implications for the Composition and Evolution of
# Mercury's Crust." In *Lunar and Planetary Science Conference*, 1--2,
# 2009.
# :::
#
# ::: {#ref-domingueAnalysisMESSENGERMASCS2019 .csl-entry role="doc-biblioentry"}
# Domingue, Deborah L., Mario D'Amore, Sabrina Ferrari, Jörn Helbert, and
# Noam R. Izenberg. "Analysis of the MESSENGER MASCS Photometric Targets
# Part I: Photometric Standardization for Examining Spectral Variability
# Across Mercury's Surface." *Icarus* 319 (February 2019): 247--263.
# doi:[gh3dp5](https://doi.org/gh3dp5).
# :::
#
# ::: {#ref-domingueAnalysisMESSENGERMASCS2019a .csl-entry role="doc-biblioentry"}
# ---------. "Analysis of the MESSENGER MASCS Photometric Targets Part II:
# Photometric Variability Between Geomorphological Units." *Icarus* 319
# (February 2019): 140--246. doi:[gh3dp6](https://doi.org/gh3dp6).
# :::
#
# ::: {#ref-donohoHessianEigenmapsLocally2003 .csl-entry role="doc-biblioentry"}
# Donoho, David L., and Carrie Grimes. "Hessian Eigenmaps: Locally Linear
# Embedding Techniques for High-Dimensional Data." *Proceedings of the
# National Academy of Sciences* 100, no. 10 (May 2003): 5591--5596.
# doi:[cnjc4z](https://doi.org/cnjc4z).
# :::
#
# ::: {#ref-e.vanderkaadenGeochemistryMineralogyPetrology2017 .csl-entry role="doc-biblioentry"}
# E. Vander Kaaden, Kathleen, Francis M. McCubbin, Larry R. Nittler,
# Patrick N. Peplowski, Shoshana Z. Weider, Elizabeth A. Frank, and
# Timothy J. McCoy. "Geochemistry, Mineralogy, and Petrology of Boninitic
# and Komatiitic Rocks on the Mercurian Surface: Insights into the
# Mercurian Mantle." *Icarus* 285 (March 2017): 155--168.
# doi:[gg3j22](https://doi.org/gg3j22).
# :::
#
# ::: {#ref-Hamilton2005 .csl-entry role="doc-biblioentry"}
# Hamilton, Victoria E., Harry Y. McSween, and Bruce Hapke. "Mineralogy of
# Martian Atmospheric Dust Inferred from Thermal Infrared Spectra of
# Aerosols." *Journal of Geophysical Research* 110, no. E12 (2005): 1--11.
# doi:[bhsb7j](https://doi.org/bhsb7j).
# :::
#
# ::: {#ref-hastieElementsStatisticalLearning2009 .csl-entry role="doc-biblioentry"}
# Hastie, Trevor, Robert Tibshirani, and Jerome Friedman. *The Elements of
# Statistical Learning: Data Mining, Inference, and Prediction, Second
# Edition*. Second. Springer Series in Statistics. New York:
# Springer-Verlag, 2009.
# doi:[10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7).
# :::
#
# ::: {#ref-Helbert2013 .csl-entry role="doc-biblioentry"}
# Helbert, Jörn, Alessandro Maturilli, Mario D'Amore, and M. D'Amore.
# "Visible and Near-Infrared Reflectance Spectra of Thermally Processed
# Synthetic Sulfides as a Potential Analog for the Hollow Forming
# Materials on Mercury." *Earth and Planetary Science Letters* 369--370
# (May 2013): 233--238. doi:[gbddt9](https://doi.org/gbddt9).
# :::
#
# ::: {#ref-hyvarinenIndependentComponentAnalysis2000 .csl-entry role="doc-biblioentry"}
# Hyvärinen, A., and E. Oja. "Independent Component Analysis: Algorithms
# and Applications." *Neural Networks* 13, no. 4 (June 2000): 411--430.
# doi:[cx35gq](https://doi.org/cx35gq).
# :::
#
# ::: {#ref-kerrWhoCanRead2006 .csl-entry role="doc-biblioentry"}
# Kerr, Richard A. "Who Can Read the Martian Clock?" *Science* 312, no.
# 5777 (May 2006): 1132--1133. doi:[b6v8tt](https://doi.org/b6v8tt).
# :::
#
# ::: {#ref-leeNonlinearDimensionalityReduction2007 .csl-entry role="doc-biblioentry"}
# Lee, John A., and Michel Verleysen. *Nonlinear Dimensionality
# Reduction*. Information Science and Statistics. New York:
# Springer-Verlag, 2007.
# doi:[10.1007/978-0-387-39351-3](https://doi.org/10.1007/978-0-387-39351-3).
# :::
#
# ::: {#ref-maatenVisualizingDataUsing2008 .csl-entry role="doc-biblioentry"}
# Maaten, Laurens van der, and Geoffrey Hinton. "Visualizing Data Using
# t-SNE." *Journal of Machine Learning Research* 9, no. 86 (2008):
# 2579--2605.
# :::
#
# ::: {#ref-Maturilli2014a .csl-entry role="doc-biblioentry"}
# Maturilli, A., J. Helbert, J. M. St. John, J. W. Head, W. M. Vaughan, M.
# D'Amore, M. Gottschalk, and S. Ferrari. "Komatiites as Mercury Surface
# Analogues: Spectral Measurements at PEL." *Earth and Planetary Science
# Letters* 398 (2014). doi:[gg3jwp](https://doi.org/gg3jwp).
# :::
#
# ::: {#ref-McClintock2007 .csl-entry role="doc-biblioentry"}
# McClintock, William E., and Mark R. Lankton. "The Mercury Atmospheric
# and Surface Composition Spectrometer for the MESSENGER Mission." *Space
# Science Reviews* 131, no. 1--4 (2007): 481--521.
# doi:[btc6f6](https://doi.org/btc6f6).
# :::
#
# ::: {#ref-mcinnesUMAPUniformManifold2018 .csl-entry role="doc-biblioentry"}
# McInnes, Leland. "UMAP: Uniform Manifold Approximation and Projection
# for Dimension Reduction Umap 0.5 Documentation."
# https://umap-learn.readthedocs.io/en/latest/index.html, 2018.
# :::
#
# ::: {#ref-mcinnesUMAPUniformManifold2020 .csl-entry role="doc-biblioentry"}
# McInnes, Leland, John Healy, and James Melville. "UMAP: Uniform Manifold
# Approximation and Projection for Dimension Reduction." *arXiv:1802.03426
# \[Cs, Stat\]* (September 2020). <https://arxiv.org/abs/1802.03426>.
# :::
#
# ::: {#ref-meslinSoilDiversityHydration2013 .csl-entry role="doc-biblioentry"}
# Meslin, P.-Y., O. Gasnault, O. Forni, S. Schröder, A. Cousin, G. Berger,
# S. M. Clegg, et al. "Soil Diversity and Hydration as Observed by ChemCam
# at Gale Crater, Mars." *Science* 341, no. 6153 (September 2013).
# doi:[f3sdqx](https://doi.org/f3sdqx).
# :::
#
# ::: {#ref-namurSilicateMineralogySurface2017 .csl-entry role="doc-biblioentry"}
# Namur, Olivier, and Bernard Charlier. "Silicate Mineralogy at the
# Surface of Mercury." *Nature Geoscience* 10, no. 1 (January 2017):
# 9--13. doi:[f9r3qp](https://doi.org/f9r3qp).
# :::
#
# ::: {#ref-nasaPDSPDS3Standards2009 .csl-entry role="doc-biblioentry"}
# NASA. "PDS: PDS3 Standards Reference."
# https://pds.nasa.gov/datastandards/pds3/standards/, 2009.
# :::
#
# ::: {#ref-Nittler2011 .csl-entry role="doc-biblioentry"}
# Nittler, L. R., R. D. Starr, S. Z. Weider, T. J. McCoy, W. V. Boynton,
# D. S. Ebel, C. M. Ernst, et al. "The Major-Element Composition of
# Mercury's Surface from MESSENGER X-Ray Spectrometry." *Science* 333, no.
# 6051 (September 2011): 1847--1850. doi:[bps3b8](https://doi.org/bps3b8).
# :::
#
# ::: {#ref-nittlerGlobalMajorelementMaps2020 .csl-entry role="doc-biblioentry"}
# Nittler, Larry R., Elizabeth A. Frank, Shoshana Z. Weider, Ellen
# Crapster-Pregont, Audrey Vorburger, Richard D. Starr, and Sean C.
# Solomon. "Global Major-Element Maps of Mercury from Four Years of
# MESSENGER X-Ray Spectrometer Observations." *Icarus* (February 2020):
# 113716. doi:[ggm5sv](https://doi.org/ggm5sv).
# :::
#
# ::: {#ref-padovanImpactinducedChangesSource2017 .csl-entry role="doc-biblioentry"}
# Padovan, Sebastiano, Nicola Tosi, Ana-Catalina Plesa, and Thomas Ruedas.
# "Impact-Induced Changes in Source Depth and Volume of Magmatism on
# Mercury and Their Observational Signatures." *Nature Communications* 8,
# no. 1 (December 2017): 1945. doi:[gcn9p2](https://doi.org/gcn9p2).
# :::
#
# ::: {#ref-Peplowski2016 .csl-entry role="doc-biblioentry"}
# Peplowski, Patrick N., Rachel L. Klima, David J. Lawrence, Carolyn M.
# Ernst, Brett W. Denevi, Elizabeth A. Frank, John O. Goldsten, Scott L.
# Murchie, Larry R. Nittler, and Sean C. Solomon. "Remote Sensing Evidence
# for an Ancient Carbon-Bearing Crust on Mercury." *Nature Geoscience* 9,
# no. 4 (March 2016): 273--276.
# doi:[10.1038/ngeo2669](https://doi.org/10.1038/ngeo2669).
# :::
#
# ::: {#ref-roweisNonlinearDimensionalityReduction2000 .csl-entry role="doc-biblioentry"}
# Roweis, Sam T., and Lawrence K. Saul. "Nonlinear Dimensionality
# Reduction by Locally Linear Embedding." *Science* 290, no. 5500
# (December 2000): 2323--2326. doi:[cbws2r](https://doi.org/cbws2r).
# :::
#
# ::: {#ref-ruixuSurveyClusteringAlgorithms2005 .csl-entry role="doc-biblioentry"}
# Rui Xu, and D. Wunsch. "Survey of Clustering Algorithms." *IEEE
# Transactions on Neural Networks* 16, no. 3 (May 2005): 645--678.
# doi:[c3pfgf](https://doi.org/c3pfgf).
# :::
#
# ::: {#ref-Sprague2009 .csl-entry role="doc-biblioentry"}
# Sprague, A. L., K. L. Donaldson Hanna, R. W. H. Kozlowski, J Helbert, A
# Maturilli, J. B. Warell, and J. L. Hora. "Spectral Emissivity
# Measurements of Mercury's Surface Indicate Mg- and Ca-Rich Mineralogy,
# K-Spar, Na-Rich Plagioclase, Rutile, with Possible Perovskite, and
# Garnet." *Planetary and Space Science* 57, no. 3 (March 2009): 364--383.
# doi:[fbd9jq](https://doi.org/fbd9jq).
# :::
#
# ::: {#ref-tenenbaumGlobalGeometricFramework2000 .csl-entry role="doc-biblioentry"}
# Tenenbaum, Joshua B., Vin de Silva, and John C. Langford. "A Global
# Geometric Framework for Nonlinear Dimensionality Reduction." *Science*
# 290, no. 5500 (December 2000): 2319--2323.
# doi:[cz8wgk](https://doi.org/cz8wgk).
# :::
#
# ::: {#ref-Vasavada1999 .csl-entry role="doc-biblioentry"}
# Vasavada, a. "Near-Surface Temperatures on Mercury and the Moon and the
# Stability of Polar Ice Deposits." *Icarus* 141, no. 2 (October 1999):
# 179--193. doi:[b9fhjd](https://doi.org/b9fhjd).
# :::
#
# ::: {#ref-Vilas2016a .csl-entry role="doc-biblioentry"}
# Vilas, Faith, Deborah L. Domingue, Jörn Helbert, Mario D'Amore,
# Alessandro Maturilli, Rachel L. Klima, Karen R. Stockstill-Cahill, et
# al. "Mineralogical Indicators of Mercury's Hollows Composition in
# MESSENGER Color Observations." *Geophysical Research Letters* 43, no. 4
# (February 2016): 1450--1456. doi:[gg3j2v](https://doi.org/gg3j2v).
# :::
# :::
#
# ::: {#footnotes .section .footnotes .footnotes-end-of-document role="doc-endnotes"}
#
# ------------------------------------------------------------------------
#
# 1.  ::: {#fn1}
#     Like Hapke, as in Hamilton, McSween, and Hapke, "Mineralogy of
#     Martian Atmospheric Dust Inferred from Thermal Infrared Spectra of
#     Aerosols."[↩︎](#fnref1){.footnote-back role="doc-backlink"}
#     :::
#
# 2.  ::: {#fn2}
#     E.g., Helbert et al., "Visible and Near-Infrared Reflectance Spectra
#     of Thermally Processed Synthetic Sulfides as a Potential Analog for
#     the Hollow Forming Materials on Mercury."[↩︎](#fnref2){.footnote-back
#     role="doc-backlink"}
#     :::
#
# 3.  ::: {#fn3}
#     Denevi et al., "MESSENGER Global Color
#     Observations."[↩︎](#fnref3){.footnote-back role="doc-backlink"}
#     :::
#
# 4.  ::: {#fn4}
#     E.g., Bland, "Crater Counting"; Kerr, "Who Can Read the Martian
#     Clock?"[↩︎](#fnref4){.footnote-back role="doc-backlink"}
#     :::
#
# 5.  ::: {#fn5}
#     McClintock and Lankton, "The Mercury Atmospheric and Surface
#     Composition Spectrometer for the MESSENGER
#     Mission."[↩︎](#fnref5){.footnote-back role="doc-backlink"}
#     :::
#
# 6.  ::: {#fn6}
#     E.g., Namur and Charlier, "Silicate Mineralogy at the Surface of
#     Mercury."[↩︎](#fnref6){.footnote-back role="doc-backlink"}
#     :::
#
# 7.  ::: {#fn7}
#     Meslin et al., "Soil Diversity and Hydration as Observed by ChemCam
#     at Gale Crater, Mars."[↩︎](#fnref7){.footnote-back
#     role="doc-backlink"}
#     :::
#
# 8.  ::: {#fn8}
#     E.g., E. Vander Kaaden et al., "Geochemistry, Mineralogy, and
#     Petrology of Boninitic and Komatiitic Rocks on the Mercurian
#     Surface"; Namur and Charlier, "Silicate Mineralogy at the Surface of
#     Mercury"; Vilas et al., "Mineralogical Indicators of Mercury's
#     Hollows Composition in MESSENGER Color Observations"; Sprague et
#     al., "Spectral Emissivity Measurements of Mercury's Surface Indicate
#     Mg- and Ca-Rich Mineralogy, K-Spar, Na-Rich Plagioclase, Rutile,
#     with Possible Perovskite, and Garnet."[↩︎](#fnref8){.footnote-back
#     role="doc-backlink"}
#     :::
#
# 9.  ::: {#fn9}
#     E.g., Padovan et al., "Impact-Induced Changes in Source Depth and
#     Volume of Magmatism on Mercury and Their Observational
#     Signatures."[↩︎](#fnref9){.footnote-back role="doc-backlink"}
#     :::
#
# 10. ::: {#fn10}
#     Blewett et al., "Hollows on Mercury."[↩︎](#fnref10){.footnote-back
#     role="doc-backlink"}
#     :::
#
# 11. ::: {#fn11}
#     Vilas et al., "Mineralogical Indicators of Mercury's Hollows
#     Composition in MESSENGER Color
#     Observations."[↩︎](#fnref11){.footnote-back role="doc-backlink"}
#     :::
#
# 12. ::: {#fn12}
#     Ibid.[↩︎](#fnref12){.footnote-back role="doc-backlink"}
#     :::
#
# 13. ::: {#fn13}
#     However, see Besse, Doressoundiram, and Benkhoff, "Spectroscopic
#     Properties of Explosive Volcanism Within the Caloris Basin with
#     MESSENGER Observations" for a successful VIS/NIR cross
#     correction.[↩︎](#fnref13){.footnote-back role="doc-backlink"}
#     :::
#
# 14. ::: {#fn14}
#     See. Domingue et al., "Analysis of the MESSENGER MASCS Photometric
#     Targets Part I"; Domingue et al., "Analysis of the MESSENGER MASCS
#     Photometric Targets Part II."[↩︎](#fnref14){.footnote-back
#     role="doc-backlink"}
#     :::
#
# 15. ::: {#fn15}
#     NASA, "PDS."[↩︎](#fnref15){.footnote-back role="doc-backlink"}
#     :::
#
# 16. ::: {#fn16}
#     We found a GDAL bug when reading 8 bytes real values that is solved
#     for version [≥]{.math .inline}2.3.0 after our report to the
#     developer. See <https://trac.osgeo.org/gdal/wiki/Release/2.3.0-News>
#     and use this version or higher when manipulating MASCS
#     data.[↩︎](#fnref16){.footnote-back role="doc-backlink"}
#     :::
#
# 17. ::: {#fn17}
#     PostgreSQL is a relational database management that controls the
#     creation, integrity, maintenance and use of a
#     database[↩︎](#fnref17){.footnote-back role="doc-backlink"}
#     :::
#
# 18. ::: {#fn18}
#     PostGIS adds support for geographic objects in geographic
#     information system and extends the database language with functions
#     to create and manipulate geographic objects. PostGIS follows the
#     Simple Features for SQL specification from the Open Geospatial
#     Consortium (OGC).[↩︎](#fnref18){.footnote-back role="doc-backlink"}
#     :::
#
# 19. ::: {#fn19}
#     MASCS VIS data have different wavelength sampling and part of the
#     global Mercury campaign had different spectral
#     binning.[↩︎](#fnref19){.footnote-back role="doc-backlink"}
#     :::
#
# 20. ::: {#fn20}
#     Domingue et al., "Analysis of the MESSENGER MASCS Photometric
#     Targets Part I"; Domingue et al., "Analysis of the MESSENGER MASCS
#     Photometric Targets Part II."[↩︎](#fnref20){.footnote-back
#     role="doc-backlink"}
#     :::
#
# 21. ::: {#fn21}
#     Hastie, Tibshirani, and Friedman, *The Elements of Statistical
#     Learning*; Hyvärinen and Oja, "Independent Component
#     Analysis."[↩︎](#fnref21){.footnote-back role="doc-backlink"}
#     :::
#
# 22. ::: {#fn22}
#     Tenenbaum, Silva, and Langford, "A Global Geometric Framework for
#     Nonlinear Dimensionality Reduction"; Roweis and Saul, "Nonlinear
#     Dimensionality Reduction by Locally Linear Embedding"; Donoho and
#     Grimes, "Hessian Eigenmaps"; Maaten and Hinton, "Visualizing Data
#     Using t-SNE"; McInnes, Healy, and Melville,
#     "UMAP."[↩︎](#fnref22){.footnote-back role="doc-backlink"}
#     :::
#
# 23. ::: {#fn23}
#     Lee and Verleysen, *Nonlinear Dimensionality
#     Reduction*.[↩︎](#fnref23){.footnote-back role="doc-backlink"}
#     :::
#
# 24. ::: {#fn24}
#     Roweis and Saul, "Nonlinear Dimensionality Reduction by Locally
#     Linear Embedding."[↩︎](#fnref24){.footnote-back role="doc-backlink"}
#     :::
#
# 25. ::: {#fn25}
#     McInnes, Healy, and Melville, "UMAP."[↩︎](#fnref25){.footnote-back
#     role="doc-backlink"}
#     :::
#
# 26. ::: {#fn26}
#     Ibid.[↩︎](#fnref26){.footnote-back role="doc-backlink"}
#     :::
#
# 27. ::: {#fn27}
#     https://umap-learn.readthedocs.io/en/latest/how_umap_works.html[↩︎](#fnref27){.footnote-back
#     role="doc-backlink"}
#     :::
#
# 28. ::: {#fn28}
#     McInnes, Healy, and Melville, "UMAP."[↩︎](#fnref28){.footnote-back
#     role="doc-backlink"}
#     :::
#
# 29. ::: {#fn29}
#     McInnes, "UMAP."[↩︎](#fnref29){.footnote-back role="doc-backlink"}
#     :::
#
# 30. ::: {#fn30}
#     The interactive tutorial \"Understanding UMAP\" give some insight in
#     how the hyperparameters influence UMAP. See [ (Coenen and Pearce,
#     "Understanding UMAP")]{.citation
#     cites="coenenUnderstandingUMAP2019a"} and
#     https://pair-code.github.io/understanding-346umap[↩︎](#fnref30){.footnote-back
#     role="doc-backlink"}
#     :::
#
# 31. ::: {#fn31}
#     see for example \"Selecting the number of clusters with silhouette
#     analysis on KMeans clustering\"
#     <https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html>[↩︎](#fnref31){.footnote-back
#     role="doc-backlink"}
#     :::
#
# 32. ::: {#fn32}
#     Rui Xu and Wunsch, "Survey of Clustering
#     Algorithms."[↩︎](#fnref32){.footnote-back role="doc-backlink"}
#     :::
#
# 33. ::: {#fn33}
#     Vasavada, "Near-Surface Temperatures on Mercury and the Moon and the
#     Stability of Polar Ice Deposits."[↩︎](#fnref33){.footnote-back
#     role="doc-backlink"}
#     :::
#
# 34. ::: {#fn34}
#     Denevi et al., "The Distribution and Origin of Smooth Plains on
#     Mercury."[↩︎](#fnref34){.footnote-back role="doc-backlink"}
#     :::
#
# 35. ::: {#fn35}
#     Ibid.[↩︎](#fnref35){.footnote-back role="doc-backlink"}
#     :::
#
# 36. ::: {#fn36}
#     Bandfield, Hamilton, and Christensen, "A Global View of Martian
#     Surface Compositions from MGS-TES."[↩︎](#fnref36){.footnote-back
#     role="doc-backlink"}
#     :::
#
# 37. ::: {#fn37}
#     Maturilli et al., "Komatiites as Mercury Surface
#     Analogues."[↩︎](#fnref37){.footnote-back role="doc-backlink"}
#     :::
#
# 38. ::: {#fn38}
#     Nittler et al., "Global Major-Element Maps of Mercury from Four
#     Years of MESSENGER X-Ray Spectrometer
#     Observations."[↩︎](#fnref38){.footnote-back role="doc-backlink"}
#     :::
#
# 39. ::: {#fn39}
#     Peplowski et al., "Remote Sensing Evidence for an Ancient
#     Carbon-Bearing Crust on Mercury."[↩︎](#fnref39){.footnote-back
#     role="doc-backlink"}
#     :::
#
# 40. ::: {#fn40}
#     Nittler et al., "The Major-Element Composition of Mercury's Surface
#     from MESSENGER X-Ray Spectrometry"; Nittler et al., "Global
#     Major-Element Maps of Mercury from Four Years of MESSENGER X-Ray
#     Spectrometer Observations."[↩︎](#fnref40){.footnote-back
#     role="doc-backlink"}
#     :::
#
# 41. ::: {#fn41}
#     https://github.com/epn-ml/MESSENGER-Mercury-Surface-Cassification-Unsupervised_DLR[↩︎](#fnref41){.footnote-back
#     role="doc-backlink"}
#     :::
# :::

# %% [markdown]
# <h2 class="unnumbered" id="sec:bibliography">Bibliography</h2>
# <div id="refs" class="references csl-bib-body hanging-indent"
# role="doc-bibliography">
# <div id="ref-Bandfield2000" class="csl-entry" role="doc-biblioentry">
# Bandfield, JL, VE Hamilton, and PR Christensen. <span>“A <span>Global
# View</span> of <span>Martian Surface Compositions</span> from
# <span>MGS</span>-<span>TES</span>.”</span> <em>Science</em> 287, no.
# March (2000): 1626–1630. doi:<a
# href="https://doi.org/fjh6x2">fjh6x2</a>.
# </div>
#

# %% [markdown] tags=[]
# ## Imports
#
# Generic imports

# %% tags=[]
import geopandas as gpd
from shapely.geometry import Polygon
import fiona

import matplotlib
# matplotlib.use('Agg') # non interactive
# #%matplotlib qt # for not-notebook
# %matplotlib inline
from   matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import seaborn as sns


matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['axes.titlesize'] = 24
matplotlib.rcParams['axes.labelsize'] = 20
matplotlib.rcParams['font.size'] = 16

pd.set_option('display.width',150)
pd.set_option('display.max_colwidth',150)
pd.set_option('display.max_rows',150)

from IPython.display import display

# %% tags=[]
import datashader as ds
import holoviews as hv
import holoviews.operation.datashader as hds
import hvplot.pandas
# hv.extension('bokeh','matplotlib')

# %% [markdown]
# Define auxiliary functions & data

# %% tags=[]
base_path = pathlib.Path('..')                     # local base path
input_data_path = base_path / 'data/processed'  # input data location 
out_figure_path = base_path / 'reports/figures' # output location <- CHANGE THIS TO YOUR LIKING
out_models_path = base_path / 'models' # output location <- CHANGE THIS TO YOUR LIKING

print(f'{base_path=}')
print(f'{input_data_path=}')
print(f'{out_models_path=}')


# this globally saves all generated plot with save_plot defind below
save_plots_bool = 0


# %% tags=[]
# make a figure
def make_map(in_data,alpha,norm,interpolation=None):
    '''
    outdf_gdf.loc[spectral_df_nona_index,'R'] = spectral_df[in_wav]
    '''
    
    outdf_gdf.loc[spectral_df_nona_index,'R'] = in_data

    map_crs = ccrs.PlateCarree(central_longitude=0.0)
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=[18,18],subplot_kw={'projection': map_crs})
    background = ax.imshow(img,
                            cmap=plt.cm.gray,
                            extent=img_extent,
                            origin='upper',
                            transform=ccrs.PlateCarree(central_longitude=0.0)
                          );
    # Here we are using the numpy reshape because we now the final image shape: it is not always the case!!
    # This is FASTER then Geopandas.GeoDataFrame.plot!!!!
    im = ax.imshow(outdf_gdf.sort_index()['R'].values.reshape(360,180).T,
               interpolation= interpolation,
               extent= data_img_extent,
               cmap=plt.cm.Spectral_r,
               transform=ccrs.PlateCarree(central_longitude=0.0),
               origin='upper',
               alpha=alpha,
               # vmax=0.065,
               norm=norm,
                  );
    return im

def save_plot(out_file,
              output_dir = out_figure_path,
              dpi=150,
              out_format='jpg',
              save=False):
    ''' helper function to save previous plot
    '''
    
    out_path = output_dir / (out_file +f'.{out_format}' )
    if save :
        plt.savefig(out_path,dpi=dpi)
        return (f'Saving image to {out_path}')
    else:
        return (f'NOT saving image to {out_path}')  

def df_shader(in_data,**kwargs):
    '''
    accept input : spectral_df[in_wav] 
    add it to outdf_gdf:
    outdf_gdf.loc[spectral_df_nona_index,'R'] = spectral_df[in_wav]
    
    returns shader(hv.Points(outdf_gdf.loc[spectral_df_nona_index,'R']))
    '''
    vdims = 'R'
    kdims=[('x','longitude'),('y','latitude')]
    outdf_gdf.loc[spectral_df_nona_index,'R'] = in_data

    return shader(hv.Points(outdf_gdf.loc[spectral_df_nona_index,['x','y','R']],kdims=kdims,vdims=vdims).opts(),**kwargs).opts()


def df_rasterer(in_data,**kwargs):
    '''
    accept input : spectral_df[in_wav] 
    add it to outdf_gdf:
    outdf_gdf.loc[spectral_df_nona_index,'R'] = spectral_df[in_wav]
    
    returns shader(hv.Points(outdf_gdf.loc[spectral_df_nona_index,'R']))
    '''
    vdims = 'R'
    kdims=[('x','longitude'),('y','latitude')]
    outdf_gdf.loc[spectral_df_nona_index,'R'] = in_data

    return rasterer(hv.Points(outdf_gdf.loc[spectral_df_nona_index,['x','y','R']],kdims=kdims,vdims=vdims).opts(),**kwargs).opts()


def shader(ppoints,aggregator=ds.mean(),x_sampling=1,y_sampling=1,cmap=plt.cm.Spectral_r, dynamic=True):
    return hds.shade(
                hds.rasterize(ppoints,
                              aggregator=aggregator,
                              x_sampling=x_sampling,
                              y_sampling=y_sampling),
                              cmap=cmap,
                              dynamic=dynamic,
                              )

def rasterer(ppoints,aggregator=ds.mean(),x_sampling=2,y_sampling=2,dynamic=True):
    return hds.rasterize(ppoints,
                         aggregator=aggregator,
                         x_sampling=x_sampling,
                         y_sampling=y_sampling,
                         dynamic=dynamic,
                        )


def colorbar_img_shader(in_data,cmap=plt.cm.Spectral_r,**kwargs):
    
    raster = df_rasterer(in_data,**kwargs).opts(alpha=1,colorbar=True,cmap=cmap)
    
    return hv.Overlay([raster,
                        hds.shade(raster,cmap=cmap,group='datashaded'),
                        background.opts(cmap='Gray',alpha=0.5),
                      ])
# .collate()

def find_nearest_in_array(
                 value,
                 array,
                 return_value=False):
    """Find nearest value in a numpy array

    Parameters
    ----------

    value : value to search for
    array : array to search in, should be sorted.
    return_value : bool, if to return the actual values

    Returns
    -------

    """

##TODO add sorting check/option to sort ? make sense?
    import numpy as np

    closest_index = (np.abs(array - value)).argmin()

    if value > np.nanmax(array):
        import warnings
        warnings.warn(f'value > np.nanmax(array) : {value} > {np.nanmax(array)}')

    if value < np.nanmin(array):
        import warnings
        warnings.warn(f'value < np.nanmin(array) : {value} < {np.nanmax(array)}')

    if not return_value: 
        return closest_index
    else:
        return closest_index, array[closest_index]


def get_mascs_geojson(file, gzipped=True):
    """return a geopandas.DataFrame from a geojson, could be compressed.

    Parameters
    ----------

    file : path of geojson compressed file to geopandas dataframe
    gzipped : bool, if compressed

    Returns
    -------

    geopandas.DataFrame
    """

    import geopandas as gpd
    import numpy as np
    import json
    import gzip
    from geopandas import GeoDataFrame

    if gzipped:
        # get the compressed geojson
        with gzip.GzipFile(file, 'r') as fin:
            geodata = json.loads(fin.read().decode('utf-8'))
    else:
        # get the uncompressed geojson
        with open(file, 'r') as fin:
            geodata = json.load(fin)

    import shapely
    # extract geometries
    geometries = [shapely.geometry.Polygon(g['geometry']['coordinates'][0]) for g in geodata['features']]
    # extract id
    ids = [int(g['id']) for g in geodata['features']]
    # generate GeoDataFrame
    out_gdf = gpd.GeoDataFrame(data=[g['properties'] for g in geodata['features']], geometry=geometries, index=ids).sort_index()
    # cast arrays to numpy
    if 'array' in out_gdf:
        out_gdf['array'] = out_gdf['array'].apply(lambda x: np.array(x).astype(float))

    return out_gdf

#CRS from  https://github.com/Melown/vts-registry/blob/master/registry/registry/srs.json
mercury_crs = {
    "geographic-dmercury2000": {
        "comment": "Geographic, DMercury2000 (iau2000:19900)",
        "srsDef": "+proj=longlat +a=2439700 +b=2439700 +no_defs",
        "type": "geographic"
    },
    "geocentric-dmercury2000": {
        "comment": "Geocentric, Mercury",
        "srsDef": "+proj=geocent +a=2439700 +b=2439700 +lon_0=0 +units=m +no_defs",
        "type": "cartesian"
    },
    "eqc-dmercury2000": {
        "comment": "Equidistant Cylindrical, DMercury2000 (iau2000:19911)",
        "srsDef": "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=2439700 +b=2439700 +units=m +no_defs",
        "type": "projected"
    },
    "merc-dmercury2000": {
        "comment": "Mercator, DMercury2000 (iau2000:19974)",
        "srsDef": "+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=2439700 +b=2439700 +units=m +no_defs",
        "type": "projected"
    },
    "steren-dmercury2000": {
        "comment": "Polar Sterographic North, DMercury2000 (iau2000:19918)",
        "srsDef": "+proj=stere +lat_0=90 +lat_ts=90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=2439700 +b=2439700 +units=m +no_defs",
        "type": "projected"
    },
    "steres-dmercury2000": {
        "comment": "Polar Stereographic South, DMercury2000 (iau2000:19920)",
        "srsDef": "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=2439700 +b=2439700 +units=m +no_defs",
        "type": "projected"
}}


# %% [markdown] tags=[]
# ## Load Data
#
# Data are too big to be included in this repo, user can find it on [Zenodo](https://zenodo.org/record/7433033) at [https://zenodo.org/record/7433033](https://zenodo.org/record/7433033).
#
# Download the datafile `grid_2D_-180_+180_-90_+90_1deg_st_median_photom_iof_sp_2nm.geojson.gz` in  `data/processed` with some variation of
#
# ```bash
# curl https://zenodo.org/record/7433033/files/grid_2D_0_360_-90_%2B90_1deg_st_median_photom_iof_sp_2nm.png --output data/processed/grid_2D_-180_+180_-90_+90_1deg_st_median_photom_iof_sp_2nm.geojson.gz
# ```
#
# This is a preview of the data cube from Zenodo.
#
# ![Preview od the data cube from Zenodo](https://zenodo.org/api/iiif/v2/c98bb0bc-cfa1-449e-94f7-95f9d074543e:f4cc114a-fac9-42b8-b5a8-380901fe8dba:grid_2D_0_360_-90_%2B90_1deg_st_median_photom_iof_sp_2nm.png/full/750,/0/default.png)
#
# Create the wavelenghts array and its helper functions

# %% tags=[]
# create the wavelenghts array
wav_grid_2nm = np.arange(260,1052,2)

# define find_nearest, with wav_grid_2nm as default array
find_nearest = lambda x : find_nearest_in_array(x,wav_grid_2nm)

# this is to index an array based on wav_grid_2nm
wavelenght = 415
print(f'         wavelenght = {wavelenght:5d} <-- number we search for')
print(f'find_nearest({wavelenght:5d}) = {find_nearest(wavelenght):5d} <-- index of {wavelenght} in wav_grid_2nm')

# %% [markdown] tags=[]
#
# Select input data. From MASCS documentation :  
#
# ```
# PHOTOM_IOF_SPECTRUM_DATA : 
#    Derived column of photometrically normalized 
#    reflectance-at-sensor spectra. One row per spectrum. NIR spectrum has up 
#    to 256 values (depending on binning and windowing), VIS has up to 512. 
#    Reflectance is a unitless parameter. Reflectance from saturated pixels, 
#    or binned pixels with one saturated element, are set to 1e32. PER 
#    SPECTRUM column."
#
# IOF_SPECTRUM_DATA : 
#    DESCRIPTION = "Derived column of reflectance-at-sensor spectra. One 
#    row per spectrum. NIR spectrum has up to 256 values (depending on binning 
#    and windowing), VIS has up to 512. Reflectance is a unitless parameter. 
#    Reflectance from saturated pixels, or binned pixels with one saturated 
#    element, are set to 1e32. PER SPECTRUM column."
# ```
#
# define the datafile with filename structure:
#
# ```
# [description from database]_[function applied to the spectra for each pixel]_[data array used]
#     [description from database] = grid_2D_0_360_-90_+90
#     [function applied to the spectra for each pixel] = avg or st_median
#     [data array used] = iof_sp_2nm or photom_iof_sp_2nm
# ```

# %% tags=[]
input_data_name = 'grid_2D_-180_+180_-90_+90_1deg_st_median_photom_iof_sp_2nm'

# %% [markdown]
# The data are in a gzipped geojson to reduce size, but geopandas doesn't like it.
#
# The function below accept a path and return a GeoDataFrame.
#
# An optional `gzipped[=True default]` keywords take care of compressed geojson.

# %% tags=[]
outdf_gdf = get_mascs_geojson( input_data_path / (input_data_name+'.geojson.gz'), gzipped=True)
# outdf_gdf = get_mascs_geojson( input_data_path / (input_data_name+'.geojson.gz'), gzipped=True, cast_to_numeric=False)

# this is to be sure that the cells are ordered in natural way == reshape with numpy
outdf_gdf = outdf_gdf.set_index('natural_index',drop=True).sort_index()
import fiona.crs

# set Mercury Lat/Lon as crs
outdf_gdf.crs = fiona.crs.from_string(mercury_crs['geographic-dmercury2000']['srsDef'])

# %% [markdown]
# Unravel spectral reflectance data

# %% tags=[]
# create wavelenghts columns: this create empy columns with np.nan (nice!)
# use a separate df, because mixed types columns are crazy. and buggy
spectral_df = pd.DataFrame(index=outdf_gdf.index,columns = wav_grid_2nm).fillna(np.nan)
## assign single wavelenght to columns, only where array vectors len !=0
spectral_df.loc[outdf_gdf['array'].apply(lambda x : len(x)) != 0, wav_grid_2nm] = np.stack(outdf_gdf.loc[outdf_gdf['array'].apply(lambda x : len(x)) != 0,'array'], axis=0).astype(np.float64)
## drop array column
outdf_gdf.drop(columns=['array'], inplace=True)

# %% tags=[]
# create x and y cols = lon and lat
outdf_gdf['x'] = outdf_gdf.apply(lambda x: x['geometry'].centroid.x , axis=1)
outdf_gdf['y'] = outdf_gdf.apply(lambda x: x['geometry'].centroid.y , axis=1)

# %% tags=[]
# drop outlier : this clean up further noisy data, instrumental effect, etc
print(spectral_df.shape)
low = .02
high = .999
quant_df = spectral_df.quantile([low, high])
spectral_df =  spectral_df[spectral_df >= 0].apply(lambda x: x[(x>quant_df.loc[low,x.name]) &\
                                       (x < quant_df.loc[high,x.name])], axis=0)\
                                       
print(spectral_df.shape)

# %% tags=[]
# cut to stop_wav
# iloc doesn't support int columns indexing!!!!
start_wav = 268 # below all NaN
stop_wav  = 975 # above a bump in NaN 

spectral_df = spectral_df.iloc[:,find_nearest(start_wav):find_nearest(stop_wav)+1]

# %% tags=[]
# count nan
print(f'{spectral_df.shape=}')

# %% tags=[]
fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(19,1.5))

spectral_df.isna().sum(axis=0).plot(ax=axs[0]);
spectral_df.isna().sum(axis=1).plot(ax=axs[1]);

# %% tags=[]
import seaborn as sns
print(spectral_df.shape, spectral_df.dropna(axis=0,how='any').shape)

display(spectral_df.sample(10))

# %% tags=[]
# whole data distribution 
fig, axarr = plt.subplots(nrows=1, ncols=2,figsize=(19,2))
sns.histplot(spectral_df.dropna(how='any').values.flatten(), ax= axarr[0],bins = 255,alpha=0.4, kde=True,edgecolor='none');
sns.histplot(spectral_df.dropna(how='any').values.flatten(), ax= axarr[1],bins = 255,alpha=0.4, kde=True,edgecolor='none',log_scale=(False, True),);

save_plot('whole_data_distribution_seaborn',save=save_plots_bool)


# %% tags=[]
img = spectral_df.dropna(how='any').values.T

from skimage import exposure

# Equalization : paramterless
plt.figure(figsize=[24,8]);
plt.imshow(exposure.equalize_hist(img),
           interpolation='bicubic',
           aspect='auto',
           cmap=plt.cm.Spectral_r);
# [Colorbar Tick Labelling Demo — Matplotlib 3.1.2 documentation](https://matplotlib.org/3.1.1/gallery/ticks_and_spines/colorbar_tick_labelling_demo.html)
cbar = plt.colorbar(ticks=[0.01, 0.5, 1], orientation='vertical')
cbar.ax.set_yticklabels(
    np.around([np.percentile(img,0.01), np.median(img), np.nanmax(img)],decimals=3)
    );

plt.tight_layout()

save_plot('spectrogram',save=save_plots_bool)

# %% tags=[]
# keep the old name? naa
spectral_df_nona_index = spectral_df.dropna(how='any').index

print(f'        outdf_gdf : {outdf_gdf.shape}')
print(f'      spectral_df : {spectral_df.shape}')
print(f'spectral_df_nonan : {spectral_df_nona_index.shape}') 

# %% tags=[]
# define 2 wavelenghts and calculate something 
in_wav = 450
en_wav = 1050

idx_in = find_nearest(in_wav)
idx_en = find_nearest(en_wav)

outdf_gdf.loc[spectral_df_nona_index,'refl'] = spectral_df[in_wav]

print(f'(wav[{idx_in}],wav[{idx_en}]) = ({wav_grid_2nm[idx_in]}, {wav_grid_2nm[idx_en]}) \nspectral[:,idx_in:idx_en].shape : {spectral_df.loc[:,in_wav:en_wav].shape}')


# %% tags=[]
# calculate rows & cols from grid properties files, assuming regular grid
rows, cols = 360, 180
rows_half_step, cols_half_step = 1, 1

data_img_extent = [-180.0, 180.0, -90.0, 90.0]

# extent = [outdf_gdf.total_bounds[i] for i in [0,2,1,3]]

# %% tags=[]
# specific wavelengths data distribution 

fig, axarr = plt.subplots(nrows=1, ncols=1,figsize=(20,4))

plot_wav = 300
ax = sns.histplot(spectral_df[plot_wav].dropna(),ax= axarr,bins = 255,stat="density", alpha=0.4, kde=True,edgecolor='none');
ax.set_xlim([0.005,0.075])
plot_wav = 700
sns.histplot(spectral_df[plot_wav].dropna(),ax= axarr,bins = 255,stat="density", alpha=0.4, kde=True,edgecolor='none');
plot_wav = 900
sns.histplot(spectral_df[plot_wav].dropna(),ax= axarr,bins = 255,stat="density", alpha=0.4, kde=True,edgecolor='none');


plt.tight_layout()

save_plot('global_300nm_700nm_900nm_distribution', out_format='png',save=save_plots_bool)

# %% [markdown]
# ## Plotting section

# %% tags=[]
import cartopy.crs as ccrs
import cartopy
# from scipy import misc
import imageio
from skimage import transform 
# read background image
img = imageio.imread( input_data_path / '1280x640_20120330_monochrome_basemap_1000mpp_equirectangular.png')
img_extent = (-180, 180, -90, 90)

# %% tags=[]
hv.extension('bokeh')

from PIL import Image, ImageEnhance
kdims=[('x','latitude'),('y','longitude')]

backimage = Image.open( input_data_path / '1280x640_20120330_monochrome_basemap_1000mpp_equirectangular.png')
## from help :  as four-tuple defining the (left, bottom, right and top) edges.
hv_img_extent = (-180, -90,180, 90)

background = hv.Image(
            np.array(backimage),
            bounds=hv_img_extent,
            kdims=kdims,
            group='backplane',
            ).opts(cmap='Gray',clone=False)

# # show the background image
# background.options(width=800,height=400)

# %% tags=[]
map_crs = ccrs.PlateCarree(central_longitude=0.0)
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=[18,18],subplot_kw={'projection': map_crs})
background = ax.imshow(img,
                        cmap=plt.cm.gray,
                        extent=img_extent,
                        origin='upper',
                        transform=ccrs.PlateCarree(central_longitude=0.0)
                      );

# %% tags=[]
hv.extension('matplotlib')
hv.output(fig='png')

# %% tags=[]
df_shader((spectral_df[970]-spectral_df[270])/700.)#.opts(fig_inches=4, aspect=2,fig_size=200)

# %% tags=[]
#     vdims = 'R'
#     kdims=[('x','longitude'),('y','latitude')]
#     outdf_gdf.loc[spectral_df_nona_index,'R'] = in_data

#     return shader(hv.Points(outdf_gdf.loc[spectral_df_nona_index,['x','y','R']],kdims=kdims,vdims=vdims).opts(),**kwargs).opts()

spectral_df[970]

# %% tags=[]
# spectral slope R[970]-R[270] / 270 -970
(background.opts(cmap='Gray',alpha=0.5)*\
 df_shader((spectral_df[970]-spectral_df[270])/700.,cmap=plt.cm.Spectral_r).opts(interpolation='bilinear',alpha=0.5)).\
    opts(
    fig_inches=4, aspect=2,fig_size=200
)

# %% tags=[]
hv.extension('matplotlib')
plot_wav = 700

out = colorbar_img_shader(spectral_df[plot_wav])
_ = out.DynamicMap.II.opts(interpolation='None',aspect=1.8,fig_size=200,alpha=0.7)
out
# hv.save(out,out_figure_path / '1b_mascs_700nm_refl.png')

# %% tags=[]
hv.extension('bokeh')
background.opts(cmap='Gray')*df_shader(spectral_df[plot_wav],cmap=plt.cm.inferno).opts(height=600,width=1000,alpha=0.7)

out = colorbar_img_shader(spectral_df[plot_wav])
_ = out.DynamicMap.II.opts(height=400,width=800,alpha=0.75)
out

# %% [markdown] tags=[]
# ## Learning

# %% tags=[]
# small_data == spectral_df[spectral_df_nonan]
X = spectral_df.loc[spectral_df_nona_index]

print(f'        outdf_gdf : {outdf_gdf.shape}')
print(f'      spectral_df : {spectral_df.shape}')
print(f'spectral_df_nonan : {spectral_df_nona_index.shape}') 


# %% [markdown]
# ### Dimensionality reduction
#
# [2.5. Decomposing signals in components (matrix factorization problems) — scikit-learn 0.20.2 documentation](https://scikit-learn.org/stable/modules/decomposition.html#decompositions)

# %%
from sklearn import decomposition
from sklearn import model_selection
from sklearn import pipeline, preprocessing

# %% [markdown] heading_collapsed=true
# #### PCA

# %% hidden=true tags=[]
# Principal components analysis : which is the data dimensionality?

# n_components = spectral_df.shape[1]//2
# pca = decomposition.PCA(n_components=n_components)
pca = decomposition.PCA(0.95)
pca.fit(X)
n_components = pca.n_components_
X_pca = pca.transform(X)

print('X.shape               : {}\n'
      'X_pca.shape           : {}\n'
      'pca.components_.shape : {}'.format(X.shape, X_pca.shape, pca.components_.shape))

print("              variance       var_ratio      cum_var_ratio")
for i in range(n_components):
    print("Component %2s: %12.10f   %12.10f   %12.10f" % (i, pca.explained_variance_[i], pca.explained_variance_ratio_[i], np.cumsum(pca.explained_variance_ratio_)[i]))


# As we can see, only the 2 first components are useful
# pca.n_components = 2
# small_data_pca = pca.fit_transform(small_data)

# %% hidden=true tags=[]

# %% [markdown] heading_collapsed=true hidden=true
# ##### PCA residual error estimation

# %% hidden=true tags=[]
pca_error = []

cmp_index = np.linspace(0, pca.n_components_, num=pca.n_components_//2, endpoint=True, dtype=int)
cmp_index[-1] -= 1

for i in cmp_index:
# for i in range(pca.n_components):
    print("Component %2s: %12.10f   %12.10f   %12.10f" % (i, pca.explained_variance_[i], pca.explained_variance_ratio_[i], np.cumsum(pca.explained_variance_ratio_)[i]))
    img = (X-np.dot(X_pca[:,:i],pca.components_[:i])-X.mean()).values
    pca_error.append({
        'min': img.min(),
        'max': img.max(),
        'mean' : img.mean(),
        'median' : np.median(img),
        'std' : np.std(img),
        'pca_scores' : np.mean(model_selection.cross_val_score(decomposition.PCA(n_components=n_components).fit(X), X,cv=2))
    })
    print(pca_error[-1])

pca_errors_df = pd.DataFrame.from_dict(pca_error)
pca_errors_df['delta'] = pca_errors_df['max']-pca_errors_df['min']
pca_errors_df['explained_variance'] = pca.explained_variance_[cmp_index]
pca_errors_df['explained_variance_ratio'] = pca.explained_variance_ratio_[cmp_index]

# %% hidden=true tags=[]
pca_errors_df.describe()

# %% hidden=true
import hvplot.pandas
a = (pca_errors_df[['std']]).hvplot().opts(width=1200,height=400)
b = (pca_errors_df['delta']).hvplot().opts(width=1200,height=400)
c = pca_errors_df['pca_scores'].hvplot().opts(width=1200,height=400)
d = pca_errors_df[['max','min']].hvplot().opts(width=1200,height=400)

c
# ((a+b+c)*hv.HLine(10000*(0.005*0.005)).opts(color='red')).cols(1)

# %% hidden=true tags=[]
# %matplotlib inline

from skimage import exposure
img = (X-np.dot(X_pca,pca.components_)-X.mean()).values

plt.figure(figsize=[20,8]);
plt.imshow(exposure.equalize_hist(img,nbins=512),
           aspect='auto',
           interpolation='bilinear',
           cmap=plt.cm.Spectral_r,
           extent = [X.columns.min(),X.columns.max(),  X.index.min(), X.index.max()],
          );
# cbar = plt.colorbar(ticks=[0.01, 0.5, 1], orientation='vertical')
# cbar.ax.set_yticklabels(
#     np.around([np.percentile(img,0.01), np.median(img), np.nanmax(img)],decimals=3)
#     );
plt.show()

# %% [markdown] hidden=true tags=[]
# recontruct initial data with choosen PCA components  and look at the difference

# %% hidden=true tags=[]
diff_img = (X-np.dot(X_pca,pca.components_)-X.mean())
fig, axs = plt.subplots(nrows=2,ncols=2,figsize=[20,8])

ax = axs.flatten()

ax[0].set_title('max')
diff_img.max().plot(ax=ax[0])

ax[1].set_title('median')
diff_img.median().plot(ax=ax[1])

ax[2].set_title('min')
diff_img.min().plot(ax=ax[2])

ax[3].set_title('std')
diff_img.std().plot(ax=ax[3])

# %% hidden=true tags=[]
# hv.extension('matplotlib')
# plot_wav = 500
# out = colorbar_img_shader((X-np.dot(X_pca,pca.components_)-X.mean())[plot_wav])
# _ = out.DynamicMap.II.opts(interpolation='bicubic',aspect=2,fig_size=400,alpha=0.7)
# out

# spectral_df.columns.min(),np.quantile(spectral_df.columns,0.25),np.quantile(spectral_df.columns,0.75),spectral_df.columns.max()
# (268, 444, 797.5, 974)

hv.extension('matplotlib')
    
NdLayout = hv.NdLayout(
            {plot_wav: df_shader( (X-np.dot(X_pca,pca.components_)-X.mean())[plot_wav],x_sampling=2,y_sampling=2).\
            opts(interpolation='bicubic',aspect=2,alpha=0.8)
                 for ind,plot_wav in enumerate([270, 470, 770, 970])}
            ,kdims='Wav')

(background.opts(cmap='Gray',alpha=1)*NdLayout.cols(2)).opts(fig_size=200,tight=True)

# %% [markdown] heading_collapsed=true hidden=true
# ##### PCA visualisation

# %% hidden=true tags=[]
# %matplotlib inline

components_shift = 0

fig = plt.figure(figsize=(12,4))
ax = plt.subplot()
ax.plot(np.arange(pca.n_components_-components_shift)+components_shift, pca.explained_variance_ratio_[components_shift:])
ax.plot(np.arange(pca.n_components_-components_shift)+components_shift, pca.explained_variance_ratio_[components_shift:], "r.")
ax.set_title("PCA explained variance")
ax.set_yscale('log')
# ax.set_xlim([components_shift-1, pca.n_components_])
# ax.set_ylim([np.min(pca.explained_variance_ratio_),np.max(pca.explained_variance_ratio_)])
plt.show()

# %% tags=[]
hv.Curve(
                   (np.arange(pca.n_components_-components_shift)+components_shift, pca.explained_variance_ratio_[components_shift:]),
                   kdims='components',vdims='var. ratio')


# %% hidden=true tags=[]
components_shift = 0
hv.extension('bokeh')

overlay = hv.Curve(
                   (np.arange(pca.n_components_-components_shift)+components_shift, pca.explained_variance_ratio_[components_shift:]),
                   kdims='components',vdims='var. ratio')

overlay2 = hv.Curve(
                    (np.arange(pca.n_components_-components_shift)+components_shift, np.cumsum(pca.explained_variance_ratio_[components_shift:])),
                    kdims='components',vdims='var. cumsum')

overlay3 = hv.Curve(
                    (np.arange(pca.n_components_-components_shift)+components_shift, np.gradient(np.cumsum(pca.explained_variance_ratio_[components_shift:]))),
                    kdims='components',vdims='gradient(var. cumsum)')

overlay4 = hv.Curve(
                    (np.arange(pca.n_components_-components_shift)+components_shift, np.gradient(np.gradient(np.cumsum(pca.explained_variance_ratio_[components_shift:])))) ,
                    kdims='components',vdims='$gradient^2$(var. cumsum)' )


layout = overlay+overlay2+overlay3+overlay4

layout.opts(
    hv.opts.Curve(line_width=3,height=500, width=600),
    hv.opts.Points(alpha=0.5, size=10),
).cols(2)

# %% hidden=true
hv.extension('bokeh')

vdims = ['~Reflectance']
kdims=['Wavelenght (um)']

width=1100
height=500

shift = 0.3
overlay_dict = {'PCA.{}'.format(ind):hv.Curve((spectral_df.columns.to_numpy(),cmp + shift*(ind+1)),vdims=vdims,kdims=kdims) for ind,cmp in 
                enumerate(preprocessing.MinMaxScaler().fit_transform(pca.components_.T).T[:4,:])}

overlay_dict['Mean'] = hv.Curve((spectral_df.columns.to_numpy(),
                  preprocessing.MinMaxScaler().fit_transform(spectral_df.mean().values.reshape(-1, 1)).squeeze()
                                ),vdims=vdims,kdims=kdims).opts(line_width=1,line_dash='solid',color='black',alpha=1)

# shift_dict = {'PCA.{}-shift'.format(ind):hv.HLine(0.5*ind).opts(
#                 line_width=0.25,line_dash='dashed',color='black') for ind in range(4)}


hv.NdOverlay(overlay_dict).opts(
                hv.opts.Curve(line_width=4, show_grid=True),
                hv.opts.NdOverlay(width=width,height=height,legend_cols=4,legend_position='bottom')
                ) #* hv.NdOverlay(shift_dict)

# %% hidden=true
hv.extension('bokeh')

max_pca_comp = 4

NdLayout = hv.NdLayout(
            {ind: df_shader(cmp[:,np.newaxis],x_sampling=2,y_sampling=2).\
             options(height=350,width=550,alpha=0.8)
                 for ind,cmp in enumerate(X_pca[:,:max_pca_comp].T)}
            ,kdims='Component')

background.opts(cmap='Gray',alpha=0.9)*NdLayout.cols(2)

# %% hidden=true
hv.extension('matplotlib')

max_pca_comp = 16

for ind,cmp in enumerate(X_pca[:,:max_pca_comp].T):
    print(f'PCA.{ind:02} min:{cmp.min():10.5} , max:{cmp.max():10.5}, delta:{cmp.max()-cmp.min():10.5}')

NdLayout = hv.NdLayout(
            {ind: df_shader(cmp[:,np.newaxis],x_sampling=2,y_sampling=2).\
            opts(interpolation='bicubic',aspect=2,alpha=0.7)
                 for ind,cmp in enumerate(X_pca[:,:max_pca_comp].T)}
            ,kdims='Component')

(background.opts(cmap='Gray',alpha=1)*NdLayout.cols(4)).opts(tight=True, vspace=0.01, hspace=0.01, fig_size=150).cols(4)

# %% hidden=true
hv.extension('bokeh')

import itertools
gridplot = {}
for x,y in itertools.combinations(range(3), 2):
    print(x,y,X_pca[:,[x,y]].shape)
    gridplot[f'PCA.{x} vs PCA.{y}'] = hv.Points(X_pca[:,[x,y]],kdims=[f'PCA.{x}',f'PCA.{y}'])


# # hds.dynspread(
hds.datashade(hv.NdLayout(gridplot),aggregator=ds.count(),cmap=plt.cm.viridis, x_sampling=0.003, y_sampling=0.003).\
opts(height=1000,width=1200,tight=True).cols(3)
# opts(fig_size=200,aspect_weight=True,tight=True).cols(3)


# %% [markdown]
# #### ICA
#
# [2.5. Decomposing signals in components (matrix factorization problems) — scikit-learn 0.20.2 documentation](https://scikit-learn.org/stable/modules/decomposition.html#independent-component-analysis-ica)
#
# Calculate the reconstruction error for increasing number of ICA components :
#
#     print(np.concatenate((np.arange(1,5),np.arange(0,160,20)[1:])))
#     array([  1,   2,   3,   4,  20,  40,  60,  80, 100, 120, 140])

# %% tags=[]
ica_rec_error_path = pathlib.Path(out_models_path / 'ica_rec_error_df.csv')

# check if we already run and stored this 
if ica_rec_error_path.is_file():
    # load reconstruction error
    ica_rec_error_df = pd.read_csv(ica_rec_error_path, index_col='ICA components n.')
else:
    # calculate reconstruction error
    ica_rec_error = {}
    for ica_n_components in np.concatenate((np.arange(1,5),np.arange(0,160,20)[1:])) : 
    #     print(ica_n_components)
        ica = decomposition.FastICA(n_components=ica_n_components,random_state=4)
        S_  = ica.fit_transform(X)
        # evaluate overall reconstruction error
        ica_rec_error[ica_n_components] = np.std((X-ica.inverse_transform(S_)).values.flatten())
        print(ica_n_components,ica_rec_error[ica_n_components])

    ica_rec_error_df = pd.DataFrame.from_dict(ica_rec_error,orient='index')
    ica_rec_error_df.index.name = 'ICA components n.'
    ica_rec_error_df.columns = ['reconstruction error']
    ica_rec_error_df.to_csv(ica_rec_error_path)

# %%
hv.extension('bokeh')
display(ica_rec_error_df.sort_index())
import hvplot.pandas

(hv.HLine(0.0015,group='line')*\
 hv.VLine(4,group='line')*\
 ica_rec_error_df.sort_index().hvplot()*\
 ica_rec_error_df.sort_index().hvplot(kind='scatter')).\
opts(
    hv.opts.Points(marker='circle',alpha=0.5),
    hv.opts.Curve(color='red') 
     )

# (hv.RGB(np.random.rand(10, 10, 4), group='A') * hv.RGB(np.random.rand(10, 10, 4), group='B')).opts(
#     hv.opts.RGB('A', alpha=0.1), hv.opts.RGB('B', alpha=0.5)
# )

# # ica_rec_error_df
# hv.extension('bokeh')

# (ica_rec_error_df/X.min().min()).hvplot()

# %%
import hvplot.pandas
hv.extension('matplotlib')

main_plot = hv.render((hv.HLine(0.0015,group='line')*\
 hv.VLine(4,group='line')*\
 ica_rec_error_df.sort_index().hvplot()*\
 ica_rec_error_df.sort_index().hvplot(kind='scatter')).\
opts(
    hv.opts.Points(marker='circle',alpha=0.5),
    hv.opts.Curve(color='red', fig_size=250, aspect=2) 
     ))

inset = hv.render((hv.HLine(0.0015,group='line')*\
 hv.VLine(4,group='line')*\
 ica_rec_error_df.sort_index().hvplot()*\
 ica_rec_error_df.sort_index().hvplot(kind='scatter')).\
opts(
    hv.opts.Points(marker='circle',alpha=0.5),
    hv.opts.Curve(color='red', fig_size=50, aspect=1) 
     ).options(xlim=(2.5, 5),ylim=(0.00144, 0.00165)))


main_ax = main_plot.get_axes()[0]
inset = inset.get_axes()

from mpl_toolkits.axes_grid1.inset_locator import inset_axes,mark_inset

axin = inset_axes(main_ax, width='50%', height='50%', loc=1)
# axin = main_ax.inset_axes([80, 0.001, 0.2, 0.5])

# # main_ax.inset_axes?
main_plot

# %% tags=[]
# %matplotlib inline
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,mark_inset

fig, ax = plt.subplots(figsize=[12,6])
ica_rec_error_df.sort_index().plot(ax = ax,label=False,marker='o',legend=False)
ax.hlines(0.0015,xmin = ica_rec_error_df.index.min(), xmax = ica_rec_error_df.index.max(), color='red')
ax.vlines(4,ymin = ica_rec_error_df.min(), ymax = ica_rec_error_df.max(), color='green')
ax.set_ylim( [ ica_rec_error_df.min().values[0], ica_rec_error_df.max().values[0]] )

axin = inset_axes(ax, width='60%', height='60%', loc=1)
ica_rec_error_df.sort_index().plot(ax = axin,label=False,marker='o',legend=False)
axin.hlines(0.0015,xmin = ica_rec_error_df.index.min(), xmax = ica_rec_error_df.index.max(), color='red')
axin.vlines(4,ymin = ica_rec_error_df.min(), ymax = ica_rec_error_df.max(), color='green')

axin.set_ylim([0.00144, 0.00165])
axin.set_xlim([2.5, 5])
axin.axes.get_xaxis().set_visible(False)
axin.axes.get_yaxis().set_visible(False)

mark_inset(ax, axin, loc1=2, loc2=3, fc="gray", alpha=0.3, ec="0.5");
plt.tight_layout()
plt.show()
save_plot('ICA_reconstruction_error_zoom_included', out_format='png',save=save_plots_bool)


# %%
# Compute ICA

ica_n_components= 4

# for ica_n_components in range(2,16):
ica = decomposition.FastICA(n_components=ica_n_components,random_state=4)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_           # Get estimated mixing matrix

# coefficients matrix
# S_ /= S_.std(axis=0)

# vector components = signal
# A_ -= A_.mean(axis=0)
# X ~= S_ x A_.T

print(f'ica_n_components : {ica_n_components}')
print(f'X : {X.shape}')
print(f'A_ : {A_.shape}')
print(f'S_ : {S_.shape}')
# print(f'square(sum(X-ICA^-1(X)) : {np.sum((X-ica.inverse_transform(S_))**2)}')
# print(f'np.norm(X-ICA^-1(X),2) : {np.linalg.norm(X-ica.inverse_transform(S_),2)}')
print(((X-ica.inverse_transform(S_)).max()-(X-ica.inverse_transform(S_)).min()).describe().T)

# %%
## reconstruction error at specific wav ma
# h.extension('matplotlib')
# plot_wav = 500
# out = colorbar_img_shader((X-np.dot(X_pca,pca.components_)-X.mean())[plot_wav])
# _ = out.DynamicMap.II.opts(interpolation='bicubic',aspect=2,fig_size=400,alpha=0.7)
# out

# spectral_df.columns.min(),np.quantile(spectral_df.columns,0.25),np.quantile(spectral_df.columns,0.75),spectral_df.columns.max()
# (268, 444, 797.5, 974)

hv.extension('matplotlib')
    
NdLayout = hv.NdLayout(
#           difference maps
            {plot_wav: df_shader( (X-ica.inverse_transform(S_))[plot_wav],x_sampling=1,y_sampling=1).\
#           only reoconstructed vectors maps
#              {plot_wav: df_shader( ica.inverse_transform(S_)[:,plot_wav//2-X.columns[0]],x_sampling=2,y_sampling=2).\

            opts(interpolation='bicubic',aspect=2,alpha=0.8)
                 for ind,plot_wav in enumerate([270, 470, 770, 970])}
            ,kdims='Wav')

(background.opts(cmap='Gray',alpha=1)*NdLayout.cols(2)).opts(fig_size=200,tight=True)

# %% tags=[]
# %matplotlib inline
np.sqrt(((X-ica.inverse_transform(S_)).max()-(X-ica.inverse_transform(S_))).apply(np.square).sum()/X.size).plot(figsize=[20,3])
plt.show()

# %%
hv.extension('bokeh')

vdims = ['Reflectance']
kdims=['wavelenght (nm)']
width=1100
height=500

overlay_dict = {f'ICA comp. n.{ind}':hv.Curve((spectral_df.columns.to_numpy(),cmp),vdims=vdims,kdims=kdims) for ind,cmp in enumerate(A_.T) }

overlay_dict['Mean'] = hv.Curve((spectral_df.columns.to_numpy(),
                  preprocessing.MinMaxScaler().fit_transform(spectral_df.mean().values.reshape(-1, 1)).squeeze()
                                ),vdims=vdims,kdims=kdims).opts(line_width=1,line_dash='dashed',color='black',alpha=1)


hv.NdOverlay(overlay_dict).opts(
                hv.opts.Curve(line_width=4, show_grid=True),
                hv.opts.NdOverlay(width=width,height=height)
                )

# %% tags=[]
hv.extension('matplotlib')

overlay_dict['Mean'].opts(linewidth=1, color='black',alpha =0.25)

out = hv.NdOverlay(overlay_dict).opts(
                hv.opts.Curve(linewidth=4, show_grid=False),
                hv.opts.NdOverlay(fig_size=500, aspect=2.5))
out
save_plot('ICA_components', out_format='png',save=save_plots_bool)


# %%
# [python - Holoviews change datashader colormap - Stack Overflow](https://stackoverflow.com/a/59837074)
from holoviews.plotting.util import process_cmap
# [Colormaps — HoloViews 1.12.7 documentation](http://holoviews.org/user_guide/Colormaps.html)
# process_cmap("Plasma")

# %%
hv.extension('matplotlib')

max_comp = ica_n_components

# sampling = 2
# cmp = S_[:,2]
# out = colorbar_img_shader(cmp[:,np.newaxis],x_sampling=sampling,y_sampling=sampling)
# _ = out.DynamicMap.II.opts(interpolation='bilinear',aspect=2,fig_size=400,alpha=1)
# out

for ind,cmp in enumerate(S_[:,:max_comp].T):
    print(f'ICA.{ind:02} min:{cmp.min():10.5} , max:{cmp.max():10.5}, delta:{cmp.max()-cmp.min():10.5}')

sampling = 1
NdLayout = hv.NdLayout(
            {ind: df_shader(cmp[:,np.newaxis],x_sampling=sampling,y_sampling=sampling).\
            opts(interpolation='bicubic',aspect=2,alpha=0.85)
                for ind,cmp in enumerate(S_.T)}
            ,kdims='ICA Component')

out = (background.opts(cmap='Gray',alpha=1)*NdLayout.cols(4)).opts(tight=True, vspace=0.01, hspace=0.01, fig_size=300).cols(2)
out
# hv.save(out, out_figure_path / 'ICA_components_map.png')

# %% tags=[]
# hv.extension('bokeh')
hv.extension('matplotlib')

df = pd.DataFrame(
    data = S_,
    columns=[f'ICA.{ica_c}' for ica_c in range(ica_n_components)]
)

# # limit to the two "major components"
# df_ds = hv.Dataset(df[['ICA.1','ICA.2']])
df_ds = hv.Dataset(df)

sampling = (df.max()-df.min()).describe().mean()/250

def local_datashade ( X,
                     aggregator=ds.count(),
                     cmap=plt.cm.Spectral_r,
                     x_sampling=sampling,
                     y_sampling=sampling,
                     **kwargs):
    return hds.datashade(X,aggregator=aggregator,cmap=cmap,x_sampling=x_sampling,y_sampling=y_sampling, **kwargs)

point_grid = hv.operation.gridmatrix(df_ds, diagonal_operation=hv.operation.histogram.instance(num_bins=50) ).map(local_datashade, hv.Scatter)
# .opts(hv.opts.RGB(interpolation='bilinear',aspect=1,fig_size=200))

out = point_grid.opts(fig_size=300).opts(hv.opts.RGB(interpolation='bilinear',aspect=1))
out
# hv.save(out, out_figure_path / 'ICA_coefficients_gridplot_density_C1_C2.png', dpi=200)
# hv.save(out, out_figure_path / 'ICA_coefficients_gridplot_density.png', dpi=200)

# %% [markdown]
# ### Manifold Learning
#
# [2.2. Manifold learning — scikit-learn 0.22.1 documentation](https://scikit-learn.org/stable/modules/manifold.html)

# %%
[p.stem.split('_') for p in pathlib.Path('embedding/').glob('*.npy')]

# %% [markdown] heading_collapsed=true
# #### TSNE embedding

# %% hidden=true
# all data
cached_tsne = pathlib.Path('tsne_embedding.npy')

# # low correlation 
# threshold=0.985
# X = CovarianceThreshold(threshold=threshold).fit_transform(spectral.values[:,idx_in:idx_en])
# cached_tsne = pathlib.Path('tsne_embedding_lowcorr-{}.npy'.format(threshold))

if cached_tsne.is_file():
    print(f'Loading: {cached_tsne}')
    X_tsne = np.load(cached_tsne)
else:
    from sklearn.manifold import TSNE
    X_tsne = TSNE(n_components=2).fit_transform(X)
    np.save(cached_tsne,X_tsne)    

print(X.shape, X_tsne.shape)

# %% hidden=true
hv.extension('matplotlib')

vdims = ['label']
kdims=['TSNE.0','TSNE.1']

perplexities = [5, 30, 50, 100]

### RANDOM SAMPPLING
# randomize = np.random.choice(X_pca.shape[0], size=1000)
# label=labels[randomize]
# X = X_pca[randomize,:]

overwrite=True
label= None
X = X_pca
print(f'TSN X.shape : {X.shape}')

filenameextra ='_pcacomponents-{}'.format(n_components)

def get_tsne(perplexity=None,filename_extra=filenameextra, overwrite=False):
    from sklearn.manifold import TSNE
    cached_tsne = pathlib.Path('embedding/tsne_embedding_perplexity-{}{}.npy'.format(perplexity,filename_extra))
    if cached_tsne.is_file():
        print(f'Loading: {cached_tsne}')
        x_tsne = np.load(cached_tsne)
    else:
        print(f'file not found: {cached_tsne} - Calculating!')
        if overwrite:
            x_tsne = TSNE(n_components=2).fit_transform(X)
            np.save(cached_tsne,x_tsne)    
        else:
            print(f'overwrite set to {overwrite} stopping')
    print('X.shape : {}, X_tsne.shape : {}, perplexity : {}'.format(X.shape, x_tsne.shape,perplexity))
    return x_tsne


def tsne_to_holocurve(*argv, **kwargs):
    xtsne = get_tsne(perplexity=kwargs['perplexity'],filename_extra=kwargs['filename_extra'], overwrite=kwargs['overwrite'])
    print(kwargs)
    if 'label' in kwargs and kwargs.get('label') is not None:
#         print('Label')
#         print(np.unique(kwargs.get('label'),return_counts=True))
        return hv.Scatter(np.hstack([xtsne ,kwargs.get('label')[:,np.newaxis]]),vdims=kwargs['vdims'],kdims=kwargs['kdims'])
    else:
        print('Nolabel')
        return hv.Scatter(xtsne,kdims=kwargs['kdims'][0], vdims=kwargs['kdims'][1])
    
curve_dict = {p:tsne_to_holocurve(label=label,
                                  perplexity=p,
                                  overwrite=overwrite,
                                  filename_extra=filenameextra,
                                  vdims=vdims,
                                  kdims=kdims)
              for p in perplexities}

# %% hidden=true
label= None
min_size, max_size = 20, 30

if label is not None:
    print('labels = ',label.shape)
    classes , s = np.unique(label, return_counts=True)
    print('classes size:',dict(zip(classes,s)))
    # marker size inverse proportional to population size
    sizes = ((1-(s-s.min())/(s.max()-s.min()))*(max_size-min_size))+min_size
    NdLayout = hv.NdLayout(curve_dict, kdims='perplexity').opts(hv.opts.Scatter(s= [dict(zip(classes,sizes)).get(l) for l in label]),hv.opts.NdLayout(fig_inches=6))
    NdLayout.opts(hv.opts.Scatter(color=vdims[0]))
else:
    print('no labels = ')
    NdLayout = hv.NdLayout(curve_dict, kdims='perplexity').opts(hv.opts.Scatter(s=min_size),hv.opts.NdLayout(fig_inches=6))

NdLayout.opts(hv.opts.Scatter(alpha=0.25, cmap='Set1'))

# %% hidden=true
perplexity = 30
# X_tsne = get_tsne(perplexity=perplexity,filename_extra=filenameextra, overwrite=False)
X_tsne = curve_dict.get(perplexity).data[:,:-1]
print('X.shape : {}, X_tsne.shape : {}, perplexity : {}'.format(X.shape, X_tsne.shape,perplexity))

# %% [markdown]
# #### UMAP embedding
#
# [Basic UMAP Parameters — umap 0.3 documentation](https://umap-learn.readthedocs.io/en/latest/parameters.html)

# %%

import umap

vdims = ['label']
kdims=['UMAP.0','UMAP.1']

X_embedd = X.values

filenameextra ='_icacomponents-{}'.format(ica_n_components)

### RANDOM SAMPPLING
randomize = np.random.choice(X.shape[0], size=100)
try:
    label=labels[randomize]
except NameError:
    label=None
# X_embedd = X_embedd[randomize,:]

overwrite=True
label= None

# neighbors = np.arange(5,X_embedd.shape[0]//6,X_embedd.shape[0]//20) # 5 to a quarter of the data each 1/8 o the data
neighbors     = [   100, 4000, 7000]
min_distances = (0.0, 0.5, 0.99)

print('X_embedd.shape : ',X_embedd.shape)
print('     neighbors : ',neighbors)
print(' min_distances : ',min_distances)

import itertools
print(list(itertools.product(neighbors,min_distances)))

def get_umap(neighbors, mindistances,filename_extra=filenameextra, label=None, overwrite=False):

    cached_umap = pathlib.Path('embedding/umap_embedding_neighbors-{}_mindist-{}{}.npy'.format(neighbors,mindistances,filename_extra))
    if cached_umap.is_file():
        print(f'Loading: {cached_umap}')
        x_umap = np.load(cached_umap)
    else:
        print(f'file not found: {cached_umap} - Calculating!')
        if overwrite: 
            x_umap = umap.UMAP(n_neighbors=neighbors, min_dist = mindistances).fit_transform(X_embedd)
            np.save(cached_umap,x_umap)    
        else:
            print(f'overwrite set to {overwrite} stopping')
    print('X_embedd.shape : {}, x_umap.shape : {}, neighbors : {}, min_dist: {}'.format(X_embedd.shape, x_umap.shape,neighbors, mindistances))
    if label is not None:
        return hv.Scatter(np.hstack([x_umap,label[:,np.newaxis]]),vdims=vdims,kdims=kdims)
    else:
        return hv.Scatter(x_umap)



# %%
hv.extension('matplotlib')

curve_dict_2D = {(n,d):get_umap(n,d,label=label, overwrite=overwrite) for n in neighbors for d in min_distances}

gridspace = hv.GridSpace(curve_dict_2D, kdims=['neighbors, local > global structure', 'minimum distance in representation']).opts(hv.opts.Scatter(s=25,alpha=0.25),hv.opts.GridSpace(fig_inches=16))

# if labels.any(): 
#     gridspace.opts(hv.opts.Scatter(cmap=plt.cm.Spectral_r,c='label'))

# gridspace

# %%
gridspace

hv.extension('matplotlib')
# hv.extension('bokeh')

out = hds.dynspread(
hds.datashade(gridspace,
              aggregator=ds.count(),
              cmap=plt.cm.Spectral_r,
              x_sampling=0.25,
              y_sampling=0.25,
             ).\
opts(aspect=1,fig_size=70).opts(hv.opts.RGB(interpolation='bilinear')))
# opts(height=1600,width=1600))
out

# hv.save(out, out_figure_path / f'UMAP_gridspace_ICA_{ica_n_components}components.png', dpi=200)

# %%
neigh_mindist =  (4000, 0.99)

# print('X.shape : {}, X_umap.shape : {}, (n_neighbors,min_dist) : {}'.format(X.shape, X_umap.shape,neigh_mindist))
X_umap_scatter = get_umap(neighbors=neigh_mindist[0], mindistances=neigh_mindist[1],filename_extra=filenameextra, label=None, overwrite=True)

X_umap = curve_dict_2D.get(neigh_mindist).data

# %%
vdims = ['label']
kdims=['UMAP.0','UMAP.1']

hv.extension('matplotlib')
# hv.extension('bokeh')
hds.datashade(X_umap_scatter,
              aggregator=ds.count(),
              cmap=plt.cm.Spectral_r,
              x_sampling=0.4,
              y_sampling=0.3,
             ).\
opts(interpolation='bilinear',aspect=1,fig_size=200)
# opts(height=600,width=600)


# %%
import pdir
a = hds.rasterize(X_umap_scatter,aggregator=ds.count(),dynamic=False,x_sampling=0.4, y_sampling=0.3).data.to_dataframe()
display(a.describe())

ax = a.plot.hist(bins=45)
ax.set_yscale('log')

# %% [markdown]
# ### Classification

# %% tags=[]
from sklearn import cluster
from sklearn import preprocessing

# data for classificatiom

# X_classification = X_pca # PCA
# X_classification = S_ # ICA
# X_classification = W # NFM # really noisy/bad!!
# X_classification = X_tsne # tsne embedding
# X_classification = X_umap # umap embedding
X_classification = z_mean
n_classification_features = X_classification.shape[1]

print('X_classification shape : ',X_classification.shape)

# %% tags=[]
##########################
# Scalers
##########
# scaler , preprocessing_type = preprocessing.StandardScaler().fit(X_classification)    , 'StandardScaler'
# scaler , preprocessing_type = preprocessing.MinMaxScaler().fit(X_classification) , 'MinMaxScaler'
# scaler , preprocessing_type = preprocessing.RobustScaler().fit(X_classification) , 'RobustScaler'
scaler, preprocessing_type = preprocessing.FunctionTransformer(lambda x:x), 'None'

# ##########################
# classifier = 'K-Means'
# n_clusters = 2
# # Kmeans estimator instance and Classify scaled data
# k_means = cluster.KMeans(n_clusters=n_clusters, random_state=0).fit(scaler.transform(X_classification))
# labels = k_means.labels_
# print('k_means.inertia_ : ',k_means.inertia_)

##########################
classifier = 'AgglomerativeClustering'
n_clusters = 13
aggclustering = cluster.AgglomerativeClustering(linkage='complete',
                                                affinity='l2',
                                                n_clusters=n_clusters).fit(scaler.transform(X_classification))
labels = aggclustering.labels_

##########################
# classifier = 'DBSCAN'
# dbscan = cluster.DBSCAN(eps=0.9, min_samples=5).fit(X_classification)
# labels = dbscan.labels_

##########################

# ##########
# classifier = 'HDBSCAN'
# import hdbscan
# hdbscan = hdbscan.HDBSCAN(
#             min_cluster_size=X_classification.shape[0]//2000,
#             min_samples=1,
#             cluster_selection_epsilon=0.75,
# #             allow_single_cluster=False,
#             )
# hdbscan.fit(scaler.transform(X_classification))
# labels = hdbscan.labels_

# %% tags=[]
##### Statistics
clust_stat_df = pd.DataFrame([{
         'size': labels[labels == val].size,
         'clAss_mean_fetures_delta': np.mean(np.max(X_classification[labels == val,:],axis=0)-np.min(X_classification[labels == val,:],axis=0)),
         'class_mean_features_std':np.max(X_classification[labels == val,:].std(axis=0)),
            }
            for val in np.unique(labels)],
          index=[val for val in np.unique(labels)]).sort_values('size',ascending=False)
print(clust_stat_df.shape[0])
display(clust_stat_df)
display(clust_stat_df[['size']].describe())

# %% code_folding=[] tags=[]
#####
# colors: relabelling the classes using the first centroids values

# calculate all the class centers in data space
y = X.groupby(labels).mean().values
# position of the data feature used to sort lables
feature_index = find_nearest(700)
# here the sorting index
centroids_sorting_index = np.argsort(y[:, feature_index])
# here the sorting labels, not the index!!
centroids_sorted_labels = np.argsort(centroids_sorting_index) 
# # use pd.Series.map(dict) di directly change values in place 
labels = pd.Series(labels).map(dict(zip(np.arange(n_clusters),centroids_sorted_labels))).values
print('index for label sort :',feature_index)
# print(' features y[:,index] :',y[:, feature_index])
# print(centroids_sorting_index)
# print(centroids_sorted_labels)
print(f'ind:y_feat  > new_index')
for i,yf,ni in zip(range(len(y[:, feature_index])),y[:, feature_index],centroids_sorted_labels):
    print(f'{i:3}:{yf:.5f} > {ni:>4}')


# %% tags=[]
sns.set(style="ticks")

scatter_df = pd.DataFrame(scaler.transform(X_classification), columns = [f'feature_{x}' for x in range(n_classification_features)])
scatter_df['class'] = labels
scatter_df['class'] = scatter_df['class'].apply(lambda x: 'classs_{}'.format(x))

if X_classification.shape[1] > 2:
    g = sns.PairGrid(scatter_df,hue="class",height=4);
    # g = g.map_offdiag(sns.kdeplot,lw=1)
    g = g.map_offdiag(plt.scatter, s=0.1 , alpha=0.5);
    g;
else: 
    plt.figure(figsize=[8,8])
    # inverselly scale scatteplot size to clas size 
    classes , s = np.unique(scatter_df['class'],return_counts=True)
    min_size, max_size = 20, 100
    sizes = (((s-s.min())/(s.max()-s.min()))*(max_size-min_size))+min_size
    sns.scatterplot(x="feature_0", y="feature_1",hue="class", size='class', sizes=dict(zip(classes,sizes)), data=scatter_df, alpha=0.1);

save_plot( f'Classification-scatter-features-n_clusters_{n_clusters}_classifier-{classifier}', out_format='png',save=save_plots_bool)

#   hv.extension('bokeh')
# hv.extension('matplotlib')

# df_ds = hv.Dataset(scatter_df,kdims=[f'feature_{x}' for x in range(n_classification_features)], vdims='class')

# sampling = (df.max()-df.min()).describe().mean()/150

# def local_datashade ( X,
#                      aggregator=ds.mean(),
#                      cmap=plt.cm.Spectral_r,
#                      x_sampling=sampling,
#                      y_sampling=sampling,
#                      **kwargs):
#     return hds.datashade(X,aggregator=aggregator,cmap=cmap,x_sampling=x_sampling,y_sampling=y_sampling, **kwargs)

# point_grid = hv.operation.gridmatrix(
#                                     df_ds, diagonal_operation=hv.operation.histogram.instance(num_bins=40)
#                                     ).map(hds.datashade, hv.Scatter)
# # # .opts(hv.opts.RGB(interpolation='bilinear',aspect=1,fig_size=200))

# point_grid.opts(fig_size=300).opts(hv.opts.RGB(interpolation='bilinear',aspect=1))

# %% tags=[]
hv.extension('bokeh')

print(np.unique(labels))
outdf_gdf.loc[spectral_df_nona_index,'R'] = labels
outdf_gdf.loc[outdf_gdf['R'] != 11,'R'] = np.nan

outdf_gdf[['x','y','R']].hvplot.scatter(x='x',y='y',c='R',
                    rasterize=True,aggregator='mean',dynamic=True,
                    x_sampling=2,y_sampling=1,cmap='rainbow',cnorm='eq_hist'
                    ).opts(height=600,width=1200,alpha=1)#*background.opts(cmap='Gray',alpha=.65)

# %% [markdown] heading_collapsed=true tags=[] jp-MarkdownHeadingCollapsed=true
# #### DBSCAN hypeparamter search

# %% hidden=true
## DBSCAN hypeparamter search

eps = np.arange(0.9,1.1,.025)
min_samples = [5,10,20,30]
# min_samples = [10] 

dbscan_stats = []

for e in eps:
    for ms in min_samples:
#         dbscan = cluster.DBSCAN(eps=e).fit(X_classification)
        dbscan = cluster.DBSCAN(eps=e,min_samples=ms).fit(X_classification)
        unique, counts = np.unique(dbscan.labels_, return_counts=True)
        stats = {'eps':e,
                 'min_samples': ms,
                 'n_clusters':len(unique),
                 'pop_mean':np.mean(counts),
                 'pop_std':np.std(counts),
                 'pop_min':np.min(counts),
                 'pop_max':np.max(counts)
                }
        
        stats_df = pd.DataFrame([{
                 'delta_feat_mean': np.mean(np.max(X_classification[dbscan.labels_ == val,:],axis=0)-np.min(X_classification[dbscan.labels_ == val,:],axis=0)),
                 'delta_feat_std' :  np.std(np.max(X_classification[dbscan.labels_ == val,:],axis=0)-np.min(X_classification[dbscan.labels_ == val,:],axis=0)),
                    }
                    for val in np.unique(dbscan.labels_)],
                  index=[val for val in np.unique(dbscan.labels_)])

        stats.update(dict(zip([x+'_min' for x in stats_df.std().to_dict().keys()],stats_df.min().to_dict().values())))
        stats.update(dict(zip([x+'_max' for x in stats_df.std().to_dict().keys()],stats_df.max().to_dict().values())))
        stats.update(dict(zip([x+'_mean' for x in stats_df.std().to_dict().keys()],stats_df.mean().to_dict().values())))
        stats.update( dict(zip( [x+'_std' for x in stats_df.std().to_dict().keys()], stats_df.std().to_dict().values())) )

        dbscan_stats.append(stats)
        print(e,ms)

# stats_df = pd.DataFrame(dbscan_stats).set_index('eps')
# display(stats_df.T)
# stats_df.plot(figsize=[20,20],subplots=True);

values='n_clusters'
print(values)
display(pd.DataFrame(dbscan_stats).pivot(index='eps', columns='min_samples', values=values).T)

values='pop_mean'
print(values)
display(pd.DataFrame(dbscan_stats).pivot(index='eps', columns='min_samples', values=values).T)

values='pop_std'
print(values)
display(pd.DataFrame(dbscan_stats).pivot(index='eps', columns='min_samples', values=values).T)

values='pop_max'
print(values)
display(pd.DataFrame(dbscan_stats).pivot(index='eps', columns='min_samples', values=values).T)

values='pop_min'
print(values)
display(pd.DataFrame(dbscan_stats).pivot(index='eps', columns='min_samples', values=values).T)

# %% [markdown] heading_collapsed=true tags=[] jp-MarkdownHeadingCollapsed=true
# ####  AgglomerativeClustering plot

# %% hidden=true
from scipy.cluster.hierarchy import dendrogram, linkage
linkage_array = linkage(aggclustering.children_)

# %% hidden=true tags=[]
plt.figure(figsize=[10,8])
dendrogram(linkage_array,
    p=40,  # show only the last p merged clusters
    truncate_mode='lastp',  # show only the last p merged clusters
    orientation='top',
    no_labels=False,
    show_contracted=True,  # to get a distribution impression in truncated branches
    leaf_rotation=90.,
    leaf_font_size=16.,
);
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
# plt.ylim([3000,6200])
# plt.xlim([-0.5,10.5])
# plt.hlines(4500,0,100,color='red')
plt.show()
save_plot(f'{classifier}_dendrogram.png', out_format='png',save=save_plots_bool)


# %% hidden=true
from scipy.cluster.hierarchy import inconsistent
depth = 3
incons = inconsistent(linkage_array, depth)
incons[-10:]


# %% hidden=true tags=[]
# see [SciPy Hierarchical Clustering and Dendrogram Tutorial | Jörn's Blog](https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/)
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

plt.figure(figsize=[10,8])
fancy_dendrogram(
    linkage_array,
    truncate_mode='lastp',
    p=20,
    leaf_rotation=90.,
    leaf_font_size=30.,
    show_contracted=True,
    annotate_above=10,  # useful in small plots so annotations don't overlap
)
plt.ylim([2000,6500])
plt.show()

# %% hidden=true
last = linkage_array[-10:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.show()
k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
print("clusters:", k)


# %% [markdown] tags=[]
# ####  Classification Vis

# %% tags=[]
[c for c in hv.Cycle.default_cycles.keys()]

# %% tags=[]
hv.extension('bokeh')

vdims = ['component']
kdims=['wavelenght']
width=1000
height=500

colors = cm(np.linspace(0,1,len(np.unique(labels))))
matplotlib.colors.LinearSegmentedColormap.from_list('Spectral',colors , N=len(np.unique(labels)))
cm = plt.cm.Spectral_r
cm_cycle = hv.Cycle([cm(c) for c in np.linspace(0,1,len(np.unique(labels)))])
hv.Cycle.default_cycles['default_colors'] = cm_cycle

overlay_dict = {ind:hv.Curve((spectral_df.columns.to_numpy(),cmp)) for ind,cmp in X.groupby(labels).mean().iterrows()}

overlay_dict['Mean'] = hv.Curve((
                    spectral_df.columns.to_numpy(),
                    spectral_df.mean()
                    ),vdims=vdims,kdims=kdims).opts(line_width=0.5,line_dash='dashed',color='black',alpha=1).relabel('mean')

hv.NdOverlay(overlay_dict).opts(
                hv.opts.Curve(line_width=4,color=cm_cycle),
                hv.opts.Curve('mean',color='black'),
                hv.opts.NdOverlay(width=width,height=height)
                )

# %% tags=[]
hv.extension('matplotlib')

hv.Cycle.default_cycles['default_colors'] = hv.Cycle([cm(c) for c in np.linspace(0,255,len(np.unique(labels)))])

vdims = ['component']
kdims=['wavelenght']

overlay_dict = {ind:hv.Curve((spectral_df.columns.to_numpy(),cmp)) for ind,cmp in X.groupby(labels).mean().iterrows()}

overlay_dict['Mean'] = hv.Curve((
                    spectral_df.columns.to_numpy(),
                    spectral_df.mean()
                    ),vdims=vdims,kdims=kdims).relabel('mean')

overlay_dict['Median'] = hv.Curve((
                    spectral_df.columns.to_numpy(),
                    spectral_df.median()
                    ),vdims=vdims,kdims=kdims).relabel('median')

out = hv.NdOverlay(overlay_dict).opts(
                hv.opts.Curve(linewidth=4, show_grid=False),
                hv.opts.Curve('mean',linewidth=6,linestyle=':',color='black',alpha =0.5),
                hv.opts.Curve('median',linewidth=3,color='black',alpha =0.5),
                hv.opts.NdOverlay(fig_size=600, aspect=2)
                )

out

# # hv.save(out,out_figure_path / f'Spectral-centroids_n_clusters-{n_clusters}_classifier-{classifier}.png', dpi = 200)

# %% tags=[]
## normalised to global mean

hv.extension('matplotlib')

hv.Cycle.default_cycles['default_colors'] = hv.Cycle([cm(c) for c in np.linspace(0,255,len(np.unique(labels)))])

vdims = ['component']
kdims=['wavelenght']

overlay_dict = {ind:hv.Curve((spectral_df.columns.to_numpy(),cmp)) for ind,cmp in (X.groupby(labels).mean()/spectral_df.median()).iterrows()}

out = hv.NdOverlay(overlay_dict).opts(
                hv.opts.Curve(linewidth=4, show_grid=False),
                hv.opts.NdOverlay(fig_size=600, aspect=2)
                )

out

# # hv.save(out,out_figure_path / f'Spectral-centroids_n_clusters-{n_clusters}_classifier-{classifier}.png', dpi = 200)

# %% tags=[]
hv.extension('bokeh')
# opts(height=600,width=1000,alpha=0.7)


out = background.opts(cmap='Gray')*df_shader(labels,cmap=cm).opts(hv.opts.RGB(aspect=2,fig_size=400,alpha=0.9,interpolation='bicubic'))
out
# hv.save(out,out_figure_path / f'Classification-map_n_clusters-{n_clusters}_classifier-{classifier}.png', dpi = 200)


# %% tags=[]
outdf_gdf['labels'] = np.nan
outdf_gdf.loc[spectral_df_nona_index,'labels'] = labels

# %% tags=[]
background.opts(cmap='Gray')*outdf_gdf[['x','y','labels']].hvplot.scatter(x='x',y='y',c='labels',
                    rasterize=True,aggregator='mean',dynamic=False,
                    x_sampling=1,y_sampling=1,cmap='rainbow',cnorm='eq_hist'
                    ).opts(height=600,width=1200,alpha=.6)

# %% tags=[]
hv.extension('bokeh')

out = background.opts(cmap='Gray')*df_shader(labels,cmap=cm).opts(height=600,width=1000,alpha=0.7)
out
# hv.save(out,out_figure_path / f'Classification-map_n_clusters-{n_clusters}_classifier-{classifier}.png', dpi = 200)


# %%
from sklearn import metrics
metrics.silhouette_score(X_classification, labels, metric='euclidean')

# %% [markdown]
# ## VAE Neural Network - Keras

# %% tags=[]
# small_data == spectral_df[spectral_df_nonan]
X = spectral_df.loc[spectral_df_nona_index]

print(f'        outdf_gdf : {outdf_gdf.shape}')
print(f'      spectral_df : {spectral_df.shape}')
print(f'spectral_df_nonan : {spectral_df_nona_index.shape}') 


# %% tags=[]
import importlib
# mdalibpy.ml.CovarianceThreshold

mdalibpy.ml = importlib.reload(mdalibpy.ml)

# %% tags=[]
# X is [observation, features]
X_corr = X.values
print(f'{X_corr.shape=}')
threshold=0.99
cov_thres = mdalibpy.ml.CovarianceThreshold(threshold=threshold).fit(X_corr)
print('threshold : {}\n'
      'Original feat. / Low Corr. feat. : {}/{}\n'
      'Ratio (Original / Low Corr) feat. : {:4.2f}'
      .format(cov_thres.threshold,
      X_corr.shape[1],
      cov_thres.n_features_,
      cov_thres.n_features_/X_corr.shape[1]))
print('Correltation matrix statistics: ')
print(cov_thres.get_statistics())


# %% tags=[]
il1 = np.tril_indices(X_corr.shape[1])
frequencies, edges = np.histogram(cov_thres.get_corr_matrix()[il1], 128)
hv.extension('bokeh')
(
 hv.Image(cov_thres.get_corr_matrix(),label='Covariance Matrix')+\
 hv.Image(cov_thres.get_corr_matrix(masked=True),label='Covariance Matrix (x>{} masked)'.format(threshold))+\
 hv.Histogram((edges, frequencies),kdims=['values'])*hv.VLine(threshold)
).opts(
   hv.opts.Image(width=600,height=600,tools=['hover'],cmap='viridis'),
   hv.opts.Points (color='black', marker='x', size=20),
   hv.opts.RGB(width=600,height=600,tools=['hover']),
   hv.opts.Histogram(width=1200,height=200),
   hv.opts.VLine(color='red', line_width=4)
).cols(1)

# %% tags=[]
# input for learning 
X = spectral_df.loc[spectral_df_nona_index]

# X = cov_thres.transform(X)
# X = X_pca # PCA

original_dim = X.shape[1] # original 784
intermediate_dim = original_dim // 3 # original 256
latent_dim = 2

print(f'{X.shape=}')


# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# %%
# # # train the VAE on spectral data
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=1/10, random_state=42)

# x_train = x_train.reshape(-1, original_dim) / 255.
# x_test = x_test.reshape(-1, original_dim) / 255.
# x_train.shape, y_train.shape, x_test.shape, y_test.shape, type(x_train), type(y_train), type(x_test), type(y_test)

# %% tags=[]
#Build the encoder

encoder_inputs = keras.Input(shape=(original_dim,))
x = layers.Dense(intermediate_dim, activation="relu")(encoder_inputs)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# %% tags=[]
#Build the decoder
from keras.models import Sequential

latent_inputs = keras.Input(shape=(latent_dim,))
decoder_outputs = keras.Sequential([
    layers.Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
    layers.Dense(original_dim, activation='sigmoid')
])(latent_inputs)

decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


# %% tags=[]
# Define the VAE as a Model with a custom train_step

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),axis=-1
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }



# %% tags=[]
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())


# %% tags=[]
vae.fit(X,
        epochs=15,
        batch_size=1280)
# display a 2D plot of the digit classes in the latent space
z_mean, _, _ = vae.encoder.predict(X)
print(f'{z_mean.std(axis=0)=}')

# %% tags=[]
plt.figure(figsize=(12, 10))
plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
# plt.scatter(z_mean[:, 0], z_mean[:, 1])
plt.colorbar()
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.show()

# %% tags=[]
z_mean_df = pd.DataFrame(data=z_mean)

hds.spread(
z_mean_df.hvplot.scatter(x='0',y='1',
                    rasterize=True,aggregator='count',dynamic=True,
                    x_sampling=1,y_sampling=1,cmap='rainbow'
                    )
          ,px=3)

