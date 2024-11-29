**Unfortunately, this LiCSBAS repository cannot be updated since 2022 for a reason. Data downloading in step01 no longer works because of a critical system change in the COMET-LiCS web portal in June 2024. Please use [LiCSBAS2](https://github.com/yumorishita/LiCSBAS2) or [COMET-version LiCSBAS](https://github.com/comet-licsar/LiCSBAS) which are maintained instead.**

# LiCSBAS

LiCSBAS is an open-source package in Python and bash to carry out InSAR time series analysis using LiCSAR products (i.e., unwrapped interferograms and coherence) which are freely available on the [COMET-LiCS web portal](https://comet.nerc.ac.uk/COMET-LiCS-portal/).

Users can easily derive the time series and velocity of the displacement if sufficient LiCSAR products are available in the area of interest. LiCSBAS also contains visualization tools to interactively display the time series of displacement to help investigation and interpretation of the results.

[<img src="https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/images/comet-lics-web.png"  height="220">](https://comet.nerc.ac.uk/COMET-LiCS-portal/) <img src="https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/images/sample_vel.png"  height="220">  <img src="https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/images/sample_ts.png"  height="220">

<img src="https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/images/LiCSBAS_plot_ts.py_demo_small.gif" alt="Demonstration Video"/>

THIS IS RESEARCH CODE PROVIDED TO YOU "AS IS" WITH NO WARRANTIES OF CORRECTNESS. USE AT YOUR OWN RISK.

## Documentation and Bug Reports

See the [**wiki**](https://github.com/yumorishita/LiCSBAS/wiki) pages and [quick start](https://github.com/yumorishita/LiCSBAS/wiki/2_0_workflow#quick-start).

There are some known issues listed [here](https://github.com/yumorishita/LiCSBAS/issues/244). If you encounter an error, please see the [known issues](https://github.com/yumorishita/LiCSBAS/issues/244), search for [other issues](https://github.com/yumorishita/LiCSBAS/issues), or report the error on the [issues page](https://github.com/yumorishita/LiCSBAS/issues).

## Sample Products and Tutorial

- Frame ID: 124D_04854_171313 (Italy)
- Time: 2016/09/09-2018/05/08 (~1.7 years, 67 epochs, ~217 interferograms)
- Clipped around Campi Flegrei (14.03/14.22/40.78/40.90)

- Tutorial: [LiCSBAS_sample_CF.pdf](https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/documents/LiCSBAS_sample_CF.pdf) (1.3MB)

- Sample batch script: [batch_LiCSBAS_sample_CF.sh](https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/documents/batch_LiCSBAS_sample_CF.sh) (2021/3/11 updated)
- Sample results: [LiCSBAS_sample_CF.tar.gz](https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/sample/LiCSBAS_sample_CF.tar.gz) (63MB) (2021/3/11 updated)

## Citations

Morishita, Y.; Lazecky, M.; Wright, T.J.; Weiss, J.R.; Elliott, J.R.; Hooper, A. LiCSBAS: An Open-Source InSAR Time Series Analysis Package Integrated with the LiCSAR Automated Sentinel-1 InSAR Processor. *Remote Sens.* **2020**, *12*, 424, https://doi.org/10.3390/RS12030424.

Morishita, Y.: Nationwide urban ground deformation monitoring in Japan using Sentinel-1 LiCSAR products and LiCSBAS. *Prog. Earth Planet. Sci.* **2021**, *8*, 6,  https://doi.org/10.1186/s40645-020-00402-7.

Lazecký, M.; Spaans, K.; González, P.J.; Maghsoudi, Y.; Morishita, Y.; Albino, F.; Elliott, J.; Greenall, N.; Hatton, E.; Hooper, A.; Juncu, D.; McDougall, A.; Walters, R.J.; Watson, C.S.; Weiss, J.R.; Wright, T.J. LiCSAR: An Automatic InSAR Tool for Measuring and Monitoring Tectonic and Volcanic Activity. *Remote Sens.* **2020**, *12*, 2430, https://doi.org/10.3390/rs12152430.

Morishita, Y., Sugimoto, R., Nakamura, R. et al. Nationwide urban ground deformation in Japan for 15 years detected by ALOS and Sentinel-1. Prog Earth Planet Sci 10, 66 (2023). https://doi.org/10.1186/s40645-023-00597-5

## Acknowledgements

This work has been accomplished during Y. Morishita’s visit at University of Leeds, funded by JSPS Overseas Research Fellowship.

COMET is the UK Natural Environment Research Council's Centre for the Observation and Modelling of Earthquakes, Volcanoes and Tectonics. LiCSAR is developed as part of the NERC large grant, "Looking inside the continents from Space" (NE/K010867/1). LiCSAR contains modified Copernicus Sentinel data [2014-] analysed by the COMET. LiCSAR uses [JASMIN](http://jasmin.ac.uk), the UK’s collaborative data analysis environment.

The [Scientific Colour Maps](http://www.fabiocrameri.ch/colourmaps.php) ([Crameri, 2018](https://doi.org/10.5194/gmd-11-2541-2018)) is used in LiCSBAS.

*Yu Morishita (PhD)\
JSPS Overseas Research Fellow (June 2018-March 2020)\
Visiting Researcher, COMET, School of Earth and Environment, University of Leeds (June 2018-March 2020)\
Chief Researcher, Geography and Crustal Dynamics Research Center, Geospatial Information Authority of Japan (GSI)*

[<img src="https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/images/COMET_logo.png"  height="60">](https://comet.nerc.ac.uk/)   [<img src="https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/images/logo-leeds.png"  height="60">](https://environment.leeds.ac.uk/see/)  [<img src="https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/images/LiCS_logo.jpg"  height="60">](https://comet.nerc.ac.uk/COMET-LiCS-portal/) 
