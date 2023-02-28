# AE2224-IGroupD07
The repository for group D07 for the project Test, Analysis &amp; Simulation. The repository contains several tools for analysing the localisation of damage in carbon fiber reinforced composites (CFRPs) using acoustic emission (AE).

#### ToA determination
---
This folder contains scripts that determine the time-of arrival (ToA) of the AE by examining the sensor reading and filtering out the background noise.

#### Least squares localisation
---
This folder contains scripts that locate the source of an AE by iteratively refining a least-square estimate of a system of overdefined equations. The equations originate from the ToAs from each sensor.

#### Damage mode
---
This folder contains scripts for qualitatively classifying certain types of damage which have occured by assessing the waveform of the AEs at their source.