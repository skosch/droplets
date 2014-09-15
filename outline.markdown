# Final thesis #################
## Introduction

## Experimental setup

## IPI

## PDPA



# IPI Paper ####################
## Introduction
* Brief history/Literature survey
* Explanation of the mechanism
* Influence of the scattering angle
  * How Scheimpflug condition can be used, but in practice it isn't
* Influence of particle size
  * Mie error
* Hot pixels on the CCD
  * How to fix them
* Optical and practical limits on fringe detection
  * How to fix them
* Finding the constants (camera dimensions etc.)
  * Why calibration is necessary
* Too much overlap
  * How to fix it: droplet detection OR slit aperture
* Droplet detection and camera mapping
  * How to fix it: use BRIEF/ORB
* Slit aperture
  * How to make/use it
  * How different authors detect it
* How to calibrate the slit aperture thing

# PDPA Paper ###################
* Brief history/Literature survey
* Explanation of the mechanism



* Trajectory Ambiguity Effect
  * See red book, and Naqwi, Optimization of the Shape of Receiving Aperture in
    a Phase Doppler System
* Different aperture shapes and sizes
  * See Naqwi, Rigorous Procedure for Design and Response Determination of Phase
    Doppler Systems
* Explanation of how misaligned beams can lead to errors
  * Fringe spacing changes with z
* Higher voltage gain leads to a greater detection volume
  * More particles falling through the wrong end are picked up
  * Too strong a laser power may also do this (Davis)
* Large particles: the farther away you get from the center, 
  * the greater the error. For particles > 1/3 diameter of measurement volume,
    the error can be great (Dantec Dynamics, LDA: Introduction to Principles and
    Appilcations). According to Davis, this isn't a problem IF sizing
    experiments are done RELATIVELY, not absolute.
* Error increases with smaller beam waists


# DropGen Paper ################
* Different types of droplet generators
  * Literature survey
  * Vibrating orifice droplet generators
* Easiest way to make a needle:
  * Rotate capillary in flame
  * Epoxy capillary into clipped-off hypodermic G16
* Easiest way to make a vibrator
  * As mentioned above, piezoelectric vibrators are often used
    * They are expensive, require manual assembly and high voltages
  * Speakers can also be used
    * They are tricky to assemble with the needle
  * Hard drive actuators are perfect
    * Cheap and require no assembly at all

