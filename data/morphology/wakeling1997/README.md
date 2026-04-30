# Morphological Data from (Wakeling, 1997)[^fn1]

## Data Description

 Data tables digitized from[^fn1]. Three data files are provided corresponding to the three tables in the source:

 - `body_data.csv`
 - `forewing_data.csv`
 - `hindwing_data.csv`

 *The image quality of the source table for the forewing data is quite poor, so contents may not be fully accurate*

 Each specimen is identified by a unique ID (column `ID` in the data tables). Each ID consists of a a species identifier followed by a numerical identifier. The species and sex of each specimen is provided in `body_data.csv` and `forewing_data.csv`. The contents of each column is indicated in the header in abbreviated notation. The notation is the same as in the source material, and is given here:

 - `L`: total length [mm]
 - `m`: total mass [mg]
 - `mb`: body mass (wings removed) [mg]
 - `mm`: thoracic muscle mass (approximate, also includes other tissue) [mg]
 - `mhm`: nondimensional muscle mass, normalized by total mass
 - `lh`: nondimensional distance from tip of head to center of mass, normalized by total length
 - `lh1`: nondimensional distance from forewing base to center of mass, normalized by total length
 - `lh2`: nondimensional radius of gyration, (Ib/(mb\*L^2))^1/2
 - `Ib`: moment of inertia of body [mg.m^2]
 - `mw`: wing mass [mg]
 - `mhw`: nondimensional wing mass, normalized by total mass
 - `R`: wing length [mm]
 - `S`: wing surface area [mm^2]
 - `AR`: wing aspect ratio (AR = 4\*R^2/S)
 - `rhkS`: nondimensional radius of kth moment of area (see reference)
 - `vh`: nondimensional virtual mass (vh = v\*(AR)^2/(2\*rho\*pi\*R^3))
 - `rhkv`: nondimensional radius of kth moment of virtual mass (see reference)
 - `hh`: nondimensional mean wing thickness (hh = mw/(rhow\*S\*R))
 - `rhkv`: nondimensional radius of kth moment of wing mass (see reference)

## References

[^fn1]: J. M. Wakeling, “Odonatan wing and body morphologies,” Odonatologica, vol. 26, no. 1, pp. 35–52, Mar. 1997. [Online](https://natuurtijdschriften.nl/pub/592186)
