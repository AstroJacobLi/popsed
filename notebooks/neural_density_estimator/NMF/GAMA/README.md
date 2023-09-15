## GAMA catalogs

I use GAMA DR3: http://www.gama-survey.org/dr4/

I use the GKV catalogs, which use the GKV ProFound photometry. The photometry catalog is from http://www.gama-survey.org/dr4/schema/table.php?id=684. The stellar masses are from http://www.gama-survey.org/dr4/schema/table.php?id=690. Notice that `logmintsfh` is the closest stellar mass to the stellar mass in PROVABGS.

All fluxes and flux errors have been corrected for foreground Galactic extinction. Note that the table only provides fluxes in units of Jy. The corresponding AB magnitudes are hence given by m = 8.9 - 2.5*log10(flux).