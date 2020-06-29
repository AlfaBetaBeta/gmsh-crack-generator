# gmsh-crack-generator

This is the procedural version of the OOP script gmsh_crack.py It is based on Pandas and uses DataFrames as the main data structure. In its current form, the script is functional and can deal with crack planes not completely going through the bulk or intersecting crack planes.

Partitioning of the solid mesh can be dealt with, as the `make_crack()` function includes code to partition the crack surface elements accordingly.
      
A preprocessing function `preproc_df()` is included to ensure that the surface elements contained in the DataFrame `df_elm` have all a unique elementary tag.
      
Only solid elements bk20 (20-noded-hexahedron) and wd15 (15-noded-wedge) are considered. Should the mesh contain other types of elements (e.g. tt10/10-noded-tetrahedron), some functions would need updating.
      
It is assumed that any crack surface element can only have 2 physical tags (`*joint` group and `s2in` group). In theory a third tag could be present if any such surface were subject to restraints (e.g. r_x+z). This version cannot handle this yet.

If a continuous crackplane is curved, and depending on the ordering of the crack surface elements (which determines the ordering in which they are processed), the orientation of the crack (determining what is its 'top' and 'bottom') may become ambiguous. For now, inspection of the crack ordering in the .msh file via gmsh is necessary until a general solution is developed.