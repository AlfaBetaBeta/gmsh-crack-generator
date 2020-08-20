# gmsh plugin for crack generation in 3D FE meshes

This is a procedural version of a program to create cracks inside a solid 3D mesh generated by gmsh by duplicating tagged surface elements. It is based on python pandas and uses DataFrames and dictionaries as the main data structures.

The `.msh` file containing the (uncracked) solid mesh is the main input argument for the python script `crack.py`, which will return an updated `.msh` file embedding the cracks upon execution. The main caveats and limitations of the program, as well as execution syntax guidelines, are showcased below by means of a series of test examples.


## Introduction

Consider an overly simplistic mesh made of two quadratic hexahedrons, as shown below. For element-based FE engines, it might be necessary to assign two different physical tags to the solid elements: one tag (`"bulk"`) would identify the group, for convenient association with constitutive models to represent the material, and the other tag (`"solid_type_1"`) would convey information about the element's self weight. Typically these two strings would be programmatically processed when integrating the data into the FE engine, though they are given descriptive dummy values here for illustration. Although some FE engines may not require this type of element characterisation, the script in its current form expects double physical tagging for all solid elements.

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/intro/pre-crack-pe-nodes-elmts.png" width=100% height=100%>

The surfaces to be duplicated need to be tagged as well. Although in this simplistic example there is only one feasible surface, in an extensive solid mesh many such surfaces may coalesce into a (possibly curved) crack plane. Each crack plane can have its own physical tag (here `"crack_horizontal"`) or they can share a common tag. The reason behind individual tagging is that, after duplication, each surface and its duplicate can be further transformed into a zero-thickness interface element and it may be convenient to differentiate between sets of crack planes (i.e. interface types) to assign them different material properties. The explicit transformation into interface elements is not addressed here, as it is FE engine dependent, but indication is made as to how it could be done wherever appropriate. Regardless of the number of physical tags for crack planes, **all** surfaces to be duplicated need to be grouped by a common tag (here `"s2in"`). Ultimately, this is necessary precisely to facilitate the definition of interface elements, as will become apparent with the examples.

Shifting focus back on the simplistic mesh, once `crack.py` is executed (guidelines on execution syntax can be found in subsequent test examples), the tagged surface is duplicated, effectively decoupling the two adjacent solid elements. This is shown below, whereby a virtual crack opening is induced for visual clarity (in reality the nodes at each crack side initially overlap in the same location). 

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/intro/post-crack-nodes-elmts.png" width=100% height=100%>

The total number of nodes in the mesh has increased by 8, as expected (nodes 33-40 form the new surface) and the number of elements has increased by 1, as highlighted below. Note that the new surface element has the general common physical tag (`4`, i.e. `"s2in"`), it is bounded by the new nodes but **shares the same elementary tag as the original surface**. 

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/intro/post-crack-elements-aux.png" width=60% height=60%>

This is not conflicting in gmsh (the resulting cracked `.msh` file can be opened for inspection without error) and is in fact a convenient way to trace the surface pairs that would form an interface should this be necessary. Indeed, elements `1`, `2` and `7` above have all the necessary information to define a quadratic 16-noded interface element and it readily allows for a programmatic approach in the case of extensive solid meshes.

In principle, the process of duplicating the tagged surface follows a simple geometric criterion. The ordering of the corner nodes determines a direction (right-hand rule) which is interpreted to point from *bottom* to *top*, whereby these terms are to be understood notionally as the opposite subdomains that the crack divides adjacent solid elements into. They do not have to be intuitive, as happens to be the case in the figure below, they purely stem from node ordering. 

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/intro/surface-normal-aux.png" width=70% height=70%>

The *bottom* solid retains the original surface nodes, whereas the *top* solid is detached from it and is assigned new duplicate nodes with the same coordinates as the originals (though in the figure the crack is open for clarity). In extensive meshes, it is important to ensure that all surface elements with a common tag (i.e. all the surfaces belonging to the same crack plane) share the same *top* and *bottom* criterion for their adjacent solid elements, and provisions are indeed in place to accommodate this.


## Test example 1

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test1/test1-pre-crack.png" width=100% height=100%>

Consider a mesh of 4 quadratic hexahedrons with a single vertical crack plane not fully cutting through the solid bulk, as shown above. The physical tags are also included in the image and follow the criteria mentioned in the previous section. With this in mind, the syntax for executing the script from the command line requires two arguments and reads as follows:
```
$ python crack.py name_of_the_mesh_file.msh [list with all (crack) surface physical tags]
```
In general, the list with the surface tags can be arbitrarily long (recall that different crack planes may be assigned different tags should this be convenient) but in any case **the last list item must be the common tag encapsulating all surfaces**. In this example, such list is simply `[3,4]`. If the arguments were passed from within an interpreter (e.g. Spyder: `Run/Configuration per file.../Command line options`) the list could be passed as is, but from the terminal it needs to prepended by `\` to avoid confusion with pattern matching. Hence, execution of the script on `test1.msh` from the terminal reads:
```
$ python crack.py test1.msh \[3,4]
``` 
This will result in a new `.msh` file stored in the working directory, retaining the original file name but prepended with `cracked_`. Thus in this case the output is `cracked_test1.msh`.

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test1/test1-post-crack.png" width=100% height=100%>

As expected, the cracked mesh comprises 8 additional nodes (52-59), which requires updating the nodal definition of two hexahedrons, as highlighted above. As ever, the crack has been artificially widened for clarity, as will be the case in all remaining test examples.


## Test example 2

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test2/test2-pre-crack.png" width=100% height=100%>

Slightly increasing the level of complexity, let's consider now the case of a mesh with various crack tags (`3`, `4` and `5` above). Note that the same tag may apply to separate crack planes (`3`) or that different tags may be coplanar (`3` and `4`), for example if they were meant to ultimately be transformed into interface elements of different material properties. Just bear in mind that, when passing the list argument, **coplanarity needs to be explicitly denoted as a sub-list**:
```
$ python crack.py test2.msh \[\[3,4],5,6]
```
<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test2/test2-post-crack.png" width=100% height=100%>

Despite involving two distinct tags, the larger (left) crack plane correctly embeds 13 new nodes.


## Test example 3

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test3/test3-pre-crack.png" width=100% height=100%>

At the time of developing `crack.py`, gmsh would not add partition physical tags to the surface elements' taglist when partitioning a three-dimensional solid mesh. This is transparently addressed by the script, which in its current form expects the lack of partition tagging on surfaces to **always be the case** and ultimately adds such tags during processing. Execution does consequently not differ from the case where the mesh would be monolithic:
```
$ python crack.py test3.msh \[3,4]
```
As can be seen below (where some nodes have been removed for clarity), the surfaces in the cracked mesh have been assigned a partition tag. If the surfaces and their duplicates are meant to be transformed into interface elements, special provisions might be necessary for the case that the crack surface is coplanar with a partition boundary, though this is FE engine dependent and is not further adressed here.

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test3/test3-post-crack.png" width=100% height=100%>

**\*** *The gmsh plugin* `SimplePartition` *does seem to assign partition tags to surfaces, as opposed to resorting to* `Modules/Mesh/Partition` *in the GUI. If the plugin were used to create the partitions, execution of* `crack.py` *would not fail but it would produce an inconsistent* `.msh` *file. This case is yet to be addressed and an upgrade will be released in due course.*


## Test example 4

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test4/test4-pre-crack.png" width=100% height=100%>


<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test4/test4-post-crack.png" width=100% height=100%>


## Caveats and shortcomings

Partitioning of the solid mesh can be dealt with, as the `make_crack()` function includes code to partition the crack surface elements accordingly.
      
A preprocessing function `preproc_df()` is included to ensure that the surface elements contained in the DataFrame `df_elm` have all a unique elementary tag.
      
Only solid elements bk20 (20-noded-hexahedron) and wd15 (15-noded-wedge) are considered. Should the mesh contain other types of elements (e.g. tt10/10-noded-tetrahedron), some functions would need updating.
      
It is assumed that any crack surface element can only have 2 physical tags (`*joint` group and `s2in` group). In theory a third tag could be present if any such surface were subject to restraints (e.g. r_x+z). This version cannot handle this yet.

If a continuous crackplane is curved, and depending on the ordering of the crack surface elements (which determines the ordering in which they are processed), the orientation of the crack (determining what is its 'top' and 'bottom') may become ambiguous. For now, inspection of the crack ordering in the .msh file via gmsh is necessary until a general solution is developed.

The main script needs 3 arguments:
* name of the `msh` file
* crack physical tags, with some pairs being possibly coupled if they belong to the same crack plane (e.g. `[1,[2,3],4]`)

