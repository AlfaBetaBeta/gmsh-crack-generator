# gmsh plugin for crack generation in 3D FE meshes

This is a procedural version of a python program to create cracks inside a solid three-dimensional mesh generated by [gmsh](https://gmsh.info) by means of duplicating tagged surface elements. It is based on the pandas library and uses DataFrames and dictionaries as the main data structures.

To run the plugin and/or hack it to your convenience, clone this repository and execute `crack.py` in the main directory:
```
$ git clone https://github.com/AlfaBetaBeta/gmsh-crack-generator.git
$ cd gmsh-crack-generator
Example:
$ python crack.py test1.msh \[3,4]
```
Guidelines for the arguments' syntax can be found in [Test example 1](https://github.com/AlfaBetaBeta/gmsh-crack-generator#test-example-1-execution-syntax-and-program-output).

The `.msh` file containing the (uncracked) solid mesh is the principal input argument for the python script, which will return an updated `.msh` file embedding the cracks upon execution. The main features of the program are showcased in the sections below:

* [Introduction](https://github.com/AlfaBetaBeta/gmsh-crack-generator#introduction)
* [Test example 1: execution syntax and program output](https://github.com/AlfaBetaBeta/gmsh-crack-generator#test-example-1-execution-syntax-and-program-output)
* [Test example 2: multi-tag crack planes](https://github.com/AlfaBetaBeta/gmsh-crack-generator#test-example-2-multi-tag-crack-planes)
* [Test example 3: partitioning](https://github.com/AlfaBetaBeta/gmsh-crack-generator#test-example-3-partitioning)
* [Test example 4: intersecting of different crack planes](https://github.com/AlfaBetaBeta/gmsh-crack-generator#test-example-4-intersecting-of-different-crack-planes)
* [Application example: multi-span arch bridge](https://github.com/AlfaBetaBeta/gmsh-crack-generator#application-example-multi-span-arch-bridge)
* [Caveats and shortcomings](https://github.com/AlfaBetaBeta/gmsh-crack-generator#caveats-and-shortcomings)


## Introduction

Consider an overly simplistic mesh made of two quadratic hexahedrons, as shown below. For some element-based FE engines, it might be necessary to assign two different physical tags to the solid elements: one tag (`"bulk"`) would identify the group, for convenient association with constitutive models to represent the material, and the other tag (`"solid_type_1"`) would convey information about the element's self weight. Typically these two strings would be programmatically processed when integrating the data into the FE engine, though they are given descriptive dummy values hereafter for simplicity. Some FE engines may not require this type of element characterisation, however, and it might suffice to assign a single tag to each solid element. Either way, **the script in its current form can accommodate *any number* (>0) of physical tags for all solid elements**.

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/intro/pre-crack-pe-nodes-elmts.png" width=100% height=100%>

The surfaces to be duplicated need to be tagged as well. Although in this simplistic example there is only one feasible surface, in an extensive solid mesh many such surfaces may coalesce into a (possibly curved) crack plane. Each crack plane can have its own physical tag (here `"crack_horizontal"`) or various crack planes can share a common tag (see the [caveats](https://github.com/AlfaBetaBeta/gmsh-crack-generator#caveats-and-shortcomings) for elaborations on how to tag surfaces appropriately). The reason behind individual tagging is that, after cracking, each surface and its duplicate peer can be further transformed into a zero-thickness interface element and it may be convenient to differentiate between sets of crack planes (i.e. interface types) to assign to them different material properties. The explicit transformation into interface elements is not addressed here, as it is FE engine dependent, but indication is made as to how it could be done wherever appropriate. Regardless of the number of physical tags for crack planes, **all** surfaces to be duplicated need to be grouped by a common tag (here `"s2in"`) as well. Ultimately, this is necessary precisely to facilitate the definition of interface elements, as will become apparent with the examples.

Shifting focus back on the simplistic mesh, once `crack.py` is executed (guidelines on execution syntax can be found in [test example 1](https://github.com/AlfaBetaBeta/gmsh-crack-generator#test-example-1)), the tagged surface is duplicated, effectively decoupling the two adjacent solid elements. This is shown below, whereby a virtual crack opening is induced for visual clarity (in reality the nodes at each crack side initially overlap in the same location). 

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/intro/post-crack-nodes-elmts.png" width=100% height=100%>

The total number of nodes in the mesh has increased by 8, as expected (nodes `33`-`40` form the new surface) and the number of elements has increased by 1, as highlighted below. Note that the new surface element has the general common physical tag (`4`, i.e. `"s2in"`), it is bounded by the new nodes but **shares the same elementary tag as the original surface**. 

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/intro/post-crack-elements-aux.png" width=60% height=60%>

This is not conflicting in gmsh (the resulting cracked `.msh` file can be opened for inspection without error) and is in fact a convenient way to trace the surface pairs that would form an interface should this be necessary. Indeed, elements `1`, `2` and `7` above have all the necessary information to define a quadratic 16-noded interface element and it readily allows for a programmatic approach in the case of extensive solid meshes.

In principle, the process of duplicating the tagged surface follows a simple geometric criterion. The ordering of the corner nodes determines a direction (right-hand rule) which is interpreted to point from *bottom* to *top*, whereby these terms are to be understood notionally as the opposite subdomains that the crack divides adjacent solid elements into. They do not have to be intuitive, as happens to be the case in the figure below, they purely stem from node ordering. In this example, the first four nodes forming the surface element are `5`,`6`,`7`,`8`.

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/intro/surface-normal-aux.png" width=70% height=70%>

The *bottom* solid retains the original surface nodes, whereas the *top* solid is detached from it and is assigned new duplicate nodes with the same coordinates as the originals (though in the figure the crack is virtually open for clarity). In extensive meshes, it is important to ensure that all surface elements with a common tag (i.e. all the surfaces belonging to the same crack plane) share the same *top* and *bottom* criterion for their adjacent solid elements, and provisions are indeed in place to accommodate this (see the [caveats](https://github.com/AlfaBetaBeta/gmsh-crack-generator#caveats-and-shortcomings)).


## Test example 1: execution syntax and program output

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test1/test1-pre-crack.png" width=100% height=100%>

Consider a mesh of 4 quadratic hexahedrons with a single vertical crack plane not fully cutting through the solid bulk, as shown above. The physical tags are also included in the image and follow the criteria mentioned in the [introduction](https://github.com/AlfaBetaBeta/gmsh-crack-generator#introduction). With this in mind, the syntax for executing the script from the command line requires up to three arguments and reads as follows:
```
$ python crack.py name_of_uncracked_file.msh [list with all crack physical tags] storing_granularity_level
```
* The first argument is the name of the `msh` file containing the entire uncracked mesh. Check the [caveats](https://github.com/AlfaBetaBeta/gmsh-crack-generator#caveats-and-shortcomings) to see the types of solid elements that are admissible, as well as the file format that is expected.
* In general, the list with the crack tags can be arbitrarily long (recall that different crack planes may be assigned different tags should this be convenient) but in any case **the last list item must always be the common tag encapsulating all surfaces**. In this example, such list is simply `[3,4]`. If the arguments were passed from within an interpreter (e.g. Spyder: `Run/Configuration per file.../Command line options`) the list could be passed as is, but from the terminal it needs to prepended by `\` to avoid confusion with pattern matching.
* The third argument is *optional*, and refers to the the granularity with which the crack surface elements shall be ultimately stored in the output `msh` file alongside the solids. The storing granularity levels are defined as:
    * `0` : all crack surface elements are stored in the output file (*default*)
    * `1` : only crack surfaces with the common tag are stored in the output file
    * `2` : no crack surfaces are stored in the output file

Hence, execution of the script on `test1.msh` from the terminal (with default storing granularity, as assumed in all test examples) reads:
```
$ python crack.py test1.msh \[3,4]
``` 
or, more explicitly:
```
$ python crack.py test1.msh \[3,4] 0
```
This will result in a new `.msh` file stored in the working directory, retaining the original file name but prepended with `cracked_`. Thus in this case the output is `cracked_test1.msh`.

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test1/test1-post-crack.png" width=100% height=100%>

As expected, the cracked mesh comprises 8 additional nodes (`52`-`59`), which requires updating the nodal definition of two hexahedrons, as highlighted above. As ever, the crack has been artificially widened for clarity, as will be the case in all remaining test examples.


## Test example 2: multi-tag crack planes

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test2/test2-pre-crack.png" width=100% height=100%>

Slightly increasing the level of complexity, let's consider now the case of a mesh with various crack tags (`3`, `4` and `5` above). Note that the same tag (`3`) may apply to separate crack planes or that different tags may be coplanar (`3` and `4`), for example if they were meant to ultimately be transformed into interface elements of different material properties. Just bear in mind that, when passing the list argument, **coplanarity needs to be explicitly denoted as a sub-list**:
```
$ python crack.py test2.msh \[\[3,4],5,6]
```
<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test2/test2-post-crack.png" width=100% height=100%>

Despite involving two distinct tags, the larger (left) crack plane correctly embeds 13 new nodes.


## Test example 3: partitioning

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test3/test3-pre-crack.png" width=100% height=100%>

At the time of developing `crack.py`, gmsh would not add partition physical tags to the surface elements' taglist when partitioning a three-dimensional solid mesh. This is transparently addressed by the script, which in its current form expects the lack of partition tagging on surfaces to **always be the case** and ultimately adds such tags during processing. Execution does consequently not differ from the case where the mesh would be monolithic:
```
$ python crack.py test3.msh \[3,4]
```
As can be seen below (where some nodes have been removed for clarity), the surfaces in the cracked mesh have been assigned a partition tag. If the surfaces and their duplicates are meant to be transformed into interface elements, special provisions might be necessary for the case that the crack surface is coplanar with a partition boundary, though this is FE engine dependent and is not further adressed here.

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test3/test3-post-crack.png" width=100% height=100%>

**\*** *The gmsh plugin* `SimplePartition` *does seem to assign partition tags to surfaces, as opposed to resorting to* `Modules/Mesh/Partition` *in the GUI. If `SimplePartition` were used to create the partitions, execution of* `crack.py` *would not fail but it would produce an inconsistent* `.msh` *file. This case is yet to be addressed and an upgrade of the script will be released in due course.*


## Test example 4: intersecting of different crack planes

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test4/test4-pre-crack.png" width=100% height=100%>

Another potential corner case accommodated by `crack.py` is that of intersecting crack planes under different physical tags. Since these planes are not coplanar they are processed at different stages and some nodes at the intersection (e.g. node `16` above) may need to have their duplicate in turn re-duplicated. However, this all happens transparently and thus execution follows the same guidelines showcased in previous examples:
```
$ python crack.py test4.msh \[3,4,5]
```
<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test4/test4-post-crack.png" width=100% height=100%>

As expected, there are 21 additional nodes in the cracked mesh. Inspecting these in more detail below, it can be seen for example that original node `16` was initially duplicated as node `129` and this one was in turn duplicated again as node `137`.

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test4/post-crack-with-nodes-transparent.png" width=70% height=70%> 


## Application example: multi-span arch bridge

Finally, and in order to showcase all previous features in a single mesh, a more realistic example is presented here, comprising a three-span arch bridge (details on how to generate this mesh can be found in [this repository](https://github.com/AlfaBetaBeta/gmsh-3D-arch-bridge#generation-of-a-macroscale-multi-span-bridge-fe-mesh)). The solid elements are either hexahedrons or wedges, and four main distinct materials are considered, as shown below (although *masonry* and *backing* are encoded with the same colour because they share the same physical tag `20` representing their self-weight):

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/bridge/bridge-materials.png" width=100% height=100%>

This mesh would be suitable for modelling at macroscale, i.e. without taking into account anisotropies or distinguishing brick and mortar explicitly. It might be of interest, however, to consider the frictional effects arising in the contact surfaces:
* between the backfill/ballast and the inner side of the spandrel walls.
* between the backing/backfill and the extrados of all arches.

To this end, all surfaces in these contact regions are programmatically retrieved and conveniently tagged, as highlighted below. Note that the corresponding surface physical tags are `"g_*2*"` and the general common tag, consistently with all previous test examples, is `"s2in"`.

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/bridge/bridge-interface-physical-tags.png" width=100% height=100%>

Additionally, and to leverage the benefits of parallel computing, the solid mesh is (arbitrarily) partitioned in 6. As mentioned in [test example 3](https://github.com/AlfaBetaBeta/gmsh-crack-generator#test-example-3), partitioning initially involves only the higher dimensional elements, in this case the hexahedrons and wedges:

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/bridge/solid-partition-labels.png" width=75% height=75%>

With all the above in mind, the cracking program can be run simply by executing:
```
$ python crack.py bridge.msh \[\[12,13],\[14,15,16,17,18],19]
```
Note the two sub-lists embedded in the second argument:
* `12` and `13` are intuitively coplanar, the only reason for their distinct tagging is that in this case cracks will be further transformed into interface elements (through another utility program not included here), and these interfaces will have different characteristics depending on the material in contact with the spandrels (ballast or backfill).
* `14` to `18` are also coplanar (using the term more loosely) in the sense that all these tags represent a single (sinuously curved) crack surface where each node is duplicated just once (check the [caveats](https://github.com/AlfaBetaBeta/gmsh-crack-generator#caveats-and-shortcomings) to see why this is suitable).

Apart from these remarks, this bridge mesh notionally replicates the same features as in the test examples: the sub-lists represent intersecting crack surfaces and these are assigned to partitions during processing, as shown below:

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/bridge/cracking-interfaces.png" width=100% height=100%>

Illustratively, partition 2 after cracking is shown in more detail below, whereby a local crack opening has been introduced on each surface sub-list for clarity:

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/bridge/cracked-partition2.png" width=75% height=75%>


## Caveats and shortcomings

### Solid element types

In its current form, the program only considers solid meshes formed by **20-noded hexahedrons** ([gmsh type](https://gitlab.onelab.info/gmsh/gmsh/blob/master/Common/GmshDefines.h) 17), **15-noded wedges** ([gmsh type](https://gitlab.onelab.info/gmsh/gmsh/blob/master/Common/GmshDefines.h) 18) or **10-noded tetrahedrons** ([gmsh type](https://gitlab.onelab.info/gmsh/gmsh/blob/master/Common/GmshDefines.h) 11). Should the `.msh` file contain other solid element types, these would just need to be added to the `solid_types` list in the function `get_IDs_same_elmt`. Additionally, if the new solid type were not quadratic and/or if it comprised more than 4 edges per face, then suitable additions would be necessary in the following dictionaries:
* `nodes_per_face` in function `get_attached_surfaces`
* `node_reordering` in function `make_crack`, whereby the relevant [node ordering](https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering) can be found in the gmsh documentation.

### `msh` file format

`crack.py` was developed for an old version of gmsh (2.2) and hence it expects the `.msh` file to comply with the [legacy version 2 format](https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format-version-2-_0028Legacy_0029), comprising only the following headers:
* `$MeshFormat`
* `$PhysicalNames`
* `$Nodes`
* `$Elements`

### Surface physical tags
      
It is assumed that **any crack surface element can only have 2 physical tags** (in the bridge example, `"g_*2*"` and `"s2in"`). In theory a third tag could be present if any such surface were subject to restraints (e.g. like `"r_x+y+z"` for the surfaces at the bottom of the bridge piers, which were assumed uncracked). This could be the case if it were intended to apply boundary conditions on one side of an interface and attach a solid element on the other. In its current form, the program cannot handle this yet.

### Crack top/bottom consistency

The way to ensure that all surfaces embedded in a crack plane share the same criterion for *top* and *bottom* adjacent solids is by means of a simple geometric check, comparing the unit normal of the surface in process with the unit normal of an adjacent already processed surface, or with the average of all previously processed surfaces belonging to the same crack plane in the absence of adjacencies. Although this method runs smoothly for crack planes without kinks or crack surfaces of moderate curvature, if such crack plane is significantly curved, and depending on the tagging strategy and the ordering of the crack surface elements (which determines the ordering in which they are processed), the orientation of the crack (determining what is its *top* and *bottom*) may become ambiguous.

To illustrate the above, consider a simple mesh like the one from [Test example 1](https://github.com/AlfaBetaBeta/gmsh-crack-generator#test-example-1-execution-syntax-and-program-output), this time with two orthogonal crack planes (whereby the uncracked state is shown to the left and the cracked one to the right). 

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test6/test6b-postcrack-surfaces.png" width=100% height=100%>

Each plane has a distinct tag (`"crack_vertical"` and `"crack_horizontal"`, processed in this order), which leads to the following node configuration:

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test6/test6b-postcrack-nodes.png" width=100% height=100%>

Note that the original nodes `5`, `37` and `12` (at the tip of the right angle) have been duplicated twice, once per tag. If instead both planes had been assigned the same physical tag (thereby introducing a right angle kink within a single crack entity), the resulting cracked surfaces would appear as below:

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test6/test6a-postcrack-surfaces.png" width=100% height=100%>

with the following node configuration:

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test6/test6a-postcrack-nodes.png" width=100% height=100%>

Note that in this case the original nodes `5`, `37` and `12` have been duplicated only once, with the final mesh having three nodes less than in the previous case. The unit normals of both planes are orthogonal (hence rendering a zero projection on each other) but the script can still accommodate this initial ambiguity because they share a common *bottom* solid element (the same would apply if they shared a common *top* element, and provisions are in place for the case that the same solid is *top* for one surface and *bottom* for the adjacent perpendicular one). 

Although both approaches work for this particular example, the former is strongly encouraged for right angle kinks and it even becomes mandatory for acute angle kinks. Thus **in general it is recommended to assign different tags to different *crack domains*, unless the curvature is moderate or the kink angle is obtuse**.

Further risks of assigning a single tag to orthogonal crack planes can be seen below, whereby in both meshes execution of `crack.py` would be aborted and a relevant error message displayed.

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test6/test6c-precrack-normals-proc-order.png" width=100% height=100%>

Following the indicated processing order (shown right), at the time of processing the second surface element no adjacencies are available, and with the unit normals (shown left) being orthogonal with no common solids, the script cannot unambiguously determine the orientation of the surface.

<img src="https://github.com/AlfaBetaBeta/gmsh-crack-generator/blob/master/img/test6/test6d-precrack-distinct-attchd-solids.png" width=100% height=100%>

Although in this mesh the orthogonal planes are adjacent and processed successively, they do not share common solids (each plane is attached to a different wedge element), and hence orientating whichever surface is processed in second place becomes ambiguous.

**Note that in both cases execution would run satisfactorily just by tagging vertical and horizontal cracks differently**.
