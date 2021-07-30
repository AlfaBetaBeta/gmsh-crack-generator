import sys
import operator
import pandas as pd
import numpy as np
import collections.abc


def mesh2df(fname):
    """
    Read through a .msh file to create relevant objects, mainly pandas DataFrames.
    
    Returns:
    
    df_phe = DataFrame with physical entities
                       dimension     name
             phys_ID   ...           ...
                .       .             .
                .       .             .
                .       .             .
             
    df_nod = DataFrame with nodes
                       coords
             node_ID   [...]
                .        .
                .        .
                .        .
            
    df_elm = DataFrame with elements
                       nodes   tags    type
             elm_ID    [...]   [...]   ...
               .         .       .      .
               .         .       .      .
               .         .       .      .
               
    mesh_format = List with msh format info
    
    NOTE: The first element of tags is already the physical tag, not the total number of tags (as in gmsh),
          as this can be readily obtained via len(tags).
    NOTE: If len(tags)>2 in a solid element, the mesh has been partitioned!
    NOTE: The msh file has to be in gmsh legacy format v2.* 

    Credit to:
    https://github.com/tjolsen/Mesh_Utilities/blob/master/gmsh_crack/gmsh_crack.py
    on which this function is based
    """
    node_ID = []
    node_coords = []

    elem_ID = []
    elem_type = []
    elem_tags = []
    elem_nodes = []

    phyent_ID = []
    phyent_name = []
    phyent_dim = []
    
    parse_dict = {"$MeshFormat": 0,
                  "$PhysicalNames": 1,
                  "$Nodes": 2,
                  "$Elements": 3}
    
    with open(file_name) as f:
        for line in f:
            line = line.strip()
            if line in parse_dict:
                parsing_section = parse_dict[line]
                
            elif line.startswith("$End"): continue

            elif parsing_section == 0:
                mesh_format = line.split()

            elif parsing_section == 1:
                if len(line.split()) == 1: continue
                tmp = line.split()
                phyent_dim.append(int(tmp[0]))
                phyent_ID.append(int(tmp[1]))
                phyent_name.append(tmp[2])

            elif parsing_section == 2:
                if len(line.split()) == 1: continue
                tmp = line.split()
                node_ID.append(int(tmp[0]))
                node_coords.append([float(t) for t in tmp[1:]])

            elif parsing_section == 3:
                if len(line.split()) == 1: continue
                tmp = line.split()
                elem_ID.append(int(tmp[0]))
                elem_type.append(int(tmp[1]))
                elem_tags.append([int(t) for t in tmp[3 : 3 + int(tmp[2])]])
                elem_nodes.append([int(t) for t in tmp[3 + int(tmp[2]) :]])

    df_phe = pd.DataFrame({"name": phyent_name, "dimension": phyent_dim},
                          columns=["dimension", "name"],
                          index=phyent_ID)

    df_nod = pd.DataFrame({"coords": node_coords},
                          index=node_ID)

    df_elm = pd.DataFrame({"type": elem_type, "tags": elem_tags, "nodes": elem_nodes},
                          columns=["nodes", "tags", "type"],
                          index=elem_ID)

    
    return (df_phe, df_nod, df_elm, mesh_format)


def flatten(L):
    """
    Flatten a list of items/sublists/nested sublists into a one-level list.
    
    Example: 
    >>> flatten([[1,2,3],4,[5,[6,7]]]) 
    [1,2,3,4,5,6,7]
    """
    for el in L:
        # Recursion if item `el` is itself an iterable (that is NOT a string)
        if isinstance(el, collections.abc.Iterable) and not isinstance(
            el, (str, bytes)
        ):
            yield from flatten(el)
        # Else just yield item `el`
        else:
            yield el
            

def check_tag_uniqueness(df_elm, L_cr, dict_crack2crack):
    """
    df_elm            = DataFrame with all elements in the mesh
    L_cr              = Flattened list with all crack IDs (surface physical tags)
    dict_crack2crack  = Dictionary linking IDs of identical surface elements (i.e. comprising the same nodes)
                        having different physical tags
    
    Returns:
    df_elm = DataFrame of elements with a unique elementary tag for each distinct surface element
    
    TODO: Look into necessary adjustments if dict_crack2crack.values() are tuples of size 2+
    """

    # Mask to subset rows with:
    # * ALL elements but the cracks
    cond_notcr = [L[0] not in L_cr for L in df_elm["tags"]]
    df_notcr = df_elm[cond_notcr]
    # * ONLY the cracks
    cond_cr = [L[0] in L_cr[:-1] for L in df_elm["tags"]]
    df_cr = df_elm[cond_cr]  # Does NOT include the common tag for s2in

    # If all crack elementary tags are already different, there is no need to proceed further
    if len(set([L[1] for L in df_cr["tags"]])) == len(df_cr):
        return df_elm

    # Set of unique values from the elementary tags of ALL elements but the cracks
    set_eltags = set([L[1] for L in df_notcr["tags"]])

    # Loop over DataFrame and amend elementary tag where appropriate
    for it in df_cr.itertuples():
        if it[2][1] not in set_eltags:
            set_eltags.add(it[2][1])
            continue
        else:
            associated_crack, = dict_crack2crack[it[0]]
            new_eltag = max(set_eltags) + 1
            df_elm.at[it[0], "tags"] = [it[2][0], new_eltag]
            df_elm.at[associated_crack, "tags"] = [L_cr[-1], new_eltag]
            set_eltags.add(new_eltag)

    return df_elm


def get_IDs_same_elmt(df_elm, L_cr, elmt_type="solid"):
    """
    df_elm    = DataFrame with all elements in the mesh
    L_cr      = Flattened list with all crack IDs (surface physical tags)
    elmt_type = "solid" or "surface"
    
    Returns:
    dict_elmt2elmt = Dictionary linking IDs of identical solid or surface elements (i.e. comprising the same nodes)
                     having different physical tags
    
    Example:
    >>> get_IDs_same_elmt(df_elm)
    {9: (10,),
     17: (18,),
     11: (12,),
     13: (14,),
     15: (16,)}
    
    NOTE: Only the following gmsh solid elements are explicitly considered here:
            * 17 = 20-noded hexahedron
            * 18 = 15-noded wedge
    """
    solid_types = [17, 18]
    if elmt_type == "solid":
        df = df_elm[df_elm["type"].isin(solid_types)]
    elif elmt_type == "surface":
        df = df_elm[[L[0] in L_cr for L in df_elm["tags"]]]
    
    # Transform df into a DataFrame of strings (to facilitate vectorisation)
    df = df.applymap(str)

    # Group the DataFrame by nodes (i.e. strings embedding lists)
    df = dict(list(df.groupby("nodes")))

    # Create ouput dictionary from the indices of the groups
    dict_elmt2elmt = {subdf.index.values[0]: tuple(subdf.index.values[1:])\
                      for subdf in df.values()\
                      if len(subdf) > 1}

    return dict_elmt2elmt


def get_adjacent_surface_ID(elm_ID, normal_dict):
    """
    elm_ID = Surface element ID (NOT included in normal_dict)
    normal_dict = Dictionary storing the outer normals of all processed surface elements
    
    Returns:
    adj_ID = ID of one surface element in `normal_dict` sharing at least one node with the given element
    
    NOTE: If `normal_dict` contains no neighbouring surface elements, then 0 is returned as ID
    """
    adj_ID = 0
    for k in normal_dict.keys():
        if bool(
            set.intersection(
                set(df_elm.loc[k, "nodes"]), set(df_elm.loc[elm_ID, "nodes"])
            )
        ):
            adj_ID = k
            break

    return adj_ID


def get_unit_normal(surface_elmt):
    """
    surface_elmt = Series of a particular surface element (= df_elm.loc[`given ID`])
    
    Returns:
    nrm = list with the [x,y,z] components of the unit normal to the surface
    
    NOTE: the orientation is given by the ordering of the surface element nodes
    """
    n3 = surface_elmt.loc["nodes"][:3]
    vec1 = [
        i - j for i, j in zip(df_nod.loc[n3[1], "coords"], df_nod.loc[n3[0], "coords"])
    ]
    vec2 = [
        i - j for i, j in zip(df_nod.loc[n3[2], "coords"], df_nod.loc[n3[1], "coords"])
    ]
    normal = [
        vec1[1] * vec2[2] - vec1[2] * vec2[1],
        vec1[2] * vec2[0] - vec1[0] * vec2[2],
        vec1[0] * vec2[1] - vec1[1] * vec2[0],
    ]

    # Scale the normal to a unit vector
    s = (sum([i**2 for i in normal])) ** (0.5)
    normal = [i / s for i in normal]

    return normal


def set_top_bottom_consistency(adj_srf_ID, normal_dict, elmt_tuple, node_reordering, nodemap):
    """
    adj_srf_ID      = ID of a surface element adjacent to the one given by `elmt_tuple`
    normal_dict     = Dictionary with
                       surface_elm_ID : [outer unit normal components]
                      as key/value pairs
    elmt_tuple      = Surface element being processed
    node_reordering = List with the adjusted node order for a surface element to reverse its unit normal
    nodemap         = Dictionary linking original with corresponding duplicated nodes across the crack lip
    
    TODO: Adjustments needed for the case that dict_crack2crack[..] returns a tuple of size 2+ !!!!
    """
    tolerance = 0.001
    
    # An adjacent previously processed surface is available
    if adj_srf_ID != 0:
        
        projection_sign = np.sign(sum([v1 * v2 for v1, v2 in zip(normal_dict[elmt_tuple[0]], normal_dict[adj_srf_ID])]))
        
        if projection_sign < -tolerance:
            # Reverse unit normal
            normal_dict[elmt_tuple[0]] = list(map(lambda x: x * projection_sign, normal_dict[elmt_tuple[0]]))
            # Update crack node ordering to comply with new normal (necessary if surfaces across crack lips become an interface)
            df_elm.at[elmt_tuple[0], "nodes"] = [elmt_tuple[1][x] for x in node_reordering]
            n_all = df_elm.loc[dict_crack2crack[elmt_tuple[0]], "nodes"]
            associated_surface, = dict_crack2crack[elmt_tuple[0]]
            df_elm.at[associated_surface, "nodes"] = [n_all[x] for x in node_reordering]
            
        elif abs(projection_sign) <= tolerance:
            print("Surface {} is perpendicular to previously processed surface {}".format(elmt_tuple[0], adj_srf_ID))
            top_adj_crack, bottom_adj_crack = get_top_bottom_elmts(adj_srf_ID, normal_dict[adj_srf_ID], nodemap)
            dupl_crack, = dict_crack2crack[adj_srf_ID]
            top_adj_dupl_crack, bottom_adj_dupl_crack = get_top_bottom_elmts(dupl_crack, normal_dict[adj_srf_ID], nodemap)
            
            top_adj = [top_adj_dupl_crack, top_adj_crack][bool(top_adj_crack)]
            bottom_adj = [bottom_adj_dupl_crack, bottom_adj_crack][bool(bottom_adj_crack)]
            
            top_current, bottom_current = get_top_bottom_elmts(elmt_tuple[0], normal_dict[elmt_tuple[0]], nodemap)
            
            if (top_adj == top_current) or (bottom_adj == bottom_current):
                print("Orientation of surface {} validated".format(elmt_tuple[0]))
                
            elif (top_adj == bottom_current) or (bottom_adj == top_current):
                # Reverse unit normal
                normal_dict[elmt_tuple[0]] = list(map(lambda x: -x, normal_dict[elmt_tuple[0]]))
                # Update crack node ordering to comply with new normal (necessary if surfaces across crack lips become an interface)
                df_elm.at[elmt_tuple[0], "nodes"] = [elmt_tuple[1][x] for x in node_reordering]
                n_all = df_elm.loc[dict_crack2crack[elmt_tuple[0]], "nodes"]
                associated_surface, = dict_crack2crack[elmt_tuple[0]]
                df_elm.at[associated_surface, "nodes"] = [n_all[x] for x in node_reordering]
                print("Orientation of surface {} reversed".format(elmt_tuple[0]))
                
            else:
                print("Unable to orientate surface {}, consider assigning it\
                a different tag to be processed separately from surface {}".format(elmt_tuple[0], adj_srf_ID))
                sys.exit(1)
            
    # There are previously processed surfaces but none is adjacent (or current surface is the first to be processed)
    elif adj_srf_ID == 0 and len(normal_dict) >= 1:
        
        # Compare current unit normal with average normal over all previously processed surfaces within the active crack plane
        avg_normal = [sum(x) / len(normal_dict) for x in zip(*normal_dict.values())]
        projection_sign = np.sign(sum([v1 * v2 for v1, v2 in zip(normal_dict[elmt_tuple[0]], avg_normal)]))
        
        if projection_sign < -tolerance:
            # Reverse unit normal
            normal_dict[elmt_tuple[0]] = list(map(lambda x: x * projection_sign, normal_dict[elmt_tuple[0]]))
            # Update crack node ordering if normal needs reversing for consistent top/bottom with adjacent surface elements
            df_elm.at[elmt_tuple[0], "nodes"] = [elmt_tuple[1][x] for x in node_reordering]
            n_all = df_elm.loc[dict_crack2crack[elmt_tuple[0]], "nodes"]
            associated_surface, = dict_crack2crack[elmt_tuple[0]]
            df_elm.at[associated_surface, "nodes"] = [n_all[x] for x in node_reordering]
            
        elif abs(projection_sign) <= tolerance:
            print("Surface {} is perpendicular to previously processed surface {}".format(elmt_tuple[0], adj_srf_ID))
            print("Unable to orientate surface {}, consider assigning it\
            a different tag to be processed separately from surface {}".format(elmt_tuple[0], adj_srf_ID))
            sys.exit(1)

            
def get_elmt_centroid(solid_elmt, nodemap):
    """
    solid_elmt = Series of a particular solid element
    nodemap    = Dictionary linking original with corresponding duplicated nodes across the crack lip,
                 in case `df_nod` makes a loc call to a duplicated node before that DataFrame could be updated
    
    Returns:
    coords = List with the average coordinates over all nodes of the element
    """
    coords = [0, 0, 0]
    for n in solid_elmt.loc["nodes"]:
        try:
            node_coords = df_nod.loc[n].tolist()[0]
        except KeyError:
            old_node = [k for k,v in nodemap.items() if v == n][0]
            node_coords = df_nod.loc[old_node].tolist()[0]

        coords = [sum(x) for x in zip(coords, node_coords)]
        
    n_elmt_nodes = len(solid_elmt.loc["nodes"])
    coords = [x / n_elmt_nodes for x in coords]

    return coords


def get_top_bottom_elmts(elm_ID, elm_normal, nodemap):
    """
    elm_ID     = Surface element ID
    elm_normal = Surface normal [x,y,z]
    nodemap    = Dictionary linking original with duplicated nodes, to be passed on to `get_elmt_centroid()`
                 in case `df_nod` makes a loc call to a duplicated node before that DataFrame could be updated
    
    Returns:
    [top,bottom] = List with the 2 solid element IDs (belonging to dict_solid2solid.keys()) attached
                   to the given surface element, always in the order (TOP elem_ID, BOTTOM elem_ID),
                   where `top` and `bottom` are given by the orientation of the surface normal
                   (it points bottom to top). Should one of the 2 solid elements be missing, then its ID
                   defaults to 0.
    """
    surface_elmt = df_elm.loc[elm_ID]
    # List with the node IDs of the first three nodes of the given surface element
    n3 = surface_elmt.loc["nodes"][:3]
    # Mask df_elm by the elements containing the first node of n3
    df = df_elm[[n3[0] in set(L) for L in df_elm["nodes"]]]# df_elm["nodes"].values.tolist()
    # Further mask by constraining the solid element ID to belong to dict_solid2solid.keys()
    df = df[[i in dict_solid2solid.keys() for i in df.index]]

    top_bottom = set(df.index)
    # Update the search with the second and third nodes of n3
    for n in range(2):
        df = df[[n3[n + 1] in set(L) for L in df["nodes"]]]
        top_bottom.intersection_update(df.index)
        
    top_bottom = list(top_bottom)

    if len(top_bottom) == 0:
        print(
            "Error: get_top_bottom_elmts() could not find solid elements attached to surface element {0}".format(
                elm_ID
            )
        )
        sys.exit(1)

    if len(top_bottom) == 1:
        corner_node = n3[0]
        if corner_node not in df_nod.index:
            corner_node = [k for k,v in nodemap.items() if v == corner_node][0]
            
        vec = [
            i - j
            for i, j in zip(
                get_elmt_centroid(df_elm.loc[top_bottom[0]], nodemap),
                df_nod.loc[corner_node, "coords"],
            )
        ]
        dot_product = sum([i * j for i, j in zip(vec, elm_normal)])
        if dot_product > 0:
            top_bottom = top_bottom + [0]
        if dot_product < 0:
            top_bottom = (top_bottom + [0])[::-1]
    else:
        vec = [
            i - j
            for i, j in zip(
                get_elmt_centroid(df_elm.loc[top_bottom[0]], nodemap),
                get_elmt_centroid(df_elm.loc[top_bottom[1]], nodemap),
            )
        ]
        dot_product = sum([i * j for i, j in zip(vec, elm_normal)])
        if dot_product < 0:
            top_bottom = top_bottom[::-1]
        elif dot_product == 0:
            print(
                "dot_product = 0 when processing surface element {}, which indicates that the same \
                solid element has been identified as top/bottom".format(elm_ID)
            )
            print("(possibly with different physical tags, check dict_solid2solid)")
            sys.exit(1)

    return top_bottom


def get_attached_surfaces(elm_ID):
    """
    elm_ID = Solid element ID
    
    Returns:
    attchd_surfs = List with IDs of all surface elements attached to the given solid element
    
    NOTE: A surface is considered 'attached' if it coincides with a full face of the solid
    NOTE: Only the following gmsh surface elements are explicitly considered here:
            * 16 = 8-noded quad
            * 9  = 6-noded triangle
    """
    nodes_per_face = {16: 8, 9: 6}
    attchd_surfs = []

    if elm_ID not in df_elm.index:
        print("There is no solid element with ID = {} in get_attached_surfaces()".format(elm_ID))
        print("Most likely caused because the active crack has no top element")
        return attchd_surfs

    for i in df_elm[df_elm["type"].isin(nodes_per_face.keys())].index:
        solid_nodes = set(df_elm.loc[elm_ID, "nodes"])
        solid_nodes.intersection_update(df_elm.loc[i, "nodes"])
        if len(solid_nodes) == nodes_per_face[df_elm.loc[i, "type"]]:
            attchd_surfs.append(i)

    return attchd_surfs


def get_active_surf_elmts(crack_ID):
    """
    crack_ID = Physical tag(s) of surface (crack) element(s) being processed
    
    Returns:
    df = DataFrame with active surface elements (active cracks, i.e. cracks undergoing processing)
    op = Check operator (to check if a surface is an active crack), corresponding to the type of `crack_ID`
    
    NOTE: Bear in mind that `crack_ID` may be an int or a list of ints!
    """
    # 
    if type(crack_ID) == int:
        df = df_elm[[crack_ID in L[:1] for L in df_elm.tags]]
        op = operator.eq
    elif type(crack_ID) == list:
        df = df_elm[
            [
                bool(set.intersection(set(crack_ID), set(L[:1])))
                for L in df_elm.tags
            ]
        ]
        op = operator.contains
    
    return df, op


def set_partition_tags(surf_top, surf_bottom, top, bottom):
    """
    surf_top    = Surface element ID representing the interface top (i.e. duplicated surface)
    surf_bottom = Surface element ID representing the interface bottom (i.e. original surface)
    top, bottom = Top/bottom solid element IDs attached to the active crack under consideration
                 
    NOTE: It is assumed that surface elements becoming interfaces are NEVER partitioned in gmsh alongside
          the solid elements, as gmsh (v2) only seems to deal with the highest dimension when partitioning.
          
    TODO: Check if this behaviour changes in v3+ and update this function accordingly!
    """
    par0 = df_elm.loc[top, "tags"][3]
    par1 = df_elm.loc[bottom, "tags"][3]
    parts = set([par0, par1])

    # Update the taglist of the crack lips' surfaces to include partition tags
    df_elm.loc[surf_top, "tags"].extend([len(parts), par0])
    df_elm.loc[surf_bottom, "tags"].extend([len(parts), par1])
    if len(parts) > 1:
        df_elm.loc[surf_top, "tags"].append(-par1)
        df_elm.loc[surf_bottom, "tags"].append(-par0)


def get_adjacent_solid_IDs(elm_ID, elm_normal, nodemap):
    """
    elm_ID           = Surface element ID
    elm_normal       = Surface normal [x,y,z]
    nodemap          = Dictionary linking original with duplicated nodes
    
    Returns:
    adj_solid_IDs = List with IDs of adjacent solid elements in contact with the given surface element
                    but NOT attached to it, provided these adjacent elements are on the 'top region' of the
                    surface element.
    
    NOTE: Adjacent solid elements in `adj_solid_IDs` do share nodes with the given surface element,
          but not an entire face (otherwise they would be attached)
    """
    surface_elmt = df_elm.loc[elm_ID]
    nodes = set(surface_elmt.loc["nodes"])
    
    df = df_elm[[bool(set.intersection(nodes, set(L))) for L in df_elm["nodes"].values.tolist()]]
    df = df[[i in dict_solid2solid.keys() for i in df.index]]

    adj_solid_IDs = list()

    for adj_ID in df.index:
        vec = [
            i - j
            for i, j in zip(
                get_elmt_centroid(df_elm.loc[adj_ID], nodemap),
                get_elmt_centroid(surface_elmt, nodemap),
            )
        ]
        
        dot_product = sum([i * j for i, j in zip(vec, elm_normal)])
        if dot_product > 0:
            adj_solid_IDs.append(adj_ID)

    return adj_solid_IDs


def get_detached_adj_solids(solid_ID, crack_nodes):
    """
    solid_ID    = ID of the solid element under consideration
    crack_nodes = Whole list of nodes conforming the active crack plane
    
    This function is meant to be used with `filter()`, to discard solid elements if they are
    attached to any active crack surface.
    """
    for surface in get_attached_surfaces(solid_ID):
        if set(df_elm.loc[surface, "nodes"]).issubset(set(crack_nodes)): return False
    return True


def duplicate_nodes(surf_elmt_tuple, node_mapping, active_crack_ID, op, normal_dict, crack_nodes):
    """
    surf_elmt_tuple = Surface element being processed
    node_mapping    = Dictionary linking original with corresponding duplicated nodes across the crack lip
    active_crack_ID = Active crack ID (i.e. physical tag of `surf_elmt_tuple`)
    op              = Operator to check if a surface element belongs to the active crack plane
    normal_dict     = Dictionary with
                       surface_elm_ID : [outer unit normal components]
                      as key/value pairs
    crack_nodes     = Whole list of nodes conforming the active crack plane
    """
    # Pair of solid top/bottom element IDs attached to crack surface `surf_elmt_tuple`
    top, bottom = get_top_bottom_elmts(surf_elmt_tuple[0], normal_dict[surf_elmt_tuple[0]], node_mapping)
        
    # TODO: Accommodate the case that an interface side is directly constrained by boundary conditions
    if not top:
        print("Active crack physical tag = {}".format(surf_elmt_tuple[2][0]))
        print("Surface element {0} of type {1} is missing a solid element on top\n".format(surf_elmt_tuple[0], surf_elmt_tuple[3]))
        return
    
    # List of surface elements attached to solid element on top of crack surface `surf_elmt_tuple`
    top_attchd_surfs = get_attached_surfaces(top)
    
    
    # Duplicate nodes:
    
    # 1) on the face of the `top` solid in contact with active crack surface `surf_elmt_tuple`
    top_nodes = df_elm.loc[top, "nodes"]
    df_elm.at[top, "nodes"] = [node_mapping[node]\
                               if node in set(surf_elmt_tuple[1]) else node\
                               for node in top_nodes]
    
    # 2) in contact with active crack surface `surf_elmt_tuple`
    #    &
    #    belonging to surfaces attached to the `top` solid that are NOT the active crack surface
    for surface in filter(lambda s: not op(active_crack_ID, df_elm.loc[s, "tags"][0]), top_attchd_surfs):
        nodes = df_elm.loc[surface, "nodes"]
        df_elm.at[surface, "nodes"] = [node_mapping[node]\
                                       if node in set(surf_elmt_tuple[1]) else node\
                                       for node in nodes]
        # Update surface partition tags if necessary
        if (len(df_elm.loc[top, "tags"]) > 2 and nodes == df_elm.loc[surf_elmt_tuple[0], "nodes"]):
            set_partition_tags(surface, surf_elmt_tuple[0], top, bottom)

    # 3) of adjacent solid elements in contact with active crack surface `surf_elmt_tuple` 
    #    &
    #    NOT attached to the crack plane (i.e. NOT attached to any active crack surface),
    #    alongside nodes of surfaces attached to these solids if these surfaces share nodes with active crack surface
    adj_solid_IDs = get_adjacent_solid_IDs(surf_elmt_tuple[0], normal_dict[surf_elmt_tuple[0]], node_mapping)
    adj_solid_IDs = list(filter(lambda ID: get_detached_adj_solids(ID, crack_nodes), adj_solid_IDs))
    for ID in adj_solid_IDs:
        attached_surfaces = get_attached_surfaces(ID)
        df_elm.at[ID, "nodes"] = [node_mapping[node]\
                                  if node in set(surf_elmt_tuple[1]) else node\
                                  for node in df_elm.loc[ID, "nodes"]]
        
        for surface in attached_surfaces:
            df_elm.at[surface, "nodes"] = [node_mapping[node]\
                                           if node in set(surf_elmt_tuple[1]) else node\
                                           for node in df_elm.loc[surface, "nodes"]]


def duplicate_surfaces(df_crack_elm, common_crack_ID):
    """
    df_crack_elm    = Subset df_elm containing only the surface elements that are active cracks
    common_crack_ID = Physical tag common to all cracks
    
    Create a new surface element per active crack surface.
    
    This new surface element has the physical tag common to all cracks. Hence, the pair of surface elements
    with physical tag `common_crack_ID` (one with original nodes and the other with new duplicated ones)
    represent the sides of an interface element (16-noded or 12-noded), the explicit creation of which is
    FE engine dependent.
    """
    new_elmt_ID = len(df_elm)
    
    for i in df_crack_elm.itertuples():
        # Nodes to be retrieved from `df_elm` as their order may have changed to maintain a consistent top/bottom
        nodes = df_elm.loc[i[0], "nodes"]  
        new_elmt_ID += 1
        elmt_type = i[3]
        tags = i[2]
        df_elm.loc[new_elmt_ID] = [nodes, [common_crack_ID] + tags[1:], elmt_type]
        
        
def update_nodes(node_mapping):
    """
    Add new duplicated nodes with overlapping coordinates to the DataFrame with all nodes in the mesh
    """
    for original, duplicate in node_mapping.items():
        df_nod.loc[duplicate] = [df_nod.loc[original, "coords"]]
        
        
def update_elmts(dict_solid2solid):
    """
    Update node lists of solid elements belonging to `dict_solid2solid.values()`
    
    NOTE: Could be optimised to update only the solids that actually had new nodes assigned, look into it!
    NOTE: This needs revisiting in case dict_solid2solid.values() are tuples of size 2+
    """
    for solid in dict_solid2solid.keys():
        associated_solid, = dict_solid2solid[solid]
        df_elm.at[associated_solid, "nodes"] = df_elm.loc[solid, "nodes"]
        
        
def make_crack(active_crack_ID, common_crack_ID):
    """
    active_crack_ID = Active crack ID (i.e. physical tag of the surface element being processed)
    common_crack_ID = Physical tag common to all cracks
    
    Duplicate relevant nodes of solid/surface elements to accommodate the given crack
    """
    df_crack_elm, op = get_active_surf_elmts(active_crack_ID)
        
    # Make a list of all distinct node IDs belonging to the active cracks
    crack_nodes = flatten([Lnod for Lnod in df_crack_elm["nodes"]])
    crack_nodes = list(set(crack_nodes))
    
    # List of new node IDs for crack nodes to be duplicated
    new_nodes = [(len(df_nod) + i + 1) for i in range(len(crack_nodes))]
    # Dictionary mapping original nodes to corresponding duplicates across the crack lip
    node_mapping = dict([(crack_nodes[i], new_nodes[i]) for i in range(len(crack_nodes))])
    
    # Initialise aux variables for orientation preprocessing
    normal_dict = dict()
    adj_srf_ID = 0
    node_reordering = {16: [0, 3, 2, 1, 7, 6, 5, 4], 9: [0, 2, 1, 5, 4, 3]}
    
    for i in df_crack_elm.itertuples():
        
        # Aux preprocessing to ensure that neighbouring crack elements have the same top/bottom consistently
        if normal_dict: adj_srf_ID = get_adjacent_surface_ID(i[0], normal_dict)
        normal_dict[i[0]] = get_unit_normal(df_elm.loc[i[0]])
        set_top_bottom_consistency(adj_srf_ID, normal_dict, i, node_reordering[i[3]], node_mapping)
        
        # Duplicate relevant nodes
        duplicate_nodes(i, node_mapping, active_crack_ID, op, normal_dict, crack_nodes)
    
    # Duplicate relevant surfaces
    duplicate_surfaces(df_crack_elm, common_crack_ID)

    # Update node and element dataframes to account for all duplicates
    update_nodes(node_mapping)
    update_elmts(dict_solid2solid) 


def df2mesh(file_name):
    """
    Create a .msh file based on the updated DataFrames and `mesh_format`.
    
    NOTE: The output file format corresponds to gmsh legacy format v2.*
    
    Credit to:
    https://github.com/tjolsen/Mesh_Utilities/blob/master/gmsh_crack/gmsh_crack.py
    on which this function is based
    """
    with open(file_name, "w") as f:
        
        # Write header
        f.write("$MeshFormat\n")
        f.write(" ".join(mesh_format) + "\n")
        f.write("$EndMeshFormat\n")

        # Write physical entities
        f.write("$PhysicalNames\n")
        f.write(str(len(df_phe)) + "\n")
        for pe in df_phe.itertuples():
            line = str(pe[1]) + " " + str(pe[0]) + " " + pe[2]
            f.write(line + "\n")
        f.write("$EndPhysicalNames\n")

        # Write nodes
        f.write("$Nodes\n")
        f.write(str(len(df_nod)) + "\n")
        for n in df_nod.itertuples():
            line = str(n[0]) + " " + " ".join([str(i) for i in n[1]])
            f.write(line + "\n")
        f.write("$EndNodes\n")

        # Write elements
        f.write("$Elements\n")
        f.write(str(len(df_elm)) + "\n")
        for e in df_elm.itertuples():
            line = (
                str(e[0])
                + " "
                + str(e[3])
                + " "
                + str(len(e[2]))
                + " "
                + " ".join([str(i) for i in e[2]])
                + " "
                + " ".join([str(i) for i in e[1]])
                + "\n"
            )
            f.write(line)
        f.write("$EndElements")


def process_arguments():
    """
    Returns:
    file_name = Name of the msh file to be processed
    crack_IDs = List object equivalent to the input string `sys.argv[2]`, e.g. [1,[2,3],4] as
                equivalent to '[1,[2,3],4]'
    """
    file_name = sys.argv[1]
    if not file_name:
        print("Error: Must specify msh file_name")
        sys.exit(1)
    
    physical_tags = sys.argv[2]
    if not physical_tags:
        print("Error: Must specify tags representing crack_IDs")
        sys.exit(1)
        
    crack_IDs = []
    sub_list = []
    
    for tag in physical_tags[1:-1].split(","):
        
        if tag.startswith("["):
            sub_list.append(int(tag.strip("[")))
            
        elif tag.endswith("]"):
            sub_list.append(int(tag.strip("]")))
            crack_IDs.append(sub_list)
            sub_list = []
            
        elif len(sub_list) != 0:
            sub_list.append(int(tag))
            
        else:
            crack_IDs.append(int(tag))
    
    return (file_name, crack_IDs)


if __name__ == "__main__":

    file_name, crack_IDs = process_arguments()
    L_cr = list(flatten(crack_IDs))
    df_phe, df_nod, df_elm, mesh_format = mesh2df(file_name)
    dict_solid2solid = get_IDs_same_elmt(df_elm, L_cr)
    dict_crack2crack = get_IDs_same_elmt(df_elm, L_cr, elmt_type="surface")
    df_elm = check_tag_uniqueness(df_elm, L_cr, dict_crack2crack)
    
    for crack_ID in crack_IDs[:-1]:
        make_crack(crack_ID, crack_IDs[-1])
    
    df2mesh("cracked_" + file_name)
