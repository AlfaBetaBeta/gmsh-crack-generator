import sys
import operator
import pandas as pd
import numpy as np

import collections.abc


def mesh2df(fname):
    """
    Function reads through a .msh file and returns three pandas.DataFrame objects
    
    df_phe = DataFrame with physical entities
                       dimension     name
             phys_ID   ...           ...
                .       .             .
                .       .             .
                .       .             .
    --------------------------------------------------------------------------------         
    df_nod = DataFrame with nodes
                       coords
             node_ID   [...]
                .        .
                .        .
                .        .
    --------------------------------------------------------------------------------        
    df_elm = DataFrame with elements
                       nodes   tags    type
             elm_ID    [...]   [...]   ...
               .         .       .      .
               .         .       .      .
               .         .       .      .
    NOTE: The first element of tags is already the physical tag, not the total number of tags (as in gmsh), as this can be readily obtained via len(tags).
    NOTE: If len(tags)>2 in a solid element, the mesh has been partitioned!

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

    nnodes = -1
    nelements = -1
    nphyents = -1

    PARSING_PHYSENTS = 0
    PARSING_NODES = 1
    PARSING_ELEMENTS = 2
    ACTION_UNSET = 3

    # Set current action in file
    ACTION = ACTION_UNSET

    # Open file
    f = open(fname, "r")
    contents = f.read()
    f.close()

    lines = contents.split("\n")
    for line in lines:
        if ACTION == ACTION_UNSET:
            if line == "$PhysicalNames":
                ACTION = PARSING_PHYSENTS
            elif line == "$Nodes":
                ACTION = PARSING_NODES
            elif line == "$Elements":
                ACTION = PARSING_ELEMENTS
            continue
        elif ACTION == PARSING_PHYSENTS:
            if line == "$EndPhysicalNames":
                ACTION = ACTION_UNSET
                continue
            if nphyents == -1:
                nphyents = int(line)
                continue
            tmp = line.split()
            phyent_dim.append(int(tmp[0]))
            phyent_ID.append(int(tmp[1]))
            phyent_name.append(tmp[2])
            phyent_dict = {"name": phyent_name, "dimension": phyent_dim}
        elif ACTION == PARSING_NODES:
            if line == "$EndNodes":
                ACTION = ACTION_UNSET
                continue
            if nnodes == -1:
                nnodes = int(line)
                continue
            tmp = line.split()
            node_ID.append(int(tmp[0]))
            node_coords.append([float(t) for t in tmp[1:]])
            node_dict = {"coords": node_coords}
        elif ACTION == PARSING_ELEMENTS:
            if line == "$EndElements":
                ACTION = ACTION_UNSET
                continue
            if nelements == -1:
                nelements = int(line)
                continue
            tmp = line.split()
            elem_ID.append(int(tmp[0]))
            elem_type.append(int(tmp[1]))
            elem_tags.append([int(t) for t in tmp[3 : 3 + int(tmp[2])]])
            elem_nodes.append([int(t) for t in tmp[3 + int(tmp[2]) :]])
            elem_dict = {"type": elem_type, "tags": elem_tags, "nodes": elem_nodes}

    # Create DataFrames
    df_phe = pd.DataFrame(phyent_dict, columns=["dimension", "name"], index=phyent_ID)
    df_nod = pd.DataFrame(node_dict, index=node_ID)
    df_elm = pd.DataFrame(elem_dict, columns=["nodes", "tags", "type"], index=elem_ID)

    return (df_phe, df_nod, df_elm)


def flatten(L):
    """
    Flatten a list of items/sublists/nested sublists into a one-level list.
    Example: [[1,2,3],4,[5,[6,7]]] is flattened to [1,2,3,4,5,6,7]
    
    If the input argument is already a one-level list, the generated object will coincide, i.e. the 'flattened' list will be equal to the original list.
    """
    for el in L:
        # Recursion if item <el> is itself an iterable (that is NOT a string)
        if isinstance(el, collections.abc.Iterable) and not isinstance(
            el, (str, bytes)
        ):
            yield from flatten(el)
        # Else just yield item <el>
        else:
            yield el


def preproc_df(df_elm, crack_ids):
    """
    df_elm    = DataFrame with all elements in the mesh
    crack_ids = List with crack physical tags, with some pairs being possibly coupled if they belong to the same crack plane (e.g. [1,[2,3],4])
                
    df_elm may be edited in here if an elementary tag is found to be repeated for different cracks

    Returns:
    dic_s2s = Dictionary with IDs of the same crack surface elements under different physical tags
    """
    L_cr = list(flatten(crack_ids))

    # Mask to subset rows with ALL elements but the cracks/ONLY the cracks
    cond_notcr = [L[0] not in L_cr for L in df_elm["tags"]]
    cond_cr = [L[0] in L_cr[:-1] for L in df_elm["tags"]]
    df_notcr = df_elm[cond_notcr]
    df_cr = df_elm[cond_cr]  # Does NOT include the common tag for s2in

    # If all crack elementary tags are already different, there is no need to proceed further
    if len(set([L[1] for L in df_cr["tags"]])) == len(df_cr):
        return

    # Set of unique values from the elementary tags of ALL elements but the cracks
    set_eltags = set([L[1] for L in df_notcr["tags"]])

    # Dict relating IDs of the same surface element under different physical tags
    dic_s2s = intercrack(df_elm, L_cr)

    # Loop over DataFrame and amend elementary tag where appropriate
    for it in df_cr.itertuples():
        if it[2][1] not in set_eltags:
            set_eltags.add(it[2][1])
            continue
        else:
            new_eltag = max(set_eltags) + 1
            df_elm.at[it[0], "tags"] = [it[2][0], new_eltag]
            df_elm.at[dic_s2s[it[0]], "tags"] = [L_cr[-1], new_eltag]
            set_eltags.add(new_eltag)

    return dic_s2s


def intercrack(df_elm, L_cr):
    """
    df_elm = DataFrame with all elements in the mesh
    L_cr   = List (flattened!) with all crack surface element IDs
    
    Returns:
    dic_s2s = Dictionary linking IDs of identical surface elements of different physical tags
    """
    # Mask to subset only surface elements representing cracks
    df_cracksurf = df_elm[[L[0] in L_cr for L in df_elm["tags"]]]
    # Transform df_cracksurf into a DataFrame of strings (to facilitate vectorisation)
    df_cracksurf = df_cracksurf.applymap(str)

    # Group the DataFrame by nodes (i.e. strings embedding lists)
    df_cracksurf_gr = df_cracksurf.groupby("nodes")
    # Exhaust the grouping generator and store into a dictionary
    df_cracksurf_gr = dict(list(df_cracksurf_gr))

    # Create ouput dictionary from the indices of the groups
    dic_s2s = []
    for subdf in df_cracksurf_gr.values():
        dic_s2s.append(tuple(subdf.index.values))
    dic_s2s = dict(dic_s2s)

    return dic_s2s


def elem_ctrd(df_nod, sr_elm, nodmap):
    """
    df_nod  = DataFrame of all nodes in the mesh
    sr_elm  = Series of a particular element (= df_elm.loc[<given ID>])
    nodmap  = Dictionary linking old with duplicated nodes (duplication at upper level) in case df_nod makes a loc call to a duplicated node
              before that DataFrame could be updated
    
    Returns a list with the average coordinates over all nodes of the element
    """
    crds = [0, 0, 0]
    for n in sr_elm.loc["nodes"]:
        try:
            nl = df_nod.loc[n].tolist()[0]
        except KeyError:
            nn = list(nodmap.keys())[list(nodmap.values()).index(n)]
            nl = df_nod.loc[nn].tolist()[0]

        crds = [sum(x) for x in zip(crds, nl)]
    crds = [x / len(sr_elm.loc["nodes"]) for x in crds]

    return crds


def ext_nrm(df_nod, sr_elm):
    """
    df_nod = DataFrame of all nodes in the mesh
    sr_elm = Series of a particular SURFACE element (= df_elm.loc[<given ID>])
    
    Returns:
    nrm = list with the components of the unit normal to the surface
    NOTE: the orientation is given by the ordering of the nodes
    """
    n3 = sr_elm.loc["nodes"][:3]
    vec1 = [
        i - j for i, j in zip(df_nod.loc[n3[1], "coords"], df_nod.loc[n3[0], "coords"])
    ]
    vec2 = [
        i - j for i, j in zip(df_nod.loc[n3[2], "coords"], df_nod.loc[n3[1], "coords"])
    ]
    nrm = [
        vec1[1] * vec2[2] - vec1[2] * vec2[1],
        vec1[2] * vec2[0] - vec1[0] * vec2[2],
        vec1[0] * vec2[1] - vec1[1] * vec2[0],
    ]

    # Normalise the vector
    s = (sum([i * j for i, j in zip(nrm, nrm)])) ** (0.5)
    nrm = [i / s for i in nrm]

    return nrm


def solid2solid(df_elm):
    """
    df_elm = DataFrame with all elements in the mesh
    
    Returns:
    dic_s2s = Dictionary linking IDs of identical SOLID elements of different physical tags
    """
    # Mask to subset only solid elements (17=bk20, 18=wd15)
    df_solid = df_elm[(df_elm["type"] == 17) | (df_elm["type"] == 18)]
    # Transform df_solid into a DataFrame of strings (to facilitate vectorisation)
    df_solid = df_solid.applymap(str)

    # Group the DataFrame by nodes (i.e. strings embedding lists)
    df_solid_gr = df_solid.groupby("nodes")
    # Exhaust the grouping generator and store into a dictionary
    df_solid_gr = dict(list(df_solid_gr))

    # Create ouput dictionary from the indices of the groups
    dic_s2s = []
    for subdf in df_solid_gr.values():
        dic_s2s.append(tuple(subdf.index.values))
    dic_s2s = dict(dic_s2s)

    return dic_s2s


def srf_split(df_nod, df_elm, elm_ID, elm_nrm, dic_s2s, nodmap):
    """
    df_nod  = DataFrame with all nodes in the mesh
    df_elm  = DataFrame with all elements in the mesh
    elm_ID  = SURFACE element ID
    elm_nrm = SURFACE normal (list with vector components)
    dic_s2s = Dictionary linking identical solid element IDs of different physical tags
    nodmap  = Dictionary linking old with duplicated nodes, to be passed on to elem_ctrd in case df_nod makes a loc call to a duplicated node
              before that DataFrame could be updated
    
    Returns:
    top_bot = list with the 2 solid element IDs (belonging to dic_s2s.keys()) attached (by at least 3 corner nodes) to the given surface element,
              always in the order [TOP elem_ID, BOTTOM elem_ID], where top and bottom are given by the orientation of the surface normal
              (it points bottom to top). Should one of the 2 elements be missing, then it is replaced by 0.
    """
    # Extract Series of the element specified by elm_ID
    sr_elm = df_elm.loc[elm_ID]
    # List with the node_IDs of all nodes of the given surface element
    n3 = sr_elm.loc["nodes"][:3]
    # Mask df_elm by the elements containing the first node of n3
    df_elm_msk = df_elm[[n3[0] in set(L) for L in df_elm["nodes"].values.tolist()]]
    # Further mask by constraining the solid element tag to belong to dic_s2s.keys()
    df_elm_msk = df_elm_msk[[i in dic_s2s.keys() for i in df_elm_msk.index]]

    top_bot = set(df_elm_msk.index)
    # Update with the second and third nodes of n3
    for n in range(2):
        df_elm_msk = df_elm[
            [n3[n + 1] in set(L) for L in df_elm["nodes"].values.tolist()]
        ]
        df_elm_msk = df_elm_msk[[i in dic_s2s.keys() for i in df_elm_msk.index]]

        top_bot.intersection_update(df_elm_msk.index)

    if len(top_bot) == 0:
        print(
            "Error: srf_split() could not find solid elements attached to surface element {0}".format(
                elm_ID
            )
        )
        sys.exit(1)

    if len(top_bot) == 1:
        vec = [
            i - j
            for i, j in zip(
                elem_ctrd(df_nod, df_elm.loc[list(top_bot)[0]], nodmap),
                df_nod.loc[n3[0], "coords"],
            )
        ]
        # dp = sum([i*j for i,j in zip(vec,ext_nrm(df_nod,sr_elm))])
        dp = sum([i * j for i, j in zip(vec, elm_nrm)])
        if dp > 0:
            top_bot = list(top_bot) + [0]
        if dp < 0:
            top_bot = (list(top_bot) + [0])[::-1]
    else:
        vec = [
            i - j
            for i, j in zip(
                elem_ctrd(df_nod, df_elm.loc[list(top_bot)[0]], nodmap),
                elem_ctrd(df_nod, df_elm.loc[list(top_bot)[1]], nodmap),
            )
        ]
        # dp = sum([i*j for i,j in zip(vec,ext_nrm(df_nod,sr_elm))])
        dp = sum([i * j for i, j in zip(vec, elm_nrm)])
        if dp > 0:
            top_bot = list(top_bot)
        if dp < 0:
            top_bot = list(top_bot)[::-1]

    if type(top_bot) == set:
        print("Error: srf_split() could not convert top_bot to list")
        print(
            "top_bot =",
            top_bot,
            "during processing of surface element {0}".format(elm_ID),
        )
        if dp == 0:
            print(
                "dp = 0, which indicates that a solid element has been counted twice in top_bot"
            )
            print("(possibly with different physical tags, check dic_s2s)")

    return top_bot


def srf_neigh(df_nod, df_elm, elm_ID, elm_nrm, dic_s2s, nodmap):
    """
    df_nod  = DataFrame with all nodes in the mesh
    df_elm  = DataFrame with all elements in the mesh
    elm_ID  = SURFACE element ID
    elm_nrm = SURFACE normal (list with vector components)
    dic_s2s = Dictionary linking identical solid element IDs of different physical tags
    nodmap  = Dictionary linking old with duplicated nodes, to be passed on to elem_ctrd in case df_nod makes a loc call to a duplicated node
              before that DataFrame could be updated
    
    Returns:
    neigh = list with IDs of neighbouring solid elements in contact with the given surface element but NOT attached to it (i.e. they are NOT
            in the output of srf_split(..,elm_ID,..)), provided these neighbouring elements are on the 'top' region of the surface element.
    
    NOTE: neighbouring solid elements in neigh do share nodes with the given surface element, but not an entire face (otherwise they would be attached)
    """
    # Extract Series of the surface element specified by elm_ID
    sr_elm = df_elm.loc[elm_ID]
    # Set with the node_IDs of ALL nodes of the given surface element
    nall = set(sr_elm.loc["nodes"])
    # Mask df_elm by the elements containing ANY node in nall
    df_elm_msk = df_elm[
        [bool(set.intersection(nall, set(L))) for L in df_elm["nodes"].values.tolist()]
    ]
    # Further mask by constraining the element tag to belong to dic_s2s.keys() (solid elements)
    df_elm_msk = df_elm_msk[[i in dic_s2s.keys() for i in df_elm_msk.index]]

    neigh = list()

    for nel in df_elm_msk.index:
        vec = [
            i - j
            for i, j in zip(
                elem_ctrd(df_nod, df_elm.loc[nel], nodmap),
                elem_ctrd(df_nod, sr_elm, nodmap),
            )
        ]
        # dp = sum([i*j for i,j in zip(vec,ext_nrm(df_nod,sr_elm))])
        dp = sum([i * j for i, j in zip(vec, elm_nrm)])
        if dp > 0:
            neigh.append(nel)

    return neigh


def surf2vol(df_elm, elm_ID):
    """
    df_elm = DataFrame with all elements in the mesh
    elm_ID = SOLID element ID
    
    Returns:
    s_val = list with IDs of all SURFACE elements attached to the given solid element
    NOTE: a surface is considered attached if it coincides with a full face of the solid
    """
    # In gmsh, type16 = 8-noded quad, type9 = 6-noded triangle
    # dict.values() = the consistent number of nodes in common with a solid
    s2inodes = {16: 8, 9: 6}

    s_val = []

    if elm_ID not in set(df_elm.index):
        print("There is no solid element with ID = {0}".format(elm_ID))
        print("Most likely caused because the active crack has no top element")
        return s_val

    for i in df_elm[(df_elm["type"] == 16) | (df_elm["type"] == 9)].index:
        n = set(df_elm.loc[elm_ID, "nodes"])
        n.intersection_update(df_elm.loc[i, "nodes"])
        if len(n) == s2inodes[df_elm.loc[i, "type"]]:
            s_val.append(i)

    return s_val


def surf2surf(df_elm, elm_ID, nrm_dict):
    """
    df_elm = DataFrame with all elements in the mesh
    elm_ID = SURFACE element ID (NOT included in nrm_dict)
    nrm_dict = Dictionary with {surface_elm_ID : [outer normal components]}
    
    Returns:
    s_adj = ID of one surface element in nrm_dict sharing at least one node with the
            given element
    NOTE: If nrm_dict contains no neighbouring surface elements, then 0 is returned as ID
    """
    s_adj = 0
    for k in nrm_dict.keys():
        if bool(
            set.intersection(
                set(df_elm.loc[k, "nodes"]), set(df_elm.loc[elm_ID, "nodes"])
            )
        ):
            s_adj = k
            break

    return s_adj


def part2surf(df_elm, srf_ID_top, srf_ID_bot, s_topbot):
    """
    df_elm     = DataFrame with all elements in the mesh
    srf_ID_top = SURFACE element ID representing the interface 'top'
    srf_ID_bot = SURFACE element ID representing the interface 'bottom'
    s_topbot   = List with the [top,bottom] SOLID element IDs attached to the active crack in consideration (srf_ID_bot, as specified at upper level)
                 
    NOTE: It is assumed that surface elements becoming interfaces are never partitioned in gmsh alongside the solid elements, as gmsh only
          seems to deal with the highest dimension when partitioning. Should this behaviour change, this function would have to be revised!
    """
    par0 = df_elm.loc[s_topbot[0], "tags"][3]
    par1 = df_elm.loc[s_topbot[1], "tags"][3]
    parts = set([par0, par1])

    df_elm.loc[srf_ID_top, "tags"].extend([len(parts), par0])
    df_elm.loc[srf_ID_bot, "tags"].extend([len(parts), par1])
    if len(parts) > 1:
        df_elm.loc[srf_ID_top, "tags"].append(-par1)
        df_elm.loc[srf_ID_bot, "tags"].append(-par0)


def make_crack(df_nod, df_elm, active_crack_ID, dic_s2s, dic_srf2srf, all_cracks_ID):
    # DataFrame with active SURFACE elements == active cracks
    # Bear in mind that active_crack_ID may be an int or a list of ints!
    if type(active_crack_ID) == int:
        df_crack_el = df_elm[[active_crack_ID in L[:1] for L in df_elm.tags]]
        op = operator.eq
    elif type(active_crack_ID) == list:
        df_crack_el = df_elm[
            [
                bool(set.intersection(set(active_crack_ID), set(L[:1])))
                for L in df_elm.tags
            ]
        ]
        op = operator.contains

    # Make set of node IDs for nodes on ALL active cracks
    crack_nodes = []
    for Lnod in df_crack_el["nodes"].values.tolist():
        crack_nodes.extend(Lnod)
    crack_nodes = list(set(crack_nodes))

    # Create list of new node IDs for nodes to be duplicated
    new_nodes = []
    for i in range(1, len(crack_nodes) + 1):
        new_nodes.append(len(df_nod) + i)

    node_mapping = dict(
        [(crack_nodes[i], new_nodes[i]) for i in range(0, len(crack_nodes))]
    )

    # Initialise aux variables for orientation preprocessing
    nrm_dict = dict()
    s_prev = 0
    reord = [0, 3, 2, 1, 7, 6, 5, 4]
    # Set nodes in each solid element "on top" of a crack to the newly created nodes
    # Loop over Surface (active crack) elements
    for i in df_crack_el.itertuples():
        # Aux preprocessing to ensure that neighbouring crack elements have the same
        # 'top'/'bottom' consistently
        if len(nrm_dict) > 0:
            s_prev = surf2surf(df_elm, i[0], nrm_dict)
        nrm_dict[i[0]] = ext_nrm(df_nod, df_elm.loc[i[0]])
        if s_prev != 0:
            sgn = np.sign(
                sum([v1 * v2 for v1, v2 in zip(nrm_dict[i[0]], nrm_dict[s_prev])])
            )
            nrm_dict[i[0]] = list(map(lambda x: x * sgn, nrm_dict[i[0]]))
            if sgn < 0:
                # Update crack node ordering if normal needs reversing
                df_elm.at[i[0], "nodes"] = [i[1][x] for x in reord]
                n_all = df_elm.loc[dic_srf2srf[i[0]], "nodes"]
                df_elm.at[dic_srf2srf[i[0]], "nodes"] = [n_all[x] for x in reord]
            if sgn == 0:
                print(
                    "Surface {} is perpendicular to previous surfaces".format(i[0] + 1)
                )
                print(
                    "Unable to orientate surface. Consider changing the surface ordering"
                )
                sys.exit(1)
        elif s_prev == 0 and len(nrm_dict) > 1:
            avg_nrm = [sum(x) / len(nrm_dict) for x in zip(*nrm_dict.values())]
            sgn = np.sign(sum([v1 * v2 for v1, v2 in zip(nrm_dict[i[0]], avg_nrm)]))
            nrm_dict[i[0]] = list(map(lambda x: x * sgn, nrm_dict[i[0]]))
            if sgn < 0:
                # Update crack node ordering if normal needs reversing
                df_elm.at[i[0], "nodes"] = [i[1][x] for x in reord]
                n_all = df_elm.loc[dic_srf2srf[i[0]], "nodes"]
                df_elm.at[dic_srf2srf[i[0]], "nodes"] = [n_all[x] for x in reord]
            if sgn == 0:
                print(
                    "Surface {} is perpendicular to previous surfaces".format(i[0] + 1)
                )
                print(
                    "Unable to orientate surface. Consider changing the surface ordering"
                )
                sys.exit(1)
        elif s_prev == 0 and len(nrm_dict) == 1:
            avg_nrm = list(nrm_dict.values())[0]
            sgn = np.sign(sum([v1 * v2 for v1, v2 in zip(nrm_dict[i[0]], avg_nrm)]))
            nrm_dict[i[0]] = list(map(lambda x: x * sgn, nrm_dict[i[0]]))
            if sgn < 0:
                # Update crack node ordering if normal needs reversing
                df_elm.at[i[0], "nodes"] = [i[1][x] for x in reord]
                n_all = df_elm.loc[dic_srf2srf[i[0]], "nodes"]
                df_elm.at[dic_srf2srf[i[0]], "nodes"] = [n_all[x] for x in reord]
            if sgn == 0:
                print(
                    "Surface {} is perpendicular to previous surfaces".format(i[0] + 1)
                )
                print(
                    "Unable to orientate surface. Consider changing the surface ordering"
                )
                sys.exit(1)

        # List of solid elements [top,bottom] attached to crack i
        s_topbot = srf_split(
            df_nod, df_elm, i[0], nrm_dict[i[0]], dic_s2s, node_mapping
        )
        # List of surface elements attached to solid element on top of crack i
        el_s = surf2vol(df_elm, s_topbot[0])
        # Duplicate nodes on the face of the solid in contact with active crack i (on top)
        if s_topbot[0] != 0:
            for j in range(len(df_elm.loc[s_topbot[0], "nodes"])):
                if df_elm.loc[s_topbot[0], "nodes"][j] in set(i[1]):
                    df_elm.loc[s_topbot[0], "nodes"][j] = node_mapping[
                        df_elm.loc[s_topbot[0], "nodes"][j]
                    ]
            # Duplicate nodes in contact with active crack i belonging to surfaces attached to the solid element s_topbot[0] that are NOT the active crack
            for ii in el_s:
                if not op(active_crack_ID, df_elm.loc[ii, "tags"][0]):
                    for j in range(len(df_elm.loc[ii, "nodes"])):
                        if df_elm.loc[ii, "nodes"][j] in set(i[1]):
                            df_elm.loc[ii, "nodes"][j] = node_mapping[
                                df_elm.loc[ii, "nodes"][j]
                            ]
                # Update surface partition tags if necessary
                if (
                    len(df_elm.loc[s_topbot[0], "tags"]) > 2
                    and df_elm.loc[ii, "tags"][0] == all_cracks_ID
                ):
                    dnid = set([node_mapping[dn] for dn in df_elm.loc[i[0], "nodes"]])
                    if dnid == set(df_elm.loc[ii, "nodes"]):
                        part2surf(df_elm, ii, i[0], s_topbot)

            # Duplicate nodes of neighbouring solid elements in contact with active crack i but NOT attached to the crack plane (i.e. NOT attached to any active crack)
            s_neigh = srf_neigh(
                df_nod, df_elm, i[0], nrm_dict[i[0]], dic_s2s, node_mapping
            )
            if not s_neigh:
                continue
            # Discard solid elements attached to other active cracks (outside of active crack i) before proceeding to duplicate nodes
            for sn in s_neigh:
                proceed = True
                s_sn = surf2vol(df_elm, sn)
                for s in s_sn:
                    s_set = set(df_elm.loc[s, "nodes"])
                    if set.intersection(s_set, set(crack_nodes)) == s_set:
                        proceed = False
                        break
                if proceed:
                    for j in range(len(df_elm.loc[sn, "nodes"])):
                        if df_elm.loc[sn, "nodes"][j] in set(i[1]):
                            df_elm.loc[sn, "nodes"][j] = node_mapping[
                                df_elm.loc[sn, "nodes"][j]
                            ]
                    # Duplicate nodes on surfaces attached to solid element sn as well, if these share nodes with active crack i:
                    for s in s_sn:
                        for j in range(len(df_elm.loc[s, "nodes"])):
                            if df_elm.loc[s, "nodes"][j] in set(i[1]):
                                df_elm.loc[s, "nodes"][j] = node_mapping[
                                    df_elm.loc[s, "nodes"][j]
                                ]
        else:
            print("Active crack physical tag = {0}".format(i[2][0]))
            print(
                "Surface element {0} of type {1} is missing a solid element on top\n".format(
                    i[0], i[3]
                )
            )

    # Create a new surface element per active crack to superimpose over the old surface element
    # Both the old and new surface elements constitute a new interface element (in16 or in12)
    newID = len(df_elm)
    for i in df_crack_el.itertuples():
        n = df_elm.loc[
            i[0], "nodes"
        ]  # NOT i[0] as node order may have changed to maintain a consistent top/bottom
        newID += 1
        eltp = i[3]  # May be 16 (8-noded quad) or 9 (6-noded triangle)
        tags = i[2]
        # Add to the element DataFrame of the entire mesh
        df_elm.loc[newID] = [n, [all_cracks_ID] + tags[1:], eltp]

    # Add new nodes with overlapping coordinates to the DataFrame with all nodes in the mesh
    for old, new in node_mapping.items():
        df_nod.loc[new] = [df_nod.loc[old, "coords"]]

    # Update node lists of solid elements belonging to dic_s2s.values()
    for ke in dic_s2s.keys():
        df_elm.at[dic_s2s[ke], "nodes"] = df_elm.loc[ke, "nodes"]


def df2mesh(fname, df_phe, df_nod, df_elm):
    """
    Credit to:
    https://github.com/tjolsen/Mesh_Utilities/blob/master/gmsh_crack/gmsh_crack.py
    """
    f = open(fname, "w")

    # Write header
    f.write("$MeshFormat\n")
    f.write("2.2 0 8\n")
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

    f.close()


def main():

    crack_ids = -1
    file_name = ""

    # Argument sys[1] = filename
    file_name = sys.argv[1]

    # Argument sys[2] = crack physical tags, with some pairs being possibly coupled if they belong to the same crack plane (e.g. [1,[2,3],4])
    crack_ids = []
    in_list = []
    for i in sys.argv[2][1:-1].split(","):
        if i.startswith("["):
            in_list.append(int(i.strip("[")))
        elif i.endswith("]"):
            in_list.append(int(i.strip("]")))
            crack_ids.append(in_list)
            in_list = []
        elif len(in_list) != 0:
            in_list.append(int(i))
        else:
            crack_ids.append(int(i))

    if crack_ids == -1:
        print("Error: Must specify crack_ids")
        sys.exit(1)
    if not file_name:
        print("Error: Must specify mesh file")
        sys.exit(1)

    df_phe, df_nod, df_elm = mesh2df(file_name)
    dic_sld2sld = solid2solid(df_elm)

    dic_srf2srf = preproc_df(df_elm, crack_ids)

    for c in crack_ids[:-1]:
        make_crack(df_nod, df_elm, c, dic_sld2sld, dic_srf2srf, crack_ids[-1])

    df2mesh("cracked_" + file_name, df_phe, df_nod, df_elm)


if __name__ == "__main__":
    main()
