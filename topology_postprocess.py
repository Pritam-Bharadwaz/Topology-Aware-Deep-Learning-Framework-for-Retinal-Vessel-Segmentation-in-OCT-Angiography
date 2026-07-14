"""
Topology-aware Post-processing for Vessel Segmentation

Cleaned version for integration into your project.

Usage:
    from topology_postprocess import full_pipeline

    cleaned_mask, flagged_points = full_pipeline(binary_mask)

binary_mask should be a NumPy array with values 0/1 or 0/255.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import (
    skeletonize,
    remove_small_objects,
    binary_closing,
    disk,
)
from skimage.measure import label


def _to_binary(mask):
    mask = mask > 0
    return mask.astype(np.uint8)


def remove_false_positives(mask, min_size=30):
    mask = _to_binary(mask)
    cleaned = remove_small_objects(mask.astype(bool),
                                   min_size=min_size,
                                   connectivity=2)
    return cleaned.astype(np.uint8)


def _find_endpoints(skel):
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    conv = ndi.convolve(skel.astype(int), kernel)
    ys, xs = np.where((skel) & (conv == 11))
    return list(zip(ys.tolist(), xs.tolist()))


def _pixel_degree_map(skel):
    kernel = np.ones((3,3), dtype=int)
    kernel[1,1] = 0
    return ndi.convolve(skel.astype(int), kernel)


def _line(y1,x1,y2,x2):
    n = int(max(abs(y2-y1), abs(x2-x1))) + 1
    yy = np.linspace(y1,y2,n).round().astype(int)
    xx = np.linspace(x1,x2,n).round().astype(int)
    return yy, xx


def bridge_small_gaps(mask, max_gap=5):
    mask = _to_binary(mask)
    skel = skeletonize(mask.astype(bool))
    endpoints = _find_endpoints(skel)

    labeled = label(mask, connectivity=2)
    result = mask.copy()
    used = set()

    for i,(y1,x1) in enumerate(endpoints):
        if i in used:
            continue

        comp1 = labeled[y1,x1]
        best = None
        best_d = max_gap + 1

        for j,(y2,x2) in enumerate(endpoints):
            if i == j or j in used:
                continue

            if labeled[y2,x2] == comp1:
                continue

            d = np.hypot(y1-y2, x1-x2)

            if d < best_d:
                best_d = d
                best = j

        if best is not None:
            y2,x2 = endpoints[best]
            rr,cc = _line(y1,x1,y2,x2)
            result[rr,cc] = 1
            used.add(i)
            used.add(best)

    return result


def _trace_branch(skel,y,x,max_len):
    path=[(y,x)]
    visited={(y,x)}
    cy,cx=y,x

    for _ in range(max_len):
        neigh=[(cy+dy,cx+dx)
               for dy in (-1,0,1)
               for dx in (-1,0,1)
               if (dy,dx)!=(0,0)
               and 0<=cy+dy<skel.shape[0]
               and 0<=cx+dx<skel.shape[1]
               and skel[cy+dy,cx+dx]
               and (cy+dy,cx+dx) not in visited]

        if len(neigh)!=1:
            break

        cy,cx=neigh[0]
        path.append((cy,cx))
        visited.add((cy,cx))

    return len(path),path


def _reconstruct(pruned, original):
    thick = binary_closing(pruned, disk(2))
    return ((thick & original.astype(bool)) | pruned).astype(np.uint8)


def prune_spurs(mask, min_branch_len=8):
    mask = _to_binary(mask)
    skel = skeletonize(mask.astype(bool)).copy()

    changed=True

    while changed:
        changed=False
        endpoints=_find_endpoints(skel)

        for y,x in endpoints:
            length,path=_trace_branch(skel,y,x,min_branch_len+1)

            if length<=min_branch_len:
                py,px=path[-1]
                if _pixel_degree_map(skel)[py,px]>=3:
                    for yy,xx in path:
                        skel[yy,xx]=False
                    changed=True

    return _reconstruct(skel, mask)


def detect_high_degree_nodes(mask, degree_threshold=4):
    mask = _to_binary(mask)
    skel = skeletonize(mask.astype(bool))
    degree = _pixel_degree_map(skel)
    ys,xs = np.where((skel) & (degree>=degree_threshold))
    return list(zip(ys.tolist(), xs.tolist()))


def full_pipeline(mask,
                  min_fp_size=30,
                  max_gap=5,
                  min_branch_len=8):

    step1 = remove_false_positives(mask, min_fp_size)
    step2 = prune_spurs(step1, min_branch_len)
    step3 = bridge_small_gaps(step2, max_gap)
    flagged = detect_high_degree_nodes(step3)

    return step3.astype(np.uint8), flagged
