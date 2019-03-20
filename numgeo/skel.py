"""
Framework and algorithm to derive linear skeletons from polygons.
"""

# Copyright 2019 Ethan I. Schaefer
#
# This file is part of numgeo.
#
# numgeo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# numgeo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with numgeo.  If not, see <https://www.gnu.org/licenses/>.

__version__ = "0.0.1a0"
__author__ = "Ethan I. Schaefer"


###############################################################################
# IMPORT                                                                      #
###############################################################################

# Import internal (intra-package).
from numgeo import geom as _geom
from numgeo import util as _util

# Import external.
try:
    import bottleneck as _bottleneck
except ImportError:
    _bottleneck = None
import copy as _copy
import collections as _collections
import itertools as _itertools
import math as _math
import numpy as _numpy
import operator as _operator
import sys as _sys
try:
    import psutil as _psutil
except ImportError:
    _psutil = None
import scipy as _scipy
import scipy.spatial  # Make _scipy.spatial available.
import scipy.sparse.csgraph  # Make _scipy.sparse.csgraph available.
del scipy  # Hide scipy.
import warnings as _warnings
import weakref as _weakref



###############################################################################
# LOCALIZATION                                                                #
###############################################################################

# Derived from built-ins.
_python_inf = float("inf")
_python_nan = float("nan")
_python_neginf = -_python_inf
_empty_tuple = ()
_marker_false = type("_marker", _empty_tuple,
                     {"__nonzero__": lambda self: False})()
_neg_1_2_tuple = (-1, 2)
_neg_1_3_tuple = (-1, 3)

# Derived from internal.
_dict_to_set = _util._dict_to_set
_is_subset = _util._is_subset
_LineString = _geom.LineString
_MultiPoint2D = _geom.MultiPoint2D
_slide_pairwise = _util.slide_pairwise
_take2 = _util._take2
_triu_indices_flat = _util._triu_indices_flat
_union_flat_arrays = _util._union_flat_arrays
_uniquify_flat_array = _util._uniquify_flat_array
_validate_string_option = _util.validate_string_option

# Derived from external.
if _bottleneck is None:
    def _bottleneck_move_mean(a, window):
        # Note: Mimic bottleneck.move_mean()'s leading nan.
        out = _numpy_empty((len(a),), dtype=_numpy_float64)
        out[0] = _numpy.nan
        _util._slide_flat(a).mean(1, out=out[1:])
        return out
    _bottleneck_nanargmax = _numpy.nanargmax
    _bottleneck_nanmin = _numpy.nanargmin
    def _bottleneck_replace(a, old, new):
        if _math_isnan(old):
            a[_numpy.isnan(old)] = new
        else:
            a[a==old] = new
        return a
else:
    _bottleneck_move_mean = _bottleneck.move_mean
    _bottleneck_nanargmax = _bottleneck.nanargmax
    _bottleneck_nanmin = _bottleneck.nanmin
    _bottleneck_replace = _bottleneck.replace
_deque = _collections.deque
_defaultdict = _collections.defaultdict
_deepcopy = _copy.deepcopy
_flatten_to_iter = _itertools.chain.from_iterable
_get_zero = _itertools.repeat(0).next
_imap = _itertools.imap
_izip = _itertools.izip
_math_isnan = _math.isnan
_numpy_abs = _numpy.abs
_numpy_add = _numpy.add
_numpy_bincount = _numpy.bincount
_numpy_concatenate = _numpy.concatenate
_numpy_divide = _numpy.divide
_numpy_equal = _numpy.equal
_numpy_fromiter = _numpy.fromiter
_numpy_bool8 = _numpy.dtype("<?")
_numpy_inf = _numpy.inf
_numpy_int8 = _numpy.dtype("<i1")
_numpy_int32 = _numpy.dtype("<i4")
_numpy_int64 = _numpy.dtype("<i8")
_numpy_isfinite = _numpy.isfinite
_numpy_empty = _numpy.empty
_numpy_float64 = _numpy.dtype("<f8")
_numpy_float64_big_endian = _numpy.dtype(">f8")
_numpy_logical_and = _numpy.logical_and
_numpy_logical_or = _numpy.logical_or
_numpy_min = _numpy.min
_numpy_multiply = _numpy.multiply
_numpy_nearly_inf = _numpy.nextafter(_numpy_inf, 0.)
_numpy_nearly_neginf = _numpy.nextafter(-_numpy_inf, 0.)
_numpy_negative = _numpy.negative
_numpy_neginf = -_numpy.inf
_numpy_ndarray = _numpy.ndarray
_numpy_not_equal = _numpy.not_equal
_numpy_ones = _numpy.ones
_numpy_sqrt = _numpy.sqrt
_numpy_square = _numpy.square
_numpy_subtract = _numpy.subtract
_numpy_zeros = _numpy.zeros



###############################################################################
# UTILITY CLASSES                                                             #
###############################################################################

class _Components(_util.Lazy):
    """
    Class that facilitates isolation of the skeleton from its complement.
    """
    _bools_array = None

    def __init__(self, skeleton, adj_dict):
        """
        Create object to aid in isolating the skeleton from its complement.

        skeleton is a Skeleton instance that specifies the skeleton whose
        isolation will be facilitated. Specifically, the current function:
            1) Contracts through binodes.
            2) Re-indexes the Voronoi vertex indices so that they have a minimal
               range and start at 0.
            3) Identifies and labels with a unique integer each connected
               component (subgraph).

        adj_dict is a dict that specifies the adjacency dictionary for skeleton.

        Attributes and otherwise undocumented methods:
            .get_reidx_for_idx(idx)
                If passed a Voronoi vertex index (or indices), returns the
                equivalent re-index (or re-indices, as an array).
            .component_labels_array
                An array that maps each re-index (by positional index) to its
                corresponding component (integer) label (by value at that
                positional index).
            .reidx_to_idx_array
                An array that maps each re-index (by positional index) to its
                corresponding Voronoi vertex index (by value at that positional
                index).
        """
        skeleton._contract_through_binodes(adj_dict)
        csr_matrix, self.reidx_to_idx_array = skeleton._make_csr_matrix(
            skeleton._graph_edge_dict, None
            )
        _, self.component_labels_array = _scipy.sparse.csgraph.connected_components(
            csr_matrix, True, "weak", True
            )
        self.get_reidx_for_idx = self.reidx_to_idx_array.searchsorted

    def get_component_label(self, vor_vert_idx):
        """
        Get (integer) component label(s) for one or more Voronoi vertex indices.

        If vor_vert_idx is an integer (sequence), an integer (array) is
        returned.

        vor_vert_idx is an integer or sequence of integers that specify the
        Voronoi vertex indices for which component labels should be returned.
        """
        return self.component_labels_array[self.get_reidx_for_idx(vor_vert_idx)]

    @staticmethod
    def _get_frequency_sorted_unique_component_labels_array(self):
        """
        Flat array of component labels in order of decreasing frequency.

        For example, the first integer in the returned array is the label for
        that component with the most Voronoi vertex indices.
        """
        return _numpy_bincount(self.component_labels_array).argsort()[::-1]

    @staticmethod
    def _get_unique_component_labels_array(self):
        """
        Flat array in which each (integer) component label occurs once.
        """
        return _util._uniquify_flat_array(self.component_labels_array)

    def get_reidxs_for_component(self, component_label):
        """
        Get a flat array of those re-indices that belong to a component.

        component_label is an integer that specifies the label for the targeted
        component.
        """
        self._bools_array = _numpy_equal(self.component_labels_array,
                                         component_label,
                                         self._bools_array)
        reidxs_so_labeled_array, = self._bools_array.nonzero()
        return reidxs_so_labeled_array

    def get_idxs_for_component(self, component_label):
        """
        Get a flat array of Voronoi vertex indices that belong to a component.

        component_label is an integer that specifies the label for the targeted
        component.
        """
        return _take2(self.reidx_to_idx_array,
                      self.get_reidxs_for_component(component_label),
                      True)


class _Frozenset2(frozenset):
    """
    A frozenset-like class with that remembers its initializing sequence.
    """

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self.tuple)

    def __init__(self, seq):
        """
        Create a frozenset-like object that remembers its initializing sequence.

        The new instance will act identical to a frozenset (and will even
        evaluate as equal to an equivalent frozenset), except:
            1) The instance has a .tuple attribute that is equal to tuple(seq):
                   _Frozenset2([0, 0]).tuple --> (0, 0)
            3) .copy() will return a _Frozenset2.
            4) .repr() is modified.
        """
        self.tuple = tuple(seq)

    @staticmethod
    def _flip(self):
        """
        Reverse order of .tuple.
        """
        i, j = self.tuple
        self.tuple = (j, i)
        return self

    def copy(self):
        """
        Copy self, preserving order.
        """
        return type(self)(self.tuple)


class _NeverEqualTuple(tuple):
    """
    A tuple-like class whose instances only compare equal if identical.
    """

    __eq__ = object.__eq__
    __hash__ = lambda self: id(self)



###############################################################################
# SKELETAL CLASSES                                                            #
###############################################################################

class Skeletal(object):
    """
    Base type of all edges and paths.
    """
    __flip_dict = {
        None: ("area", "length3D", "segment_count", "skeleton",
               "spatial_reference", "stem_width", "untailed_length2D",
               "vertex_count"),
        lambda a: a[::-1]: ("coords_array3D", "segment_widths",
                            "_tail_coords_arrays", "vor_vert_idxs_array"),
        _Frozenset2._flip: ("_aligned_key",),
        lambda n: -n: ("delta_z",)
        }
    # Note: Using _empty_tuple instead of None permits .flip() to work
    # without extra code.
    _tail_coords_arrays = _empty_tuple

    def __init__(self, skeleton, vor_vert_idxs_array):
        """
        Create a LineString-like instance with special attributes.

        skeleton is the Skeleton instance with which the LineString-like
        instance is associated.

        vor_vert_idxs_array is a flat integer array that specifies the Voronoi
        vertex indices that are the vertices of the LineString-like instance.
        """
        self.skeleton = skeleton
        self.vor_vert_idxs_array = vor_vert_idxs_array

    @staticmethod
    def _get__aligned_key(self, pre_tail=None):
        """
        Frozenset of end Voronoi vertex indices whose .tuple is aligned.

        That is,
            (self.vor_vert_idxs_array[0],
             self.vor_vert_idxs_array[-1]) == self._aligned_key.tuple --> True
        """
        if pre_tail is not None:
            return pre_tail
        return _Frozenset2(
            self.vor_vert_idxs_array[::len(self.vor_vert_idxs_array) - 1].tolist()
            )

    @staticmethod
    def _get_area(self, pre_tail=None):
        """
        Approximate area of the subset of the input polygon represented by self.

        This approximation excludes the area around the tail, which is typically
        smaller than interval**2. Additionally, wherever the input polygon
        narrows to a width smaller than interval, the area may be poorly
        approximated.
        """
        if pre_tail is not None:
            return pre_tail
        segment_lengths = self.segment_lengths
        # Remove tail segments, if necessary.
        if self._tail_coords_arrays:
            (start_tail_coords_array,
             end_tail_coords_array) = self._tail_coords_arrays
            if start_tail_coords_array is not None:
                segment_lengths = segment_lengths[1:]  # *REASSIGNMENT*
            if end_tail_coords_array is not None:
                segment_lengths = segment_lengths[:1]  # *REASSIGNMENT*
        return (segment_lengths * self.segment_widths).sum()

    def _fetch_out_array(self, column_count):
        """
        Get an array whose base has an extra row at top and bottom.

        The current method is used to create arrays that may later be expanded
        to include tail data.

        column_count is an integer that specifies the number of column that the
        returned array should have.
        """
        shape_list = [len(self.vor_vert_idxs_array) + 2]
        if column_count is not None:
            shape_list.append(column_count)
        base_array = _numpy_empty(tuple(shape_list), _numpy_float64)
        return base_array[1:-1]

    # Note: This method is not named ._get_coords_array() for reasons
    # explained in numgeo.geom.Geometry._process_aray().
    @staticmethod
    def _get__arg0(self, pre_tail=None):
        if pre_tail is None:
            # *REASSIGNMENT*
            pre_tail = (_take2(self.skeleton.voronoi_vertex_coords_array,
                               self.vor_vert_idxs_array,
                               out=self._fetch_out_array(2)),
                        False)
        if not self._tail_coords_arrays:
            return pre_tail
        (start_tail_coords_array,
         end_tail_coords_array) = self._tail_coords_arrays
        coords_array, _ = pre_tail
        base = coords_array.base
        if start_tail_coords_array is None:
            base = base[1:]  # *REASSIGNMENT*
        else:
            base[0] = start_tail_coords_array[:2]
        if end_tail_coords_array is None:
            base = base[:-1]  # *REASSIGNMENT*
        else:
            base[-1] = end_tail_coords_array[:2]
        return (base, False)

    @staticmethod
    def _get_coords_array3D(self, pre_tail=None):
        """
        Vertex coordinates, including interpolated z-coordintes.

        Each vertex of self is also a Voronoi vertex. Consequently, except in
        cases of extreme local symmetry, exactly 3 samples will be nearest each
        vertex, and all of these will be equidistant from that vertex. The z-
        coordinate interpolated for each vertex is simply the mean of the
        respective z-coordinates of these 3 samples. (If more than 3 such
        samples exist, an arbitrary set of 3 is chosen from among them.)

        Warning: The original input polygon must be 3D.

        Warning: The interpolated z-coordinates are approximate, and they may be
        especially crude (relative to reality) if the terrain across the input
        polygon's width is not approximately planar locally.

        Warning: This is a relatively expensive operation. Use .delta_z
        instead if it is sufficient.
        """
        if pre_tail is None:
            # *REASSIGNMENT*
            pre_tail = self._get_delta_z(self, "coords_array3D")
        if not self._tail_coords_arrays:
            return pre_tail
        (start_tail_coords_array,
         end_tail_coords_array) = self._tail_coords_arrays
        base = pre_tail.base
        if start_tail_coords_array is None:
            base = base[1:]  # *REASSIGNMENT*
        else:
            base[0] = start_tail_coords_array
        if end_tail_coords_array is None:
            base = base[:-1]  # *REASSIGNMENT*
        else:
            base[-1] = end_tail_coords_array
        # Release memory claimed by .coords_array2D.
        self.coords_array2D = base[:,:2]
        return base

    @staticmethod
    def _get_delta_z(self, name="delta_z", pre_tail=None):
        """
        Approximate total change in the z-coordinate from start to end node. A
        negative value indicates a downslope orientation.

        Note: See .coords_array3D documentation for more details.
        """
        # If .coords_array3D was already generated (and to support
        # the special case in which the user manually drapes the
        # LineString independent of the input polygon), use that
        # attribute.
        if "coords_array3D" in self.__dict__:
            return float(
                self.coords_array3D[-1][2] - self.coords_array3D[0][2]
                )

        # Check that the original input polygon was 3D, and raise an
        # error if not.
        skel = self.skeleton
        sampled_coords_array = skel.sampled_coords_array
        if sampled_coords_array.shape[1] < 3:
            # Polygon was not 3D, so raise standard error.
            object.__getattribute__(self, name)

        # Subset the vertex indices to be used, if appropriate.
        vor_vert_idxs_array = self.vor_vert_idxs_array
        if name == "delta_z":
            # Keep only the end vertex or vertices where no tail was
            # added.
            tail_coords_arrays = self._tail_coords_arrays
            if not tail_coords_arrays:
                # *REASSIGNMENT*
                vor_vert_idxs_array = vor_vert_idxs_array[::len(vor_vert_idxs_array) - 1]
            else:
                (start_tail_coords_array,
                 end_tail_coords_array) = self._tail_coords_arrays
                if start_tail_coords_array is not None:
                    if end_tail_coords_array is not None:
                        # In the special case that a tail has been added to
                        # both ends (e.g., a trunk), return immediately with
                        # the delta z.
                        return float(end_tail_coords_array[2] -
                                     start_tail_coords_array[2])
                    # Tail was added to starting end only.
                    vor_vert_idxs_array = vor_vert_idxs_array[-1:]
                else:
                    # By elimination, tail was added to ending end only.
                    vor_vert_idxs_array = vor_vert_idxs_array[:1]

        # Find the 3 sample indices associated with each vertex (or
        # arbitrarily use the first 3 such samples).
        # Note: Because hyperhub cracking does not directly modify the
        # Voronoi vertex indices registered to (the temporary)
        # ._vor_vert_idx_pairs_array, it is *not* guaranteed that each
        # vertex is associated with exactly 3
        # samples.
        fetch_associated_sample_index = skel._fetch_associated_sample_index
        nested_sample_idxs = [
            tuple(fetch_associated_sample_index(vor_vert_idx, True))[:3]
            for vor_vert_idx in vor_vert_idxs_array.tolist()
            ]
        # Note: The performance of 32-bit integer arrays is similar to
        # or better than that of 64-bit integer arrays for creation +
        # one application of fancy indexing.
        sample_idxs_array = _numpy_fromiter(
            _flatten_to_iter(nested_sample_idxs), _numpy_int32
            )
        del nested_sample_idxs  # Release memory.
        sample_idxs_array.shape = _neg_1_3_tuple

        # Retrieve the respective z-coordinates associated with each
        # vertex, and average them.
        zs_array = _take2(
            sampled_coords_array[:,2], sample_idxs_array,
            out=None if name == "delta_z" else self._fetch_out_array(3)
            )
        del sample_idxs_array  # Release memory.
        # Note: Storing the results of .mean() to the array over
        # which the mean is taken yields erroneous results.
        z_array = zs_array.mean(1)

        # Return the z difference, if appropriate.
        if name == "delta_z":
            if not tail_coords_arrays:
                z0, z1 = z_array.tolist()
            elif start_tail_coords_array is None:
                # Tail was added to ending end only.
                z0, = z_array.tolist()
                z1 = float(end_tail_coords_array[2])
            else:
                # By elimination, tail was added to starting end only
                # (because the special case of tail addition at each end
                # was addressed further above).
                z0 = float(start_tail_coords_array[2])
                z1, = z_array.tolist()
            return z1 - z0

        # Populate and return a coordinate array with x-, y-, and
        # (the just interpolated) z-coordinates, ignoring any tails.
        # Note: ._get_coords_array3D() addresses tail coordinates.
        # Note: Re-use zs_array for efficiency and to accommodate tail
        # addition.
        zs_array[:,2] = z_array
        zs_array.base[:,:2] = self.coords_array.base
        return zs_array

    @staticmethod
    def _get_length3D(self, pre_tail=None):
        """
        Approximate 3D length.

        Note: See .coords_array3D documentation for more details.
        """
        return _geom.LineString3D(self.coords_array3D).length

    @staticmethod
    def _get_segment_count(self, pre_tail=None):
        """
        Segment count, excluding any tails.
        """
        if pre_tail is not None:
            return pre_tail
        return len(self.vor_vert_idxs_array) - 1

    @staticmethod
    def _get_segment_widths(self, pre_tail=None):
        """
        Approximate local width of the input polygon along the length.

        More precisely, the approximated "width" at each segment is double the
        mean distance from each of the two vertices to the respective nearest
        sample (on the input polygon's boundary). As a result, the width vector
        may not be oriented perpendicular to self, especially where the input
        polygon is markedly widening or narrowing.

        Warning: The approximated widths do not include tail segments, so if any
        tails have been added len(self.segment_widths) !=
        len(self.segment_lengths).
        """
        # Note: The current function is optimized under the assumption
        # that the width at the same Voronoi vertex (index) may be
        # calculated many times. For example, this can happen while
        # searching for a trunk with at least a specified mean width.
        ## If the above use-case is ever deprecated, revert to 
        ## optimizing under the assumption that the width at most 
        ## Voronoi vertices will only be calculated once, which is
        ## probably the more common case in practice.
        if pre_tail is not None:
            return pre_tail

        # Find the distance from each vertex to the nearest sample.
        vor_vert_idx_to_sample_dist = self.skeleton._vor_vert_idx_to_sample_dist
        dists_array = _numpy_fromiter(
            [vor_vert_idx_to_sample_dist[vor_vert_idx]
             for vor_vert_idx in self.vor_vert_idxs_array.tolist()],
            _numpy_float64
            )

        # Estimate width at each segment as double the moving mean
        # distance from each vertex to the nearest sample.
        seg_dists_array = _bottleneck_move_mean(dists_array, 2)[1:]
        del dists_array  # Release memory.
        return _numpy_multiply(seg_dists_array, 2., seg_dists_array)

    @staticmethod
    def _get_spatial_reference(self, pre_tail=None):
        """
        Spatial reference well-known text. Inherited from input polygon.
        """
        if pre_tail is not None:
            return pre_tail
        return self.skeleton.polygon.spatial_reference

    @staticmethod
    def _get_stub_coords_array(self, pre_tail=None):
        """
        2D coordinates at stub end of a partitioned-out branch.

        If self was not partitioned out as a branch, an error is raised. If a
        tail was added, its end coordinates are returned.

        Warning: If a tail was added, the definition of "stub" here differs from
        that used elsewhere in the current module.
        """
        # Identify stub if possible (or error).
        # Note: This call will error if self is not a (partitioned out)
        # branch.
        stub_vor_vert_idx = self._fetch_end_vor_vert_idx("stub")

        # If tail has been added, return its coordinates.
        if self._tail_coords_arrays:
            vor_vert_idx0, vor_vert_idxN = self._aligned_key.tuple
            tail_is_after_last_node = stub_vor_vert_idx == vor_vert_idxN
            return self._tail_coords_arrays[0 + tail_is_after_last_node][:2]

        # Otherwise, return non-stem end.
        return self.skeleton.voronoi_vertex_coords_array[stub_vor_vert_idx]

    @staticmethod
    def _get_stem_coords_array(self, pre_tail=None):
        """
        2D coordinates at stem end of a partitioned-out branch or loop.

        If self was not partitioned out as a branch or loop, an error is raised.
        """
        # Stem node does not change with tail-adding.
        if pre_tail is not None:
            return pre_tail

        # Return stem coordinates array (or error if self is not a
        # (partitioned-out) branch or loop.
        return self.skeleton.voronoi_vertex_coords_array[self._fetch_end_vor_vert_idx("stem")]

    def _fetch_end_vor_vert_idx(self, kind):
        """
        Return Voronoi vertex index of the specified kind. Branches, loops only.

        kind is a string that specifies the Voronoi vertex whose index will be
        returned. It should be "stub" or "stem".

        Warning: An AttributeError is raised if self is neither a branch nor
        loop, and also raised for loops if kind is "stub".
        """
        try:
            stem_vor_vert_idx = self._stem_vor_vert_idx
        except AttributeError:
            raise AttributeError(
                "this LineString was not partitioned out as a branch or loop"
                )
        if kind == "stem":
            return stem_vor_vert_idx
        try:
            return self._stub_vor_vert_idx
        except AttributeError:
            raise AttributeError(
                "this LineString was not partitioned out as a branch"
                )

    @staticmethod
    def _get_stem_width(self, pre_tail=None):
        """
        Approximate local width of the input polygon at self's stem end.

        If self was not partitioned out as a branch or loop, an error is raised.
        See .segment_widths for a description of how the width is approximated.
        """
        # If pre-tail stem width was already calculated, return that
        # result.
        if pre_tail is not None:
            return pre_tail

        # Return twice the distance from the stem vertex to the nearest
        # sample (or error if self is not a (partitioned out) branch.
        return (
            2. *
            self.skeleton._vor_vert_idx_to_sample_dist[self._fetch_end_vor_vert_idx("stem")]
            )

    @staticmethod
    def _get_untailed_length2D(self, pre_tail=None):
        """
        Length in 2D, excluding any tails.
        """
        # Note: It is guaranteed that .untailed_length2D is assigned
        # prior to tail addition, so that the simple code here is
        # sufficient.
        if pre_tail is not None:
            return pre_tail
        return self.length

    @staticmethod
    def _get_vertex_count(self, pre_tail=None):
        """
        Vertex count, excluding any tails.
        """
        if pre_tail is not None:
            return pre_tail
        return len(self.vor_vert_idxs_array)

    def add_tails(self, cut=True, fast=False):
        """
        Add one or more tails, if either end is a stub.

        A tail, which extends self to a boundary of the input polygon, can only
        be added to a given end if that end is a stub in the partitioned
        skeleton. If self has two such stubs (e.g., it is a trunk), a tail will
        be sought for each end. A tuple of 5 booleans is returned:
            idx    Meaning
            0      whether tails were added with this call
            1      whether self.coords_array[0] is the end of a tail
            2      whether self.coords_array[-1] is the end of a tail
            3      whether self.coords_array[0] is a stub in the partitioned
                   skeleton
            4      whether self.coords_array[-1] is a stub in the partitioned
                   skeleton
        Any tails are only added to self on the first call of the current
        function, but subsequent calls can be used to describe the results of
        that first call.

        cut is a boolean that specifies whether tails should be sought for stubs
        that were formed by the cutting involved in the "CUT" isolation_mode
        option specified at initialization.

        fast is a boolean that specifies whether, if tails were already added,
        the function should return False instead of a tuple.
        """
        # Honor fast argument.
        if fast and self._tail_coords_arrays is not _empty_tuple:
            return False

        # Determine which end(s) are stubs.
        skel = self.skeleton
        vor_vert_idx_to_partitioned_degree_array = skel._vor_vert_idx_to_partitioned_degree_array
        start_vor_vert_idx, end_vor_vert_idx = self._aligned_key.tuple
        start_is_stub = vor_vert_idx_to_partitioned_degree_array[start_vor_vert_idx] == 1
        end_is_stub = vor_vert_idx_to_partitioned_degree_array[end_vor_vert_idx] == 1

        # Return immediately if tail addition was already attempted.
        if self._tail_coords_arrays is not _empty_tuple:
            tail_coords_arrays = self._tail_coords_arrays
            return (False,
                    (tail_coords_arrays[0] is not None if tail_coords_arrays
                     else False),
                    (tail_coords_arrays[1] is not None if tail_coords_arrays
                     else False),
                    start_is_stub,
                    end_is_stub)

        # Ensure that .untailed_length2D is retrievable after tail
        # addition.
        self.untailed_length2D

        # Prepare for finding tails.
        self._tail_coords_arrays = tail_coords_arrays = []
        fetch_associated_sample_index = skel._fetch_associated_sample_index
        polygon_boundary = skel.polygon.boundary
        polygon_has_a_hole = len(polygon_boundary) > 1
        sampled_coords_array = skel.sampled_coords_array
        if polygon_has_a_hole:
            get_ring_idx_from_sample_idx = skel._get_ring_idx_from_sample_idx
            last_sample_idx_each_ring_array = skel._last_sample_idx_each_ring_array
        else:
            # Note: Invariant values.
            ring, = polygon_boundary
            first_samp_idx_this_ring = 0
            samp_count_this_ring = len(sampled_coords_array) + 1
        tolerance = 2 + bool(cut)
        interval = skel.interval

        # Find tail for each stub end, as applicable.
        added_tail_count = 0  # Initialize.
        for is_stub, vor_vert_idx in ((start_is_stub, start_vor_vert_idx),
                                      (end_is_stub, end_vor_vert_idx)):
            try:
                assert is_stub

                # Find the minimum and maximum associated sample
                # indices.
                samp_idxs_set = fetch_associated_sample_index(vor_vert_idx,
                                                              True)
                min_samp_idx = min(samp_idxs_set)
                max_samp_idx = max(samp_idxs_set)

                # If input polygon has holes, check that both samples
                # come from the same ring.
                if polygon_has_a_hole:
                    min_ring_idx = get_ring_idx_from_sample_idx(min_samp_idx)
                    max_ring_idx = get_ring_idx_from_sample_idx(max_samp_idx)
                    assert min_ring_idx == max_ring_idx
                    ring = polygon_boundary[min_ring_idx]
                    if min_ring_idx:
                        first_samp_idx_this_ring = last_sample_idx_each_ring_array[min_ring_idx - 1] + 1
                    else:
                        first_samp_idx_this_ring = 0
                    samp_count_this_ring = (
                        last_sample_idx_each_ring_array[min_ring_idx]
                        - first_samp_idx_this_ring
                        + 1)

                # If the difference between the minimum and maximum
                # associated sampled indices is greater than allowed,
                # check whether "rotating" the indices resolves this
                # issue.
                # Note: For example, consider an input polygon that has
                # only one ring, which has 100 samples. If the stub is
                # associated with samples at indices 1 and 99, the
                # apparent index difference is large but, in fact, the
                # sample at index 0 neighbors both of these samples. The
                # code below would convert these associated sample
                # indices to 1 (unchanged) and -1, respectively.
                if max_samp_idx - min_samp_idx > tolerance:
                    rotated_samp_idxs = [samp_idx - samp_count_this_ring
                                         if samp_idx - min_samp_idx > tolerance
                                         else samp_idx
                                         for samp_idx in samp_idxs_set]
                    # *REASSIGNMENT*
                    min_samp_idx = min(rotated_samp_idxs)
                    # *REASSIGNMENT*
                    max_samp_idx = max(rotated_samp_idxs)
                    assert max_samp_idx - min_samp_idx <= tolerance

                # Set tail end.
                # Note: In essence, the number of samples lying between
                # the two furthest separated associated samples is
                # counted. If that number is odd, the middle sample is
                # used as the tail end. Otherwise a new coordinate is
                # interpolated along the boundary. Usually that
                # coordinate lies midway between the two central samples
                # from the span between the two furthest separated
                # associated samples, but in the special case that those
                # two central samples bracket the closing vertex of the
                # ring, it is merely guaranteed that the tail end is
                # within the sampling interval of the midpoint between
                # those two central samples.
                one_float_array = _numpy_empty((1,))
                if (max_samp_idx - min_samp_idx) % 2:
                    dist_along_ring = (0.5*(min_samp_idx + max_samp_idx)
                                       - first_samp_idx_this_ring) * interval
                    if dist_along_ring < 0.:
                        # Note: Account for index rotation.
                        dist_along_ring += ring.length
                    one_float_array[0] = dist_along_ring
                    tail_coords_arrays.append(
                        ring.interpolate(one_float_array)[0]
                        )
                else:
                    targ_samp_idx = (min_samp_idx + max_samp_idx) // 2
                    if targ_samp_idx < first_samp_idx_this_ring:
                        # Ex: min_samp_idx is -2 and max_samp_idx is 0,
                        # so targ_samp_idx is initially -1.
                        targ_samp_idx += samp_count_this_ring
                    tail_coords_arrays.append(
                        sampled_coords_array[targ_samp_idx]
                        )
                added_tail_count += 1

            # If any requirements failed, register no tail.
            # Note: Namely, end may not be a stub, or its associated
            # sample coordinates may come from different rings, or the
            # number of samples separating those two coordinates may be
            # greater than tolerance.
            except AssertionError:
                tail_coords_arrays.append(None)
                continue

        # If no tails were added, return immediately.
        if not added_tail_count:
            # Note: In many places in the code, a line without any tails
            # should be treated the same whether or not tail addition
            # has been attempted. Therefore, empty containers are used
            # in both cases, for quick testing, but an empty list is
            # used if tail addition was attempted whereas an empty tuple
            # is used otherwise, to permit discrimination between the
            # two cases.
            self._tail_coords_arrays = []
            return (False, False, False, start_is_stub, end_is_stub)

        # Update any Skeletal-level lazy attributes that were previously
        # generated without tails, and clear all other lazy attributes
        # (as they may need to be updated now that tails have been
        # added).
        pre_tail_self_dict = self.__dict__.copy()
        self._Lazy__clear_lazy()
        # Note: Because ._get__arg0() (rather than ._get_coords_array())
        # is defined, manually clear .coords_array(2D) to force re-
        # initialization.
        if "coords_array" in pre_tail_self_dict:
            delattr(self, "coords_array")
            if "coords_array2D" in pre_tail_self_dict:
                delattr(self, "coords_array2D")
        # Note: There is a slight optimization in that if both
        # .coords_array3D and .delta_z were generated without tails,
        # .coords_array3D will be re-generated with tails and then
        # .delta_z derived therefrom, since lazy attribute names are
        # iterated over in alphabetical order.
        for lazy_attr_name in _util.Lazy.__dir__.__func__(Skeletal, True, True):
            if lazy_attr_name not in pre_tail_self_dict:
                continue
            setattr(self, lazy_attr_name,
                    getattr(self, "_get_" + lazy_attr_name)(
                        self, pre_tail=pre_tail_self_dict[lazy_attr_name]))

        # Return.
        return (True,
                tail_coords_arrays[0] is not None,
                tail_coords_arrays[1] is not None,
                start_is_stub,
                end_is_stub)

    # Note: .normalized_length is defined as a property because length
    # can change with tail addition.
    @property
    def normalized_length(self):
        """
        Calculate 2. * self.length / self.stem_width.

        If self was not partitioned out as a branch or loop, None is returned.
        See discussion in documentation for .prune().
        """
        try:
            return 2. * self.length / self.stem_width
        except AttributeError:
            return None

    @property
    def description(self):
        """
        Description. Equivalent to self.skeleton.describe_line(self, False).
        """
        return self.skeleton.describe_edge(self, False)


class SkeletalLineString2D(Skeletal, _geom.LineString2D):
    pass

class SkeletalLineSegment2D(Skeletal, _geom.LineSegment2D):
    pass



###############################################################################
# VORONOI CLASSES                                                             #
###############################################################################

class Voronoi2D(_util.Lazy):

    def __init__(self, coords, merge_option="IGNORE", translate="MEAN",
                 localize=True, furthest_site=False, qhull_options=None):
        """
        Generate a 2D Voronoi diagram for the specified point coordinates.

        In 2D, Voronoi analysis identifies a polygonal neighborhood about each
        input coordinate that includes all coordinates closer to that input
        coordinate than to any other input coordinate. Consequently, coordinates
        along the boundary of a neighborhood are equidistant from two input
        coordinates, and the vertices of the polygonal neighborhoods are
        equidistant from three (or more) input coordinates. As a simple example,
        the Voronoi diagram for a grid of equally spaced points is a grid of
        squares, with each square centered on one of the input points.

        coords is an array of shape (-1, 2) or a MultiPoint2D that specifies the
        point coordinates for which the Voronoi diagram should be generated. If
        coords is a MultiPoint2D for which a .spatial_reference is assigned, the
        results will also have that .spatial_reference.

        merge_option is a string that specifies the behavior when two or more
        input coordinates are so close that they must be merged. If
        merge_option is "IGNORE", no action is triggered by such merging. If
        merge_option is instead "ERROR", an error is raised in the event of such
        merging.

        translate is a string that specifies how coords, or a copy thereof,
        should be translated prior to Voronoi analysis. Translation to lower
        magnitude values (i.e., to a position near the origin) improves
        precision and can avoid merging. The following options are supported
        (where each code snippet assumes coords is an array):
            None:
                Do not translate.
            "FIRST":
                Translate the copy of coords so that the first coordinate
                (coords_copy[0]) is at (0, 0).
            "MEAN":
                Translate the copy of coords so that the mean coordinate
                (coords_copy.mean(0)) is at (0, 0).
            "MEDIAN":
                Translate the copy of coords so that the median coordinate
                (numpy.percentile(coords_copy, 50., 0)) is at (0, 0).
            "MIN":
                Translate the copy of coords so that the minimum coordinate
                (coords_copy.min(0)) is at (0, 0).
        In addition, "_NO_COPY" may be appended to any of the above options to
        avoid copying coords prior to translation. In that case, coords is
        translated approximately back to its original values before the function
        returns. The similarity of these restored values to their original
        counterparts depends on the vagaries of floating point arithmetic. More
        generally, if any translation is applied (i.e., translate is not None),
        the results for the translated (copy of) coords are similarly translated
        back before the function returns.

        localize is a boolean that specifies whether four carefully chosen
        coordinates should be added to those specified by coords. In the Voronoi
        diagram, the most exterior points in coords have infinite neighborhoods,
        because no matter how far one goes out, one finds a coordinate closer to
        one of those exterior points than to any other point in coords. If
        localize is True, the four points added to the diagram are each the most
        exterior in that direction, and therefore ensure that the most exterior
        *finite* neighborhood boundaries circumscribe the original coords,
        thereby localizing the Voronoi diagram. Regardless of whether "_NO_COPY"
        is appended to the translate argument, these points are never added in
        places to coords.

        See scipy.spatial.Voronoi for documentation of the furthest_site and
        qhull_options arguments.

        Note: Specifying "Q7" for qhull_options may improve performance.

        Attributes:
            .in_coords_array
                Same as coords unless localize is True, in which case it also
                includes the four added coordinates.
            .out_coords_array
                All the unique vertices of finite Voronoi neighborhoods.
            .used_in_indices
                None if no merging occurred. Otherwise is a flat array of those
                row indices in coords that were not merged (though neighboring
                points may have been merged into them).
            .voronoi
                The scipy.spatial.Voronoi used to generated the diagram.
        """

        # Initially process input coordinates. If input coordinates are
        # specified by a Geometry, store its spatial reference, if any,
        # for later use.
        if isinstance(coords, _geom.Geometry):
            if hasattr(coords, "spatial_reference"):
                self.spatial_reference = coords.spatial_reference
            ## Modify code when explode to points is implemented.
            # Note: .coords_array is not defined for all Geometry's,
            # which is why the documentation only states support for
            # MultiPoint2D's.
            coords = coords.coords_array  # *REASSIGNMENT*
        in_multipoint = _MultiPoint2D(coords)
        del coords  # Prime for potential later release of memory.
        in_coords_array = in_multipoint.coords_array

        # Validate merge option.
        # *REASSIGNMENT*
        merge_option = _validate_string_option(
            merge_option, "merge_option", ("IGNORE", "ERROR")
            )

        # Translate input coordinates, if necessary.
        e_padding = None  # This is the default.
        if translate is not None:

            # Validate translate value and find the implied shift.
            # *REASSIGNMENT*
            translate = translate.upper().replace(" ", "_")
            copy = translate[-8:] != "_NO_COPY"
            if not copy:
                translate = translate[:-8]  # *REASSIGNMENT*
            if translate == "FIRST":
                neg_deltas = in_coords_array[0].tolist()
            elif translate == "MEAN":
                neg_deltas = in_coords_array.mean(0).tolist()
            elif translate == "MEDIAN":
                neg_deltas = _numpy.percentile(in_coords_array, 50., 0).tolist()
            elif translate == "MIN":
                neg_deltas = in_coords_array.min(0).tolist()
            else:
                _validate_string_option(
                    translate, "translate", ("FIRST", "MEAN", "MEDIAN", "MIN")
                    )

            # Copy input coordinates array if requested.
            if copy:
                orig_coords_array = in_coords_array
                if localize:
                    # Note: Because localizing coordinates will be added
                    # later, it is more efficient to create an oversized
                    # copy of the input coordinates now, with a padded
                    # area that can fit the localizing coordinates. The
                    # MultiPoint2D is then re-created further below
                    # after those localizing coordinates are generated
                    # and inserted.
                    e = _numpy_empty((len(in_coords_array) + 4, 2),
                                     _numpy_float64)
                    # *REASSIGNMENT*
                    in_coords_array = e_main = e[:-4]
                    e_padding = e[-4:]
                    in_coords_array[:] = in_multipoint.coords_array
                else:
                    # *REASSIGNMENT*
                    in_coords_array = in_coords_array.copy()
                # *REASSIGNMENT*
                in_multipoint = _MultiPoint2D.make_fast(in_coords_array, False)

            # Perform translation.
            in_multipoint.translate(neg_deltas, True)

        # Optionally add localizing coordinates.
        if localize:

            # Calculate localizing coordinates.
            # Note: Although the approach below does not optimally
            # localize the Voronoi diagram to the minimum possible area,
            # it does guarantee (based on numerical analysis) that the
            # essential portion of the Voronoi diagram is reasonably
            # well localized. Namely, in the worst case scenario, x_span
            # and y_span are equal and an input coordinate lies at a
            # corner of the input coordinates' bounding box. In that
            # case, the localized portion of the Voronoi diagram extends
            # ~1.71 * x_span from that corner.
            min_x, min_y, max_x, max_y = in_multipoint.envelope_coords
            x_span = max_x - min_x
            y_span = max_y - min_y
            offset = 1.15 * max(x_span, y_span)
            mid_x = min_x + 0.5*x_span
            mid_y = min_y + 0.5*y_span
            loc_coords_array = _numpy_fromiter((min_x - offset, mid_y,
                                                max_x + offset, mid_y,
                                                mid_x, min_y - offset,
                                                mid_x, max_y + offset),
                                                _numpy_float64, 8)
            loc_coords_array.shape = _neg_1_2_tuple

            # Combine localizing coordinates with input coordinates.
            if e_padding is None:
                # *REASSIGNMENT*
                in_coords_array = _numpy_concatenate((in_coords_array,
                                                      loc_coords_array))
            else:
                e_padding[:] = loc_coords_array
                in_coords_array = e  # *REASSIGNMENT*

        # Perform Voronoi analysis.
        self.in_coords_array = in_coords_array
        self.voronoi = voronoi = _scipy.spatial.Voronoi(
            in_coords_array, furthest_site, False, qhull_options
            )
        self.out_coords_array = voronoi.vertices

        # If input coordinates were translated, restore them (as much as
        # possible) and likewise "un-translate" the output coordinates.
        if translate is not None:
            if copy:
                if localize:
                    # Note: e_main and e_padding are slices of
                    # self.in_coords_array. Therefore, .translate()
                    # modifies self.in_coords_array in-place.
                    e_main[:] = orig_coords_array
                    del orig_coords_array  # Release memory.
                    _MultiPoint2D.make_fast(e_padding, False).translate(
                        neg_deltas, False
                        )

                else:
                    # *REASSIGNMENT*
                    self.in_coords_array = orig_coords_array
            else:
                # Note: This line also modifies self.in_coords_array in-
                # place.
                in_multipoint.translate(neg_deltas, False)
            # Note: This line modifies self.out_coords_array in-place.
            _MultiPoint2D.make_fast(self.out_coords_array, False).translate(
                neg_deltas, False
                )

        # Apply merge option, if necessary.
        # Note: Only one input coordinate from each set of coplanar
        # (possibly duplicate) coordinates is used. The ignored
        # coordinates from each set are considered "merged."
        if merge_option == "ERROR" and self.used_in_indices is not None:
            raise TypeError(
                "{} coordinates are coplanar (possibly duplicate)".format(
                    len(in_coords_array) - len(self.used_in_indices)
                    )
                )

    @staticmethod
    def _get_used_in_indices(self):
        """
        Row indices of self.in_coords_array that were not merged in analysis.

        This attribute is None if no indices were merged.
        """
        regions = self.voronoi.regions
        # Note: At scipy version 1.1.0, one empty region is included if
        # incremental is specified as False.
        empty_region_count = regions.count([])
        # Note: If no coordinates were merged, each input coordinate
        # will have its own Voronoi cell.
        merged_coord_count = (len(self.in_coords_array) - len(regions) +
                              empty_region_count)
        if not merged_coord_count:
            return None
        return _uniquify_flat_array(self.voronoi.ridge_points.flatten(), False)

    def iter_lines(self):
        """
        Lazily generate each neighborhood boundary segment.

        Attributes of generated LineSegment2D's:
            .in_idxs_array  [array]
                The row indices in self.in_coords_array of the two coordinates
                for which the LineSegment2D is the perpendicular bisector and to
                which the LineSegment2D is closest.
            .out_idxs  [list]
                The row indices in self.out_coords_array of the two coordinates
                that define the vertices of the LineSegment2D, in order.
        """
        LineSegment2D = _geom.LineSegment2D
        out_coords_array = self.out_coords_array
        assign_spat_ref = hasattr(self, "spatial_reference")
        if assign_spat_ref:
            spatial_reference = self.spatial_reference
        for out_idxs, in_idxs_array in _izip(self.voronoi.ridge_vertices,
                                             self.voronoi.ridge_points):
            # Note: Ignore points at infinity.
            if -1 in out_idxs:
                continue
            line = LineSegment2D(out_coords_array[out_idxs])
            if assign_spat_ref:
                line.spatial_reference = spatial_reference
            line.out_idxs = out_idxs
            line.in_idxs_array = in_idxs_array
            yield line

    def iter_points(self):
        """
        Lazily generate each neighborhood boundary vertex.

        Attributes of generated Point2D's:
            .out_idx  [int]
                The row index in self.out_coords_array of the coordinate.
        """
        assign_spat_ref = hasattr(self, "spatial_reference")
        if assign_spat_ref:
            spatial_reference = self.spatial_reference
        for out_idx, point in enumerate(
            _geom.MultiPoint2D(self.out_coords_array).iterate_only()
            ):
            if assign_spat_ref:
                point.spatial_reference = spatial_reference
            point.out_idx = out_idx
            yield point



###############################################################################
# SKELETON CLASSES                                                            #
###############################################################################

class Skeleton(_util.Lazy):
    """
    Base class for (vector) skeletonization.
    """
    # Various attributes used internally.
    _get_length = _operator.attrgetter("length")
    _lines_were_manually_deleted = False  # This is the default.
    _shortest_path_args = {"directed": True, "return_predecessors": True,
                           "unweighted": False}
    ## Is qhull option "Q7" faster? If so, maybe default to using it?
    default_voronoi_kwargs = {"merge_option": "IGNORE",
                              "translate": "MEAN",
                              "localize": True,
                              "furthest_site": False,
                              "qhull_options": None}

    # Attributes related (perhaps indirectly) to implicit/brute-search
    # intervals, including those attributes relating to general memory
    # safeguards.
    _cutting_was_attempted = False
    _max_known_unsafe_interval = None
    _min_cutting_failure_interval = _python_inf
    _min_known_safe_interval = _python_inf
    target_memory_footprint = None

    # User-facing attribute defaults.
    had_hyperhub = False  # This is the default.
    initialized = False  # This is the default.

    def __init__(self, polygon, interval, isolation_mode="SAFE",
                 memory_option=0, targ_GB=None, **kwargs):
        """
        Initiate skeleton derivation for a polygon. (Compare EasySkeleton.)

        In simplest terms, the skeleton of a polygon collapses an elongate
        polygon to a linear form. For example, the outline of a river is a
        polygon (with a nonzero area). However, we often think of a river in
        terms of its skeleton, such as when a river is drawn as a line on a map.
        Even though that line does not represent the river's width, it still
        captures a useful expression of the river's geometry. The specific
        algorithm implemented by the current class has a lot of moving parts,
        but in effect, these are the highlights:
            1) Sample points along the boundary of polygon, including around any
               holes.
            2) Compute a Voronoi diagram. (See Voronoi2D.)
            3) Isolate the "graph skeleton" from the other extraneous bits
               (which together make up the graph skeleton's "complement") in the
               Voronoi diagram. (For example, each hole in polygon has its own
               skeleton that should be discarded.)
            4) "Partition out" paths from the graph skeleton to incrementally
               construct the "partitioned skeleton". (See
               .partition_out_trunk().)
            5) Optionally add "tails" to the paths so that they extend to
               polygon's boundary. (See .add_tails().)
            6) Optionally prune away undesired paths. (See .prune().)
        At initialization (i.e., when the current function is called), the above
        steps are executed partway through step 3 (but see note further below).
        The curious should consider reading the following paper for a more
        detailed, albeit technical, description of the algorithm, including some
        very helpful figures:
            [placeholder]

        polygon is a Polygon for which skeleton derivation will be executed.

        interval is a float that specifies the interval along polygon's boundary
        at which coordinates are sampled, in map units. Its effect is similar to
        a resolution: finer (smaller) values yield results with finer precision
        but also entail higher memory footprints. Ideally, interval should be no
        coarser than half the width of polygon's narrowest constriction. (See
        "CUT" isolation_mode option.) If interval is specified as 0, an
        approximately "optimized" interval is set automatically (but see note
        further below).

        isolation_mode is a string that specifies several aspects of how the
        graph skeleton is isolated from its complement and general memory
        safeguards. The available options, which may be combined as long as they
        are delimited by underscores, are:
            "NAIVE"
                If polygon and its complement (i.e., the holes in polygon and
                the area surrounding polygon) everywhere have a width equal to
                at least twice interval, the skeleton can be more efficiently
                isolated by assuming that 1) only the skeleton lies within
                polygon and 2) the skeleton is completely contained within
                polygon.
            "VERIFY"
                "VERIFY" is ignored unless "NAIVE" is also specified. In that
                case, (something similar to) the second assumption in "NAIVE" is
                tested and, upon failure, the algorithm resorts to non-naive
                skeleton isolation.
            "CUT"
                If interval is anywhere coarser than half the polygon's local
                width, the skeleton may not be contained to polygon (Theorem 4.1
                of Brandt and Algazi (1992)). If there is any ambiguity as to
                whether a (degree 3) hub belongs to the skeleton, it is "cut"
                along with all edges terminating at that hub. This is a
                heuristic resolution to facilitate the isolation of the skeleton
                from its complement when interval is coarser than ideal and can
                have undesired effects on the resulting skeleton. Cutting is
                only applied if not cutting would cause skeleton isolation to
                fail. If any nodes are cut, they are stored to
                .cut_vor_vert_idxs.
            "SAFE"
                At the two steps (2 and 4, as numbered further above) with the
                greatest memory footprints, it is tested whether continued
                processing would exhaust targ_GB. If such exhaustion is
                predicted, processing is aborted and a MemoryError is raised.

        Remaining arguments are documented with EasySkeleton.__init__().

        Note: Although it can be very useful to specify a zero interval or
        template_interval, there are some important factors to bear in mind:
            1) If you do not specify targ_GB, the "optimized" interval will
               depend on the memory currently available on your computer at that
               specific time. That available memory does *not* include any
               memory currently claimed by Python that is, in fact, available
               for reuse, and therefore can be significantly underestimated.
               Consuming additional memory during processing (e.g., by opening
               other programs) could also cause the algorithm to fail and/or
               slow your computer.
            2) You should always describe your results in terms of the interval
               value that is ultimately used, possibly after optimization
               (.interval), and, if applicable, the corresponding template
               interval's value (.template_interval). Doing so ensures that (1)
               your results can be easily reproduced without the optimization
               process, by directly specifying those values, and (2) that any
               such reproduction is not dependent on the individual system
               (e.g., physical memory), system state (e.g., available memory),
               or optimization code (which may be updated in the future) used.
            3) The optimization of interval (but not template_interval) involves
               a brute-force search that can be very computationally expensive,
               especially (and perhaps counterintuitively) if polygon has a
               relatively simple geometry (e.g., shaped like a "Y" rather than a
               more complicated tree or even mesh-like shape).
            4) During the brute-first search, step 3 (as enumerated further
               above) is executed to completion rather than only partway.
            5) Some details of the brute-search for an optimized interval can be
               retrieved by .estimate_safe_interval() and .get_safe_interval().
            6) The "optimized" value is no coarser than double that interval
               that would approximately exhaust targ_GB. However, because the
               relationship between memory footprint and interval is not linear,
               the optimized interval may use much less than half of the
               targeted memory footprint.
        """
        ## Add attributes list to docstring. Include fact that .voronoi 
        ## is unavailable for memory_option > 0.        
        # Store calling arguments.
        self._calling_args = calling_args = locals().copy()
        del calling_args["self"]
        del calling_args["kwargs"]
        calling_args.update(kwargs)
        self.polygon = polygon
        self.interval = interval
        self.voronoi_kwargs = kwargs

        # Process memory_option and targ_GB.
        user_memory_option = memory_option
        memory_option = int(user_memory_option)  # *REASSIGNMENT*
        if (user_memory_option != memory_option or
            memory_option not in (0, 1)):
            raise TypeError("memory_option must be 0 or 1")
        self.memory_option = memory_option
        if targ_GB is not None:
            self.target_memory_footprint = int(float(targ_GB) * 2.**30.)

        # Process isolation mode.
        isolation_options = []
        for option in isolation_mode.split("_"):
            isolation_option = _validate_string_option(
                option, "option in isolation_mode",
                ("CUT", "NAIVE", "VERIFY", "SAFE", "SAFETEST1", "SAFETEST2"),
                True
                )
            if isolation_option.startswith("SAFETEST"):
                isolation_options.append("SAFE")
            isolation_options.append(isolation_option)
        self.isolation_mode = isolation_mode = "_".join(isolation_options)

        # Set target memory footprint and related attributes if possible
        # and not already set.
        # Note: This is performed for all calls in case
        # .estimate_max_node_count() is called later.
        boundary = polygon.boundary
        total_length = sum([ring.length for ring in boundary])
        sample_count_per_byte = 3.26e-4  # From testing.
        if self.target_memory_footprint is not None or _psutil is not None:
            if self.target_memory_footprint is None:
                self.target_memory_footprint = _psutil.virtual_memory().available
            # Note: This interval would approximately consume the entire
            # targeted memory footprint during non-partitioning
            # processing, based on testing. The main memory sink during
            # that period is the Voronoi analysis. Partitioning out also
            # has a significant memory footprint (primarily due to the
            # predecessors and cost arrays) but depends on the number of
            # nodes, excluding (contracted) binodes, which is difficult
            # to predict a priori.
            # Note: ~200 MB is set aside for general overhead.
            min_interval = self._min_nonpartitioning_interval = (
                total_length / ((self.target_memory_footprint - 2e8)
                                * sample_count_per_byte)
                )
            # Note: Because the memory footprint for non-partitioning
            # roughly depends linearly on the interval, 1.2 is a
            # reasonable safety margin.
            min_safe_interval = self._min_safe_nonpartitioning_interval = (
                1.2 * min_interval
                )

        # Apply brute search, "SAFE", and "SAFETEST1" options.
        brute_search = interval == 0.
        # Note: "SAFETEST*" will satisfy the second condition.
        if brute_search or "SAFE" in isolation_mode:
            if self.target_memory_footprint is None:
                self._raise_psutil_error()
            if self.target_memory_footprint < 2e8:
                self.interval_is_nonpartitioning_safe = False
                raise MemoryError(
                    "available/specified memory footprint must be > 200 MB"
                    )
            if "SAFETEST1" in isolation_options:
                return
            if brute_search:
                # *REASSIGNMENT*
                interval = self.interval = min_interval * 10.
            elif interval < min_safe_interval:
                self.interval_is_nonpartitioning_safe = False
                raise MemoryError(
                    "interval is smaller than the estimated smallest interval that can be safely processed for the input polygon within available memory, ignoring partitioning out: {}".format(
                        min_safe_interval
                        )
                    )
            self.interval_is_nonpartitioning_safe = True

        # Now that interval is guaranteed to be explicit (nonzero),
        # store approximate non-partitioning memory footprint.
        # Note: This is performed for all calls in case
        # .estimate_max_node_count() is called later.
        self._approx_nonpartitioning_mem = total_length/interval * sample_count_per_byte**-1.

        # Determine whether generating a proxy polygon is required.
        # Note: In ._isolate_graph_skeleton(), testing against this
        # proxy (rather than the input polygon) is especially necessary
        # when interval is much coarser than the typical boundary
        # segment length for the input polygon. In that case, even hubs
        # of a complement can lie within the input polygon but outside
        # the proxy polygon, which, conceptually, represents the polygon
        # "seen" by boundary sampling of the input polygon. More
        # precisely, the proxy polygon is a polygon with fewer vertices
        # than the input polygon but on whose boundary all samples from
        # the input polygon's boundary lay, except for any samples from
        # a ring of the input polygon that were sampled fewer than 3
        # times and therefore have zero "seen" area. On the other hand,
        # if interval is smaller than the smallest segment length in the
        # input polygon's boundary, a proxy is not required.
        proxy_poly_reqd = False  # Default.
        for ring in boundary:
            if ring.segment_lengths.min() < interval:
                proxy_poly_reqd = True
                poly_is_3D = polygon.is_3D
                break

        # Regularly sample each ring in the input polygon and generate
        # proxy polygon rings if required.
        proxy_rings = []
        LineString2D = _geom.LineString2D
        sampled_coords_each_ring = []
        sampled_coords_each_ring_append = sampled_coords_each_ring.append
        for ring in boundary:

            # Sample regularly along ring boundary.
            sampled_coords_this_ring, seg_idxs_this_ring = ring.sample_regularly(
                interval, False, False, True
                )
            sampled_coords_each_ring_append(sampled_coords_this_ring)

            # If a corresponding proxy ring must be generated, do so.
            # Note: It would also be sufficient to merely use all
            # sampled coordinates, but only using at most two (the first
            # and last samples) from each segment makes for faster
            # containment testing.
            if not proxy_poly_reqd or len(sampled_coords_this_ring) < 3:
                continue
            mask = _numpy_empty((len(seg_idxs_this_ring),), _numpy_bool8)
            mask[0] = mask[-1] = True
            _numpy_not_equal(seg_idxs_this_ring[1:-1],
                             seg_idxs_this_ring[:-2], mask[1:-1])
            _numpy_logical_or(
                seg_idxs_this_ring[1:-1] != seg_idxs_this_ring[2:],
                mask[1:-1], mask[1:-1]
                )
            proxy_ring_sample_idxs, = mask.nonzero()
            del mask  # Release memory.
            proxy_ring_coords_array = _numpy_empty(
                (len(proxy_ring_sample_idxs) + 1, 2),
                _numpy_float64
                )
            if poly_is_3D:
                # *REASSIGNMENT*
                sampled_coords_this_ring = sampled_coords_this_ring[:,:2]
            _take2(sampled_coords_this_ring, proxy_ring_sample_idxs,
                   out=proxy_ring_coords_array[:-1])
            del proxy_ring_sample_idxs  # Release memory.
            proxy_ring_coords_array[-1] = proxy_ring_coords_array[0]
            proxy_rings.append(LineString2D(proxy_ring_coords_array))
        # Reduce local namespace and release memory.
        del sampled_coords_this_ring, seg_idxs_this_ring

        # Store sampled coordinates.
        self.sampled_coords_array = sampled_coords_array = _numpy_concatenate(
            sampled_coords_each_ring
            )

        # Construct proxy polygon, if required.
        if proxy_poly_reqd:
            self.test_poly = proxy_poly = _geom.Polygon2D(proxy_rings)
            proxy_poly.spatial_reference = polygon.spatial_reference
        else:
            self.test_poly = polygon._2D

        # Create a function to convert a sample coordinate index to a
        # ring index (and its supporting array).
        last_sample_idx_each_ring_array, get_ring_idx_from_sample_idx = self._convert_counts(
            _imap(len, sampled_coords_each_ring)
            )
        del sampled_coords_each_ring  # Release memory.
        self._last_sample_idx_each_ring_array = last_sample_idx_each_ring_array
        # Note: As currently coded, this function is only used if
        # polygon has a hole.
        self._get_ring_idx_from_sample_idx = get_ring_idx_from_sample_idx

        # Perform Voronoi analysis.
        # Note: Even if polygon is 3D, its outline nature restricts it
        # to representing a surface, and that surface is only well
        # described if it is locally planar. It is also expected that,
        # in practice, analyzed polygons will typically be approximately
        # planar locally. For these reasons, the underlying Voronoi
        # analysis is 2D. Restricting Voronoi analysis to 2D also
        # permits spatial indexing to be leveraged to avoid expensive
        # geometric tests, as explained furhter below.
        voronoi_kwargs = self.default_voronoi_kwargs.copy()
        if kwargs:
            # Prevent overriding fundamental arguments.
            for voronoi_arg_name in ("coords", "furthest_site"):
                if voronoi_arg_name in kwargs:
                    raise TypeError(
                        "cannot override this Voronoi2D argument: {!r}".format(
                            voronoi_arg_name
                            )
                        )
            voronoi_kwargs.update(kwargs)
        # For Voronoi analysis only, strip z-coordinate from sample
        # coordinates.
        # Note: sample_mp (as opposed to merely its array) is only
        # created so that the input polygon's spatial reference is
        # preserved in the Voronoi2D.
        sample_mp = _geom.MultiPoint2D.make_fast(sampled_coords_array[:,:2],
                                                 False)
        if hasattr(polygon, "spatial_reference"):
            sample_mp.spatial_reference = polygon.spatial_reference
        voronoi_kwargs["coords"] = sample_mp
        voronoi = Voronoi2D(**voronoi_kwargs)

        # Localize Voronoi analysis outputs required for Skeleton
        # initialization, and, if memory use should be reduced, retain
        # only those outputs that might be later required internally.
        used_sample_idxs = voronoi.used_in_indices
        self._bracketing_sample_idx_pairs_array = bracketing_sample_idx_pairs_array = voronoi.voronoi.ridge_points
        vor_vert_idx_pairs_array = _numpy_fromiter(
            _flatten_to_iter(voronoi.voronoi.ridge_vertices),
            _numpy_int32 if memory_option else _numpy_int64
            )
        if memory_option:
            # Adjust estimated memory footprint for the use of 32-bit
            # rather than 64-bit integers.
            self._approx_nonpartitioning_mem -= _sys.getsizeof(
                vor_vert_idx_pairs_array
                )
        self._vor_vert_idx_pairs_array = vor_vert_idx_pairs_array
        vor_vert_idx_pairs_array.shape = _neg_1_2_tuple
        if memory_option:
            self._voronoi = _util.Object()
            if voronoi.used_in_indices is None:
                self._voronoi.used_in_indices = None
            elif memory_option == 1:
                self._voronoi.used_in_indices = voronoi.used_in_indices
            if memory_option == 1:
                self._voronoi.out_coords_array = voronoi.out_coords_array
            # Adjust estimated memory footprint for the deletion of
            # .voronoi.
            # Note: voronoi.ridge_points is stored to
            # ._bracketing_sample_idx_pairs_array further above, and
            # voronoi.vertices is stored to ._voronoi.out_coords_array
            # immediately above. Furthermore, the memory space freed by
            # deletion of the lists (voronoi.point_region,
            # voronoi.regions, and .ridge_vertices) may be
            # underestimated as sys.getsizeof() returns the size of the
            # list structure but integer objects may also be freed.
            self._approx_nonpartitioning_mem -= sum(
                map(_sys.getsizeof,
                    (voronoi.voronoi.point_region, voronoi.voronoi.regions,
                     voronoi.voronoi.ridge_vertices))
                )
            del voronoi  # Release memory.
        else:
            self.voronoi = self._voronoi = voronoi

        # Each Voronoi segment must ultimately be categorized as one of
        # the following:
        #   1) skeletal segment: a segment that belongs to the skeleton
        #      of the input polygon; in the simplest (well-sampled)
        #      case, all skeletal segments are wholly contained by the
        #      input polygon
        #   2) complementary segment: a segment that belongs to the
        #      skeleton's complement; in the simplest (well-sampled)
        #      case, all complementary segments are completely outside
        #      the input polygon (including within holes)
        #   3) peripheral segment: a segment that is the perpendicular
        #      bisector of the line between two consecutive samples on
        #      a single ring of the input polygon's boundary; in the
        #      simplest (well-sampled) case, each peripheral segment
        #      intersects the input polygon's boundary

        # Instead of categorizing segments by expensive geometric
        # operations, spatial indexing is used wherever possible. As a
        # first pass, we identify those segments that are bracketed by
        # numerically consecutive sample coordinates (after any merged
        # indices are removed; see Voronoi2D documentation for merging).
        # Note: Such segments are "nearly always" peripheral, but
        # special care is taken further below to account for exceptions.

        # Note: "re-index" (and its abbreviation, "reidx") is used
        # hereinafter to refer to sample indices that have been
        # re-indexed to include only those that were not merged. For
        # example, if sample index 1 was merged into sample index 0 but
        # sample index 2 was not merged, sample index 0's re-index would
        # remain 0 but sample index 2's re-index would be 1.
        if used_sample_idxs is None:
            # No merging occurred, so no re-indexing is necessary.
            bracketing_sample_reidx_pairs_array = bracketing_sample_idx_pairs_array
        else:
            # Merging occurred, so re-index.
            bracketing_sample_reidx_pairs_array = used_sample_idxs.searchsorted(
                bracketing_sample_idx_pairs_array
                )
        bracketing_sample_reidx_pair_diffs_array = (
            bracketing_sample_reidx_pairs_array[:,0] -
            bracketing_sample_reidx_pairs_array[:,1]
            )
        abs_bracketing_sample_reidx_pair_diffs_array = _numpy_abs(
            bracketing_sample_reidx_pair_diffs_array,
            bracketing_sample_reidx_pair_diffs_array
            )
        # Note: The array name is temporarily a misnomer, though
        # typically accurate for most segments.
        is_peri_array = abs_bracketing_sample_reidx_pair_diffs_array == 1

        # The only just identified segments that can be non-peripheral
        # (i.e., skeletal or complementary) are those that are bracketed
        # by numerically consecutive inter-ring samples. If polygon has
        # holes, find these exceptions. (If polygon does not have holes,
        # no inter-ring samples can exist.)
        # Note: For example, consider a polygon with one hole and for
        # which no merging occurred during Voronoi anlaysis. If that
        # polygon's outermost ring has 100 sample coordinates (with
        # indices 0...99), a segment bracketed by sample coordinates
        # with indices 99 and 100 is located between (the last sample
        # from) the outermost ring and (the first sample from) the hole,
        # and is therefore skeletal.
        polygon_has_a_hole = len(boundary) > 1
        if polygon_has_a_hole:
            ring_idx_pairs_array = get_ring_idx_from_sample_idx(
                bracketing_sample_idx_pairs_array
                )
            is_same_ring_array = ring_idx_pairs_array[:,0] == ring_idx_pairs_array[:,1]
            del ring_idx_pairs_array  # Release memory.
            _numpy_logical_and(is_peri_array, is_same_ring_array, is_peri_array)
            del is_same_ring_array  # Release memory.

        # Create a set of the sample coordinate index pairs that "close"
        # each ring in polygon (in both orders).
        # Note: Such closing index pairs represent the only case where
        # indices can be spatially consecutive but not numerically
        # consecutive. For example, if the outermost ring of polygon has
        # 100 sample coordinates (with indices 0...99), a segment
        # bracketed by sample coordinates with indices 99 and 0 is
        # peripheral despite these indices being numerically non-
        # consecutive.
        closing_sample_idxs_set = set()
        closing_sample_idxs_set_add = closing_sample_idxs_set.add
        low_idx = 0  # Initialize.
        for high_idx in last_sample_idx_each_ring_array.tolist():
            closing_sample_idxs_set_add((low_idx, high_idx))
            closing_sample_idxs_set_add((high_idx, low_idx))
            low_idx = high_idx + 1

        # Create and populate an "adjacency" dictionary of non-
        # peripheral segments (and finalize categorization).
        # Note: This array name is strictly a misnomer, though typically
        # accurate for most ridges.
        is_nonperi_array = _numpy.logical_not(is_peri_array, is_peri_array)
        del is_peri_array  # Avoid accidental reuse.
        is_nonperi_row_idxs_array, = is_nonperi_array.nonzero()
        del is_nonperi_array  # Release memory.
        nonperi_vor_vert_idx_pairs = _take2(vor_vert_idx_pairs_array,
                                            is_nonperi_row_idxs_array).tolist()
        nonperi_bracketing_sample_idx_pairs = _take2(
            bracketing_sample_idx_pairs_array, is_nonperi_row_idxs_array
            ).tolist()
        ## ._nonperi_adj_dict accounts for a significant fraction (up to
        ## 15%) of the peak memory footprint in testing. This dict-of-
        ## lists is keyed to integers only and its list values contain 
        ## only integers, so a drop-in replacement backed by an integer 
        ## array would dramatically decrease the size of the container.
        self._nonperi_adj_dict = nonperi_adj_dict = _defaultdict(list)
        for (vor_vert_idx0, vor_vert_idx1), (sample_idx0, sample_idx1) in _izip(
            nonperi_vor_vert_idx_pairs, nonperi_bracketing_sample_idx_pairs
            ):
            # If segment is bracketed by ring-closing samples, it is
            # peripheral.
            if (sample_idx0, sample_idx1) in closing_sample_idxs_set:
                # Note: The line below would finalize segment
                # categorization, where row_idx comes from iterating
                # over is_nonperi_array.tolist().
                # is_nonperi_array[row_idx] = False
                continue
            nonperi_adj_dict[vor_vert_idx0].append(vor_vert_idx1)
            nonperi_adj_dict[vor_vert_idx1].append(vor_vert_idx0)
        # Discard any entries referring to Voronoi vertices at infinity.
        if -1 in nonperi_adj_dict:
            for vor_vert_idx1 in nonperi_adj_dict.pop(-1):
                nonperi_adj_dict[vor_vert_idx1].remove(-1)

        # Prepare for quarantining loops, partitioning out, and width
        # calculations.
        self._loop_dict = {}
        self._kind_to_partitioned_path_dict = _defaultdict(dict)
        self._self_proxy = _weakref.proxy(self)
        self._vor_vert_idx_to_sample_dist = _util.defaultdict2(
            self._calc_vor_dist
            )

        # Optionally execute a brute search for the smallest interval
        # that can be fully processed in available (or user-permitted)
        # memory.
        # Note: The search must be executed through completion now to
        # prevent the user from interacting with the instance (which is
        # about to be returned) before the search completes. If such
        # interaction were permitted, the user's operations would be
        # lost with the next iteration of the search.
        if brute_search:
            self._apply_brute_and_safe_options(brute=True)

        # Note: Initialization (or "Phase 1") is effectively split into
        # (1) Voronoi analysis and peripheral segment removal (the
        # current function) and (2) isolation of the skeleton and
        # contraction through binodes (executed lazily on first use of
        # ._graph_edge_dict) to potentially permit the user access to
        # the "raw" data output by the current function.

    def _apply_brute_and_safe_options(self, brute=False, safe=False):
        """
        Apply "SAFE" isolation_mode option and execute interval optimization.

        brute is a boolean that specifies whether the current call should
        execute a brute search optimization of the sampling interval.

        safe is a boolean that specifies whether the current call should test
        whether partitioning out would exceed .target_memory and raise a
        MemoryError if that is the case. If safe is specified as True but "SAFE"
        does not exist in .isolation_mode, the current function returns without
        executing the test.

        Note: Only brute or safe can be True.
        """
        # Test that exactly one mode is specified.
        assert bool(brute) != bool(safe)

        # If mode (for the current function) is brute, set isolation
        # mode appropriately.
        # Note: Permitting "cutting" of skeletons during the brute
        # search is necessary to ensure that processing of each coarse
        # trial interval can complete and therefore inform the next
        # iteration.
        if brute:
            orig_calling_args = self._calling_args
            orig_isolation_mode = orig_calling_args["isolation_mode"]
            user_permits_cutting = "CUT" in orig_isolation_mode
            if user_permits_cutting:
                brute_calling_args = orig_calling_args
            else:
                brute_calling_args = orig_calling_args.copy()
                if orig_isolation_mode:
                    brute_calling_args["isolation_mode"] = orig_isolation_mode + "_CUT"
                else:
                    brute_calling_args["isolation_mode"] = "CUT"

        # Cancel processing if mode (for the current function) is safe
        # but isolation mode (set at initialization) does not specify
        # that mode.
        elif safe and "SAFE" not in self.isolation_mode:
            return

        # Approximate the number of uncontracted nodes that can be
        # safely processed in the target memory.
        # Note: Because the memory footprint of partitioning out roughly
        # depends on the square of the number of uncontracted nodes, 0.9
        # is a reasonable safety margin.
        max_safe_node_count = 0.9 * self.estimate_max_node_count()

        # Enter iterative brute search.
        # Note: If mode is safe (for the current function), only one
        # (partial) iteration will be executed.
        brute_search_failed = False  # Initialize.
        brute_search_finished = False  # Initialize.
        while not brute_search_finished:

            # If the immediately previous trial interval failed because
            # cutting did not enable skeleton isolation, halve interval
            # or fail, as appropriate.
            # Note: It is assumed that reducing interval is the only
            # option for enabling skeleton isolation, whether because
            # cutting then enables skeleton isolation or because no
            # cutting is necessary.
            cur_interval = self.interval
            next_interval = None  # This is the default.
            if self._min_cutting_failure_interval == cur_interval:
                # If the immediately previous trial interval was already
                # the finest that can be supported for non-partitioning,
                # fail.
                if cur_interval == self._min_safe_nonpartitioning_interval:
                    brute_search_failed = True
                else:
                    # Provisionally halve interval or set it to the
                    # smallest value that can be supported for non-
                    # partitioning. However, if that provisional value
                    # is smaller than the coarsest interval known to
                    # exceed the target memory, fail.
                    next_interval = max(0.5 * cur_interval,
                                        self._min_safe_nonpartitioning_interval)
                    if next_interval <= self._max_known_unsafe_interval:
                        brute_search_failed = True

            else:

                # If non-binode count generated by current interval is
                # unsafe, error or prepare for the next iteration, as
                # appropriate.
                node_count = self.get_node_count("end", loops=False)
                if node_count > max_safe_node_count:
                    # Error if merely checking that non-binode count is
                    # safe.
                    if safe and "SAFETEST2" not in self.isolation_mode:
                        raise MemoryError(
                            "only ~{:n} nodes (excluding contracted binodes) can be processed in memory, but {:n} exist at the following interval (which would require ~{:.1f} GB); smooth/simplify input polygon, increase interval, or use a (coarser) template: {}".format(
                                int(max_safe_node_count), node_count,
                                (self._approx_nonpartitioning_mem + 18.*node_count**2.) / (0.9 * 2.**30),
                                cur_interval
                                )
                            )
                    # If the smallest interval known to be safely
                    # processible is "about" double the size of the
                    # current unsafe interval, resort to it. Otherwise,
                    # multiply the current interval by 10 (as possible).
                    # Note: Because each successive interval is
                    # multiplied by either 10 (if unsafe) or a value in
                    # the range [0.5, 1) (if overly conservative or
                    # cutting did not enable skeleton isolation),
                    # testing whether the smallest known safe interval
                    # is no more than 3x larger than the current
                    # interval basically tests whether that interval was
                    # the previously tried interval and whether it
                    # permitted skeleton isolation (with or without
                    # cutting), with plenty of allowance for resolution.
                    self._max_known_unsafe_interval = cur_interval
                    if self._min_known_safe_interval / cur_interval < 3.:
                        next_interval = self._min_known_safe_interval
                        # Note: ._cutting_was_attempted is assigned
                        # merely to simplify code further below and code
                        # for .estimate_safe_interval(). It is
                        # overwritten on the next (final)
                        # initialization.
                        self._cutting_was_attempted = self._cutting_was_attempted_for_min_known_safe_interval
                        brute_search_finished = True
                    else:
                        # Provisionally adopt the finer of 10x the
                        # current interval and half the finest interval
                        # for which cutting failed to enable skeleton
                        # isolation. However, if the latter is no
                        # coarser than the current (unsafe) interval,
                        # fail.
                        next_interval = min(
                            10. * cur_interval,
                            0.5 * self._min_cutting_failure_interval
                            )
                        if next_interval <= cur_interval:
                            brute_search_failed = True

                # The current interval is safe. If this is a brute
                # search, proceed accordingly.
                elif brute:
                    # If the current (safe) interval is (1) already the
                    # finest that can be supported for non-partitioning
                    # or (2) no more than double the coarsest known
                    # unsafe interval, treat the search as complete.
                    # Otherwise halve (or reduce) the current interval
                    # on the next iteration (if possible).
                    # Note: The latter condition covers the case where
                    # the first-guess interval was only a little too
                    # small to be accommodated. For example, if the
                    # first guess was 1 m and was found to be too small,
                    # 10 m, 5 m, 2.5 m, and 1.25 m would be tried in
                    # order. If 1.25 m can be accommodated, further
                    # searching is canceled.
                    if (cur_interval == self._min_safe_nonpartitioning_interval or
                        (self._max_known_unsafe_interval is not None and
                         cur_interval / self._max_known_unsafe_interval <= 2.)):
                        brute_search_finished = True
                    else:
                        self._min_known_safe_interval = cur_interval
                        self._cutting_was_attempted_for_min_known_safe_interval = self._cutting_was_attempted
                        next_interval = max(
                            0.5 * cur_interval,
                            self._min_safe_nonpartitioning_interval
                            )

            # If brute search just failed or finished, proceed
            # accordingly.
            if brute_search_failed:
                raise TypeError(
                    "interval (or template interval) could not be optimized because skeleton could not be isolated (despite cutting)"
                    )
            if brute_search_finished:
                if "SAFETEST2" in orig_isolation_mode:
                    if next_interval is not None:
                        # Note: .interval is assigned to support code in
                        # .estimate_safe_interval().
                        self.interval = next_interval
                    return
                if not user_permits_cutting and self._cutting_was_attempted:
                    raise TypeError(
                        "no interval was found that can both (1) be safely processed in targeted memory footprint and (2) does not require cutting to isolate the skeleton (i.e., 'CUT' in isolation_mode)"
                        )
                if next_interval is None:
                    self._calling_args = orig_calling_args
                else:
                    # *REASSIGNMENT* (possibly)
                    brute_calling_args = orig_calling_args

            # If no future iterations are called for, return
            # immediately.
            # Note: Either mode is safe and interval can be fully
            # processed or mode is brute and current interval is
            # approximately optimized.
            if next_interval is None:
                return None

            # Release memory from current iteration.
            # Note: Other public attribute names are retained to provide
            # some support for debugging and subclasses.
            ## Maybe go with a similar blacklist approach for private
            ## attribute names, in case of subclassing?
            big_public_attr_names = ("sampled_coords_array", "voronoi",
                                     "voronoi_vertex_coords_array")
            # Note: ._cutting_was_attempted is intentionally cleared so
            # that it defaults to False (class attribute) with each re-
            # initialization.
            retain_private_attr_names = (
                "_approx_nonpartitioning_mem", "_calling_args",
                "_cutting_was_attempted_for_min_known_safe_interval",
                "_max_known_unsafe_interval", "_min_cutting_failure_interval",
                "_min_known_safe_interval", "_min_safe_nonpartitioning_interval"
                )
            for attr_name in self.__dict__.keys():
                if attr_name[0] == "_":
                    if attr_name in retain_private_attr_names:
                        continue
                elif attr_name not in big_public_attr_names:
                    continue
                delattr(self, attr_name)

            # Re-initialize (e.g., prime for next iteration).
            brute_calling_args["interval"] = next_interval
            try:
                # Note: Use Skeleton explicitly (rather than self) so
                # that subclasses can still use the current function.
                Skeleton.__init__(self, **brute_calling_args)
            except:
                # Note: If failure was due to a skeleton that could not
                # be isolated despite cutting, do not fail (yet).
                if self._min_cutting_failure_interval != next_interval:
                    raise

    @staticmethod
    def estimate_safe_interval(polygon, isolation_mode="SAFE_CUT",
                               memory_option=0, targ_GB=None,
                               include_partitioning=True, **kwargs):
        """
        Return bounds on what intervals can be safely processed for a polygon.

        The current function iteratively initializes a Skeleton for the
        specified polygon and isolation_mode at each of multiple sampling
        intervals as part of a brute-force search for an "optimized" interval
        (which here means the smallest interval that can be comfortably fully
        processed within the targeted memory footprint) up to the point where
        the memory footprint ultimately required for processing can be
        estimated. A tuple with five items is returned:
            idx    Meaning
            0      That sampling interval (within a factor of 2) that can
                   approximately completely consume the targeted memory. (See
                   documentation for Skeleton.__init__().) If no such interval
                   can be identified, an error will be raised as though
                   Skeleton.__init__() were called. [float]
            1      That coarsest sampling interval that was directly tested and
                   would have consumed more memory than targeted. By definition,
                   this value is less than the value at idx=0 and represents an
                   approximate lower bound on the intervals that can be
                   successfully processed for the specified polygon and
                   isolation_mode within the targeted memory. [float]
            2      The finest sampling interval that was directly tested and for
                   which the resulting skeleton could not be isolated despite
                   cutting (see "CUT" isolation_mode option in
                   Skeleton.__init__().) May be infinity if every tested
                   skeleton could be isolated, whether by cutting or without it.
                   By definition, this value is greater than the value at idx=0
                   and represents an approximate upper bound on the intervals
                   that can be successfully processed for the specified polygon
                   and isolation_mode. [float]
            3      A boolean indicating whether cutting was required for the
                   interval at idx=0. In essence, "_CUT" is automatically
                   appended to isolation_mode prior to testing, so the boolean
                   at idx=3 may be True even if you do not specify "CUT" in
                   isolation_mode. [bool]
            4      The target memory footprint in GB. [float]

        include_partitioning is a boolean that specifies whether the memory
        required for partitioning out should be included in testing. If
        include_partitioning is False, the current function returns very quickly
        with a tuple of three items:
            idx    Meaning
            0      That sampling interval that would completely consume the
                   targeted memory. [float]
            1      The finest sampling interval that will be initially permitted
                   if the "SAFE" isolation_mode option is specified. (The
                   estimated memory footprint of partitioning is accounted for
                   later, and processing may be aborted at that time, and an
                   error raised, if the combined memory footprint for non-
                   partitioning and partitioning exceeds the targeted memory
                   footprint.)
            2      The target memory footprint in GB. [float]

        The remaining arguments not documented here have the same meaning as in
        Skeleton.__init__().

        See also:
            .estimate_max_node_count()
        """
        calling_kwargs = locals().copy()
        del calling_kwargs["include_partitioning"]
        del calling_kwargs["kwargs"]
        calling_kwargs.update(kwargs)
        if include_partitioning:
            test_isolation_option = "SAFETEST2"
        else:
            test_isolation_option = "SAFETEST1"
        calling_kwargs["isolation_mode"] = "{}_{}".format(
            calling_kwargs["isolation_mode"], test_isolation_option
            ).lstrip("_")
        skel = Skeleton(interval=0., **calling_kwargs)
        if not include_partitioning:
            return skel.get_safe_interval()
        return (skel.interval, skel._max_known_unsafe_interval,
                skel._min_cutting_failure_interval, skel._cutting_was_attempted,
                skel.target_memory_footprint / 2.**30.)

    def get_safe_interval(self):
        """
        Return info on the finest interval that can be initially processed.

        The current function returns a tuple of the same form as is returned by
        .estimate_safe_interval(..., include_partitioning=False). The only
        difference between the current function and that function is that the
        current function returns the internally stored values that were
        calculated during processing. (Whereas the current function is an
        instance method, .estimate_safe_interval() is a static method.) If the
        relevant values were not calculated, a TypeError is raised.

        Note: If valid arguments were specified at initialization, the relevant
        values should be available, even if processing was internally aborted
        (e.g., a MemoryError was raised).
        """
        try:
            return (self._min_nonpartitioning_interval,
                    self._min_safe_nonpartitioning_interval,
                    self.target_memory_footprint / 2.**30.)
        except AttributeError:
            raise TypeError(
                "the requested safe interval bounds were not calculated"
                )

    def _calc_vor_dist(self, vor_vert_idx,
                       calc_dist=_geom.Point2D.calculate_distance.__func__):
        """
        Approximate the local half-width of .polygon at a Voronoi vertex.

        More precisely, the current function calculates the distance from a
        Voronoi vertex to the closest sample on .polygon's boundary.

        vor_vert_idx is an integer that specifies the Voronoi vertex index
        (i.e., row index in .voronoi_vertex_coords_array).
        """
        # Note: Since all samples associated with a vertex are
        # equidistant from that vertex, evaluating just one such sample
        # will suffice.
        return calc_dist(
            self.sampled_coords_array[self._fetch_associated_sample_index(vor_vert_idx)].tolist()[:2],
            self.voronoi_vertex_coords_array[vor_vert_idx].tolist()
            )

    def add_loops_to_partitioned(self, test_func=None):
        """
        Add uninterrupted terminal loops to the partitioned skeleton.

        Other than .partition_out*()'s, the current function is the only
        function that adds paths to the partitioned skeleton. Specifically, it
        adds uninterrupted terminal loops, which are those edges (contracted
        through their binodes only) that would have been added to the graph
        skeleton except that they begin and end at the same node (which is
        initially a degree-3 hub. They can only exist if .polygon contains one
        or more holes, but even then, are typically rare and may not exist for
        most skeletons with holes as looped paths (also called "cycles") around
        holes are commonly interrupted by the terminations of other edges.
        Because such uninterrupted terminal loops create issues for later
        processing, they are quarantined as soon as they are recognized, which
        is immediately before finalization of the graph skeleton. Note that,
        after this quarantining, the node that was the stem hub for the loop
        becomes a stub (hence why such loops are called "terminal"). Somewhat
        analogous to .partition_out*()'s, the current function:
            1) Iterates over each uninterrupted terminal loop whose stem is
               present in the partitioned skeleton.
            2) Removes (accepted) uninterrupted terminal loops from their
               quarantine.
            3) Adds uninterrupted terminal loops to the partitioned skeleton.
            4) Returns an integer that indicates the total number of
               uninterrupted terminal loops added to the partitioned skeleton

        test_func is a function that specifies a test function to be applied to
        each loop. It serves the same role as test_func in .partition_out_*()'s
        except that 1) only a Type 3 call is executed, with kind specified as
        "loop" (see .make_test_func() and the .make_deep_test_func() of the 
        function returned by that method), and 2) the abort value returned by 
        *_test_func() is ignored, because loops are iterated over in arbitrary 
        order. If test_func is not specified (None), every loop is considered 
        acceptable.

        Warning: After calling the current function, the deletion of paths from
        the partitioned skeleton could result in the isolation of uninterrupted
        terminal loops so that they no longer touch the rest of the partitioned
        skeleton. For this reason, it is suggested that the current function be
        called only after all path deletions have been executed.
        """
        quarantined_loop_dict = self._loop_dict
        orig_quarantined_loop_count = len(quarantined_loop_dict)
        vor_vert_idx_to_partitioned_degree_array = self._vor_vert_idx_to_partitioned_degree_array
        partitioned_loop_dict = self._kind_to_partitioned_path_dict["loop"]
        for loop in quarantined_loop_dict.keys():
            if not vor_vert_idx_to_partitioned_degree_array[loop._stem_vor_vert_idx]:
                continue
            if test_func is None or test_func(loop, True, "loop")[0]:
                partitioned_loop_dict[loop] = loop
                del quarantined_loop_dict[loop]
        return orig_quarantined_loop_count - len(quarantined_loop_dict)

    def isolate_graph_skeleton(self):
        """
        Force graph skeleton to be isolated (i.e., completion of Phase 1).

        Phase 1 ends with the isolation of the graph skeleton. It is begun at
        initialization but may only be completed lazily, to support interposed
        behavior in subclasses. Because it is guaranteed that Phase 1 will be
        completed when necessary, there is never a need to call the current
        function. However, because completing Phase 1 can be a computationally
        expensive operation, it can make the method call or even attribute use
        that triggers it appear desceptively expensive. If you are profiling
        code or otherwise want to avoid this behavior, calling the current
        function will complete Phase 1, or do nothing if Phase 1 is already
        completed.
        """
        self._graph_edge_dict

    @staticmethod
    def interpolate_3D(lines, copy_data=True):
        """
        Convert one or more lines to an equal number of LineString3D copies.

        A LineString3D or list of LineString3D's is returned, depending on
        whether lines is a LineString or sequence, respectively. The
        corresponding line is assigned to .original for each returned
        LineString3D.

        lines is a SkeletalLineString2D or sequence of that type that specifies
        the lines to be converted.

        copy_data is a boolean that specifies whether the .data of each returned
        line should be updated by the .data of the corresponding line from
        lines.

        Note: The current function only returns converted copies. The edges in
        the graph skeleton and paths in the partitioned skeleton remain
        SkeletalLineString2D's at all times.

        Warning: Conversion to 3D can have a significant memory footprint.
        """
        return_one_line = isinstance(lines, Skeletal)
        if return_one_line:
            lines = (lines,)  # *REASSIGNMENT*
        LineString3D = _geom.LineString3D
        lines3D = []
        lines3D_append = lines3D.append
        for line in lines:
            line3D = LineString3D(line.coords_array3D)
            line3D.original = line
            line3D.spatial_reference = line.spatial_reference
            if copy_data:
                line3D.data = line.data.copy()
            lines3D_append(line3D)
        if return_one_line:
            return line3D
        return lines3D

    @staticmethod
    def _get__graph_edge_dict(self):
        """
        Dictionary representing the graph skeleton.

        The current object has the same form as the edge_dict argument in
        ._make_csr_matrix(), and its creation triggers the completion of Phase 1
        (see ._isolate_graph_skeleton()).
        """
        self._isolate_graph_skeleton()
        return self._graph_edge_dict

    @staticmethod
    def _get__vor_vert_idx_to_graph_degree_array(self):
        """
        Array mapping Voronoi vertex index to node degree in graph skeleton.

        The creation of the current object triggers the completion of Phase 1
        (see ._isolate_graph_skeleton()).
        """
        self._isolate_graph_skeleton()
        return self._vor_vert_idx_to_graph_degree_array

    def _isolate_graph_skeleton(self):
        """
        Isolate the graph skeleton from its complement.

        The current function completes Phase 1, which was begun at
        initialization, by
            1) Contracting through binodes and quarantining any loops (see
               ._contract_through_binodes()).
            2) "Cracking" hyperhubs, if necessary (see ._crack_hyperhubs()).
            3) Isolating the graph skeleton from its complement, cutting if
               necessary and allowed. In doing so, it implements the "CUT",
               "NAIVE", and "VERIFY" isolation_mode options specified at
               initialization.
        """
        # Contract adjacency dictionary of the skeleton and its
        # complement through binodes to create raw edge dictionary, and
        # quarantine any uninterrupted terminal loops.
        # Note: Any hyperhubs are also cracked. As a consequence,
        # uninterrupted non-terminal loops, which must stem from
        # hyperhubs, are also addressed and remain in the raw skeleton.
        nonperi_adj_dict = self._nonperi_adj_dict
        del self._nonperi_adj_dict
        components = _Components(self, nonperi_adj_dict)

        # Interpret isolation mode.
        isolation_mode = self.isolation_mode
        naive = "NAIVE" in isolation_mode
        verify = naive and "VERIFY" in isolation_mode
        # Note: A maximum of 2 cut iterations are allowed. See note
        # further below.
        cut = 2 if "CUT" in isolation_mode else False

        # Setup test polygon.
        # Note: The test polygon is either the input polygon or a proxy
        # constructed from sampled coordinates on the input polygon's
        # boundary. In either case, the test polygon is intended to
        # represent the polygon "seen" by Voronoi analysis.
        test_poly = self.test_poly
        test_poly.optimize()
        test_poly_contains_point = test_poly._contains_point

        # Iterate until skeleton is identified.
        # If non-naive:
        #     Test every hub in each component to determine if that hub
        #     lies within the test polygon. Identify as the skeleton
        #     candidate that component with the greatest number of hubs
        #     contained by the test polygon.
        # If naive:
        #     Iterate over each component, in order of decreasing non-
        #     binode count. Test only one hub (chosen arbitrarily) for
        #     each component, and identify as the skeleton candidate the
        #     first component for which the tested hub is contained by
        #     the test polygon.
        cand_skel_component_label = None  # Initialize.
        # Note: ._vor_vert_idx_to_raw_degree_array was assigned when
        # _Components was instantiated.
        vor_vert_idx_to_graph_degree_array = self._vor_vert_idx_to_graph_degree_array
        # Note: In this array, 0 = node is not tested, 1 = node is not
        # contained by test polygon, and 2 = node is contained by test
        # polygon.
        vor_vert_idx_to_contained_int_array = _numpy_zeros(
            (len(vor_vert_idx_to_graph_degree_array),), _numpy_int8
            )
        vor_vert_coords_array = self.voronoi_vertex_coords_array
        cut_vor_vert_idxs = []
        while True:

            # Determine what components to iterate over, and their order.
            if cand_skel_component_label is not None:
                component_labels = (cand_skel_component_label,)
            elif naive:
                component_labels = components.frequency_sorted_unique_component_labels_array
            else:
                component_labels = components.unique_component_labels_array
            max_contained_hub_count = 0  # Initialize.

            # Iterate over each component.
            for component_label in component_labels:
                contained_hub_count = 0  # Initialize.
                uncontained_hub_vor_vert_idxs = []  # Initialize.
                uncontained_hub_vor_vert_idxs_append = uncontained_hub_vor_vert_idxs.append
                vor_vert_idxs_so_labeled_array = components.get_idxs_for_component(
                    component_label
                    )

                # Test each hub for containment within test polygon.
                for vor_vert_idx_so_labeled in vor_vert_idxs_so_labeled_array:
                    if vor_vert_idx_to_graph_degree_array[vor_vert_idx_so_labeled] < 3:
                        continue
                    contained_int = vor_vert_idx_to_contained_int_array[vor_vert_idx_so_labeled]
                    if contained_int:
                        point_is_in_test_poly = contained_int == 2
                    else:
                        point_is_in_test_poly = test_poly_contains_point(
                            vor_vert_coords_array[vor_vert_idx_so_labeled].tolist()
                            )
                        vor_vert_idx_to_contained_int_array[vor_vert_idx_so_labeled] = point_is_in_test_poly + 1
                    if point_is_in_test_poly:
                        contained_hub_count += 1
                    else:
                        uncontained_hub_vor_vert_idxs_append(
                            vor_vert_idx_so_labeled
                            )
                    if naive:
                        break
                if contained_hub_count > max_contained_hub_count:
                    max_contained_hub_count = contained_hub_count
                    cand_skel_vor_vert_idxs_array = vor_vert_idxs_so_labeled_array
                    cand_skel_uncontained_hub_vor_vert_idxs = uncontained_hub_vor_vert_idxs
                    if naive:
                        break

            # If no skeleton candidate was identified and iteration was
            # naive, abandon naive isolation and restart iteration.
            # Otherwise error.
            if not max_contained_hub_count:
                if naive:
                    naive = False
                    continue
                raise TypeError(
                    "skeleton could not be isolated from its complement for unknown reason"
                    )

            # A skeleton candidate was identified. If it was naively
            # isolated, optionally start a verification iteration in
            # which the candidate is treated non-naively but as though
            # it were the only component (i.e., every one of its hubs
            # must be contained by the test polygon).
            if naive and verify:
                cand_skel_component_label = component_label
                naive = False
                continue

            # If no (tested) hubs lie outside the test polygon, accept
            # the candidate.
            if not cand_skel_uncontained_hub_vor_vert_idxs:
                break

            # The skeleton candidate has one or more hubs outside the
            # test polygon. If the just completed iteration was a
            # verification iteration following a naive skeleton
            # identification, restart a full non-naive iteration from
            # scratch.
            # Note: This scenario necessarily implies (Theorem 4.1 of
            # Brandt and Algazi, 1992) that at least some portion of the
            # input polygon is insufficiently sampled. More precisely,
            # the input polygon is somewhere locally narrower than twice
            # the sampling interval. In such cases, the skeleton and its
            # complement can become linked, forming a single component.
            if verify:  # naive was previously True.
                cand_skel_component_label = None
                verify = False
                continue

            # If cutting is not (or no longer) permitted, error.
            if not cut:
                # Note: ._min_cutting_failure_interval is used during
                # brute search for an optimal interval.
                interval = self.interval
                if self._min_cutting_failure_interval > interval:
                    self._min_cutting_failure_interval = interval
                raise TypeError(
                    "skeleton could not be isolated from its complement, likely because interval is too large{}: {}".format(
                        " and cutting was not permitted" if "CUT" not in isolation_mode else "",
                        interval
                        )
                    )

            # As a heuristic resolution, "cut" (delete) each hub in the
            # skeleton candidate that lies outside the test polygon as
            # well as any edges that touch that hub.
            # Note: This "fix" is far from ideal, as explained in the
            # documentation.
            self._cutting_was_attempted = True
            for uncontained_hub_vor_vert_idx in cand_skel_uncontained_hub_vor_vert_idxs:
                for adj_vor_idx in nonperi_adj_dict[uncontained_hub_vor_vert_idx]:
                    nonperi_adj_dict[adj_vor_idx].remove(
                        uncontained_hub_vor_vert_idx
                        )
                del nonperi_adj_dict[uncontained_hub_vor_vert_idx]
            cut_vor_vert_idxs.extend(cand_skel_uncontained_hub_vor_vert_idxs)

            # Rebuild contracted graph skeleton and restart iteration.
            # *REASSIGNMENT*
            components = _Components(self, nonperi_adj_dict)
            # Note: It is conceivable that the next iteration could
            # identify a different component as the skeleton candidate,
            # which could be cut and, in some cases, yield a desirable
            # result. However, this seems like a highly unlikely
            # scenario, would only arise for flawed (e.g., undersampled)
            # skeletons, and would risk infinite looping. Therefore,
            # only two cut iterations are allowed.
            cut -= 1

        # Store any cut Voronoi vertex indices in an array.
        self._cut_vor_vert_idxs_array = _numpy_fromiter(cut_vor_vert_idxs,
                                                        _numpy_int32)
        del cut_vor_vert_idxs  # Release memory.

        # Permanently discard all complement edges.
        # Note: Although this and the following block could use
        # ._register_line(), the existing code is faster.
        skeleton_vor_vert_idxs_set = set(cand_skel_vor_vert_idxs_array.tolist())
        # Note: ._graph_edge_dict was assigned when the most recent
        # _Components was instantiated.
        graph_edge_dict = self._graph_edge_dict
        for key in graph_edge_dict.keys():
            if key <= skeleton_vor_vert_idxs_set:
                continue
            del graph_edge_dict[key]
        loop_dict = self._loop_dict
        for line in loop_dict.keys():
            if line._stem_vor_vert_idx in skeleton_vor_vert_idxs_set:
                continue
            del loop_dict[line]
        del skeleton_vor_vert_idxs_set  # Release memory.

        # Create new record of node degrees that includes only nodes in
        # the identified skeleton.
        new_vor_vert_idx_to_graph_degree_array = self._vor_vert_idx_to_graph_degree_array = _numpy_zeros(
            vor_vert_idx_to_graph_degree_array.shape, _numpy_int8
            )
        new_vor_vert_idx_to_graph_degree_array[cand_skel_vor_vert_idxs_array] = vor_vert_idx_to_graph_degree_array[cand_skel_vor_vert_idxs_array]

    def _clear_graph_edge_dict_derivations(self):
        """
        Clear each unsynchronized attribute derived from ._graph_edge_dict.

        Each of these attributes is lazy and therefore will be lazily re-created
        when next needed (if ever).
        """
        self_dict_pop = self.__dict__.pop
        for attr_name in ():
            self_dict_pop(attr_name, None)

    def _clear_partitioned_edge_dicts_derivations(self):
        """
        Clear each attribute derived from ._kind_to_partitioned_edge_dict.

        More specifically, clear each unsychronized attribute derived from
        ._kind_to_partitioned_edge_dict. Each of these attributes is lazy and
        therefore will be lazily re-created when next needed (if ever).
        """
        self_dict_pop = self.__dict__.pop
        for attr_name in ():
            self_dict_pop(attr_name, None)

    def recontract_through_binodes(self):
        """
        Re-contract edges of graph skeleton through (degree 2) binodes.

        A (degree 2) "binode" exists wherever exactly two edges meet.
        Immediately prior to the start of the partitioning out, any nodes that
        are then binodes are irrelevant to the routing that underlies
        partitioning out. In essence, these binodes are like stations that a
        train passes through but at which it never stops. Because the train does
        not deviate when passing through these stations, the conductor doesn't
        require any specific knowledge of the stations. Similarly, prior to
        partitioning out, these binodes can be dropped from the graph skeleton.
        More accurately, in our train example, this would be like removing the
        pass-through stations from the train route's map but, of course,
        retaining the geometry of the links between stations at which the train
        does stop. Contracting through binodes can improve performance and (more
        significantly) decrease the memory footprint.

        Note: Edges are always contracted through binodes during initialization,
        but later processing (e.g., manual deletion of edges by .delete_lines())
        may render the graph skeleton "ripe" for another such contraction. After
        the start of partitioning out, however, any node that was a (degree 3)
        "hub" when partitiong out began may be reduced to a binode in the
        (residual) graph skeleton by the partitioning out of a path that
        terminated there. Nonetheless, these nodes remain essential to continued
        partitioning out. Therefore, calling the current function after the
        start of partitioning out will raise an error and leave the graph
        skeleton intact.

        Warning: The current function may itself be very expensive in terms of
        computation time.

        See also:
            .describe_edge()
            .describe_node()
        """
        self._contract_through_binodes(
            self._derive_adjacency_dict(self._graph_edge_dict),
            self._graph_edge_dict
            )

    def _crack_hyperhubs(self, adj_dict):
        """
        Replace (degree >3) hyperhubs with loops perpendicular to the x-y plane.

        Additional nodes with duplicate (x, y) coordinates are added to the
        graph skeleton (and relevant records) to ensure that no hub has degree
        >3. Effectively, this is identical to projecting a loop at the hub
        upward, out of the x-y plane. At various points along this loop,
        adjacent nodes in the x-y plane are attached so that no juncture has
        more than three edges. A loop is used to ensure that no stubs are added
        to the graph skeleton. The only divergence of the actual solution from
        this mental picture is that the loop has 0 size in x, y, and z
        coordinates. adj_dict, which is updated in place, is returned.

        adj_dict is a dictionary with the same form as the adj_dict argument in
        ._contract_through_binodes().

        Note: The current function is only called if a hyperhub is detected.

        Note: The current function is designed with the thought that it could be
        overridden (e.g., in a subclass). However, one should note the
        attributes assigned by the current function and likewise assign to these
        attributes compatible data in any overriding function. (See source
        code.)
        """
        # Create loop of duplicate nodes to crack each hyperhub in the
        # adjacency dictionary.
        orig_vor_vert_count = len(self._voronoi.out_coords_array)
        next_vor_vert_idx = orig_vor_vert_count - 1  # Initialize.
        dup_orig_tuples = []
        dup_orig_tuples_append = dup_orig_tuples.append
        orig_dup00_dupN_tuples = []
        orig_dup00_dupN_tuples_append = orig_dup00_dupN_tuples.append
        for orig_vor_vert_idx in adj_dict.keys():
            adj_idxs = adj_dict[orig_vor_vert_idx]
            if len(adj_idxs) > 3:

                # As an illustrative example, consider a hyperhub with
                # the following edges: 0-1, 0-2, 0-3, and 0-4. In this
                # example, the original Voronoi vertex index of the
                # hyperhub (orig_vor_vert_idx) is 0.
                dup_vor_vert_idxs = [orig_vor_vert_idx]
                # Only retain the first edge. In our example, that edge
                # is 0-1.
                deleted_adj_idxs = adj_idxs[1:]
                del adj_idxs[1:]
                # Store the first new duplicate node index.
                dup00 = next_vor_vert_idx + 1
                for adj_idx in deleted_adj_idxs:
                    # Generate a new node index. It will ultimately be
                    # assigned the same (x, y) coordinates as node 0, so
                    # we'll call the index 0' (though in reality,
                    # next_vor_vert_idx is numeric).
                    next_vor_vert_idx += 1
                    dup_vor_vert_idxs.append(next_vor_vert_idx)
                    # Replace edge 0-2 with 0'-2 in all records (where
                    # adj_idx is 2).
                    # Note: Records 0-2 (and 0-3 and 0-4) were deleted
                    # further above. Therefore add 0'-2 record...
                    adj_dict[next_vor_vert_idx] = [adj_idx]
                    # ...delete 2-0 record...
                    adj_adj_idxs = adj_dict[adj_idx]
                    adj_adj_idxs.remove(orig_vor_vert_idx)
                    # ...and replace it with 2-0' record.
                    adj_adj_idxs.append(next_vor_vert_idx)
                    # Also store the (0', 0) pair for later use.
                    dup_orig_tuples_append(
                        (next_vor_vert_idx, orig_vor_vert_idx)
                        )
                # Span new duplicate nodes with a path 0-0'-0''-...
                for dup0, dup1 in _slide_pairwise(dup_vor_vert_idxs):
                    adj_dict[dup0].append(dup1)
                    adj_dict[dup1].append(dup0)
                # Close path to form loop 0-0'-0''-...-0.
                adj_dict[dup1].append(orig_vor_vert_idx)
                adj_dict[orig_vor_vert_idx].append(dup1)
                # Note: The following edges now exist: 0-1, 0'-2, 0''-3,
                # 0'''-4, 0-0', 0'-0'', 0''-0''', and 0'''-0. Note that
                # no node links to >3 other nodes.

                # Store the Voronoi vertex indices that bracket the
                # block of duplicates.
                orig_dup00_dupN_tuples_append((orig_vor_vert_idx,
                                               dup00, dup1 + 1))

        # Record newly generated Voronoi vertex indices and their
        # corresponding original Voronoi vertex indices.
        vor_vert_idx_addendum = self._vor_vert_idx_addendum = _numpy_fromiter(
            _flatten_to_iter(dup_orig_tuples), _numpy_int32
            )
        vor_vert_idx_addendum.shape = _neg_1_2_tuple
        del dup_orig_tuples  # Release memory.
        self._approx_nonpartitioning_mem += _sys.getsizeof(
            vor_vert_idx_addendum
            )

        # Record the coordinates of the newly generated nodes.
        total_vor_vert_idx_count = next_vor_vert_idx + 1
        if self.memory_option:
            voronoi_vertex_coords_array = self._voronoi.out_coords_array.resize(
                (total_vor_vert_idx_count, 2), False
                )
            del self._voronoi.out_coords_array
            # Note: Non-partitioning memory footprint approximations are
            # assumed to scale linearly with post-cracking node count.
            # Because self.voronoi.out_coords_array does not exist and
            # self._voronoi.out_coords_array was just deleted, no
            # adjustment need be made to the memory footprint
            # accounting. (Compare note further below.)
        else:
            voronoi_vertex_coords_array = _numpy_empty(
                (total_vor_vert_idx_count, 2), _numpy_float64
                )
            voronoi_vertex_coords_array[:orig_vor_vert_count] = self._voronoi.out_coords_array
            # Note: self.voronoi.out_coords_array is now distinct from
            # self.voronoi_vertex_coords_array (assigned further below),
            # and both are retained. Only the latter would be accounted
            # for without the line below. (Compare note further above.)
            self._approx_nonpartitioning_mem += orig_vor_vert_count * 16.
        for orig, dup00, dupN in orig_dup00_dupN_tuples:
            voronoi_vertex_coords_array[dup00:dupN] = voronoi_vertex_coords_array[orig]
        del orig_dup00_dupN_tuples  # Release memory.
        # Note: Assignment to .voronoi_vertex_coords_array is
        # intentionally delayed until that array is fully populated.
        self.voronoi_vertex_coords_array = voronoi_vertex_coords_array

        # Return adjacency dictionary.
        # Note: Though adj_dict is updated in place in the current
        # function, allow for replacement of the current function with
        # one that instead populates a new dictionary.
        return adj_dict

    def _contract_through_binodes(self, adj_dict, source_edge_dict=None):
        """
        Contract edges of a graph through (degree 2) binodes.

        Commonly, many nodes link exactly two Voronoi segments (and therefore
        have degree 2 in a graph sense). These "binodes" are inconsequential for
        the path-finding between non-binodes executed during partitioning out.
        To minimize memory use during such path-finding (which can be
        prohibitive), the current function merges segments through binodes so
        that only (degree 1) "stubs" and (degree >=3) "hubs" remain. In graph
        theory, this process could be properly called path contraction between
        nodes whose respectives degrees are not 2. For brevity, the process is
        here termed "[path] contraction through binodes" or simply "binode
        contraction." Also from graph theory, links between nodes are called
        "edges". That term is likewise used here. Note that each edge comprises
        >=1 Voronoi segments. The current function can also be used to re-
        contract a graph skeleton's edges (which are no longer guaranteed to be
        Voronoi segments) through binodes, presumably after editing of that
        skeleton (so that the re-contraction has some effect). On each call, the
        following attributes are updated as described:
            ._graph_edge_dict                       created or re-created
            ._loop_dict                             extended
            ._vor_vert_idx_to_graph_degree_array    created or reset+repopulated

        adj_dict is a dict in which each key is an integer and each value is a
        list of integers. This "adjacency dictionary" represents a graph in
        which a node (indicated by its Voronoi index, stored as the dict key)
        can reach each of n other nodes (indicated by a list of their respective
        Voronoi indices, stored as the dict value). Beacuse this graph is
        invariably symmetric, each edge is doubly represented:
            all_indices_adjacent_to_i = adj_dict[i]
            nbr_idx = all_indices_adjacent_to_i[0]  # Arbitrary neighbor.
            all_indices_adjacent_to_nbr_idx = adj_dict[nbr_idx]
            i in all_indices_adjacent_to_nbr_idx --> True

        source_edge_dict is a dict with the same format as the edge_dict
        argument in ._make_csr_matrix(). If adj_dict was derived from an edge
        dictionary (so that the graph described by adj_dict may contain edges
        comprised of multiple Voronoi segments), that edge dictionary must be
        specified for the source_edge_dict argument to avoid erroneous results.

        Note: In addition, the current function "cracks" (degree >3)
        "hyperhubs" (via ._crack_hyperhubs()), that is, adds duplicate nodes to
        simulate the connectivity of each hyperhub but ensure that the maximum
        degree of any node is 3. The absence of hyperhubs is a fundamental
        assumption made throughout the current class.
        """
        # Abort (re)contraction if any partitioning-out has occurred.
        for path_dict in self._kind_to_partitioned_path_dict.itervalues():
            if path_dict:
                raise TypeError("binodes cannot be contracted through after partitioning-out has begun")

        # Check for and crack hyperhubs, if possible.
        for was_cracked in (False, True):
            for adj_idxs in adj_dict.itervalues():
                # If a hyperhub is detected, attempt cracking or error
                # if cracking was already attempted.
                # Note: It is not anticipated that cracking would ever
                # fail, unless ._crack_hyperhubs() is overridden.
                if len(adj_idxs) > 3:
                    if was_cracked:
                        raise TypeError(
                            "(degree >3) hyperhubs exist despite cracking"
                            )
                    self.had_hyperhub = True
                    # *REASSIGNMENT* (unless adj_dict updated in place)
                    adj_dict = self._crack_hyperhubs(adj_dict)
                    break
            else:
                # If no hyperhub was encountered (on the current
                # iteration), no further iteration is necessary.
                break
        # Warn if hyperhubs were cracked. Otherwise, update attributes.
        if was_cracked:
            _warnings.warn_explicit(
                "Voronoi diagram contained one or more (degree >3) hyperhubs. Duplicate nodes and 0-length edges have been inserted to reduce the degree of these nodes to 3.",
                UserWarning, __file__, 0
                )
        # Note: If the current function has been called previously,
        # no need to reassign to .voronoi_vertex_coords_array.
        elif "voronoi_vertex_coords_array" not in self.__dict__:
            # Note: .voronoi_vertex_coords_array is set
            # correctly, and ._voronoi.out_coords_array deleted,
            # if appropriate, by ._crack_hyperhubs() if it was
            # called.
            self.voronoi_vertex_coords_array = self._voronoi.out_coords_array
            if self.memory_option:
                del self._voronoi.out_coords_array

        # Localize some useful information.
        polygon_has_a_hole = len(self.polygon.boundary) > 1
        merge_path = self._merge_path

        # Walk from each not-yet-walked binode in each direction, and
        # also walk single-edge paths between non-binodes.
        # Note: Although this approach may not be the most obvious one,
        # it is robust to unusual situations like a skeleton that is a
        # single uninterrupted loop.
        if "_vor_vert_idx_to_graph_degree_array" in self.__dict__:
            # Note: Before re-contracting through binodes, both the node
            # degree records and the graph skeleton edge dictionary are
            # reset. The former is necessary because that array is also
            # used to ensure that each binode chain is walked exactly
            # once, and the latter keeps the code for ._register_line()
            # simpler as a result (by avoiding special-casing).
            vor_vert_idx_to_degree_array = self._vor_vert_idx_to_graph_degree_array
            vor_vert_idx_to_degree_array.fill(0)
        else:
            vor_vert_idx_to_degree_array = self._vor_vert_idx_to_graph_degree_array = _numpy_zeros(
                (len(self.voronoi_vertex_coords_array),), _numpy_int8
                )
        targ_edge_dict = self._graph_edge_dict = {}
        skeleton_is_a_loop = False  # This is the default.
        for start_idx, next_idxs in adj_dict.iteritems():

            # If starting at a non-binode, walk to each neighboring non-
            # binode.
            # Note: Because the edges linking neighboring non-binode
            # pairs are the only edges that do not belong to a binode
            # chain (including its ends), registering them here is
            # necessary.
            if len(next_idxs) != 2:
                for next_idx in next_idxs:
                    if len(adj_dict[next_idx]) != 2:

                        # Be careful to only register each edge once.
                        # Note: Each neighboring non-binode pair is seen
                        # twice: once for for each node in the pair.
                        vor_vert_idx_pair = (start_idx, next_idx)
                        # Note: "Faux" key is a frozenset whereas the
                        # keys in edge dictionaries are _Frozenset2's.
                        faux_key = frozenset(vor_vert_idx_pair)
                        if faux_key in targ_edge_dict:
                            continue

                        # Reuse preexisting LineString if possible.
                        if source_edge_dict is None:
                            edge = merge_path(vor_vert_idx_pair,
                                              source_edge_dict)
                        else:
                            edge = source_edge_dict[faux_key]
                        self._register_line(edge, None, vor_vert_idx_pair,
                                            start_idx, next_idx)
                continue

            # Only start walking a binode chain at a binode that has not
            # previously been walked.
            if vor_vert_idx_to_degree_array[start_idx]:
                continue

            # Prime relevant objects.
            # Note: "backward" and "forward" directions depend on the
            # current iteration. For example, during the second
            # iteration, variables with "backward" relate to what was
            # "forward" during the first iteration.
            first_idx_backward, next_idx_forward = next_idxs
            walked_path_idxs_deque = _deque((first_idx_backward, start_idx,
                                             next_idx_forward))
            prev_idx = start_idx  # Initialize.

            # Walk binode chain in one consistent (but otherwise
            # arbitrary) direction to its end ("first direction"), and
            # then repeat in the opposite direction ("second
            # direction").
            for is_first_direction, walked_path_nodes_deque_append in (
                (True, walked_path_idxs_deque.append),
                (False, walked_path_idxs_deque.appendleft)
                ):
                while True:

                    # End this walk if stub or hub is encountered.
                    next_idxs = adj_dict[next_idx_forward]
                    if len(next_idxs) != 2:
                        # If the chain was being walked in the first
                        # direction, prep for walking in the second
                        # direction.
                        if is_first_direction:
                            prev_idx = start_idx
                            next_idx_forward = first_idx_backward
                            break
                        break

                    # Be careful not to double back.
                    next_or_prev_idx1, next_or_prev_idx2 = next_idxs
                    if next_or_prev_idx1 == prev_idx:
                        prev_idx = next_idx_forward
                        next_idx_forward = next_or_prev_idx2
                    else:
                        prev_idx = next_idx_forward
                        next_idx_forward = next_or_prev_idx1

                    # If the chain is being walked in the first
                    # direction and the next node is the binode at which
                    # walking started, the skeleton must comprise a
                    # single uninterrupted loop. In this case, cancel
                    # walking in the second direction.
                    # Note: For example, polygon could be donut-shaped
                    # with a circle for its skeleton. Although loops can
                    # also occur as "leaves" to a skeleton, such loops
                    # would have to stem from a hub, which is not the
                    # case here.
                    # Note: The loop is already closed without adding
                    # next_idx_forward because of how
                    # walked_path_nodes_deque was initialized.
                    if (polygon_has_a_hole and is_first_direction and
                        next_idx_forward == start_idx):
                        skeleton_is_a_loop = True
                        break
                    walked_path_nodes_deque_append(next_idx_forward)

                # An end of the binode chain was reached. If walking in
                # the second direction is cancelled, break now.
                if skeleton_is_a_loop:
                    break

            # The binode chain has been fully walked (in both
            # directions, if necessary). Create edge and register it.
            edge = merge_path(walked_path_idxs_deque, source_edge_dict)
            self._register_line(edge, None, walked_path_idxs_deque)

            # If the skeleton comprises a single uninterrupted loop,
            # only binodes exist (including its start/end node), and all
            # of them have been contracted. Therefore, cease binode
            # contraction.
            if skeleton_is_a_loop:
                break

    @staticmethod
    def _convert_counts(counts, copy=False):
        """
        Convert an array of counts to an array of last indices and function.

        If counts is the number of students in each of len(counts) classes, the
        returned array would represent the last (0-indexed) index for the last
        student in each class, if all students were cumulatively enumerated and
        each class was counted in turn. For example, if the first three classes
        contain 0, 100, and 20 students, respectively, the returned array would
        begin: -1, 99, 119. The returned function is simply the .searchsorted of
        the returned array and can be used to identify the class given a
        student's index. In the earlier example, func(104) --> 1, the 0-indexed
        class to which the 104th student belongs, and func(0) --> 1, the 0-
        indexed class to whith the 0th (or "first") student belongs. The array
        and function are returned in a tuple of the form (last_indices_array,
        func).

        counts is an iterable of intgers.

        copy is a boolean that specifies whether a copy of counts should be made
        internally, and that copy populated with the last indices. If copy is
        False (the default) and counts is an array, the returned array is the
        same array as count but modified in place. copy only applies if counts
        is an array.
        """
        if not isinstance(counts, _numpy_ndarray):
            # *REASSIGNMENT*
            counts = _numpy_fromiter(counts, _numpy_int64)
        else:
            # Note: counts dtype must be compatible with -1.
            assert counts.dtype.kind != "u"
            if copy:
                counts = counts.copy()
        counts[0] -= 1
        last_idx_each_x_array = counts.cumsum(0, counts.dtype, counts)
        return last_idx_each_x_array, last_idx_each_x_array.searchsorted

    @staticmethod
    def _derive_adjacency_dict(edge_dict):
        """
        Derive an adjacency dictionary from an edge dictionary.

        The returned adjacency dictionary has the same form as the adj_dict
        argument in ._contract_through_binodes().

        edge_dict is a dict with the same form as the edge_dict argument in
        ._make_csr_matrix()
        """
        adj_dict = _defaultdict(list)
        for key in edge_dict:
            idx0, idxN = key.tuple
            adj_dict[idx0].append(idxN)
            adj_dict[idxN].append(idx0)
        return adj_dict

    def _fetch_associated_sample_index(self, vor_vert_idx, full_set=False):
        """
        Get one or more associated sample indices given a Voronoi vertex index.

        Sampled coordinates from the input polygon's boundary (and their
        indices) are considered "associated" with a Voronoi vertex if no other
        sampled coordinate is closer. Typically, there are exactly three, all
        equidistant from the Voronoi vertex, and there must always be at least
        that many.

        vor_vert_idx is an integer that specifies the Voronoi vertex index.

        full_set is a boolean that specifies whether all associated sample
        indices should be returned, in a Python set. If full_set is False (the
        default), only one (arbitrarily chosen) index is returned.
        """
        sorted_vor_vert_idx_sample_idx_pairs_array, search = self._sorted_vor_vert_idx_sample_idx_pairs_array_and_search
        idx0 = search(vor_vert_idx)
        if self.had_hyperhub:
            # Note: If the Voronoi diagram had a hub of degree >3, it
            # may be that the associated sample index found immediately
            # below is actually a placeholder necessitated by the
            # "cracking" of such hyperhubs (see ._crack_hyperhubs()). If
            # so, this placeholder represents the negative of the
            # (original) Voronoi vertex index minus 1. (Note that such
            # placeaholders are the only negative values in the array
            # and therefore can be easily recognized.) The query must be
            # redirected to that Voronoi vertex index.
            assoc_samp_idx = sorted_vor_vert_idx_sample_idx_pairs_array[idx0][1]
            if assoc_samp_idx < 0:
                vor_vert_idx = -assoc_samp_idx - 1  # *REASSIGNMENT*
                idx0 = search(vor_vert_idx)  # *REASSIGNMENT*
            elif not full_set:
                return assoc_samp_idx
        if full_set:
            idxN = search(vor_vert_idx, "right")
            return set(
                sorted_vor_vert_idx_sample_idx_pairs_array[idx0:idxN][:,1:].ravel().tolist()
                )
        return sorted_vor_vert_idx_sample_idx_pairs_array[idx0][1]
    
    @staticmethod
    def _test_pool_criteria(
        post_filter, pred_reidx_to_vor_vert_idx_array, pred_col_reidx,
        vor_vert_idxs_pool1_array, must_test_both_pools,
        vor_vert_idxs_pool2_array, truncate, pred_row_reidx,
        terminal_pred_reidx=_numpy_inf, pre_walking=True, **kwargs
        ):
        """
        Test whether the nodes at either end of a path satisfy pool criteria.

        The current function is simply a convenience function to facilitate the
        "post-filtering" of nodes and is used internally by
        ._find_longest_shortest_paths(). It relies heavily on local variables
        within that method and therefore is useless otherwise. A boolean is
        returned indicating whether the path does, or (if pre_walking is True)
        at least has the potential to, satisfy the (node) pool criteria.

        pre_walking is a boolean that specifies whether the function is being
        called prior to walking the path (so that it is not yet known whether
        this path may converge with another).
        """
        # Note: The current function is not necessarily optimized, but
        # clear and simple code is preferable here since the tests
        # enabled by the current function are considered "advanced"
        # functionality.
        # Note: terminal_pred_reidx will not be used unless pre_walking
        # is False. To avoid forcing calls to specify a dummy value for
        # terminal_pred_reidx, default terminal_pred_reidx to a value
        # that, if not overridden, will raise an error if it is used.
        # Note: Although the current function may be called within a
        # conditional statement in ._find_longest_shortest_paths(), it
        # is designed to behave appropriately (e.g., raising an error)
        # if it is called under any scenario within that method.

        # To be as general as possible, check whether post-filtering is
        # to be performed at all.
        if not post_filter:
            return True

        # Identify "start" Voronoi vertex index.
        # Note: Hereinafter, the "start" node is the initial "far" node
        # for walking. Effectively, walking initializes with a near and
        # far node, and then incrementally walks the far node back until
        # it either converges with the (unchanging) near node (for a
        # complete path) or converges with another path.
        start_vor_vert_idx = pred_reidx_to_vor_vert_idx_array[pred_col_reidx]

        # Determine whether the start node is in the first and second
        # pools, as necessary.
        start_in_pool1 = vor_vert_idxs_pool1_array[vor_vert_idxs_pool1_array.searchsorted(start_vor_vert_idx)] == start_vor_vert_idx
        if must_test_both_pools:
            start_in_pool2 = vor_vert_idxs_pool2_array[vor_vert_idxs_pool2_array.searchsorted(start_vor_vert_idx)] == start_vor_vert_idx

        # Determine whether the end node is in the first and second
        # pools, as necessary.
        if truncate and pre_walking:
            # Because truncation is allowed and the path has not yet
            # been walked, the ultimate end node is not yet known.
            # Therefore, the potential of that ultimate end node to
            # satisfy the pool criteria is unconstrained.
            end_in_pool1 = end_in_pool2 = True
        else:
            # Identify the end node.
            if pre_walking:
                end_vor_vert_idx = pred_reidx_to_vor_vert_idx_array[pred_row_reidx]
            else:
                end_vor_vert_idx = pred_reidx_to_vor_vert_idx_array[terminal_pred_reidx]
            end_in_pool1 = vor_vert_idxs_pool1_array[vor_vert_idxs_pool1_array.searchsorted(end_vor_vert_idx)] == end_vor_vert_idx
            if must_test_both_pools:
                end_in_pool2 = vor_vert_idxs_pool2_array[vor_vert_idxs_pool2_array.searchsorted(end_vor_vert_idx)] == end_vor_vert_idx

        # Return the result, which may only reflect the potential of the
        # path to fulfill the pool criteria.
        if must_test_both_pools:
            return ((start_in_pool1 and end_in_pool2) or
                    (start_in_pool2 and end_in_pool1))
        return start_in_pool1 or end_in_pool1

    def _convert_vor_vert_idxs_to_coords2D(self, vor_vert_idx1, vor_vert_idx2):
        """
        Convert two Voronoi vertex indices to their 2D coordinates.

        A tuple of arrays, each of shape (1, 2), are returned. The order of the
        arrays is the same as the order of arguments to the current function.

        vor_vert_idx1 is an integer that specifies the first Voronoi vertex
        index.

        vor_vert_idx1 is an integer that specifies the second Voronoi vertex
        index.
        """
        voronoi_vertex_coords_array = self.voronoi_vertex_coords_array
        return (voronoi_vertex_coords_array[vor_vert_idx1],
                voronoi_vertex_coords_array[vor_vert_idx2])

    def _convert_vor_vert_idxs_to_coords3D(self, vor_vert_idx1, vor_vert_idx2):
        """
        Convert two Voronoi vertex indices to their interpolated 3D coordinates.

        A tuple of arrays, each of shape (1, 3), are returned. The order of the
        arrays is the same as the order of arguments to the current function.
        See Skeletal._get_coords_array3D for documentation on the interpolation.

        vor_vert_idx1 is an integer that specifies the first Voronoi vertex
        index.

        vor_vert_idx1 is an integer that specifies the second Voronoi vertex
        index.

        Warning: The original input polygon must be 3D.

        Warning: The interpolated z-coordinates are approximate, and they may be
        especially crude (relative to reality) if the terrain across the input
        polygon's width is not approximately planar locally.
        """
        vor_vert_idxs_array = _numpy_fromiter((vor_vert_idx1, vor_vert_idx2),
                                              _numpy_int64)
        return tuple(
            SkeletalLineSegment2D(self, vor_vert_idxs_array).coords_array3D
            )

    def _find_longest_shortest_paths(self, kind, cost_func,
                                     from_vor_vert_idxs_array=None,
                                     to_vor_vert_idxs_array=None,
                                     vor_vert_idxs_pool1_array=None,
                                     vor_vert_idxs_pool2_array=None,
                                     test_func=None, one_path_only=False,
                                     truncate=True,
                                     forbidden_nodes_arrays=None):
        """
        Partition out successively shorter full paths from the graph skeleton.

        The current function executes a search (based on the Dijkstra or, if any
        costs are negative, the Johnson algorithm) for the minimum cumulative
        cost ("shortest") paths from each specified from-node to each specified
        to-node. (Edge costs are calculated using cost_func.) It then considers
        the highest cost ("longest") of these paths, followed by successively
        lower cost (shorter) paths (unless one_path_only is True and path is
        acceptable as determined by test_func() and
        vor_vert_idxs_pool*_array's). Note that this decreasing trend is cost is
        only guaranteed for full paths; if a path converges with an earlier
        accepted path, it is truncated at that point of convergence (if truncate
        is True) and therefore may have lower cost (be shorter) than other paths
        considered thereafter. This sequence continues until all paths are
        exhausted (or, if truncate is False, until the first convergence). After
        walking each path, it registers that path to the partitioned skeleton
        under the specified kind if that path is acceptable (as determined by
        test_func() and vor_vert_idxs_pool*_array's). Finally, it returns a
        tuple of the form
            (added_path_count, cumulative_cost, includes_truncated,
             considered_full_path_count, deleted_path_count)
        where
            added_path_count:      Total number of paths partitioned out
                                   (including each truncated part, if any).
            cumulative_cost:       Total cost of all paths (including at least
                                   those full paths that were not truncated).
            includes_truncated:    Boolean (True/False) indicating whether
                                   cumulative_cost includes the cost of any
                                   truncated parts that were partitioned out.
            considered_path_count: Total number of full paths that were
                                   considered. If search was aborted, this count
                                   includes the path that was being considered
                                   when the abort was triggered.
            deleted_path_count:    Total number of paths (including each
                                   truncated part, if any) deleted from the
                                   graph skeleton. This may be greater than
                                   added_path_count if test_func()'s returned
                                   values indicating that a path should be
                                   discarded without being partitioned out.

        kind is a string that specifies the key in
        self._kind_to_partitioned_path_dict to which any accepted paths will be
        registered.

        cost_func is a function that is called to assign a cost to each edge.
        See .make_cost_func() and._make_csr_matrix() for more details.

        from_vor_vert_idxs_array is a flat array of integers (but see note
        further below) that specifies the indices of Voronoi vertices that are
        permitted to be path starts.

        to_vor_vert_idxs_array is a flat array of integers (but see note further
        below) that specifies the indices of Voronoi vertices that are permitted
        to be path ends. If to_vor_vert_idxs_array is equivalent to
        from_vor_vert_idxs_array, specifying both arguments by the same array
        will improve performance.

        test_func is a function that is called to determine whether a particular
        path should be accepted and partitioned out, merely deleted from the
        graph skeleton, or skipped, and whether path-processing should continue.
        See .make_test_func().

        vor_vert_idxs_pool1_array is a flat array of integers (but see note
        further below) that specifies the pool of Voronoi vertex indices that
        are acceptable for the start or (possibly truncated) end of a path. If
        both vor_vert_idxs_pool1_array and vor_vert_idxs_pool2_array are
        specified (not None), acceptable paths must start at a Voronoi vertex
        with its index in one of these arrays and end at a Voronoi vertex with
        its index in the other array.

        vor_vert_idxs_pool2_array is a flat array of integers (but see note
        further below) that specifies the pool of Voronoi vertex indices that
        are acceptable for the start or (possibly truncated) end of a path. If
        both vor_vert_idxs_pool1_array and vor_vert_idxs_pool2_array are
        specified (not None), acceptable paths must start at a Voronoi vertex
        with its index in one of these arrays and end at a Voronoi vertex with
        its index in the other array.

        one_path_only is a boolean that specifies whether only one path should
        be processed. If one_path_only is True and test_func is specified (not
        None), multiple paths may be considered but iteration will cease upon
        the first accepted path. (.max and .min of cost_func are still honored.)

        truncate is a boolean that specifies whether paths may be truncated
        where they converge with an earlier path accepted on the current call.
        If truncate is False, path processing is aborted when the first such
        convergence is encountered.

        forbidden_nodes_arrays is a sequence of flat arrays of Voronoi vertex
        indices whose union specifies what nodes cannot be touched by any path.
        For convenience, forbidden_nodes_arrays can also be specified by a
        single flat array.

        Note: All arguments whose names end with "array" (therefore excluding
        forbidden_nodes_arrays) can be specified by any of the following:
            1) a flat array of integers
            2) a (non-array) flat sequence of integers
            3) a string corresponding to a node kind recognized by .get_nodes()
               for the graph skeleton
        Internally, the interdependence of these arguments is leveraged to
        optimize processing. As a simple example, if from_vor_vert_idxs_array
        is a subset of vor_vert_idxs_pool1_array and vor_vert_idxs_pool2_array
        is not specified, from_vor_vert_idxs_array guarantees satisfaction of
        the vor_vert_idxs_pool1_array criterion, which therefore need never be
        tested. To maximize this optimization, you should specify equivalent
        arguments by identical objects. For example, if you specify
        from_vor_vert_idxs_array=a, it is faster to specify
        to_vor_vert_idxs_array=a instead of to_vor_vert_idxs_array=a.copy().

        Note: A path is considered unacceptable as soon as it is known that any
        specified criterion cannot be satisfied: vor_vert_idxs_pool1_array,
        vor_vert_idxs_pool2_array, or test_func. Also, the current function
        returns immediately after convergence if truncate is True (or after
        acceptance of a single path if one_path_only is True). One consequence
        is that test_func() is not guaranteed to see every considered path.
        """
        # Create csr matrix for the graph skeleton, or its permitted
        # subset.
        graph_edge_dict = src_edge_dict = self._graph_edge_dict
        if (forbidden_nodes_arrays is not None
            and len(forbidden_nodes_arrays) > 0):
            # Exclude forbidden nodes by excluding their associated
            # edges.
            # Note: The code in this block, as well as the allowance to
            # specify forbidden_nodes_arrays by a sequence of flat
            # arrays, are intended to help minimize the memory
            # footprint, which can be prohibitive later in the current
            # function.
            if isinstance(forbidden_nodes_arrays, _numpy_ndarray):
                forbidden_nodes_array = forbidden_nodes_arrays
            else:
                forbidden_nodes_array = _numpy_concatenate(
                    forbidden_nodes_arrays
                    )
            del forbidden_nodes_arrays  # At least reduce namespace.
            forbidden_nodes = forbidden_nodes_array.tolist()
            del forbidden_nodes_array  # At least reduce namespace.
            forbidden_nodes_isdisjoint = set(forbidden_nodes).isdisjoint
            del forbidden_nodes  # Release memory.
            src_edge_dict = graph_edge_dict.copy()  # *REASSIGNMENT*
            for key in graph_edge_dict:
                if forbidden_nodes_isdisjoint(key.tuple):
                    continue
                del src_edge_dict[key]
            del forbidden_nodes_isdisjoint  # Release memory.
        src_csr_matrix, pred_reidx_to_vor_vert_idx_array = self._make_csr_matrix(
            src_edge_dict, cost_func
            )
        del src_edge_dict  # Release memory or reduce local namepsace.
        convert_vor_vert_idx_to_pred_reidx = pred_reidx_to_vor_vert_idx_array.searchsorted

        # Note: The next several blocks focus on interpreting and
        # optimizing the "input 'arrays'" (i.e., *_array arguments).

        # Convert each input "array" to a unique (and true) array (that
        # is also sorted).
        # Note: When ultimately interpreted, each input "array" is
        # either  an array or unspecified. Each may initially be an
        # array, an array-like object, or a node kind specified as a
        # string.
        input_arrays = [from_vor_vert_idxs_array, to_vor_vert_idxs_array,
                        vor_vert_idxs_pool1_array, vor_vert_idxs_pool2_array]
        # Note: This dictionary is used to ensure that identical
        # values are uniquified (and, if applicable, generated) only
        # once.
        input_array_ID_to_unique_array = {id(None): None}
        std_null_result = (0, 0., True, 0, 0)
        unique_input_arrays = []
        for input_array in input_arrays:
            if isinstance(input_array, basestring):
                ID = input_array
                if ID not in input_array_ID_to_unique_array:
                    unique_array = self.get_nodes(input_array, coords=False)[0]
                    if unique_array is None:
                        # Generated array is empty and therefore
                        # excludes all possible paths.
                        return std_null_result
            else:
                ID = id(input_array)
                if ID not in input_array_ID_to_unique_array:
                    if not len(input_array):
                        # Input array is empty and therefore excludes
                        # all possible paths.
                        return std_null_result
                    unique_array = _uniquify_flat_array(
                        _numpy.array(input_array), False
                        )
            if ID in input_array_ID_to_unique_array:
                unique_array = input_array_ID_to_unique_array[ID]
            else:
                input_array_ID_to_unique_array[ID] = unique_array
            unique_input_arrays.append(unique_array)
        # *REASSIGNMENTS* (except for None's)
        (from_vor_vert_idxs_array,
         to_vor_vert_idxs_array,
         vor_vert_idxs_pool1_array,
         vor_vert_idxs_pool2_array) = unique_input_arrays
        # Release memory.
        del input_array_ID_to_unique_array, unique_input_arrays

        # In the special case that only a self-referential path can
        # satisfy the node criteria, return immediately.
        if (from_vor_vert_idxs_array is to_vor_vert_idxs_array and
            from_vor_vert_idxs_array is not None and
            len(from_vor_vert_idxs_array) == 1) or (
                vor_vert_idxs_pool1_array is vor_vert_idxs_pool2_array and
                vor_vert_idxs_pool1_array is not None and
                len(vor_vert_idxs_pool1_array) == 1
                ):
            return std_null_result

        # Clear each Voronoi vertex index pool, if possible.
        # Note: Be careful to only clear both pools if from- and to-
        # nodes together guarantee that each path will start at a node
        # from one pool and end at a node from the other pool.
        # Note: The code below is simple and may result in one or both
        # pools being unnecessarily "re-cleared" (i.e., re-assigned
        # None) if from- and/or to-nodes are None.
        # Note: If both pools are specified, special care must be taken
        # to ensure that (erroneous) simplification of the pool criteria
        # doesn't permit, for example, a path to start at a node that is
        # common to both pools but end at a node found in neither pool.
        # More generally, if both pools are specified, they must be
        # treated as a pair rather than independently. This principle is
        # important throughout the next several blocks.
        specified_pool_count = ((vor_vert_idxs_pool1_array is not None) +
                                (vor_vert_idxs_pool2_array is not None))
        if specified_pool_count == 2:
            if ((from_vor_vert_idxs_array is vor_vert_idxs_pool1_array and
                 to_vor_vert_idxs_array is vor_vert_idxs_pool2_array) or
                (from_vor_vert_idxs_array is vor_vert_idxs_pool2_array and
                 to_vor_vert_idxs_array is vor_vert_idxs_pool1_array)):
                vor_vert_idxs_pool1_array = vor_vert_idxs_pool2_array = None
        elif from_vor_vert_idxs_array is vor_vert_idxs_pool1_array:
            vor_vert_idxs_pool1_array = None
            if to_vor_vert_idxs_array is vor_vert_idxs_pool2_array:
                vor_vert_idxs_pool2_array = None
        elif from_vor_vert_idxs_array is vor_vert_idxs_pool2_array:
            vor_vert_idxs_pool2_array = None
            if to_vor_vert_idxs_array is vor_vert_idxs_pool1_array:
                vor_vert_idxs_pool1_array = None

        # If cost is not directed, it is further possible to replace
        # unspecified from- and to-nodes with (possibly specified)
        # pools (and then clear the relevant pools).
        # Note: In effect, this exchanges post-filtering of nodes in
        # preference for (faster) pre-filtering. (See notes further
        # below.) It also continues the effort from the previous block
        # to clear each pool, as possible.
        cost_is_directed = getattr(cost_func, "is_directed", True)
        # *REASSIGNMENT*
        specified_pool_count = ((vor_vert_idxs_pool1_array is not None) +
                                (vor_vert_idxs_pool2_array is not None))
        if not cost_is_directed:
            if specified_pool_count == 2:
                if (from_vor_vert_idxs_array is None and
                    to_vor_vert_idxs_array is None):
                    from_vor_vert_idxs_array = vor_vert_idxs_pool1_array
                    to_vor_vert_idxs_array = vor_vert_idxs_pool2_array
                    vor_vert_idxs_pool1_array = vor_vert_idxs_pool2_array = None
            else:
                # Note: Because from- and to-nodes are swapped further
                # below if necessary for optimization, no care is
                # required here in specifying one versus the other.
                if (from_vor_vert_idxs_array is None and
                    vor_vert_idxs_pool1_array is not None):
                    from_vor_vert_idxs_array = vor_vert_idxs_pool1_array
                    vor_vert_idxs_pool1_array = None
                if (to_vor_vert_idxs_array is None and
                    vor_vert_idxs_pool2_array is not None):
                    to_vor_vert_idxs_array = vor_vert_idxs_pool2_array
                    vor_vert_idxs_pool2_array = None

        # Identify those Voronoi vertex indices that are permitted for
        # use, temporarily ignoring explicit from- and to-node criteria.
        # Note: Routing can only utilize the ends of edges in the graph
        # skeleton. Furthermore, if both pools of indices are specified,
        # their union may further subset the permitted indices.
        # Note: This dictionary is used to ensure that identical arrays
        # are processed only once.
        unique_input_array_ID_to_processed_array = {id(None): None}
        if (vor_vert_idxs_pool1_array is None or
            vor_vert_idxs_pool2_array is None):
            permitted_end_vor_vert_idxs_array = pred_reidx_to_vor_vert_idx_array
        else:
            if vor_vert_idxs_pool1_array is vor_vert_idxs_pool2_array:
                full_vor_vert_idxs_pool_array = vor_vert_idxs_pool1_array
            else:
                full_vor_vert_idxs_pool_array = _union_flat_arrays(
                    vor_vert_idxs_pool1_array, vor_vert_idxs_pool2_array
                    )
            # If both pools of indices were generated, they are each
            # guaranteed to be a subset of the edge ends in the graph
            # skeleton (i.e., pred_reidx_to_vor_vert_idx_array).
            if (isinstance(input_arrays[2], basestring) and
                isinstance(input_arrays[3], basestring)):
                permitted_end_vor_vert_idxs_array = full_vor_vert_idxs_pool_array
            else:
                permitted_end_vor_vert_idxs_array = _numpy.intersect1d(
                    full_vor_vert_idxs_pool_array,
                    pred_reidx_to_vor_vert_idx_array, True
                    )
            # In the special case that both pools are identical:
            # 1) It is guaranteed that all "processed" arrays (as
            #    described further below) will satisfy the pool
            #    requirement, so both pools may be cleared.
            # 2) Each pool has now been processed.
            if vor_vert_idxs_pool1_array is vor_vert_idxs_pool2_array:
                vor_vert_idxs_pool1_array = vor_vert_idxs_pool2_array = None
                unique_input_array_ID_to_processed_array[id(vor_vert_idxs_pool1_array)] = permitted_end_vor_vert_idxs_array
            del full_vor_vert_idxs_pool_array  # Release memory.

        # Process each unique (and sorted) input array, ensuring that
        # the values in each come only from the permitted Voronoi vertex
        # indices just found.
        # *REASSIGNMENT*
        unique_input_arrays = [from_vor_vert_idxs_array, to_vor_vert_idxs_array,
                               vor_vert_idxs_pool1_array,
                               vor_vert_idxs_pool2_array]
        processed_input_arrays = []
        for input_array, unique_input_array in _izip(input_arrays,
                                                     unique_input_arrays):
            # If an input array identical to the current one (i.e.,
            # represented by the same object) was already seen,
            # reuse the earlier processed result.
            ID = id(unique_input_array)
            if ID in unique_input_array_ID_to_processed_array:
                processed_array = unique_input_array_ID_to_processed_array[ID]
            # If permitted Voronoi vertex indices are identical to some
            # subset of edge ends in the graph skeleton, it is
            # guaranteed that any generated array is already processed.
            elif (permitted_end_vor_vert_idxs_array is pred_reidx_to_vor_vert_idx_array and
                  isinstance(input_array, basestring)):
                processed_array = unique_input_array
            elif _is_subset(unique_input_array,
                            permitted_end_vor_vert_idxs_array,
                            a_is_unique=True, b_is_unique=True, sort_a=False,
                            sort_b=False):
                processed_array = unique_input_array
            else:
                processed_array = _numpy.intersect1d(
                    unique_input_array, permitted_end_vor_vert_idxs_array, True
                    )
            if ID not in unique_input_array_ID_to_processed_array:
                unique_input_array_ID_to_processed_array[ID] = processed_array
            processed_input_arrays.append(processed_array)
        # *REASSIGNMENTS* (except for special cases)
        (from_vor_vert_idxs_array, to_vor_vert_idxs_array,
         vor_vert_idxs_pool1_array,
         vor_vert_idxs_pool2_array) = processed_input_arrays
        # Release memory and reduce local namespace.
        del (input_arrays, processed_input_arrays,
             permitted_end_vor_vert_idxs_array,
             unique_input_array_ID_to_processed_array, unique_input_arrays)

        # If no permitted from- and/or to-nodes are available, return
        # now.
        if ((from_vor_vert_idxs_array is not None and
             not len(from_vor_vert_idxs_array)) or
            (to_vor_vert_idxs_array is not None and
             not len(to_vor_vert_idxs_array))):
            return std_null_result

        # Minimize (internal) post-filtering of nodes, as possible.
        # Note: In essence, this is the third block that attempts to
        # clear each pool, as possible, but it is the most thorough.
        # Note: Post-filtering is slower but more versatile than pre-
        # filtering. Both are separately described in detail in notes
        # further below.
        # *REASSIGNMENT*
        specified_pool_count = ((vor_vert_idxs_pool1_array is not None) +
                                (vor_vert_idxs_pool2_array is not None))
        internal_post_filter = specified_pool_count > 1
        if internal_post_filter:
            # If either from- or to-nodes are a subset of a pool, that
            # pool's criterion is guaranteed to be satisfied (but
            # clearing both pools is more complicated).
            # Note: Be careful to only clear both pools if from- and
            # to-nodes together guarantee that each path will start at
            # a node from one pool and end at a node from the other
            # pool. Prefer to clear the larger pool if both pools are
            # specified.
            if specified_pool_count == 2:
                if ((_is_subset(from_vor_vert_idxs_array,
                                vor_vert_idxs_pool1_array,
                                a_is_unique=True, b_is_unique=True,
                                sort_a=False, sort_b=False) and
                     _is_subset(to_vor_vert_idxs_array,
                                vor_vert_idxs_pool2_array,
                                a_is_unique=True, b_is_unique=True,
                                sort_a=False, sort_b=False)) or
                    (_is_subset(from_vor_vert_idxs_array,
                                vor_vert_idxs_pool2_array,
                                a_is_unique=True, b_is_unique=True,
                                sort_a=False, sort_b=False) and
                     _is_subset(to_vor_vert_idxs_array,
                                vor_vert_idxs_pool1_array,
                                a_is_unique=True, b_is_unique=True,
                                sort_a=False, sort_b=False))):
                    vor_vert_idxs_pool1_array = vor_vert_idxs_pool2_array = None
                    internal_post_filter = False  # *REASSIGNMENT*
            else:
                # Only one pool is specified. Ensure that it is the first
                # first pool, swapping pools if necessary.
                if vor_vert_idxs_pool1_array is None:
                    vor_vert_idxs_pool1_array, vor_vert_idxs_pool2_array = vor_vert_idxs_pool2_array, vor_vert_idxs_pool1_array
                if (_is_subset(from_vor_vert_idxs_array,
                               vor_vert_idxs_pool1_array,
                               a_is_unique=True, b_is_unique=True,
                               sort_a=False, sort_b=False) or
                    _is_subset(to_vor_vert_idxs_array,
                               vor_vert_idxs_pool1_array,
                               a_is_unique=True, b_is_unique=True,
                               sort_a=False, sort_b=False)):
                    vor_vert_idxs_pool1_array = None
                    internal_post_filter = False  # *REASSIGNMENT*

            # Prepare for internal post-filtering, if necessary.
            if internal_post_filter:
                test_pool_criteria = self._test_pool_criteria
                # Note: The variable(s) below are used within
                # ._test_pool_criteria().
                must_test_both_pools = vor_vert_idxs_pool2_array is not None
                vor_vert_idxs_pool1_array.sort()
                if must_test_both_pools:
                    vor_vert_idxs_pool2_array.sort()

        # If cost is not directed and there are more from- than to-
        # nodes, swap these variables.
        # Note: This swapping, if executed, permits the memory footprint
        # to be reduced earlier during path-finding (further below).
        from_and_to_nodes_are_swapped = (
            not cost_is_directed and
            (from_vor_vert_idxs_array is None or
             (to_vor_vert_idxs_array is not None and
              len(to_vor_vert_idxs_array) < len(from_vor_vert_idxs_array)))
            )
        if from_and_to_nodes_are_swapped:
            from_vor_vert_idxs_array, to_vor_vert_idxs_array = (
                to_vor_vert_idxs_array, from_vor_vert_idxs_array
                )

        # Note: *_array arguments (i.e., "input 'arrays'") are now
        # interpreted and optimized.

        # Prepare for external post-filtering, if necessary.
        # Note: Post-filtering is slower but more versatile than pre-
        # filtering. Both are separately described in detail in notes
        # further below.
        test_func_is_specified = test_func is not None
        external_post_filter = test_func_is_specified and getattr(
            test_func, "test_ends", True
            )
        if external_post_filter:
            if self.polygon.is_3D:
                convert_vor_vert_idxs_to_coords = self._convert_vor_vert_idxs_to_coords3D
            else:
                convert_vor_vert_idxs_to_coords = self._convert_vor_vert_idxs_to_coords2D

        # Convert (processed) input arrays to use sorted re-indices.
        # *REASSIGNMENT*
        processed_input_arrays = [from_vor_vert_idxs_array,
                                  to_vor_vert_idxs_array,
                                  vor_vert_idxs_pool1_array,
                                  vor_vert_idxs_pool2_array]
        # Reduce local namespace.
        del (from_vor_vert_idxs_array, to_vor_vert_idxs_array,
             vor_vert_idxs_pool1_array, vor_vert_idxs_pool2_array)
        # Note: This dictionary is used to ensure that identical arrays
        # are converted (and sorted) only once.
        processed_input_array_ID_to_pred_reidxs_array = {id(None): None}
        pred_reidxs_arrays = []
        for processed_input_array in processed_input_arrays:
            # Note: If a processed input array identical to the current
            # one (i.e., represented by the same object) was already
            # seen, reuse the earlier converted result.
            ID = id(processed_input_array)
            if ID in processed_input_array_ID_to_pred_reidxs_array:
                pred_reidxs_array = processed_input_array_ID_to_pred_reidxs_array[ID]
            else:
                pred_reidxs_array = processed_input_array_ID_to_pred_reidxs_array[ID] = convert_vor_vert_idx_to_pred_reidx(
                    processed_input_array
                    )
                pred_reidxs_array.sort()
            pred_reidxs_arrays.append(pred_reidxs_array)
        (from_pred_reidxs_array, to_pred_reidxs_array,
         pred_reidxs_pool1_array, pred_reidxs_pool2_array) = pred_reidxs_arrays
        # Release memory and reduce local namespace.
        del (pred_reidxs_arrays, processed_input_arrays,
             processed_input_array_ID_to_pred_reidxs_array)

        # Determine which shortest-path algorithm to use and its
        # arguments.
        shortest_path_args = self._shortest_path_args.copy()
        maximum = getattr(
            test_func, "max", _numpy_nearly_inf
            ) if test_func_is_specified else _numpy_nearly_inf
        # Note: As noted in the documentation to .make_test_func(),
        # (positive) infinity is invariably excluded.
        if maximum == _numpy_inf:
            # *REASSIGNMENT*
            maximum = _numpy_nearly_inf
        minimum = getattr(
            test_func, "min", _numpy_nearly_neginf
            ) if test_func_is_specified else _numpy_nearly_neginf
        # Note: As noted in the documentation to .make_test_func(),
        # negative infinity is invariably excluded.
        if minimum == _numpy_neginf:
            # *REASSIGNMENT*
            minimum = _numpy_nearly_neginf
        if (not getattr(cost_func, "negative", True) or
            src_csr_matrix.data.min() >= 0):
            shortest_path_func = _scipy.sparse.csgraph.dijkstra
            if maximum < _numpy_nearly_inf:
                shortest_path_args["limit"] = maximum
        else:
            shortest_path_func = _scipy.sparse.csgraph.johnson
        ## Note: In the special case that the input arrays constrain
        ## routes to being only hub-hub, the "indices" argument could be
        ## specified to exclude all stubs, but it is not clear how much
        ## performance gain would result.

        # Find all shortest paths between each pair of nodes (possibly
        # subset by the "indices" argument to the path-finding
        # function).
        # Note: The call below returns two dense square arrays that,
        # in typical cases, result in the maximum memory footprint
        # throughout all processing until they are deleted or subset.
        # Edge contraction through binodes is implemented during
        # initialization to reduce this footprint.
        cost_array, predecessors_array = shortest_path_func(
            src_csr_matrix, **shortest_path_args
            )

        # Subset the costs array, if appropriate.
        # Note: Elsewhere in the current function, this subsetting is
        # called node "pre-filtering", because nodes are filtered out
        # prior to path-walking (or any other inspection of the
        # results). As is demonstrated below, this pre-filtering reduces
        # the memory footprint (after momentarily increasing it),
        # ensuring that the peak memory footprint is smaller. It also
        # increases performance, as it reduces the number of paths that
        # are analyzed further below. Pre-filtering is also much more
        # computationally efficient (on a per-node basis) than post-
        # filtering, which is applied further below.
        cost_array_is_rereindexed = False  # This is the default.
        if from_pred_reidxs_array is not None:
            # *REASSIGNMENT*
            cost_array = _take2(cost_array, from_pred_reidxs_array)
            cost_array_is_rereindexed = True
        retest_on_convergence = (
            test_func_is_specified and
            getattr(test_func, "retest_on_convergence", True)
            )
        if retest_on_convergence:
            # Note: In case cost_array is redefined futher below (by
            # slicing out to-nodes), preserve it now for calculating
            # truncated costs.
            cost_array_full_rows = cost_array
        if to_pred_reidxs_array is not None:
            # *REASSIGNMENT*
            cost_array = _take2(cost_array, to_pred_reidxs_array, axis=1)
            cost_array_is_rereindexed = True
        cost_array_col_count = cost_array.shape[1]
        cost_array_flat = cost_array.ravel()

        # Identify the indices in the costs array over which to iterate.

        # Find only the single highest cost path if it's guaranteed that
        # only that path need be considered.
        # Note: Even if only one path should be processed, if a test
        # function is specified, multiple paths need to be identified in
        # case the first one is rejected.
        if ((one_path_only and not test_func_is_specified) or
            len(cost_array_flat) == 1):
            # Apply maximum cost restriction, if necessary.
            # Note: If "limit" argument was specified for the path-
            # finding function, costs that would have exceeded the
            # allowed maximum are instead set to positive infinity. If
            # any such positive infinity values exists, they are
            # addressed further below.
            if "limit" not in shortest_path_args:
                cost_array_flat[cost_array_flat > maximum] = _numpy_neginf

            # Find the index of the finite maximum in the costs array
            # (if any).
            try:
                cost_array_flat_idx = _bottleneck_nanargmax(cost_array_flat)
            except ValueError:
                # Note: No permitted paths exist (i.e., cost_array_flat
                # has 0 length or contains only nan's).
                cost_array_flat_idx = None
            else:
                max_cost = cost_array_flat[cost_array_flat_idx]
                if not _numpy_isfinite(max_cost):
                    # If the non-nan maximum is positive infinity, set
                    # positive infinite values to negative infinity and
                    # try again.
                    # Note: For example, if "limit" argument was
                    # specified for the path-finding function, costs
                    # that would have exceeded the allowed maximum were
                    # set to positive infinity.
                    if max_cost == _numpy_inf:
                        _bottleneck_replace(cost_array_flat, _numpy_inf,
                                            _numpy_neginf)
                        # *REASSIGNMENT*
                        cost_array_flat_idx = _bottleneck_nanargmax(
                            cost_array_flat
                            )
                        # *REASSIGNMENT*
                        max_cost = cost_array_flat[cost_array_flat_idx]
                        if not _numpy_isfinite(max_cost):
                            # Note: No permitted paths exist.
                            cost_array_flat_idx = None  # *REASSIGNMENT*
                    else:
                        # Note: No permitted paths exist.
                        cost_array_flat_idx = None  # *REASSIGNMENT*

            # If no permitted path exists, return now.
            if cost_array_flat_idx is None or max_cost < minimum:
                return std_null_result
            cost_array_flat_idxs = (cost_array_flat_idx,)

        # In the most general case, find all paths that cannot be
        # immediately rejected (e.g., as having lower cost that the
        # minimum allowed).
        else:

            # If cost is not directed and pre-filtering was symmetric
            # (or not applied), disregard equivalent (but reversed)
            # paths.
            # Note: Paths (defined by start and end indices) are
            # duplicated across the diagonal of the current costs array
            # if either (1) no pre-filtering was applied (in which case,
            # from_pred_reidxs_array and to_pred_reidxs_array are
            # identical, because each is None) or (2) pre-filtering was
            # symmetric (in which case from_pred_reidxs_array and
            # to_pred_reidxs_array are identical but not None). In these
            # cases, removing indices lying above the diagonal will
            # remove the duplicate paths. However, if pre-filtering was
            # asymmetric, it could be that only one of these duplicate
            # paths is retained in the sliced costs array, and therefore
            # the simple diagonal-based optimization implemented below
            # is not applicable.
            # Note: If only one path is to be found, and therefore only
            # one iteration will be executed, this "optimization" is
            # unnecessary (and obviously not an optimization). The same
            # logic would apply if only a few paths are to be found, but
            # that case is not tested for.
            if (not cost_is_directed and
                from_pred_reidxs_array is to_pred_reidxs_array):
                cost_array_flat[_triu_indices_flat(cost_array_col_count)] = _numpy_neginf

            # Find the indices that would sort the costs array.
            cost_array_flat_argsort = cost_array_flat.argsort()

            # Count the number of costs that are too small or too large.
            if cost_array_flat[cost_array_flat_argsort[0]] < minimum:
                too_small_count = cost_array_flat.searchsorted(
                    minimum, "left", cost_array_flat_argsort
                    )
            else:
                too_small_count = 0
            # Note: Although nan > x --> False regardless of x, nan
            # sorts higher than positive infinity. Therefore, it must be
            # tested that the maximum sorted costs is both finite (which
            # excludes nan) and less than or equal the permitted
            # maximum.
            # Note: If "limit" argument was specified for the path-
            # finding function, values that were too large were replaced
            # with positive infinity, which is necessarily too large, so
            # the same code can be used regardless.
            max_cost = cost_array_flat[cost_array_flat_argsort[-1]]
            if not _numpy_isfinite(max_cost) or max_cost > maximum:
                # Note: Because nan sorts higher than positive infinity,
                # too_large_count is guaranteed to count both nan and
                # positive infinity values (if the permitted maximum is
                # not nan).
                too_large_idx = cost_array_flat.searchsorted(
                    maximum, "right", cost_array_flat_argsort
                    )
            else:
                too_large_idx = len(cost_array_flat)

            # Subset and reverse the sorting indices so that only
            # permitted costs, in descending order, will be iterated
            # over.
            # *REASSIGNMENT*
            cost_array_flat_argsort = cost_array_flat_argsort[too_small_count:too_large_idx]
            # If no permitted path exists, return now.
            if len(cost_array_flat_argsort) == 0:
                return std_null_result
            # Note: Don't convert this array to a list because the
            # performance gain is small but the memory cost could be
            # significant. Recall that the memory footprint may be
            # precariously large at this point, as explained further
            # above.
            cost_array_flat_idxs = cost_array_flat_argsort[::-1]
            # Reduce local namespace (and prime for later memory
            # release).
            del cost_array_flat_argsort
        # Reduce local namespace (and prime for later memory release).
        del cost_array

        # Tag the self-referential diagonal of the predecessors array as
        # unwalkable if there are many paths to walk.
        # Note: The 10:1 ratio is somewhat arbitrary. The goal is to
        # avoid tagging the entire diagonal if only a few paths are to
        # be walked.
        # Note: Using a -1 of the same type as the predecessors array to
        # tag that array is slightly faster.
        neg_1 = predecessors_array.dtype.type(-1)
        tag_each_start = len(predecessors_array) // len(cost_array_flat_idxs) > 10
        if not tag_each_start:
            predecessors_array.ravel()[::len(predecessors_array) + 1] = neg_1

        # Prepare to iterate over each path.
        # Only perform reverse walks (further below) if cost is
        # undirected.
        if cost_is_directed:
            walking_tuple = (False,)
        else:
            walking_tuple = (False, True)
        # Localize some useful objects.
        merge_path = self._merge_path
        unravel_index2D = _util.unravel_index2D
        register_line = self._register_line
        # Initialize and set defaults.
        terminate = False
        added_path_count = 0
        cum_cost = 0.
        cum_cost_is_complete = True
        considered_full_path_count = 0
        deleted_path_count = 0
        cost_this_part = 0.  # Default for unknown costs.
        if test_func is None:
            accept = True
            abort = False
            delete_from_graph_skel = True

        # Iterate over each path in order of descending full path cost.
        # Note: The current function typically constitutes the bulk of
        # the total processing time. For a full partitioning, this is
        # primarily due to how many paths are iterated over (not, e.g.,
        # the path-finding).
        ## Check whether this is still true.
        for cost_array_flat_idx in cost_array_flat_idxs:

            # Test that the complete path cost (assuming no convergence)
            # is acceptable.
            considered_full_path_count += 1
            cost = cost_array_flat[cost_array_flat_idx]
            if test_func_is_specified:
                # Note: This is a Type 1a call as documented for
                # for .make_test_func().
                accept, abort = test_func(cost, True)
                if abort:
                    break
                if not accept:
                    continue

            # Find the row and column indices that match the path to the
            # predecessors array.
            pred_row_reidx, pred_col_reidx = unravel_index2D(
                cost_array_flat_idx, cost_array_col_count
                )
            if retest_on_convergence:
                cost_row_idx = pred_row_reidx
            if cost_array_is_rereindexed:
                if from_pred_reidxs_array is not None:
                    # *REASSIGNMENT*
                    pred_row_reidx = from_pred_reidxs_array[pred_row_reidx]
                if to_pred_reidxs_array is not None:
                    # *REASSIGNMENT*
                    pred_col_reidx = to_pred_reidxs_array[pred_col_reidx]
            # Note: It seems sufficiently safe to break on the first
            # encountered self-referential path, though this case should
            # usually be addressed by a well-chosen test_func.min.
            # (Also, if tag_each_start is False, each self-referential
            # path would be individually skipped further below because
            # of the tagging of the self-referential diagonal.)
            if pred_row_reidx == pred_col_reidx:
                break

            # Make up to two passes along the path, forward and reverse.
            # Note: Such "double"-walking is desirable for branch-
            # finding in which convergence does not terminate iteration,
            # because it ensures that the complete marginal ("stub-
            # facing") portion of the graph is walked. For example, if
            # the local branch-path with the greatest cost only
            # converges with the trunk over an interval of that path's
            # "midsection", the path will be walked from both ends and
            # terminated at the point of convergence with the trunk in
            # each case (resulting in two branches). Because of such
            # double-walking, the graph's branches can be fully
            # partitioned out in a single call of the current function
            # (if partitioning out is exhaustive, so that one_path_only
            # is False, truncate is True, etc., as also assumed in the
            # next example). Consider another example, in which a single
            # branch-path converges along two or more earlier found (and
            # hence higher cost, if complete) branches. In such a case,
            # only the stub-facing portions must be captured (and will
            # be captured, due to the double-walking) during the branch-
            # partitioning-out call, as the inter-hub (interior)
            # portions of such a branch-path are bridges. It is not
            # clear whether there is a similar benefit for bridge-
            # finding, but any cost should be small, as any attempt to
            # rewalk the same edge is quickly recognized.
            for is_second_walk in walking_tuple:

                # If on the second walk, prepare for walking in the
                # reverse sense.
                if is_second_walk:
                    pred_row_reidx, pred_col_reidx = (pred_col_reidx,
                                                      pred_row_reidx)

                # Unless only one path is to be found or the paths are
                # otherwise severely subset (e.g., by an extreme test
                # function), there will typically be a very large
                # number of paths to potentially walk for any but the
                # smallest datasets. However, after some small fraction
                # of these have been walked, nearly all remaining paths
                # will be subsets of earlier paths, and therefore should
                # be skipped. (These walked edges are tagged with -1 in
                # the predecessors_array.) This case is so common that
                # it is most efficient to test for it immediately rather
                # than do any preliminary preparation or optimization.
                if predecessors_array[pred_row_reidx, pred_col_reidx] != neg_1:

                    # Post-filter on start and possibly end nodes, as
                    # appropriate.
                    # Note: Post-filtering evaluates whether the start
                    # and end nodes in a path satisfy the pool criteria
                    # (cf. pre-filtering) and/or a user-specified
                    # function (see ends_test_func argument of
                    # .make_test_func()), called "internal" and
                    # "external" post-filtering, respectively. Post-
                    # filtering here ensures that a path is walked only
                    # if it is not "destined" (known a priori) to be
                    # disqualified by these criteria. If a path is not
                    # so disqualified, internal post-filtering may again
                    # be performed once the path has has been walked.
                    # Note: Only one end node is currently known for the
                    # path if truncation is permitted (because the other
                    # end may then be truncated). Therefore, post-
                    # filtering is only possible if truncation is
                    # disallowed or, in the case of internal post-
                    # filtering only, if two pools are specified (so
                    # that both end nodes must be tested).
                    if internal_post_filter and (
                        (must_test_both_pools or not truncate) and
                         not test_pool_criteria(**locals())
                        ):
                        continue
                    if external_post_filter and not truncate:
                        # Note: This is one of two places in the code
                        # where a Type 2 call, as documented for
                        # .make_test_func(), may be executed.
                        accept, abort = test_func(
                            cost, True, None,
                            *convert_vor_vert_idxs_to_coords(
                                pred_reidx_to_vor_vert_idx_array[pred_col_reidx],
                                pred_reidx_to_vor_vert_idx_array[pred_row_reidx]
                                )
                            )
                        if abort:
                            terminate = True  # *REASSIGNMENT*
                            break
                        if not accept:
                            continue

                    # Prepare/optimize for path walking.
                    path_pred_reidx_deque = _deque()
                    if is_second_walk != from_and_to_nodes_are_swapped:
                        path_pred_reidx_deque_appendleft = path_pred_reidx_deque.append
                        path_pred_reidx_deque_newest_idx = -1
                    else:
                        path_pred_reidx_deque_appendleft = path_pred_reidx_deque.appendleft
                        path_pred_reidx_deque_newest_idx = 0
                    predecessors_row = predecessors_array[pred_row_reidx]
                    far_reidx = pred_col_reidx  # Initialize
                    # Note: Whether tagging of the self-referential
                    # index (on the diagonal of the predecessor's array)
                    # was done above or will be done below, it handily
                    # allows the walking of a path to be arrested at
                    # that path's end in exactly the same way that a
                    # path is truncated at a convergence.
                    if tag_each_start:
                        predecessors_row[pred_row_reidx] = neg_1

                    # Walk.
                    while far_reidx != neg_1:
                        path_pred_reidx_deque_appendleft(far_reidx)
                        far_reidx = predecessors_row[far_reidx]

                    # Post-filter nodes, if necessary.
                    # Note: Internal post-filtering ensures that any
                    # walked path that is further considered (e.g.,
                    # passed to the test function) has one end sourced
                    # from each of the permitted pools (if both are
                    # specified). This test is termed post-filtering
                    # because it occurs after path-walking (cf. pre-
                    # filtering). External post-filtering ensures that
                    # any walked path that is further processed
                    # satisfies a user-specified function (called
                    # internally by test_func()).
                    # Note: If truncation is disallowed, both end nodes
                    # have already passed post-filtering (or the path
                    # converged and will be rejected further below).
                    terminal_pred_reidx = path_pred_reidx_deque[path_pred_reidx_deque_newest_idx]
                    path_is_complete = terminal_pred_reidx == pred_row_reidx
                    if truncate:
                        if internal_post_filter and not test_pool_criteria(
                            pre_walking=False, **locals()
                            ):
                            continue
                        if external_post_filter:
                            # Note: This is one of two places in the 
                            # code where a Type 2 call, as documented
                            # for .make_test_func(), may be executed.
                            accept, abort = test_func(
                                cost, path_is_complete, None,
                                *convert_vor_vert_idxs_to_coords(
                                    pred_reidx_to_vor_vert_idx_array[terminal_pred_reidx],
                                    pred_reidx_to_vor_vert_idx_array[pred_row_reidx]
                                    )
                                )
                            if abort:
                                terminate = True  # *REASSIGNMENT*
                                break
                            if not accept:
                                continue

                    # If path was truncated by convergence, terminate
                    # iteration if necessary or else call test function
                    # and proceed as specified by the result.
                    if not path_is_complete:
                        if not truncate:
                            terminate = True  # *REASSIGNMENT*
                            break
                        if retest_on_convergence:
                            # If test function result indicates that
                            # iteration should be permanently
                            # terminated, do so.
                            # *REASSIGNMENT*
                            cost_this_part = cost_array_full_rows[cost_row_idx, terminal_pred_reidx]
                            # Note: This is a Type 1b call as documented
                            # for .make_test_func().
                            # *REASSIGNMENTS*
                            accept, abort = test_func(cost_this_part, False)
                            if abort:
                                terminate = True  # *REASSIGNMENT*
                                break
                            # If test function result indicates that the
                            # cost of the truncated path is not
                            # satisfactory, continue (if appropriate) to
                            # try that truncated path on the opposite
                            # end of the complete path (i.e., try
                            # walking from the opposite end in reverse).
                            if not accept:
                                continue
                        else:
                            # Cost is unknown.
                            # *REASSIGNMENT*
                            cum_cost_is_complete = False

                    # Merge walked path to a line.
                    path_pred_reidx_array = _numpy_fromiter(
                        path_pred_reidx_deque, _numpy_int64
                        )
                    path_vor_vert_idx_array = _take2(
                        pred_reidx_to_vor_vert_idx_array, path_pred_reidx_array,
                        True
                        )
                    del path_pred_reidx_array  # Reduce local namespace.
                    path_vor_vert_idxs = path_vor_vert_idx_array.tolist()
                    merged_line = merge_path(path_vor_vert_idxs,
                                             graph_edge_dict)

                    # Optionally test merged line.
                    if test_func_is_specified:
                        # Note: This is a Type 3 call as documented for 
                        # .make_test_func().
                        accept, abort = test_func(merged_line, path_is_complete, 
                                                  kind)
                        delete_from_graph_skel = accept or accept is None

                    # Proceed based on result of test or default
                    # behavior.
                    if accept:
                        register_line(merged_line, kind, path_vor_vert_idxs)
                    elif delete_from_graph_skel:
                        register_line(merged_line, False, path_vor_vert_idxs)

                    # Update counters as necessary.
                    if delete_from_graph_skel:
                        deleted_path_count += 1
                        if accept:
                            added_path_count += 1
                            cum_cost += (cost if path_is_complete
                                         else cost_this_part)

                    # Terminate if appropriate.
                    if (one_path_only and accept) or abort:
                        terminate = True  # *REASSIGNMENT*
                        break

                    # If the path was deleted from the graph skeleton,
                    # tag its edges as unwalkable.
                    if delete_from_graph_skel:
                        for pred_reidx1, pred_reidx2 in _slide_pairwise(
                            path_pred_reidx_deque
                            ):
                            _bottleneck_replace(
                                predecessors_array[pred_reidx1],
                                pred_reidx2, neg_1
                                )
                            _bottleneck_replace(
                                predecessors_array[pred_reidx2],
                                pred_reidx1, neg_1
                                )
                            _bottleneck_replace(
                                predecessors_array[:, pred_reidx1],
                                pred_reidx2, neg_1
                                )
                            _bottleneck_replace(
                                predecessors_array[:, pred_reidx2],
                                pred_reidx1, neg_1
                                )

                    # Skip reverse walk if path was completely walked
                    # (i.e., did not converge).
                    if path_is_complete:
                        break

            # Casacde a permanent termination, if appropriate.
            if terminate:
                break

        # Return results.
        return (added_path_count, cum_cost,
                cum_cost_is_complete if added_path_count else True,
                considered_full_path_count, deleted_path_count)

    @staticmethod
    def _get__vor_vert_idx_to_partitioned_degree_array(self):
        return _numpy_zeros(
            (len(self.voronoi_vertex_coords_array),), _numpy_int8
            )

    @staticmethod
    def _get_voronoi_vertex_coords_array(self):
        self._isolate_graph_skeleton()
        return self.voronoi_vertex_coords_array

    @staticmethod
    def _get__sorted_vor_vert_idx_sample_idx_pairs_array_and_search(self):
        """
        Array and function to support finding associated sample indices.

        The returned array has three columns corresponding to Voronoi vertex
        index, associated boundary sample index 1, and associated boundary
        sample index 2. More precisely, the first column is sorted and indicates
        a Voronoi vertex (by its index), whereas the next two columns indicate
        sample coordinates from the boundary of the input polygon (by their
        indices) that bracket a Voronoi segment emanating from that Voronoi
        vertex. There will typically be two rows that share the same value in
        the first column, because two peripheral (plus one non-peripheral)
        ridges typically meet at a Voronoi vertex. A tuple of the form
        (sorted_vor_vert_idx_sample_idx_pairs_array, search) is returned, where
        sorted_vor_vert_idx_sample_idx_pairs_array is the array just described
        and search is a function that, when passed a Voronoi vertex index,
        returns the index of the row in
        sorted_vor_vert_idx_sample_idx_pairs_array at which the corresponding
        records start. (search is simply
        sorted_vor_vert_idx_sample_idx_pairs_array[:,0].searchsorted.)
        """
        ## It may be more accurate to state "typically... three rows 
        ## that share the same value in the first column", but should 
        ## check this.
        # Get and release relevant attributes (as they will not be used
        # again).
        vor_vert_idx_pairs_array = self._vor_vert_idx_pairs_array
        del self._vor_vert_idx_pairs_array
        bracketing_sample_idx_pairs_array = self._bracketing_sample_idx_pairs_array
        del self._bracketing_sample_idx_pairs_array
        vor_vert_idx_addendum = getattr(self, "_vor_vert_idx_addendum", None)
        if vor_vert_idx_addendum is not None:
            del self._vor_vert_idx_addendum

        # Create empty array to be populated.
        vor_seg_count = len(vor_vert_idx_pairs_array)
        dbl_vor_seg_count = 2 * vor_seg_count
        if vor_vert_idx_addendum is None:
            addendum_length = 0
        else:
            addendum_length = len(vor_vert_idx_addendum)
        # Note: If the number of sampled coordinates (and hence Voronoi
        # vertices) is large, the array ultimately returned by the
        # current function can have a large memory footprint. To reduce
        # this footprint somewhat, 32-bit integers are used if a nonzero
        # memory option was specified. Although unsigned integers would
        # support a still larger maximum value, the maximum value
        # supported by 32-bit integers should not, in practice, ever be
        # exceeded by node indices, .searchsorted() is much faster for
        # signed integers, and negative "redirect" indices are used in
        # the even of hyperhyb cracking. (See further below.)
        vor_vert_idx_sample_idx_pairs_array = _numpy_empty(
            (dbl_vor_seg_count + addendum_length, 3),
            _numpy_int32 if self.memory_option else _numpy_int64
            )

        # Populate addendum, if one exists.
        # Note: The addendum records Voronoi vertex information for
        # nodes generated during hyperhub cracking. Like the rest of the
        # array returned by the current function, the addendum's first
        # column records a Voronoi vertex index, albeit not one
        # outputted during Voronoi analysis. However, the second
        # ("redirection") column records the negative of the
        # corresponding original Voronoi vertex index, minus 1. (The
        # minus 1 ensures that even the original Voronoi vertex index 0
        # will have a negative redirector.) The third column is unused.
        if vor_vert_idx_addendum is not None:
            addendum = vor_vert_idx_sample_idx_pairs_array[-addendum_length:]
            addendum[:,:2] = vor_vert_idx_addendum
            del vor_vert_idx_addendum  # Release memory.
            addendum_redir_col = addendum[:,1]
            _numpy_negative(addendum_redir_col, addendum_redir_col)
            _numpy_subtract(addendum_redir_col, 1)

        # Populate the first column with Voronoi vertex indices.
        vor_vert_idx_col = vor_vert_idx_sample_idx_pairs_array[:,0]
        vor_vert_idx_col[:vor_seg_count] = vor_vert_idx_pairs_array[:,0]
        vor_vert_idx_col[vor_seg_count:dbl_vor_seg_count] = vor_vert_idx_pairs_array[:,1]
        del vor_vert_idx_pairs_array  # Release memory.

        # Populate the second and third columns with sample coordinate
        # indices
        bracketing_sample_idx_pairs_cols = vor_vert_idx_sample_idx_pairs_array[:,1:]
        bracketing_sample_idx_pairs_cols[:vor_seg_count] = bracketing_sample_idx_pairs_array
        bracketing_sample_idx_pairs_cols[vor_seg_count:dbl_vor_seg_count] = bracketing_sample_idx_pairs_array
        del bracketing_sample_idx_pairs_array  # Release memory.

        # Sort a copy of the just populated array by its first column
        # (which contains Voronoi vertex indices) and return it.
        sorted_first_col_idxs = vor_vert_idx_col.argsort()
        sorted_vor_vert_idx_sample_idx_pairs_array = _take2(
            vor_vert_idx_sample_idx_pairs_array, sorted_first_col_idxs
            )
        if self.memory_option:
            # Note: When a 64-bit integer, or array thereof, is used as
            # the first argument to the .searchsorted() of a 32-bit
            # integer array, that entire array is converted to 64-bit on
            # each call. To avoid that potentially significant
            # computational expense, coerce the argument to 32-bit
            # (since sorted_vor_vert_idx_sample_idx_pairs_array is 32-
            # bit for a nonzero memory option.
            search = lambda v, side="left", sorter=None, int32=_numpy_int32.type: sorted_vor_vert_idx_sample_idx_pairs_array[:,0].searchsorted(
                int32(v), side, sorter
                )
        else:
            search = sorted_vor_vert_idx_sample_idx_pairs_array[:,0].searchsorted
        return (sorted_vor_vert_idx_sample_idx_pairs_array, search)

    @staticmethod
    def _make_csr_matrix(edge_dict, cost_func):
        """
        Create scipy.sparse.csr_matrix for a graph given its edge dictionary.

        The csr matrix and a flat array are returned in a 2-tuple of the form
        (csr_matrix, reidx_to_idx_array). The flat array is populated with the
        unique Voronoi vertex indices in edge_dict and is sorted. To minimize
        the size of the csr matrix (and, more importantly, any dense arrays
        derived therefrom, such as the costs and predecessors arrays generated
        during path-finding), its row and column indices do not directly
        correspond to the original Voronoi vertex indices. However, any array
        populated with these row and column "re-indices", or the original
        Voronoi vertex indices, can be easily inter-converted:
            original_indices_array = reidx_to_idx_array[reindices_array]
            # ...or, equivanet but possibly faster...
            original_indices_array = _take2(reidx_to_idx_array, reindices_array)
            # ...and...
            reindices_array = reidx_to_idx_array.searchsorted(
                original_indices_array
                )

        edge_dict is a dict in which each key is a _Frozenset2 of length 2 and
        each value is a LineString. The value represents the edge between the
        two Voronoi vertices whose indices populate the corresponding key, and
        the directionality of the edge is implied by the order of those indices
        in key.tuple (hence why a _Frozenset2 is used instead of a standard
        frozenset). Usually, edge_dict is self._graph_edge_dict.

        cost_func is a function that is called to populate values in the
        returned csr matrix. Specifically, cost_func(edge, key, flipped) is
        called, where edge is the LineString returned by edge_dict[key] and
        flipped is a boolean (True or False) indicating whether the edge's cost
        should be measured in reverse (that is, the edge is walked from key[1]
        to key[0]). If costs are independent of direction so that the flipped
        argument can be safely ignored, greater performance can be achieved by
        assigning
            cost_func.is_directed = False
        If cost_func is not specified, all weights are assumed to be 1.

        Warning: Conversion from original Voronoi vertex indices to reindices
        by
            reidx_to_idx_array.searchsorted(original_indices_array)
        is only guaranteed to yield correct results if original_indices_array
        exclusively contains original Voronoi vertex indices represented in
        edge_dict.keys().

        Note: The returned csr matrix will be symmetric if and only if
        cost_func() is direction-indepedent. Nonetheless, for simplicity, the
        matrix can be treated as directed regardless of the nature of
        cost_func(). This is (of course) necessary if cost_func() is direction-
        dependent, but it also "generally leads to more efficient computation"
        even if cost_func() is direction-independent (per scipy.sparse.csgraph
        documentation at scipy version 1.1.0).
        """
        # Identify the paired and unique Voronoi vertex indices in
        # edge_dict.
        vor_vert_idx_pairs_array = _numpy_fromiter(
            _flatten_to_iter([key.tuple for key in edge_dict]), _numpy_int64
            )
        unique_vor_vert_idxs_array = _uniquify_flat_array(
            vor_vert_idx_pairs_array
            )
        vor_vert_idx_pairs_array.shape = _neg_1_2_tuple

        # Create the row and column index arrays required to construct
        # the csr matrix.
        # Note: The row and column indices are reindexed to avoid gaps
        # that would occur if the original Voronoi vertex indices were
        # used (e.g., [3, 7, 9...] --> [0, 1, 2...]). These gaps would
        # result in unnecessarily large dense arrays during path-
        # finding, which in turn would (in common cases) constitute the
        # single most restrictive limit on the effective resolution
        # (sampling interval) at which a polygon can be analyzed within
        # a given memory footprint.
        vor_vert_reidx_pairs_array = unique_vor_vert_idxs_array.searchsorted(
            vor_vert_idx_pairs_array
            )
        # Note: The lines below are equivalent to simply vertically
        # stacking vor_vert_reidx_pairs_array atop a column-swapped copy
        # of that array, then taking the respective columns of the
        # combined output as row_idxs_array and col_idxs_array. Put
        # another way, the first half of row_idxs_array is equal to the
        # second half of col_idxs_array, and the second half of
        # row_idxs_array is equal to the first half of col_idxs_array.
        # (This can clearly be seen in the final two lines.) The
        # ultimate goal is simply that row-column index pairs will
        # recreate vor_vert_reidx_pairs_array.
        edge_count = len(edge_dict)
        # Reuse array.
        row_idxs_array = vor_vert_idx_pairs_array.ravel()
        row_idxs_array[:edge_count] = vor_vert_reidx_pairs_array[:,0]
        row_idxs_array[edge_count:] = vor_vert_reidx_pairs_array[:,1]
        # Reuse array.
        col_idxs_array = vor_vert_reidx_pairs_array.ravel()
        col_idxs_array[:edge_count] = row_idxs_array[edge_count:]
        col_idxs_array[edge_count:] = row_idxs_array[:edge_count]

        # Create the weight arrays required to construct the csr matrix.
        if cost_func is None:
            full_edge_costs_array = _numpy_ones((edge_count * 2,), _numpy_int8)
        else:
            # Note: The line below (and its counterpart further below)
            # demonstrate the main reason why _Frozenset2's are used as
            # keys in edge dict's instead of frozenset's, namely, so
            # that the direction of the edge's LineString relative to
            # the order of tuple(key) is known. Note also that whether
            # this directionality is reversed is specified by the third
            # argument to cost_func().
            # Note: The corresponding order of vor_vert_idx_pairs_array
            # and edge_costs_array1 (that is, of edge_dict.iterkeys()
            # and edge_dict.iteritems()) is guaranteed as long as
            # edge_dict is not modified between the population of these
            # arrays, which it is not.
            edge_costs_array1 = _numpy_fromiter(
                [cost_func(edge, key, False)
                 for key, edge in edge_dict.iteritems()],
                _numpy_float64, edge_count
                )
            if getattr(cost_func, "is_directed", True):
                edge_costs_array2 = _numpy_fromiter(
                    [cost_func(edge, key, True)
                     for key, edge in edge_dict.iteritems()],
                    _numpy_float64, edge_count
                    )
            else:
                edge_costs_array2 = edge_costs_array1
            full_edge_costs_array = _numpy_concatenate((edge_costs_array1,
                                                        edge_costs_array2))
            del edge_costs_array1, edge_costs_array2  # Release memory.

        # Create the csr matrix.
        reidx_count = len(unique_vor_vert_idxs_array)
        csr_matrix = _scipy.sparse.csr_matrix(
            (full_edge_costs_array, (row_idxs_array, col_idxs_array)),
            (reidx_count, reidx_count), _numpy_float64
            )
        return (csr_matrix, unique_vor_vert_idxs_array)

    def _register_line(self, line, kind=None, vor_vert_idxs=None, idx0=None,
                       idxN=None, from_kind=None):
        """
        Register (or unregister) a graph edge or partitioned path.

        The current function can
            1) register (or unregister) a graph edge to the graph skeleton
            2) register (or unregister) a path to the partitioned skeleton,
               including deleting its components from the graph skeleton
            3) recategorize a path already in the partitioned skeleton
            4) quarantine a loop
        In each case, all relevant node degree records are updated. In addition,
        ._stem_vor_vert_idx and ._stub_vor_vert_idx are assigned as applicable.
        The edge or path dictionary to which the line was registered is
        returned, or None if the line was not registered.

        line is a LineString that specifies the edge or path.

        kind is a string that specifies the partitioned category to which the
        line should be registered. It must have nonzero length.
            kind       from_kind    Action
            None       None         register line to graph skeleton (or
                                    quarantine as a loop; see 1a below)
            [str]      None         register line to partitioned skeleton, and
                                    unregister line's components from graph
                                    skeleton (see 1b, 2, and 3 below)
            [str]      [str]        recategorize line within partitioned
                                    skeleton
            None       [str]        unregister line from partitioned skeleton
            False      None         unregister line, which must belong to the
                                    graph skeleton or be quarantined as a loop
        There are three special cases in which kind is reset internally (or a
        similar redirection occurs):
            1) If idx0 and idxN are the same (whether specified or found
               internally) and
               a) kind is None: line is quarantined as a loop.
               b) kind is a string: kind is reset to "loop".
            2) If kind is "branch" but the node at neither end of line exists in
               the partitioned skeleton, kind is reset to "trunk". (This permits
               opportunistic partitioning out of commingled trunks and branches
               formed by truncating trunk paths.)
            3) If kind is "branch" but both the start and end nodes exist in the
               partitioned skeleton, kind is reset to "bridge". (This permits
               opportunistic partitioning out of commingled branches and bridges
               formed by truncating branch paths.)

        vor_vert_idxs is a sequence of integers that implies line's components.
        vor_vert_idxs must be specified if line's components are relevant. (See
        documentation for kind argument.)

        idx0 is an integer that specifies the start Voronoi vertex index. If
        idx0 is not specified (None), the correct integer is found
        automatically.

        idxN is an integer that specifies the end Voronoi vertex index. If idxN
        is not specified (None), the correct integer is found automatically.

        from_kind is a string that specifies the partitioned category from which
        the line or its components should be removed. See documention for kind
        argument.

        Warning: Specifying idx0, idx1, or vor_vert_idxs incorrectly will cause
        issues. These arguments are merely provided to avoid the cost of
        internal generation of the correct objects when possible.
        """
        # Determine whether components will be unregistered with this
        # call.
        unregister_components = from_kind is None and kind
        if unregister_components and vor_vert_idxs is None:
            raise TypeError("vor_vert_idxs must be specified")

        # Localize source and target line dictionaries and node degree
        # arrays.
        if kind is None:
            if from_kind is None:
                src_line_dict = None
                # Note: Target line dictionary is changed further below
                # if line is then determined to be a loop.
                targ_line_dict = self._graph_edge_dict
            else:
                src_line_dict = self._kind_to_partitioned_path_dict[from_kind]
                targ_line_dict = None
        elif kind:
            if from_kind is None:
                src_line_dict = self._graph_edge_dict
                targ_line_dict = self._kind_to_partitioned_path_dict[kind]
            else:
                src_line_dict = self._kind_to_partitioned_path_dict[from_kind]
                targ_line_dict = self._kind_to_partitioned_path_dict[kind]
        else:
            # Note: Source line dictionary is changed further below if
            # line is then determined to be a loop.
            src_line_dict = self._graph_edge_dict
            targ_line_dict = None
        if src_line_dict is None:
            src_vor_vert_idxs_to_degree_array = None
        else:
            if src_line_dict is self._graph_edge_dict:
                src_vor_vert_idxs_to_degree_array = self._vor_vert_idx_to_graph_degree_array
            else:
                src_vor_vert_idxs_to_degree_array = self._vor_vert_idx_to_partitioned_degree_array
        if targ_line_dict is None:
            targ_vor_vert_idxs_to_degree_array = None
        else:
            if targ_line_dict is self._graph_edge_dict:
                targ_vor_vert_idxs_to_degree_array = self._vor_vert_idx_to_graph_degree_array
            else:
                targ_vor_vert_idxs_to_degree_array = self._vor_vert_idx_to_partitioned_degree_array

        # Derive key.
        vor_vert_idxs_array = line.vor_vert_idxs_array
        if idx0 is None or idxN is None:
            if idx0 is None:
                if vor_vert_idxs is None:
                    idx0 = vor_vert_idxs[0]  # *REASSIGNMENT*
                else:
                    # *REASSIGNMENT*
                    idx0 = int(vor_vert_idxs_array[0])
            if idxN is None:
                if vor_vert_idxs is None:
                    idxN = vor_vert_idxs[-1]  # *REASSIGNMENT*
                else:
                    # *REASSIGNMENT*
                    idxN = int(vor_vert_idxs_array[-1])
        if targ_line_dict is None:
            # Note: A _Frozenset2 would also work but is slower to
            # create.
            key = frozenset((idx0, idxN))
        else:
            key = _Frozenset2((idx0, idxN))

        # If line is a loop, modify assignments from above as necessary.
        if idx0 == idxN:
            key = line  # *REASSIGNMENT*
            if src_line_dict is self._graph_edge_dict:
                src_line_dict = self._loop_dict
                src_vor_vert_idxs_to_degree_array = None
            elif src_line_dict in self._kind_to_partitioned_path_dict:
                src_line_dict = self._kind_to_partitioned_path_dict["loop"]
            if targ_line_dict is self._graph_edge_dict:
                targ_line_dict = self._loop_dict
                targ_vor_vert_idxs_to_degree_array = None
                line._stem_vor_vert_idx = idx0
            elif targ_line_dict in self._kind_to_partitioned_path_dict:
                targ_line_dict = self._kind_to_partitioned_path_dict["loop"]

        # Set a branch's special attributes or change kind from "branch"
        # to "trunk" or "bridge", if necessary.
        elif kind == "branch":
            if targ_vor_vert_idxs_to_degree_array[idxN]:
                if targ_vor_vert_idxs_to_degree_array[idx0]:
                    kind = "bridge"
                    # *REASSIGNMENT*
                    targ_line_dict = self._kind_to_partitioned_path_dict["bridge"]
                else:
                    line._stem_vor_vert_idx = idxN
                    line._stub_vor_vert_idx = idx0
            elif targ_vor_vert_idxs_to_degree_array[idx0]:
                line._stem_vor_vert_idx = idx0
                line._stub_vor_vert_idx = idxN
            else:
                kind = "trunk"  # *REASSIGNMENT*
                # *REASSIGNMENT*
                targ_line_dict = self._kind_to_partitioned_path_dict["trunk"]

        # Unregister line.
        # Note: No need to update node degrees if merely recategorizing
        # line.
        update_node_degrees = src_vor_vert_idxs_to_degree_array is not targ_vor_vert_idxs_to_degree_array
        if src_line_dict is not None:
            if unregister_components:
                # Note: Conversion from an array is for performance
                # only. It does not change the final result.
                if isinstance(vor_vert_idxs, _numpy_ndarray):
                    # *REASSIGNMENT*
                    vor_vert_idxs = vor_vert_idxs.tolist()
                # Note: _Frozenset2's would also work but are slower to
                # create. Strictly, the order within each key is also
                # not known.
                for component_faux_key in _imap(frozenset,
                                                _slide_pairwise(vor_vert_idxs)):
                    del src_line_dict[component_faux_key]
            else:
                del src_line_dict[key]
            if (update_node_degrees
                and src_vor_vert_idxs_to_degree_array is not None):
                src_vor_vert_idxs_to_degree_array[vor_vert_idxs_array] -= 2
                src_vor_vert_idxs_to_degree_array[idx0] += 1
                src_vor_vert_idxs_to_degree_array[idxN] += 1
                if src_line_dict is self._graph_edge_dict:
                    self._clear_graph_edge_dict_derivations()

        # Register line.
        if targ_line_dict is not None:
            targ_line_dict[key] = line
            if (update_node_degrees
                and targ_vor_vert_idxs_to_degree_array is not None):
                targ_vor_vert_idxs_to_degree_array[vor_vert_idxs_array] += 2
                targ_vor_vert_idxs_to_degree_array[idx0] -= 1
                targ_vor_vert_idxs_to_degree_array[idxN] -= 1

        # Return.
        return targ_line_dict

    # @staticmethod
    # def _get__vor_vert_idx_to_partitioned_degree_array(self):
    #     """
    #     Dict mapping Voronoi vertex index to degree in partitioned skeleton.
    #     """
    #     vor_vert_idx_to_partitioned_degree_array = _numpy_zeros(
    #         (len(self.voronoi_vertex_coords_array),), _numpy_int8
    #         )
    #     for kind, path_dict in self._kind_to_partitioned_path_dict.iteritems():
    #         for key, path in path_dict.iteritems():
    #             # Note: Because non-loop edges can only be added by
    #             # .partition_out*()'s and it is enforced at
    #             # initialization that no node in the graph skeleton has
    #             # degree > 3, it is guaranteed that no two partitioned
    #             # edges can span the same pair of nodes. (Because the
    #             # trunk and branches must each have at least one stub
    #             # end, clearly these kinds cannot span the same pair of
    #             # nodes as any other edge. In cases of high symmetry,
    #             # two bridges could span the same pair of nodes, but in
    #             # order to be bridges, that pair of nodes would also
    #             # have to stem from a third edge (or form an
    #             # uninterrupted loop, which would have been quarantined
    #             # at initialization), implying nodes of degree 4 at
    #             # those stem points, which are not allowed.)
    #             self._register_line(path, kind,
    #     return vor_vert_idx_to_partitioned_degree_array

    def _merge_path(self, vor_vert_idxs, from_line_dict=None, reuse=True):
        """
        Merge path along Voronoi vertices to a single LineString.

        Given Voronoi vertex indices along a path and the corresponding line
        dictionary (from_line_dict), the current function constructs and returns
        a merged LineString with all spanned Voronoi vertices, even if the
        vertices passed to the function (vor_vert_idxs) are incomplete due to
        contraction. The current function is used both during contraction
        through binodes and partitioning out.

        vor_vert_idxs is a sequence that specifies the Voronoi vertex indices
        spanned by the path within from_line_dict. They can be incomplete due to
        contraction.

        from_line_dict is the source line dictionary used to interpret and
        possibly "fill out" to completion the Voronoi vertex indices spanned by
        the path. If from_line_dict is None, vor_vert_idxs must be complete
        (without any contraction through bindoes). from_line_dict is not
        modified by the current function.

        reuse is a boolean that specifies whether it is safe to permanently
        attach vor_vert_idxs to the returned line. If vor_vert_idxs may be
        modified in the future, reuse should be False. If vor_vert_idxs is not
        an array or from_line_dict is specified (not None), reuse is ignored.
        """
        # If no line_dict is specified, use vor_vert_idxs as directly as
        # possible.
        if from_line_dict is None:
            if isinstance(vor_vert_idxs, _numpy_ndarray):
                if reuse:
                    merged_vor_vert_idxs_array = vor_vert_idxs
                else:
                    merged_vor_vert_idxs_array = vor_vert_idxs.copy()
            else:
                merged_vor_vert_idxs_array = _numpy_fromiter(vor_vert_idxs,
                                                             _numpy_int64)

        # Iterate over components in line_dict implied by vor_vert_idxs
        # to fill out a complete record (array) of all spanned Voronoi
        # vertex indices.
        else:

            # Convert vor_vert_idxs from an array, if necessary, for
            # sake of performance.
            if isinstance(vor_vert_idxs, _numpy_ndarray):
                vor_vert_idxs = vor_vert_idxs.tolist()  # *REASSIGNMENT*

            # Iterate over components.
            vor_vert_idx_arrays = []
            vor_vert_idx_arrays_append = vor_vert_idx_arrays.append
            for vor_vert_idx_pair in _slide_pairwise(vor_vert_idxs):

                # Add array of Voronoi vertex indices for component,
                # correctly ordered, minus the final index, to the
                # growing list.
                # Note: Each component "faux" key is a frozenset whereas
                # the keys in from_line_dict are _Frozenset2's.
                component_faux_key = frozenset(vor_vert_idx_pair)
                edge = from_line_dict[component_faux_key]
                edge_vor_vert_idxs_array = edge.vor_vert_idxs_array
                if vor_vert_idx_pair != edge._aligned_key.tuple:
                    # *REASSIGNMENT*
                    edge_vor_vert_idxs_array = edge_vor_vert_idxs_array[::-1]
                # Note: The final index is removed to avoid redundant
                # vertices when the next component's array is added to
                # the list.
                vor_vert_idx_arrays_append(edge_vor_vert_idxs_array[:-1])

            # Undo truncation of the last index from the most recently
            # added Voronoi vertex indices array, so that the path
            # terminates where it should, and create complete merged
            # Voronoi vertex array.
            vor_vert_idx_arrays[-1] = edge_vor_vert_idxs_array
            del vor_vert_idxs  # Release memory.
            merged_vor_vert_idxs_array = _numpy_concatenate(vor_vert_idx_arrays)
            del vor_vert_idx_arrays  # Release memory.

        # Create new merged line.
        return SkeletalLineString2D(self._self_proxy,
                                    merged_vor_vert_idxs_array)

    def get_node_degree(self, index, partitioned=False):
        """
        Get a node's degree within either graph or partitioned skeletons.

        An integer is returned. If index is unrecognized, 0 is returned. See
        .describe_node() for further documentation.
        """
        if partitioned is None:
            try:
                degree = self._nonperi_adj_dict.get(index, 0)
            except AttributeError:
                raise TypeError("graph skeleton was already isolated")
        else:
            if partitioned:
                vor_vert_idx_to_degree_array = self._vor_vert_idx_to_partitioned_degree_array
            else:
                vor_vert_idx_to_degree_array = self._vor_vert_idx_to_graph_degree_array
            try:
                degree = vor_vert_idx_to_degree_array[index]
            except IndexError:
                degree = 0
        return degree

    def describe_node(self, index, partitioned=False):
        """
        Describe the connectedness of a node within either skeleton.

        The returned description depends on the node's degree, which is the
        number of edges or paths that touch the node:
            degree       description
            1            "stub"
            2            "binode"
            3            "hub"
            >3           "hyperhub" (applies only to partitioned="raw")
            [otherwise]  "unrecognized"

        index is an integer that specifies the Voronoi vertex index (i.e., row
        in .voronoi.out_coords_array, where .voronoi is a Voronoi2D) of the node
        to be examined.

        partitioned is a boolean that specifies for which skeleton the node's
        connectivity should be described:
            True    partitioned skeleton
            False   graph skeleton
            None    [see note below]

        Note: If partitioned is None, an attempt is made to return the node's
        connectivity within the "raw" data that exist prior to isolation of the
        graph skeleton. This is a rather advanced use and most users can safely
        ignore the option (except to be sure not to specify partitioned=None).
        Note that isolation of the graph skeleton occurs automatically during or
        soon after initialization, depending on the specified arguments. If
        partitioned is None and the graph skeleton has already been isolated, an
        error is raised.

        Note: Uninterrupted terminal loops are quarantined when the graph
        skeleton is isolated. Because they are quarantined, they have no effect
        on a node's degree. See .add_loops_to_partitioned().

        See also:
            .describe_line()
            .get_node_degree()
        """
        degree = self.describe(index, partitioned)
        if degree == 1:
            return "stub"
        if degree == 2:
            return "binode"
        if degree == 3:
            return "hub"
        if degree > 3:
            return "hyperhub"
        return "unrecognized"

    def describe_line(self, line, safe=True):
        """
        Describe a line from either the graph or partitioned skeleton.

        One of the following descriptions is returned:
            1) if line is from the graph skeleton, its description depends on
               the combination of node degrees at either end (see
               .describe_node()):
                   description   degree at one end   degree at other end
                   "isolated"    1                   1
                   "link"        2                   2
                   "interior"    3                   3
                   "tip"         2                   1
                   "marginal"    3                   1
                   "spoke"       3                   2
                   "unknown"     [degree at one or both ends is unknown]
            2) if line is a (quarantined) uninterrupted terminal loop that has
               not been partitioned out (see .add_loops_to_partitioned()), its
               description is "quarantined_loop"
            3) if line is from the partitioned skeleton, its description is its
               kind (e.g., "branch", "loop")
            4) otherwise:
                   "invalid"       line was not generated by skeletal analysis
                   "foreign"       line is from a different skeletal analysis
                   "unrecognized"  line could not be categorized (e.g., was
                                   deleted from the current skeletal analysis)

        line is a SkeletalLineString that specifies the line to be described.

        safe is a boolean that specifies whether it should be tested that line
        belongs to the current skeletal analysis. If safe is False and line
        belongs to another skeletal analysis (i.e., is "foreign"), the
        returned description is meaningless.

        See also:
            .describe_node()
        """
        try:
            key = line._aligned_key
        except AttributeError:
            return "invalid"
        # Note: Allow for .skeleton to be a weakref.proxy.
        if safe and line.skeleton.__hash__() != hash(self):
            return "foreign"
        if len(key) == 1:
            if line in self._loop_dict:
                return "quarantined_loop"
            if line in self._kind_to_partitioned_path_dict["loop"]:
                return "loop"
        else:
            if key in self._graph_edge_dict:
                idx0, idxN = key.tuple
                degrees_set = {self._vor_vert_idx_to_graph_degree_array[idx0],
                               self._vor_vert_idx_to_graph_degree_array[idxN]}
                if 0 in degrees_set:
                    return "unknown"
                if len(degrees_set) == 1:
                    degree, = degrees_set
                    if degree == 1:
                        return "isolated"
                    if degree == 2:
                        return "link"
                    if degree == 3:
                        return "interior"
                elif 1 in degrees_set:
                    if 2 in degrees_set:
                        return "tip"
                    if 3 in degrees_set:
                        return "marginal"
                elif 2 in degrees_set:
                    if 3 in degrees_set:
                        return "spoke"
                return "unknown"
            for kind, path_dict in self._kind_to_partitioned_path_dict.iteritems():
                if key in path_dict:
                    return kind
        return "unrecognized"

    def get_lines(self, kind="partitioned", include_loops=True,
                  hardness_level=None):
        """
        Get skeleton lines as a list of LineString's, subject to criteria.

        Each line is a SkeletalLineString2D (a subtype of LineString2D) with the
        following attributes beyond those generally supported for LineString2D:
            add_tails*         [1]   extend line to input polygon's boundary
            area**                   approx area of polygon that line represents
            coords_array*      [2]   usual meaning
            coords_array3D**   [2]   z-coords interpolated from polygon vertices
            delta_z*           [2]   change in z-coords from start to end node
            description              result of skeleton.describe_line(...)
            length             [2]   2D length
            length3D**         [2]   interpolated 3D length
            normalized_length* [2-4] see Skeleton.prune()
            segment_count            len(vor_vert_idxs_array) - 1
            segment_widths**         approx 2D width's (cf. segment_lengths)
            skeleton                 the current Skeleton or its weakref.proxy
            stem_coords_array  [5]   stem's coordinates as flat array
            stem_width*        [5]   approx 2D width at line's stem
            stub_coords_array  [2,4] stub's (or tail's) coordinates, flat array
            untailed_length2D        2D length, not including any tails
            vertex_count             len(vor_vert_idxs_array)
            vor_vert_idxs_array      array of Voronoi vertex indices
            _aligned_key.tuple       tuple of Voronoi vertex indices at each end
            _stem_vor_vert_idx [5]   Voronoi vertex index for .stem_coords_array
            _stub_vor_vert_idx [4,6] Voronoi vertex index for stub or tail end


        [1] Only supported for (partitioned out) trunks and branches.
        [2] May be modified (and not in place) if .add_tails() is called. Also
            applies to standard attributes of LineString2D's.
        [3] Any operations involving .normalized_length performed prior to tail
            addition effectively use .untailed_length2D instead .length
        [4] Only assigned for (partitioned out) branches.
        [5] Only assigned for (partitioned out) branches and loops.
        [6] This is the Voronoi vertex index of stub_coords_array if no tail
            has been added. Otherwise, it is the Voronoi vertex index of the
            stub to which that tail was added (because a tail's end is never at
            a Voronoi vertex). In other words, it is determined prior to tail
            addition and is unmmodified by tail addition.
        Except for .skeleton, all of the listed public data attributes (i.e.,
        those without a leading underscore), even .coords_array, are lazily
        generated (from .vor_vert_idxs_array). Those marked with * (**) are
        somewhat (especially) computationally expensive. .coords_array3D,
        .delta_z, and .length3D are only supported if the input polygon is 3D.

        kind is a string that specifies the category of lines to return:
            "graph"                 edges that have not yet been partitioned out
            "quarantined_loop"      uninterrupted terminal loops that have not
                                    yet been added to the partitioned skeleton
            "partitioned"           paths that have been partitioned out
            "trunk"                 partitioned out trunk(s)
            "branch"                partitioned out branches
            "bridge"                partitioned out bridges
            "loop"                  uninterrupted terminal loops that have been
                                    added to the partitioned skeleton (see
                                    .add_loops_to_partitioned())
        If kind is not recognized, an empty list is returned.

        include_loops is a boolean that specifies whether uninterrupted terminal
        loops should be included. include_loops is ignored unless kind is
        "graph" or "partitioned".

        hardness_level is an integer that specifies how the returned lines
        should behave upon deletion of the current Skeleton.
            0  Pro:  Minimizes memory footprint and the time required to
                     initially generate the returned lines.
               Con:  Each returned line will be unusable once self is deleted.
            1  Pro:  Each line will permanently support all regular LineString2D
                     attributes and functionality.
               Con:  Maximies total memory footprint (i.e., that shared between
                     lines and self) and the time required to initially generate
                     the returned lines. Additionally, each line is only
                     guaranteed to support the special attributes listed above
                     (e.g., .area) if self has not been deleted or that
                     attribute was used, whether directly or indirectly.
               Note: Calling line.minimize_memory(True) will reduce a line of
                     hardness 1 to hardness 0 (but have no hardness effect if
                     line's hardness is 2). Lines are automatically elevated to
                     hardness 1 if .coords_array is used prior to the deletion
                     of self.
            2  Pro:  Each line will permanently support all regular LineString2D
                     attributes and functionality, as well as those special
                     attributes listed above (e.g., .area).
               Con:  Deleting self releases no memory (unlike all other hardness
                     levels).
               Note: No call of line.minimize_memory(...) can modify the
                     hardness of lines with this hardness.
        Strictly, hardness_level specifies the minimum permitted hardness level.
        If a line already has a higher hardness than that specified, it is not
        demoted. If hardness_level is unspecified (None), it defaults to 1 for
        partitioned paths and 0 otherwise. If you ultimately write out any of
        these lines, they will be promoted to a hardness level of 1 if they are
        only hardness level 0 at the time.

        See also:
            .delete_lines()
        """
        # Compile lines.
        if kind == "graph":
            lines = self._graph_edge_dict.values()
            if include_loops:
                lines.extend(self._loop_dict)
        elif kind == "partitioned":
            lines = []
            for (partitioned_kind,
                 path_dict) in self._kind_to_partitioned_path_dict.iteritems():
                if not include_loops and partitioned_kind == "loop":
                    continue
                lines.extend(path_dict.itervalues())
        elif kind == "quarantined_loop":
            lines = self._loop_dict.keys()
        elif kind in self._kind_to_partitioned_path_dict:
            lines = self._kind_to_partitioned_path_dict[kind].values()
        else:
            lines = []

        # Make hardness level explicit, if necessary.
        if hardness_level is None:
            if kind in ("graph", "quarantined_loop"):
                hardness_level = 0  # *REASSIGNMENT*
            else:
                hardness_level = 1  # *REASSIGNMENT*

        # Apply hardness level.
        if hardness_level == 1:
            for line in lines:
                line.coords_array
        elif hardness_level == 2:
            for line in lines:
                line.skeleton = self
        elif hardness_level != 0:
            raise TypeError(
                "hardness_level is not recognized: {!r}".format(hardness_level)
                )

        # Return lines.
        return lines

    def get_nodes(self, kind, graph=True, partitioned=False, coords=True,
                  degrees=False, loops=False, interpolate_3D=False):
        """
        Get node information from either the graph or partitioned skeleton.

        To describe the targeted nodes, a tuple of the form
            (indices_array, coords_array, degrees_array)
        is returned in which indices_array is a flat array whose values are each
        node's Voronoi vertex index, that is, the corresponding row number from
        .voronoi.out_coords_array; coords_array is a 2-column (3-column) array
        if interpolate_3D is False (True) whose values are each node's x-  and
        y- (and z-)coordinates (or None); and degrees_array is a flat array
        whose values are each node's degree, that is, the number of edges and/or
        paths that touch it (or None). indices_array is always sorted
        (ascending), and coords_array and degrees_array are in the same order so
        that the nth value in each array describes the same node. If no nodes of
        the specified kind exist, None is returned instead of a tuple. A major
        use for the current method is to help the user specify the nodes
        argument in .partition_out*()'s.

        kind is a string that specifies the category of nodes to return:
            kind                Description
            "stub"        [1]   degree 1 nodes
            "hub"         [1,2] degree >=3 nodes
            "nonbinode"   [1]   "stub" + "hub", commingled
            "binode"      [1]   degree 2 nodes
            "hyberhub"    [1,2] degree >3 nodes
            "all"         [1]   all nodes
            "bridge"      [3]   current/potential bridge stem nodes
            "bridge2"     [4]   "bridge" + current bridge binodes, commingled
            "branch_stem" [3]   current/potential branch stem nodes
            "branch_stub" [3]   current/potential branch stub nodes
            "branch"      [4]   "branch_stub" + "branch_stem", commingled
            "branch2"     [4]   "branch" + current branch binodes, commingled
            "trunk"       [3]   current/potential trunk stub nodes
            "trunk2"      [4]   "trunk" + current trunk binodes, commingled
            "loop"        [3,5] current/potential loop stem nodes
            "loop2"       [3,5] "loop" + current/potential loop binodes
            "end"         [6]   nodes from either end of each edge or path
            "cut"         [7]   hubs removed by cutting
        [1] If both graph and partitioned are True, the degree of a given node
            in each skeleton is summed. For example, a node that is a stub in
            the graph skeleton and a binode in the partitioned skeleton would be
            included in get_nodes(kind="hub", graph=True, partitioned=True).
            This summation also applies to degrees_array, if generated.
        [2] Hyperhubs, with degree >3, only exist in the raw data. (See note
            further below.)
        [3] Exactly one of graph and partitioned must be specified as True, or
            an error is raised. If partitioned is True, the described nodes
            reflect the current categorization within the partitioned skeleton.
            If graph is instead True, only nodes from the graph skeleton are
            described and represent those nodes that are suitable for use in the
            specified way based on a simple local analysis of the current node
            degree in each skeleton. For example, a node that is a stub in the
            graph skeleton and a binode in the partitioned skeleton could
            potentially serve as a branch stem. However, consider a more
            ambiguous example: could a node that is currently a hub in the graph
            skeleton and absent from the partitiond skeleton likewise
            potentially serve as a branch stem? Eventually, maybe, but not
            immediately. The kinds listed above will honor the "eventually"
            interpretation. If you wish to instead impose the "immediately"
            interpretation, prepend "next_" (e.g., "next_branch_stem"). The
            following kinds are therefore equivalent aliases if graph is True:
                "branch_stub", "trunk", "next_branch_stub", and "next_trunk"
                "branch_stem" and "bridge"
                "next_branch_stem" and "next_bridge"
            Finally, note that this simple analysis is incomplete (even if one
            were to ignore the possibility of deletion/pruning), because it
            relies on node degree only. For example, a node that is a stub in
            the graph skeleton and a binode in the partitioned skeleton could
            potentially serve as a bridge stem, given only that information.
            However, it could be that the edge in the graph skeleton for which
            that node is a stub also has a stub at the other end, which is
            absent in the partitioned skeleton. In that case, the node could
            never serve as a bridge stem (but it could potentially serve as a
            branch stem).
        [4] For this kind, graph must be False and partitioned must be True, or
            else an error is raised. The described nodes reflect the current
            categorization within the partitioned skeleton.
        [5] Loops are a special case of the scenario described in [3]. Namely,
            if graph is True, nodes from quarantined uninterrupted terminal
            loops are described for each loop whose stem node currently exists
            in the graph or partitioned skeletons, whereas if partitioned is
            instead True, nodes from uninterrupted terminal loops that were
            added to the partitioned skeleton are described (see
            .add_loops_to_partitioned()).
        [6] Exactly one of graph and partitioned must be specified as True, or
            an error is raised. If graph is True, the described nodes are the
            ends of every edge in the graph skeleton and therefore include all
            stubs and hubs, and may also include binodes, especially after
            partitioning out has begun. If partitioned is instead True, the
            described nodes are the ends of every path in the partitioned
            skeleton, which should be equivalent to all stubs and hubs and
            include no binodes.
        [7] Refers to nodes deleted during isolation of the graph skeleton.
            Because these nodes belong to neither the graph nor partitioned
            skeleton, both the graph and partitioned arguments must be specified
            as False.

        graph is a boolean that specifies whether nodes in the graph skeleton
        should be among those described.

        partitioned is a boolean that specifies whether nodes in the partitioned
        skeleton should be among those described.

        coords is a boolean that specifies whether coords_array should be
        generated. If coords is False, coords_array in the returned tuple is
        instead None.

        degrees is a boolean that specifies whether degrees_array should be
        generated. If degrees is True, an error is raised unless kind explicitly
        depends on node degrees (i.e., has a [1] next to its name further above)
        or line ends (i.e., has a [6] next to its name further above). If
        degrees is instead False, degrees_array in the returned tuple is None.

        loops is a boolean that specifies whether uninterrupted terminal loops
        should be included when node degree is calculated. This argument is
        ignored unless kind explicitly depends on a node degrees (i.e., has a
        [1] next to its name further above) or line ends (i.e., has a [6] next
        to its name further above).

        interpolate_3D is a boolean that specifies whether interpolated z-
        coordinates should be included in the returned coords_array as an
        additional column. These coordinates can only be interpolated if the
        input polygon was 3D. They are interpolated in three steps:
            1) z- (and x- and y-) coordinates are linearly interpolated between
               vertices of the input polygon at the spatial sampling interval
               specified at initialization.
            2) Voronoi vertices are computed by planar (2D) Voronoi analysis on
               the polygon boundary samples from the first step.
            3) The z-coordinate for each Voronoi vertex is then approximated by
               the mean of the z-coordinates of the three closest polygon
               boundary samples. Note that, due to the nature of Voronoi
               analysis, each of these three polygon boundary samples is
               equidistant from the Voronoi vertex.
        Note that the first two of these steps were already completed during
        initialization. If coords is False, interpolate_3D is ignored.

        Warning: The z-coordinates interpolated if interpolate_3D is True are
        approximate, and they may be especially crude (relative to reality) if
        the terrain across polygon's width is not approximately planar locally.
        Additionally, such interpolation is a relatively expensive operation,
        and the results are not saved, so each z-coordinate is re-interpolated
        whenever it is requested.

        Note: For kinds that explicitly depend on node degrees (i.e., have a [1]
        next to their names further above), an advanced functionality is
        supported. Namely, if both graph and partitioned are False, an attempt
        is made to describe nodes from the raw data that exist prior to
        isolation of the graph skeleton. These data exists only temporarily, and
        are automatically deleted either during or soon after initialization. If
        they no longer exist, an error is raised. Also note that retrieving
        these data has a significant per-call computational cost. Therefore, you
        might consider a single call with degrees specified as True rather than
        multiple calls.
        """
        return self._get_nodes(**locals())

    @staticmethod
    def _get_nodes(self, kind, graph, partitioned, coords, degrees, loops,
                   interpolate_3D, validate_only=False, count_only=False):
        # Validate arguments.
        if coords and interpolate_3D and not self.polygon.is_3D:
            raise TypeError("input polygon is not 3D")
        explicit_degree = kind in ("stub", "hub", "nonbinode", "binode",
                                   "hyperhub", "all")
        explicit_degree_or_ends = explicit_degree or kind == "end"
        predictive_kinds = ("bridge", "branch_stem", "branch_stub", "trunk",
                            "loop", "loop2")
        kind_is_predictive_compatible = (kind in predictive_kinds
                                         or (kind[:5] == "next_"
                                             and kind[5:] in predictive_kinds))
        if degrees and not explicit_degree_or_ends:
            raise TypeError(
                "degrees must be False for this kind: {!r}".format(kind)
                )
        if kind == "cut":
            if graph or partitioned:
                raise TypeError(
                    "cut nodes belong to neither graph nor partitioned skeleton"
                    )
        elif bool(graph) == bool(partitioned):
            if not explicit_degree and kind != "all2":
                raise TypeError(
                    "must specify either partitiond or graph (not both, nor neither) for this kind: {!r}".format(
                        kind
                        )
                    )
            if not graph:
                # Raw data are targeted.
                if not hasattr(self, "_nonperi_adj_dict"):
                    raise TypeError(
                        "raw data were deleted when graph skeleton was isolated"
                        )
                if kind == "all2":
                    # Note: For raw data, "all2" and "end" are
                    # equivalent, because no contraction through binodes
                    # has been applied. Reassign "all2" to "end" to
                    # simplify code further below.
                    kind = "end"  # *REASSIGNMENT*
            elif "all2":
                # Note: Unlike "all", the diagnostic "all2" is
                # restricted to only one target in order to keep code
                # further below as simple as possible: the graph or
                # partitioned skeletons (not both), or the raw data.
                # "all2" is further discussed in a note further below.
                raise TypeError(
                    "kind does not support summation across both skeletons: 'all2'"
                    )
        elif (graph and not explicit_degree_or_ends
              and not kind_is_predictive_compatible and kind != "all2"):
            raise TypeError(
                "this kind (presumably) reflects the current categorization within the partitioned skeleton, so partitioned must be True (and graph must be False): {!r}".format(kind)
                )
        elif partitioned:
            # These errors are included to address possible user
            # misunderstanding. The kinds addressed are not directly
            # implied by the documentation.
            if kind.startswith("next_"):
                raise TypeError(
                    "the following kind was specified with partitioned as True, but kinds that begin with 'next_' must instead have graph as True: {!r}".format(kind)
                    )
            if kind in ("branch_stem2", "branch_stub2", "end2", "cut2"):
                raise TypeError(
                    "the following kind is disallowed because it paradoxically implies both a specific (e.g., end) node (the letters in the kind) and binodes (the terminal '2' in the kind): {!r}".format(kind)
                    )
        if validate_only:
            return None

        # For all kinds, the ultimate goal is to identify the
        # corresponding Voronoi vertex indices.

        # Identify the corresponding Voronoi vertex indices directly for
        # the special case that only cut nodes are requested.
        if kind == "cut":
            if hasattr(self, "_cut_vor_vert_idxs_array"):
                # Note: Do not return persistent record (to avoid its
                # modification.)
                vor_vert_idxs_array = self._cut_vor_vert_idxs_array.copy()
            else:
                vor_vert_idxs_array = _empty_tuple  # Placeholder.

        # For kinds that depend explicitly on node degree or concern
        # *potential* partitioning categorization, an array that maps
        # Voronoi vertex index to node degree must be identified first.
        # Note: loops argument is applied further below.
        elif explicit_degree or (graph and kind_is_predictive_compatible):

            # For kinds that depend explicitly on node degree...
            if explicit_degree:

                # Begin by finding node degrees, including summing
                # across both skeletons if necessary.
                if graph:
                    vor_vert_idx_to_degree_array = self._vor_vert_idx_to_graph_degree_array
                    if partitioned:
                        # *REASSIGNMENT*
                        vor_vert_idx_to_degree_array = (
                            vor_vert_idx_to_degree_array
                            + self._vor_vert_idx_to_partitioned_degree_array
                            )
                    elif loops and self._loop_dict:
                        # Note: Because vor_vert_idx_to_degree_array
                        # will be modified when loop nodes are
                        # registered, make it independent from the
                        # persistent record.
                        # *REASSIGNMENT*
                        vor_vert_idx_to_degree_array = vor_vert_idx_to_degree_array.copy()
                elif partitioned:
                    vor_vert_idx_to_degree_array = self._vor_vert_idx_to_partitioned_degree_array
                    if not loops and self._kind_to_partitioned_path_dict["loop"]:
                        # Note: Because vor_vert_idx_to_degree_array
                        # will be modified when loop nodes are
                        # unregistered, make it independent from the
                        # persistent record.
                        # *REASSIGNMENT*
                        vor_vert_idx_to_degree_array = vor_vert_idx_to_degree_array.copy()
                else:
                    vor_vert_idx_array = _numpy_fromiter(self._nonperi_adj_dict,
                                                         _numpy_int64)
                    degree_array = _numpy_fromiter(
                        _imap(len, self._nonperi_adj_dict.itervalues()),
                        _numpy_int8
                        )
                    vor_vert_idx_to_degree_array = _numpy_zeros(
                        (len(self._voronoi.out_coords_array),), _numpy_int8
                        )
                    vor_vert_idx_to_degree_array[vor_vert_idx_array] = degree_array
                    del vor_vert_idx_array, degree_array

                # Apply loops argument.
                # Note: If vor_vert_idx_to_degree_array will be modified
                # in this block, it was made independent from persistent
                # records in the previous block.
                if graph and loops:
                    for loop in self._loop_dict:
                        vor_vert_idx_to_degree_array[loop.vor_vert_idxs_array] += 2
                        # Note: Because this is a loop, the stem is
                        # repeated twice in .vor_vert_idxs_array.
                        vor_vert_idx_to_degree_array[loop._stem_vor_vert_idx] -= 2
                elif partitioned and not loops:
                    for loop in self._kind_to_partitioned_path_dict["loop"]:
                        vor_vert_idx_to_degree_array[loop.vor_vert_idxs_array] -= 2
                        # Note: Because this is a loop, the stem is
                        # repeated twice in .vor_vert_idxs_array.
                        vor_vert_idx_to_degree_array[loop._stem_vor_vert_idx] += 2

            # For kinds that instead concern potential partitioning
            # categorization...
            else:

                # Begin by summing node degrees across both skeletons.
                vor_vert_idx_to_degree_array = (
                    self._vor_vert_idx_to_graph_degree_array
                    + self._vor_vert_idx_to_partitioned_degree_array
                    )

                # Zero out summed node degree if node is no longer
                # present in the graph skeleton (except for predictions
                # for uninterrupted terminal loops, which only need to
                # have their stem node in either skeleton).
                if kind not in ("loop", "loop2", "next_loop"):
                    _numpy_multiply(
                        vor_vert_idx_to_degree_array,
                        self._vor_vert_idx_to_graph_degree_array != 0,
                        vor_vert_idx_to_degree_array
                        )

                # Also zero out node degree if node must already be present
                # in the partitioned skeleton but is not.
                # Note: Unlike "next_bridge", "next_branch_stem",
                # "next_loop", and "next_loop2", "next_branch_stub" and
                # "next_trunk" instead require the *absence* of a node
                # from the partitioned skeleton. The requirements that
                # such a node have a summed degree of 1 (imposed further
                # below) and be present in the graph skeleton (imposed
                # above) are sufficient to guarantee this requirement.
                # Therefore, "branch_stub", "trunk", "next_branch_stub",
                # and "next_trunk" are all aliases when graph is True.
                if kind in ("next_bridge", "next_branch_stem", "next_loop",
                            "next_loop2"):
                    _numpy_multiply(
                        vor_vert_idx_to_degree_array,
                        self._vor_vert_idx_to_partitioned_degree_array != 0,
                        vor_vert_idx_to_degree_array
                        )

            # For the degree-explicit and notpotential categorization
            # cases described above, proceed with shared code to
            # identify the target Voronoi vertex indices.
            if kind == "all":
                if count_only:
                    return (vor_vert_idx_to_degree_array != 0).sum()
                vor_vert_idxs_array, = vor_vert_idx_to_degree_array.nonzero()
            elif kind in ("loop", "loop2" "next_loop", "next_loop2"):
                potential_loops = [
                    loop for loop in self._loop_dict
                    if vor_vert_idx_to_degree_array[loop._stem_vor_vert_idx]
                    ]
                if kind[-1] == "2":
                    vor_vert_idxs_array_with_dups = _numpy_concatenate(
                        [loop.vor_vert_idxs_array for loop in potential_loops]
                        )
                else:
                    vor_vert_idxs_array_with_dups = _numpy_fromiter(
                        [loop._stem_vor_vert_idx for loop in potential_loops],
                        _numpy_int64
                        )
                # Note: Because uninterrupted terminal loops are
                # quarantined prior to hyperhub cracking, it is possible
                # that any number of loops could share the same stem
                # node. Therefore, allow for duplicate (stem) nodes.
                vor_vert_idxs_array = _uniquify_flat_array(
                        vor_vert_idxs_array_with_dups
                        )
                del vor_vert_idxs_array_with_dups  # Release memory.
            else:
                if kind in ("stub", "branch_stub", "trunk", "next_branch_stub",
                            "next_trunk"):
                    mask = vor_vert_idx_to_degree_array == 1
                elif kind in ("hub", "bridge", "branch_stem", "next_bridge",
                              "next_branch_stem"):
                    mask = vor_vert_idx_to_degree_array >= 3
                elif kind == "nonbinode":
                    is_stub_mask = vor_vert_idx_to_degree_array == 1
                    is_hub_mask = vor_vert_idx_to_degree_array >= 3
                    mask = _numpy_logical_or(is_stub_mask, is_hub_mask,
                                             is_stub_mask)
                    # Reduce local namespace and release memory.
                    del is_stub_mask, is_hub_mask
                elif kind == "binode":
                    mask = vor_vert_idx_to_degree_array == 2
                elif kind == "hyperhub":
                    mask = vor_vert_idx_to_degree_array > 3
                else:
                    raise TypeError("kind is not recognized: {!r}".format(kind))
                if count_only:
                    return mask.sum()
                vor_vert_idxs_array, = mask.nonzero()
                del mask  # Release memory.

        # Compile the Voronoi vertex indices currently used by the
        # specified partitioned-out category or categories (and a few
        # other special cases).
        # Note: The special cases are when kind is "all2" or "end".
        # "all2" is an (intentionally undocumented) alias for "all" that
        # leverages the lines themselves rather than the node degree
        # records, as a potential diagnostic. Both "all2" and "end" are
        # supported for the graph skeleton alone, the partitioned
        # skeleton alone, and the raw data alone. If raw data are
        # targeted, "all2" and "end" reduce to aliases, and "all2" is
        # reassigned to "end" (further above) to simplify the code
        # further below.
        # Note: This code intentionally allows for user-defined
        # categories that do not conflict with other kinds.
        else:
            if kind[-1] == "2":  # Excludes raw data. See further above.
                if graph:  # Note: kind is "all2".
                    lines_iter = self._graph_edge_dict.itervalues()
                    if loops:
                        # *REASSIGNMENT*
                        lines_iter = _itertools.chain(lines_iter,
                                                      self._loop_dict)
                elif kind == "all2":  # Note: partitioned is True.
                    lines_iter = _flatten_to_iter(
                        [path_dict.itervalues()
                         for path_kind, path_dict in
                         self._kind_to_partitioned_path_dict.iteritems()
                         if loops or path_kind != "loop"]
                        )
                else:
                    lines_iter = self._kind_to_partitioned_path_dict[kind[:-1]].itervalues()
                vor_vert_idxs_array_with_dups = _numpy_concatenate(
                        [line.vor_vert_idxs_array for line in lines_iter]
                        )
                vor_vert_idxs_array = _uniquify_flat_array(
                        vor_vert_idxs_array_with_dups
                        )
                del vor_vert_idxs_array_with_dups  # Release memory.

            else:
                duplicates_may_exist = False  # This is the default.
                if kind == "end":
                    if graph:
                        # Note: Each hub end is duplicated across each
                        # terminating edge.
                        duplicates_may_exist = True
                        vor_vert_idx_iter = _flatten_to_iter(
                            [key.tuple for key in self._graph_edge_dict]
                            )
                        if loops:
                            # *REASSIGNMENT*
                            vor_vert_idx_iter = _itertools.chain(
                                vor_vert_idx_iter,
                                [loop._stem_vor_vert_idx
                                 for loop in self._loop_dict]
                                )
                    else:
                        # Note: The only duplicated end nodes in the
                        # partitioned skeleton are the stem nodes of
                        # uninterrupted terminal loops.
                        duplicates_may_exist = loops and self._kind_to_partitioned_path_dict["loop"]
                        vor_vert_idx_iter = _flatten_to_iter(
                            _flatten_to_iter(
                                [key.tuple for path_kind, path_dict in
                                 self._kind_to_partitioned_path_dict.iteritems()
                                 if path_kind != "loop"
                                 for key in path_dict]
                                )
                            )
                        if duplicates_may_exist:  # Note: Include loops.
                            # *REASSIGNMENT*
                            vor_vert_idx_iter = _itertools.chain(
                                vor_vert_idx_iter,
                                [loop._stem_vor_vert_idx for loop in
                                 self._kind_to_partitioned_path_dict["loop"]]
                                )
                elif kind == "branch_stem":
                    # Note: The only duplicated end nodes in the
                    # partitioned skeleton are the stem nodes of
                    # uninterrupted terminal loops.
                    duplicates_may_exist = False
                    vor_vert_idx_iter = [
                        branch._stem_vor_vert_idx for branch in
                        self._kind_to_partitioned_path_dict["branch"].itervalues()
                        ]
                elif kind == "branch_stub":
                    # Note: The only duplicated end nodes in the
                    # partitioned skeleton are the stem nodes of
                    # uninterrupted terminal loops.
                    duplicates_may_exist = False
                    vor_vert_idx_iter = [
                        branch._stub_vor_vert_idx for branch in
                        self._kind_to_partitioned_path_dict["branch"].itervalues()
                        ]
                elif kind == "loop":
                    # Note: Each stem node is duplicated within each
                    # uninterrupted terminal loop.
                    duplicates_may_exist = True
                    vor_vert_idx_iter = [
                        loop._stem_vor_vert_idx for loop in
                        self._kind_to_partitioned_path_dict["loop"]
                        ]
                else:
                    # Note: No end nodes are repeated in the partitioned
                    # skeleton (except for uninterrupted terminal loops,
                    # which are treated separately above).
                    duplicates_may_exist = False
                    vor_vert_idx_iter = _flatten_to_iter(
                        [key.tuple for key in
                         self._kind_to_partitioned_path_dict[kind]]
                        )
                vor_vert_idxs_array = _numpy_fromiter(vor_vert_idx_iter,
                                                      _numpy_int64)
                del vor_vert_idx_iter  # Reduce local namespace.
                if duplicates_may_exist:
                    # *REASSIGNMENT*
                    vor_vert_idxs_array = _uniquify_flat_array(
                        vor_vert_idxs_array
                        )
                elif not count_only:
                    # Note: Sort for consistency with other results.
                    vor_vert_idxs_array.sort()

        # If no matching Voronoi vertex indices were found, return None.
        if len(vor_vert_idxs_array) == 0:
            return None

        # If only a count is requested, return it now.
        if count_only:
            return len(vor_vert_idxs_array)

        # Construct required arrays.
        if coords:
            if interpolate_3D:
                coords_array = SkeletalLineString2D(
                    self, vor_vert_idxs_array
                    ).coords_array3D
            else:
                coords_array = self.voronoi_vertex_coords_array[vor_vert_idxs_array]
        else:
            coords_array = None  # Placeholder.
        if degrees:
            degrees_array = vor_vert_idx_to_degree_array[vor_vert_idxs_array]
        else:
            degrees_array = None  # Placeholder.

        # Return.
        return (vor_vert_idxs_array, coords_array, degrees_array)

    def _delete_keys(self, keys_set, return_lines=False):
        """
        Low-level helper function for .delete_lines().

        keys_set is a set of the keys to be deleted from the line dictionaries.
        Each key must compare equal to targeted_line._aligned_key, except for
        loops, for which key must compared equal to targeted_line itself, where
        targeted_line is the line to be deleted in each case. If a key does not
        correspond to a line in any line dictionary, no error is raised.

        return_lines is a boolean that specifies whether the deleted lines
        should be returned as a list.
        """
        # Set up from- and (to-)kinds...
        # ...for deleting from the partitioned skeleton...
        from_kinds = self._kind_to_partitioned_path_dict.keys()
        kinds = [None] * len(from_kinds)
        # ...for deleting from the graph skeleton...
        # Note: self._graph_edge_dict is effectively a placeholder for
        # the true from-kind (None).
        from_kinds.append(self._graph_edge_dict)
        kinds.append(False)
        # ...and for deleting uninterrupted terminal loops from their
        # quarantine.
        # Note: self._loop_dict is effectively a placeholder for the
        # true from-kind (None).
        from_kinds.append(self._loop_dict)
        kinds.append(False)

        # Find and unregister each line implied by a key.
        register_line = self._register_line
        if return_lines:
            deleted_lines = []
            deleted_lines_append = deleted_lines.append
        for kind, from_kind in _izip(kinds, from_kinds):
            if kind is None:
                line_dict = self._kind_to_partitioned_path_dict[from_kind]
            else:
                line_dict = from_kind
                # Note: Here is where the placeholder from-kinds are
                # replaced.
                from_kind = None  # *REASSIGNMENT*
            if keys_set.isdisjoint(line_dict):
                continue
            for key in keys_set:
                if key in line_dict:
                    line = line_dict[key]
                    idx0, idxN = key.tuple
                    register_line(line, kind, None, idx0, idxN, from_kind)
                    if return_lines:
                        deleted_lines_append(line)

        # Optionally return lines.
        if return_lines:
            return deleted_lines
        return None

    def delete_lines(self, lines, safe=True):
        """
        Delete any line or lines from either or both skeletons.

        Any edge or path--including uninterrupted terminal loops, whether
        quarantined or added to the partitioned skeleton--can be deleted.
        However, deleting lines, especially from the partitioned skeleton, can
        have unexpected consequences, including splitting the partitioned
        skeleton into two or more connected components. It is highly recommended
        that you consider .prune() instead for deleting paths from the
        partitioned skeleton.

        lines is a SkeletalLineString or sequence of SkeletalLineString's that
        specifies the edges and/or paths to be deleted. If any line is
        unrecognized (e.g., was already deleted), that line is simply ignored.
        However, if any object other than a SkeletalLineString is encountered,
        an error is raised.

        safe is a boolean that specifies whether it should be tested that each
        line in lines belongs to the current skeletal analysis. If safe is False
        and line belongs to another skeletal analysis, a line from the current
        skeletal analysis could be deleted in its place. safe is False is a bit
        faster.

        Warning: The current function does *not* undo partitioning-out. Deleting
        a partitioned-out path edge will not restore it (nor its components) to
        the graph skeleton.
        """
        # Convert lines to a sequence, if necessary.
        if isinstance(lines, Skeletal):
            lines = (lines,)  # *REASSIGNMENT*

        # Create a set of the appropriate keys.
        keys_set = set()
        keys_set_add = keys_set.add
        for line in lines:
            # Note: Allow for .skeleton to be a weakref.proxy.
            if safe and line.skeleton.__hash__() != hash(self):
                continue
            aligned_key = line._aligned_key
            # Note: If line is an uninterrupted (terminal) loop, its key
            # is itself.
            if len(aligned_key) == 1:
                keys_set_add(line)
                continue
            keys_set_add(aligned_key)

        # Delete lines by their keys.
        if keys_set:
            self._delete_keys(keys_set)
            self._lines_were_manually_deleted = True

    @staticmethod
    def _raise_psutil_error():
        raise TypeError(
            "psutil module must be available or an explicit target memory footprint specified in isolation_mode (at initialization)"
            )

    def estimate_max_node_count(self):
        """
        Estimate how many nodes can be safely processed in memory.

        There are two potential memory bottlenecks during processing. The first
        occurs during initialization, primarily due to Voronoi analysis, and
        scales directly with the number of boundary samples and hence inversely
        with the sampling interval. At that bottleneck, the smallest sampling
        inteval that can be accommodated within the targeted memory footprint
        (allowing no margin) bottleneck is given by
            .estimate_safe_interval(..., include_partitioning=False)[0]
        The second bottleneck occurs during partitioning-out and depends most
        directly on the number of nodes in the graph skeleton, excluding those
        binodes that have been contracted through. With both bottlenecks
        considered, the smallest interval that can be fully processed within the
        targeted memory footprint (with some margin) is given by
            .estimate_safe_interval(..., include_partitioning=True)[0]
        but requires a brute-force search because of the complex relationship
        between sampling interval and node count. Alternatively, the current
        function can be called after initialization to directly (and quickly)
        return the critical number of nodes that can be fully processed
        (allowing no margin). With this knowledge, it may be possible to
        simplify the skeleton by some means other than coarsening of the
        sampling interval so that no more than this number of nodes remains.

        See also:
            .estimate_safe_interval()
        """
        if self.target_memory_footprint is None:
            self._raise_psutil_error()
        avbl_partitioning_mem = (self.target_memory_footprint
                                 - self._approx_nonpartitioning_mem)
        # Note: This calculation is estimated from testing. If it is
        # ever updated, the inverse calculation (used to generate a
        # helpful error message) should be updated in
        # ._apply_brute_and_safe_options().
        return int((avbl_partitioning_mem / 18.)**0.5)

    def get_node_count(self, kind, graph=True, partitioned=False, loops=True):
        """
        Count nodes in the graph or partitioned skeleton.

        For the same arguments, the current function is equivalent to
            len(self.get_nodes(...)[0])
        The current function is never meaningfully slower than that code and can
        be much faster.

        All arguments have the same meaning as in .get_nodes().
        """
        calling_kwargs = locals().copy()
        if not graph:
            if not partitioned:
                if kind == "all":
                    return len(self._nonperi_adj_dict)
            elif kind == "loop":
                return len(self._kind_to_partitioned_path_dict["loop"])
            elif not loops or not self._kind_to_partitioned_path_dict["loop"]:
                if kind == "end":
                    count = 0  # Initialize.
                    for (path_kind,
                         path_dict) in self._kind_to_partitioned_path_dict:
                        if path_kind == "loop":
                            continue
                        count += 2*len(path_dict)
                    return count
                if kind in self._kind_to_partitioned_path_dict:
                    return len(self._kind_to_partitioned_path_dict[kind]) * 2
                if kind in ("branch_stem", "branch_stub"):
                    return len(self._kind_to_partitioned_path_dict["branch"])
        return self._get_nodes(count_only=True, coords=False, degrees=False,
                               interpolate_3D=False, **calling_kwargs)

    def gobble_paths(self, indices, invert=False, return_paths=False):
        """
        Delete partitioned-out paths with a specified stub at either end.

        The current function deletes (from the partitioned skeleton) each
        partitioned-out path with a specified stub at either end. It then
        deletes any path that now has a "new" stub at either end due to the
        previous series of deletions. (Although the "new" stub's node existed
        previously, it is only newly a stub.) This step repeats until no more
        paths can be deleted (i.e., no new stubs are formed by a series of
        deletions).

        indices is a container of integers that specifies the Voronoi vertex
        indices (see .get_nodes()) marked for gobbling (or their complement, if
        invert is True). If any value in indices is not the Voronoi vertex index
        of a stub in the partitioned skeleton at the time the function is
        called, that value is ignored.

        invert is a boolean that specifies whether those stubs whose Voronoi
        vertex indidces are not included in indices should be gobbled instead.

        return_paths is a boolean that specifies whether the gobbled paths
        should be returned as a list. Otherwise, None is returned.
        """
        # Convert indices to a set.
        if isinstance(indices, _numpy_ndarray):
            indices = indices.tolist()  # *REASSIGNMENT*
        indices_set = set(indices)
        del indices  # Reduce local namespace.
        must_subset_indices = True  # Initialize.

        # Prepare to compile gobbled paths, if necessary.
        if return_paths:
            gobbled_paths = []
            gobbled_paths_extend = gobbled_paths.extend

        while True:

            # Identify those stubs to be gobbled.
            cur_stub_indices_array, = (
                self._vor_vert_idx_to_partitioned_degree_array == 1
                ).nonzero()
            cur_stub_indices_set = set(cur_stub_indices_array.tolist())
            del cur_stub_indices_array  # Release memory.

            # Subset indices to include only nodes that were stubs when
            # the current function was called, if this has not already
            # been done (i.e., if this is the first iteration).
            # Note: This is done primarily to retain similarity to
            # .gobble_edges().
            if must_subset_indices:
                indices_set &= cur_stub_indices_set
                # Note: must_subset_indices is permanently set to False
                # further below.
            if invert:
                cur_stub_indices_set -= indices_set
            elif must_subset_indices:
                # Note: The implied intersection was already performed
                # further above.
                cur_stub_indices_set = indices_set  # *REASSIGNMENT*
                must_subset_indices = False
            else:
                cur_stub_indices_set &= indices_set
            # Note: cur_stub_indices_set was reused further above for
            # efficiency. Change its name to represent its current
            # contents.
            gobble_stub_indices_set = cur_stub_indices_set
            del cur_stub_indices_set  # Reduce local namespace.

            # If no stubs remain to be gobbled, break loop.
            if not gobble_stub_indices_set:
                break

            # Find those path keys that include a stub to be gobbled.
            gobble_stub_indices_set_isdisjoint = gobble_stub_indices_set.isdisjoint
            gobble_keys_set = set()
            gobble_keys_set_add = gobble_keys_set.add
            for path_dict in self._kind_to_partitioned_path_dict.itervalues():
                for key in path_dict:
                    if gobble_stub_indices_set_isdisjoint(key):
                        continue
                    gobble_keys_set_add(key)

            # Delete keys to be gobbled.
            result = self._delete_keys(gobble_keys_set, return_paths)
            if return_paths:
                gobbled_paths_extend(result)

        # Optionally return gobbled paths.
        if return_paths:
            return gobbled_paths
        return None

    def gobble_edges(self, indices, invert=False, return_edges=False,
                     contract=True):
        """
        Delete sequences of marginal graph edges from particular stubs.

        The current function walks the graph skeleton inward from each specified
        (or implied) starting stub, deleting each edge it encounters until a
        (degree >1) non-stub is encountered, where each node's degree is
        continuously updated by the progressive deletions. Consequently, nodes
        that are (degree 2) binodes or even (degree 3) hubs at the time that the
        function is called may still be deleted.

        indices is a container of integers that specifies the Voronoi vertex
        indices (see .get_nodes()) from which gobbling should start (or their
        complement, if invert is True). If any value in indices is not the
        Voronoi vertex index of a stub in the graph skeleton at the time the
        function is called, that value is ignored.

        invert is a boolean that specifies whether gobbling should instead start
        from every stub whose Voronoi vertex index is not included in indices.

        return_edges is a boolean that specifies whether the gobbled edges
        should be returned as a list. Otherwise, None is returned.

        contract is a boolean that specifies whether the graph skeleton should
        be (re)contracted through its binodes after gobbling. (See
        .recontract_through_binodes().)
        """
        # Note: Although the code from .gobble_paths() could be largely
        # reused here, the code below is faster if the creation time for
        # the adjacency dictionary is excluded (or possibly even if that
        # time is included). Because (re)contraction through binodes
        # requires just such an adjacency dictionary, and the contract
        # argument will likely usually be True, the code below is
        # typically (or always?) faster.

        # Convert indices to a set.
        if isinstance(indices, _numpy_ndarray):
            indices = indices.tolist()  # *REASSIGNMENT*
        indices_set = set(indices)
        del indices

        # Identify those stubs to be gobbled.
        cur_stub_indices_array, = (
            self._vor_vert_idx_to_graph_degree_array == 1
            ).nonzero()
        cur_stub_indices_set = set(cur_stub_indices_array.tolist())
        del cur_stub_indices_array  # Release memory.
        if invert:
            cur_stub_indices_set -= indices_set
        else:
            cur_stub_indices_set &= indices_set
        # Note: cur_stub_indices_set was reused further above for
        # efficiency. Change its name to represent its current
        # contents.
        gobble_stub_indices_set = cur_stub_indices_set
        del cur_stub_indices_set  # Reduce local namespace.

        # Derive adjacency dictionary.
        graph_edge_dict = self._graph_edge_dict
        adj_dict = self._derive_adjacency_dict(graph_edge_dict)

        # Prepare to compile gobbled paths, if necessary.
        if return_edges and contract:
            gobbled_edges = []
            gobbled_edges_append = gobbled_edges.append

        # Gobble depth-wise from each targeted stub.
        if not contract:
            faux_keys_set = set()
            faux_keys_set_add = faux_keys_set.add
        for start_idx in gobble_stub_indices_set:
            adj_idxs = adj_dict[start_idx]
            while len(adj_idxs) == 1:
                del adj_dict[start_idx]
                decremented_idx, = adj_idxs
                adj_dict[decremented_idx].remove(start_idx)
                if return_edges or not contract:
                    faux_key = frozenset(start_idx, decremented_idx)
                    if not contract:
                        faux_keys_set_add(faux_key)
                    else:
                        gobbled_edges_append(graph_edge_dict[faux_key])
                start_idx = decremented_idx
                adj_idxs = adj_dict[start_idx]

        # Re-create graph skeleton or delete gobbled edges only, as
        # specified.
        if contract:
            self._clear_graph_edge_dict_derivations()
            self._contract_through_binodes(adj_dict, graph_edge_dict)
        else:
            result = self._delete_keys(faux_keys_set, return_edges)

        # Optionally return gobbled edges.
        if return_edges:
            if contract:
                return gobbled_edges
            return result
        return None

    @classmethod
    def make_test_func(cls, min_cost=None, max_cost=None,
                       min_trunc_cost=None, max_trunc_cost=None,
                       path_test_func=None, ends_test_func=None,
                       no_negative_costs=False, force_accuracy=False):
        """
        Make a test function, as used by .partition_out*()'s.

        When the test function returned by the current method is passed to a
        .partition_out*(), the test function is called to determine whether a
        particular path should be accepted and whether path-processing should
        continue. It is called at up to four different points in the processing
        of each "considered" path (see documentation for the opportunistic
        argument of .partition_out*()'s):
            1) test_func(full_path_cost, True) is called prior to walking the
               path. (Type 1a)
            2) test_func(path_cost, bool, None, path_start_array, 
               path_end_array) is called as soon as both the start and end nodes 
               are known, where bool is True (False) if path is complete 
               (truncated), and the call signature is the same as for 
               ends_test_func. (Type 2)
            3) test_func(trunacted_path_cost, False) is called if the path
               being walked converges with another path, at which it truncates.
               (Type 1b)
            4) test_func(linestring, bool, kind) is called once the path is 
               merged to a LineString, where bool is True (False) if path is 
               complete (truncated); kind describes linestring and may be 
               "branch", "bridge", or "loop"; and the call signature is the same 
               as for path_test_func. (Type 3)
        Each time test_func() is called, it should return a tuple of the form
        (accept, abort), which triggers the following behavior:
            accept     abort  Types  Behavior
            True       False  1,2    Continue processing current path.
            False      False  1,2    Stop processing current path. Skip to next
                                     path.
            [ignored]  True   1,2    Do not process current or any future path
                                     on this call of .partition_out*().
            True       False  3      Partition out current path.
            True       True   3      Partition out current path, then abort all
                                     future processing of any path for this
                                     call of .partition_out*().
            False      False  3      Do not partition out current path, leaving
                                     its edges in the graph skeleton. Continue
                                     to processing next path.
            False      True   3      Do not partition out current path, leaving
                                     its edges in the graph skeleton. Abort all
                                     future processing of any path for this
                                     call of .partition_out*().
            None       False  3      Do not partition out current path and
                                     delete its edges from the graph skeleton.
                                     Continue to processing next path.
            None       True   3      Do not partition out current path and
                                     delete its edges from the graph skeleton.
                                     Abort all future processing of any path for
                                     this call of .partition_out*().
        The following attributes are automatically assigned to the returned test
        function to enhance performance and/or support functionality:
            Attribute [type]              Meaning
            .max [float]                  complete (untruncated) paths with
                                          costs > .max (or >= .max, if max is
                                          inf) are ignored
            .min [float]                  complete (untruncated) paths with
                                          costs < .min (or <= .min, if min is
                                          -inf) are ignored (default is -inf
                                          unless no_negative_costs is True)
                                          (also see note further below)
            .retest_on_convergence [bool] if False:
                                          test_func(trunacted_path_cost, False)
                                          is never called, assumed to return
                                          (True, False);
                                          if True:
                                          guarantees that includes_truncated
                                          returned by .partition_out*() is True
            .test_ends [bool]             indicates whether
                                          test_func(path_cost, [bool],
                                                    path_start_array,
                                                    path_end_array)
                                          should (ever) be called; defaults to
                                          True
            .make_deep_test_func [func]   see the documentation attached to this
                                          function
        Note that .min defaults to 0 if no_negative_costs is True, which means
        that 0-cost paths are not strictly excluded. However, it is assumed
        throughout this module that no path that should be considered can have a
        cost less than that of the highest-cost self-referential path (e.g., the
        path from a node to itself that never leaves that node). In many cases,
        such as length, the cost of a self-referental path is 0 and therefore,
        in effect, .min in these cases defaults to a value just above 0 (i.e.,
        numpy.nextafter(0., 1.)).

        min_cost is a float that specifies the minimum cost permitted for a full
        path. If a full path has a cost smaller than min_cost, that path will
        not be walked at all (and therefore will not be truncated, either). If
        min_cost is None (the default), it will be interpreted as 0 if
        no_negative_costs is True, otherwise as negative infinity.

        max_cost is a float that specifies the maximum cost permitted for a full
        path. If a full path has a cost greater than max_cost, that path will
        not be walked at all (and therefore will not be truncated, either). If
        max_cost is None (the default), it will be interpreted as positive
        infinity.

        min_trunc_cost is a float that specifies the minimum cost permitted for
        a truncated path. If min_trunc_cost is not specified (None), no minimum
        cost criterion will be applied to truncated paths.

        max_trunc_cost is a float that specifies the maximum cost permitted for
        a truncated path. If max_trunc_cost is not specified (None), or it is no
        larger than max_cost (which would logically guarantee satisfaction of
        the max_trunc_cost criterion, if cost positively correlates with
        length), no maximum cost criterion will be applied to truncated paths.

        path_test_func is a function that is called once for each path (whether
        full or truncated) upon generation of a LineString to represent that
        path. It effectively provides an opportunity to make a "final" decision
        about how to proceed. Its call signature and expected return tuple are
        described further above. Note that, like all edges and paths in the
        current class, the LineString supports some special attributes (see
        .get_lines()). Note further that if path_test_func was created by an
        earlier call of the current function and has more restrictive criteria
        (e.g., min_cost, min_trunc_cost), these will be honored.

        ends_test_func is a function that is called once for each path (whether
        full or truncated) when both ends of the path have been identified. It
        effectively provides an opportunity to avoid the processing that would
        otherwise take place prior to the calling of path_test_func(). The
        function's call signature is
            ends_test_func(path_cost, bool, path_start_array, path_end_array)
        where path_cost is the traversal cost of the (possibly truncated) path,
        bool is a a boolean that indicates whether the path was truncated, and
        path_start_array and path_end_array are the coordinates of the first and
        last nodes in the path, respectively, as arrays. Z-coordinates are
        included if .polygon is 3D. The function must return a tuple of the form
        (accept, abort), as described further above.

        no_negative_costs is a boolean that specifies whether negative costs are
        possible. Its only use is to interpret min_cost and min_trunc_cost, as
        described further above.

        force_accuracy is a boolean that specifies whether the cumulative added
        cost returned by .partition_out*()'s must include the costs of any
        truncated (converging) paths. If force_accuracy is True, the process of
        partitioning out may not be quite as fast and may use significantly more
        memory.

        Note: min_cost and max_cost can potentially dramatically reduce analysis
        time if specified.

        Note: The filtering achieved by min_trunc_cost and max_trunc_cost can be
        moved to path_test_func() without compromising the final result. Whether
        it is "better" to use min/max_trunk_cost or path_test_func() depends on
        your particular use case, but in general, applying this filtering in
        path_test_func() will slow performance but could save memory relative to
        applying the same filtering via min/max_trunc_cost.

        Note: force_accuracy may be set to True (regardless of how it is
        specified) if this is implicity required by other arguments (e.g.,
        min_trunc_cost and max_trunc_cost). If this occurs, the returned test
        function's .retest_on_convergence will be True.

        Warning: It is very easy to go wrong if you try to create a test
        function from scratch (i.e., without calling the current method), and it
        is also completely unnecessary in most cases. At a bare minimum, it is
        strongly advised that you carefully view the current method's code
        before attempting to create a custom test function on your own.
        """
        # Standardize min/max full path cost criteria, as possible.
        if min_cost is None:
            if no_negative_costs:
                min_cost = 0.
            else:
                min_cost = _python_neginf
        if max_cost is None:
            max_cost = _python_inf

        # Honor path_test_func's optimizing/support attributes, if they
        # are more restrictive than the criteria just found.
        if ends_test_func:
            ends_test_funcs = [ends_test_func]
        else:
            ends_test_funcs = []
        if path_test_func is None:
            test_trunc_cost_by_path_test = False
        else:
            if (hasattr(path_test_func, "min") and
                path_test_func.min > min_cost):
                min_cost = path_test_func.min
            if (hasattr(path_test_func, "max") and
                path_test_func.max < max_cost):
                max_cost = path_test_func.max
            test_trunc_cost_by_path_test = getattr(path_test_func,
                                                   "retest_on_convergence",
                                                   False)
            if getattr(path_test_func, "test_ends", False):
                ends_test_funcs.append(path_test_func)

        # Change truncated cost criteria to None if they are guaranteed
        # to be satisfied. Also determine whether any such criteria
        # remain.
        if min_trunc_cost is not None:
            if no_negative_costs:
                if min_trunc_cost <= 0.:
                    min_trunc_cost = None  # *REASSIGNMENT*
            elif min_trunc_cost == _python_neginf:
                min_trunc_cost = None  # *REASSIGNMENT*
        # Note: It can only be safely inferred that a truncated path
        # must have a lower cost than the full path for which it is a
        # truncation if no traversal cost can be negative.
        if (no_negative_costs and max_trunc_cost is not None
            and max_trunc_cost >= max_cost):
            max_trunc_cost = None
        retest_on_convergence = not (min_trunc_cost is None and
                                     max_trunc_cost is None and
                                     not force_accuracy and
                                     not test_trunc_cost_by_path_test)

        # Define test function.
        def test_func(
            path_or_cost, complete, kind=None, start_array=None, end_array=None,
            ends_test_funcs=ends_test_funcs, path_test_func=path_test_func,
            retest_on_convergence=retest_on_convergence,
            min_trunc_cost=min_trunc_cost, max_trunc_cost=max_trunc_cost,
            test_trunc_cost_by_path_test=test_trunc_cost_by_path_test
            ):
            # If kind is not None, call must be Type 3.
            if kind is not None:
                # Call path_test_func() if it is specified. Otherwise,
                # return the default.
                if path_test_func is not None:
                    return path_test_func(path_or_cost, complete, kind)
                return (True, False)
            
            # If start_array was passed, call must be Type 2. Call each 
            # ends_test_func() and return the most restrictive result.
            if start_array is not None:
                accepts = []
                for ends_test_func in ends_test_funcs:
                    accept, abort = ends_test_func(path_or_cost, complete,
                                                   start_array, end_array)
                    if abort:
                        # Note: accept is ignored.
                        return (None, True)
                    accepts.append(accept)
                for accept in accepts:
                    if not accept:
                        return (False, False)
                return (True, False)

            # If path is complete, call must be Type 1a. Assuming that 
            # min/max cost criteria are the only criteria that could 
            # possibly be applied and were already satisfied (for 
            # complete paths) by code within 
            # ._find_longest_shortest_paths(), immediately return 
            # the default.
            if complete:
                return (True, False)

            # By process of elimination, call must be Type 1b. Test 
            # truncated cost against corresponding criteria, if any.
            if min_trunc_cost is None or path_or_cost >= min_trunc_cost:
                if (max_trunc_cost is None or
                    path_or_cost <= max_trunc_cost):
                    if test_trunc_cost_by_path_test:
                        # Note: Honor path_test_func()'s Type 1b 
                        # behavior.
                        return path_test_func(path_or_cost, False)
                    return (True, False)
                return (False, False)
            return (False, False)

        # Define deep-test function.
        test_ends = bool(ends_test_funcs)
        def make_deep_test_func(cost_func, _test_func=test_func,
                                _cost_func_to_deep_test_func={}):
            """
            Create a deep test function.

            The returned function deep_test_func() simulates the role of
            test_func() within Skeleton.partition_out*()'s as much as possible
            and has a Type 3 call signature (see Skeleton.make_test_func()).
            Specifically, deep_test_func() first tests whether the traversal
            cost of the path passed to it satisfies the cost_func.min and
            cost_func.max bounds. Even though test_func() is not actually called
            to implement this test (neither within deep_test_func() nor within
            Skeleton.partition_out*()'s), we will refer to this test as a "Type
            0" call for simplicity hereinafter. Then, if the passed path is
            complete:
                test_func() calls of Types 1a, 2, and 4 are executed, in order.
            Otherwise, if the passed path is truncated:
                test_func() calls of Types 2, 1b, and 4 are executed, in order.
            The only notable departure of this simulation from what is
            implemented in Skeleton.partition_out*()'s is that if a truncated
            path is passed, a Type 1a call is never executed, because the
            corresponding full (non-truncated) path traversal cost is unknown.
            Equivalent to the implementation in Skeleton.partition_out*()'s, as
            soon as a call indicates that a path should be rejected or that an
            abort should be triggered, the sequences of calls enumerated above
            is exited. In all cases, the result of the final test_func() call is
            returned, where the failure of a Type 0 "call" returns (False,
            False). A major use for deep_test_func() is as the value for the
            *test_func arguments in Skeleton.sweep()

            cost_func is a function that specifies a cost function as returned
            by Skeleton.make_test_func().
            """

            # If cost_func was already seen, return the deep_test_func()
            # stored at that time.
            if cost_func in _cost_func_to_deep_test_func:
                return _cost_func_to_deep_test_func[cost_func]

            # Define deep_test_func().
            def deep_test_func(
                path, complete, kind, cost_func=cost_func,
                minimum=max(cost_func.min, float(_numpy_nearly_neginf)),
                maximum=min(cost_func.max, float(_numpy_nearly_inf)),
                test_func=_test_func,
                retest_on_convergence=cost_func.retest_on_convergence,
                test_ends=cost_func.test_ends
                ):
                """
                Simulate the role of test_func within Skeleton.partition_out*()'s.

                A tuple of booleans of the form (accept, abort) are returned.
                See .make_deep_test_func() of any cost_func() returned by
                Skeleton.make_test_func() for more details.

                path is a SkeletalLineString2D that specifies a path suitable for
                partitioning out.

                complete is a boolean that specifies whether path is complete (rather
                than truncated).
                """
                # If path is complete, simulate a Type 0 "call".
                # Note: test_func() is never called within
                # Skeleton._find_longest_shortest_paths() to implement a
                # bounds check because the permitted bounds are enforced
                # prior to path-walking.
                path_cost = cost_func(path)
                if complete and path_cost < minimum or path_cost > maximum:
                    return (False, False)

                # If path is complete, simulate a Type 1a call.
                if complete:
                    accept, abort = test_func(path_cost, True)
                    if not accept or abort:
                        return (accept, abort)

                # If end coordinates should be tested, simulate a Type 2
                # call.
                skeleton = path.skeleton
                if test_ends:
                    if skeleton.polygon.is_3D:
                        ends_coords = skeleton._convert_vor_vert_idxs_to_coords3D(
                            *path._aligned_key.tuple
                            )
                    else:
                        ends_coords = skeleton._convert_vor_vert_idxs_to_coords2D(
                            *path._aligned_key.tuple
                            )
                    accept, abort = test_func(path_cost, complete, None,
                                              *ends_coords)
                    if not accept or abort:
                        return (accept, abort)

                # If path is incomplete, simulate a Type 1b call.
                if not complete:
                    accept, abort = test_func(path_cost, False)
                    if not accept or abort:
                        return (accept, abort)

                # Simulate a Type 3 call.
                return test_func(path, complete, kind)

            # Store and return deep_test_func().
            _cost_func_to_deep_test_func[cost_func] = deep_test_func
            return deep_test_func

        # Assign attributes to enhance performance and/or support
        # functionality.
        # Note: The following attribute assignments enhance performance
        # (e.g., in ._find_longest_shortest_paths()), dramatically in
        # some cases.
        test_func.min = min_cost
        test_func.max = max_cost
        test_func.retest_on_convergence = retest_on_convergence
        # Note: This attribute indicates whether test_func() should be
        # called once path ends are (or can be) identified.
        test_func.test_ends = test_ends
        # Note: This attribute is used to simulate the role of
        # test_func() in certain contexts.
        test_func.make_deep_test_func = make_deep_test_func
        return test_func

    @classmethod
    def make_cost_func(cls, hint, directed=False, get_flipped_cost=None,
                       infinite=True, negative=True, avoid_nan=True):
        """
        Make a cost function and related priority and orientation functions.

        The current function creates and returns a cost function, with
        attributes:
            cost_func(line, flipped=False)
                Calculate the cost of traversing line. If directed is True,
                specifying flipped=True instead calculates the cost of
                traversing the line in the direction opposite to its current
                orientation.
            cost_func.get_cost(line)
                Same as hint, if hint is a function, or a function that returns
                the attribute specified by hint.
            cost_func.is_directed
                Same as the directed argument.
            cost_func.avoid_nan
                Same as the avoid_nan argument.
            cost_func.infinite
                Same as the infinite argument.
            cost_func.negative
                Same as the negative argument.
            cost_func.priority_func(line)
                Get the minimum cost of traversing line in either direction, if
                directed is True. Otherwise identical to cost_func.get_cost().
            if directed:
                cost_func.get_flipped_cost(line)
                    Same as get_flipped_cost argument, if it is specified,
                    otherwise equivalent to cost_func(line, True).
                cost_func.orient_func(line)
                    Return line oriented in-place such that its traversal cost
                    is minimized.

        hint is a function or string that specifies how a line's traversal cost
        should be calculated. If hint is a function, it must accept a LineString
        as its first or only argument and return the traversal cost of that
        line, which must be coercible to a float (including nan, inf, and -inf,
        but excluding None, for example). If that cost depends on direction, the
        cost corresponding to the LineString's current orientation should be
        returned. If hint is instead a string, it is interpreted to be an
        attribute of the LineString and replaced with a function that
        efficiently fetches that attribute (using operator.attrgetter()). For
        example, the following will return a cost function based on the lengths
        of a LineString:
            make_cost_func(".length")
            make_cost_func(".length", avoid_nan=None)  # Faster.
        "Sub-attributes" may also be specified (though the particular example
        below is probably useless):
            make_cost_func(".skeleton.polygon")

        directed is a boolean that specifies whether traversal costs should be
        treated as dependent on direction. If directed is False (the default),
        the returned cost function may be much more efficient than if directed
        were specified as True. For an example in which direction matters,
        consider .delta_z (see .get_lines()).

        get_flipped_cost is a function that specifies how a line's traversal
        cost in the direction opposite its current orientation should be
        calculated. It has the same call signature as hint and the same
        expectations for the returned value. As an example, consider the
        following:
            make_cost_func(".delta_z", True, lambda line: -line.delta_z)
        If directed is True and get_flipped_cost is not specified (None), a
        suitable function based on hint() is automatically generated.
        Nonetheless, specifying your own function can be much more efficient.
        If directed is instead False, this argument is ignored.

        infinite is a boolean that specifies whether infinite traversal costs
        should be supported. If infinite is False and an infinite cost is
        calculated, behavior may not be as expected.

        negative is a boolean that specifies whether negative costs should be
        supported. If negative is False and a negative cost is calculated,
        behavior may not be as expected.

        avoid_nan is a boolean that specifies how costs that evaluate to nan
        should be handled by cost_func.orient_func(). If cost_func.get_cost()
        and cost_func.get_flipped_cost() will never return nan, you should
        specify avoid_nan as None for some performance benefit. If avoid_nan is
        True (the default), cost_func.orient_func() will return nan only if the
        traversal cost in both directions is nan. If avoid_nan is instead False,
        cost_func.orient_func() will return nan if the traversal cost in either
        direction is nan.

        Note: You can use a cost function to force traversal to follow a
        particular direction by returning a special value to disallow the
        undesired direction. For maximum efficiency, nan is recommended, but
        positive infinity (i.e., float("inf")) works as well if the infinite
        argument is specified as True in the current function. For example, if
        you only want paths to go downslope and follow the steepest gradient,
        you could specify the following for hint:
            def hint(line, nan=float("nan")):
                if line.delta_z > 0:
                    return nan  # Disallow upslope direction.
                return line.length  # Route by shortest (2D) path.
        Note that returning line.length would result in the shortest 2D paths
        (i.e., in the xy plane) being preferred, and the shortest 2D path
        between any two points on a 3D surface implies the maximum gradient.
        cost_func.orient_func() leverages the just described special value
        behavior to determine an acceptable orientation for a line. The line
        will be flipped in place up to one time to achieve a traversal cost that
        is not specially valued, and then returned with the corresponding
        orientation. If both orientations have specially-valued traversal costs,
        the line is oriented arbitrarily. Note that you may be able to create a
        more efficient orientation function depending on your use case,
        especially if you did not specify get_flipped_cost or you can devise a
        faster orientation test than fully calculating the traversal cost.

        Warning: It is strongly suggested that neither hint() nor
        get_flipped_cost() return negative traversal costs. Although negative
        costs are, strictly, nominally supported, they are usually unwise. If
        .polygon has one or more holes, negative costs raise the possibility
        that negative cycles will occur, in which a path winds around a hole ad
        infinitum to minimize its cumulative traversal cost, i.e., make it ever
        more negative. If this happens, analysis will not get stuck in that
        infinite loop, but partitioning will fail. On the other hand, if
        .polygon does not have holes, routing by negative values usually yields
        results that are either undesirable or identical to what could be
        achieved using traversal costs >= 0, and is less computationally
        efficient in the latter case.
        """
        # Convert hint to a function to get the cost of a LineString in
        # its "forward" (current) direction, if necessary.
        if isinstance(hint, basestring):
            get_cost = _operator.attrgetter(hint.lstrip("."))
        elif hasattr(hint, "__call__"):
            get_cost = hint
        else:
            raise TypeError("cost_func must be a function or string")

        # Determine what function to use to find the minimum of an
        # iterable of values and honor avoid_nan (while maximizing
        # efficiency).
        if avoid_nan:
            # Note: A potential problem is that numpy.nanmin() raises a
            # warning when it "must" return nan (because only nan's were
            # passed to it). If that behavior is ever replicated by
            # bottleneck.nanmin(), it would be undesirable.
            get_min = _bottleneck_nanmin
        elif avoid_nan is None:
            # Note: Python's min() does not have consistent behavior: it
            # may or may not return nan if nan is present, depending on
            # the order of the passed iterable. On the other hand, min()
            # is much faster (for the non-array sequences passed to it).
            get_min = min
        else:
            # Note: In testing, bottleneck.nanmin() was usually faster
            # than numpy.min(). However, numpy.min() must be used to
            # replicate the behavior implied by avoid_nan=False.
            get_min = _numpy_min

        # Define functions to orient a LineString, find the cost of the
        # flipped LineString, and calculate a LineString's "priority"
        # (minimum cost in either direction), respectively.

        # If cost is not directed, orientation is irrelevant and the
        # other two functions reduce to get_cost().
        if not directed:
            # *REASSIGNMENT* (for get_flipped_cost())
            get_flipped_cost = priority_func = get_cost

        else:

            # If get_flipped_cost is not specified, leverage
            # LineString.flip() to derive it and an orientation
            # function.
            if get_flipped_cost is None:
                def orient_func(line, get_cost=get_cost):
                    # If line, as currently oriented, has a finite cost
                    # (indicating that it is acceptably oriented),
                    # return line.
                    # Note: The inequality below evaluates False if
                    # get_cost() returns nan or positive infinity
                    # (whether from standard Python or numpy).
                    forward_cost = get_cost(line)
                    if forward_cost < _python_inf:
                        return line
                    # Current (original) orientation is not acceptable,
                    # so flip line and test its cost again.
                    line.flip(False)
                    result = None  # This is the default.
                    try:
                        # If (now flipped) line is accceptably oriented,
                        # set it as the result to be returned.
                        flipped_cost = get_cost(line)
                        if flipped_cost < _python_inf:
                            result = line  # *REASSIGNMENT*
                    finally:
                        if result is None:
                            # Either an error occurred, or neither
                            # orientation is acceptable for line, so
                            # restore its original orientation.
                            line.flip(False)
                        return result

                # *REASSIGNMENT*
                def get_flipped_cost(line, get_cost=get_cost):
                    line.flip(False)
                    try:
                        flipped_cost = get_cost(line)
                    finally:
                        line.flip(False)
                    return flipped_cost
            else:

                # User specified get_flipped_cost. Define orientation
                # function that only flips line if necessary.
                def orient_func(line, get_cost=get_cost,
                                get_flipped_cost=get_flipped_cost):
                    # If line, as currently oriented, has a finite cost
                    # (indicating that it is acceptably oriented),
                    # return line.
                    # Note: The inequality below evaluates False if
                    # get_cost() returns nan or positive infinity
                    # (whether from standard Python or numpy).
                    forward_cost = get_cost(line)
                    if forward_cost < _python_inf:
                        return line
                    # Current (original) orientation is not acceptable,
                    # so test flipped cost.
                    flipped_cost = get_flipped_cost(line)
                    if flipped_cost < _python_inf:
                        # Flipped cost is acceptable, so flip and return
                        # line.
                        return line.flip(False)
                    # Neither orientation is acceptable for line, so
                    # return None.
                    return None

                # If no nan's are expected, opt for a simple (and more
                # computationally efficient) inequality.
                if avoid_nan is None:
                    def priority_func(line, get_min=get_min,
                                      get_cost=get_cost,
                                      get_flipped_cost=get_flipped_cost):
                        forward_cost = get_cost(line)
                        backward_cost = get_flipped_cost(line)
                        if backward_cost < forward_cost:
                            return backward_cost
                        return forward_cost

                # If nan's are expected, treat finding the minimum cost
                # with care.
                # Note: Any inequality involving nan evaluates False, so
                # get_min() must be used instead of an inequality-based
                # test if any nan's are expected.
                else:
                    def priority_func(line, get_min=get_min, get_cost=get_cost,
                                      get_flipped_cost=get_flipped_cost):
                        result = get_min((get_cost(line),
                                          get_flipped_cost(line)))
                        # Note: Because nan-cost paths cannot be walked,
                        # it is assumed that they should have the lowest
                        # possible priority score (which maximizes their
                        # potential to be deleted).
                        if _math_isnan(result):
                            return _python_neginf
                        return result

        # Define cost function.
        def cost_func(line, flipped, directed=directed, get_cost=get_cost,
                      get_flipped_cost=get_flipped_cost):
            if directed and flipped:
                return get_flipped_cost(line)
            return get_cost(line)

        # Assign attributes to cost function and return it.
        # Note: cost_func.is_directed may be tested to enhance
        # performance (e.g., in ._make_csr_matrix()).
        cost_func.is_directed = directed
        cost_func.avoid_nan = avoid_nan
        cost_func.infinite = infinite
        cost_func.negative = negative
        cost_func.priority_func = priority_func
        cost_func.get_cost = get_cost
        if directed:
            cost_func.get_flipped_cost = get_flipped_cost
            cost_func.orient_func = orient_func
        return cost_func

    def minimize_memory(self, deep=False):
        """
        Minimize the memory footprint of the current instance.

        The current function primarily deletes objects that optimize performance
        but do so at the cost of increased memory use. In these cases, the
        relevant object is automatically ("lazily") re-generated at the moment
        it is next needed (if ever), but this re-generation can be
        computationally expensive. It is therefore not advisable to call the
        current function unless necessary.

        deep is a boolean that specifies whether a deeper (and more time
        consuming) memory release should be performed. (See warning below!)

        Warning: A call with deep specified as True has the following
        consequences:
            1) .voronoi is deleted, if it exists, and cannot be re-generated.
            2) Effectively zeroes the "hardness" of any line that has a hardness
               of 1 at the time that the function is called, which can have
               serious consequences if self is deleted. (See .get_lines().) Any
               line with a hardness of 2 is unaffected.
        """
        self._clear_graph_edge_dict_derivations()
        self._clear_partitioned_edge_dicts_derivations()
        if not deep:
            return
        if hasattr(self, "voronoi"):
            del self.voronoi
        for kind in ("partitioned", "unpartitioned"):
            for line_dict in self._fetch_line_dicts_by_kind(kind, False):
                for line in line_dict.itervalues():
                    # Note: The line below is equivalent to
                    # line.minimize_memory(True).
                    line._Lazy__clear_lazy()

    @staticmethod  # Note: self is passed when called.
    def _partition_out(self, kind, node_kind1, node_kind2, cost_func,
                       opportunistic, target, test_func, from_nodes, to_nodes,
                       disjoint_kind=None, **kwargs):
        """
        Partition out paths from the graph skeleton.

        The current function effectively implements .partition_out*()'s as well
        as wraps ._find_longest_shortest_paths(). Its main contribution is to
        support nonzero target values (recursion) and facilitate the
        disjoint_kind argument.

        kind is a string that specifies the key in
        self._kind_to_partitioned_path_dict to which the results will be
        registered, with an exception for the special case that kind is "branch"
        but the path is really a trunk (see ._register_line()).

        node_kind1 is a string that specifies the kind of node that must be
        found at one end of each full path. .get_nodes(node_kind1) must describe
        these nodes.

        node_kind2 is a string that specifies the kind of node that must be
        found at the other end of each full path. .get_nodes(node_kind2) must
        desribe these nodes.

        disjoint_kind is a string that specifies the kind of edge whose nodes
        are forbidden for any path to touch. If opportunistic is True, this
        disjoint relationship is only guaranteed for complete (non-truncated)
        paths.

        The remaining arguments are passed directly from .partition_out*()'s and
        are documented in those methods.
        """
        # If a cost function is not specified, default to using length.
        if cost_func is None:
            # *REASSIGNMENT*
            cost_func = EasySkeleton.length2D_cost_func

        # Execute recursive behavior, if required.
        if target is None or target > 1:
            kwargs.update(locals())
            kwargs["target"] = 0
            new_added_path_count = True  # Initialize.
            if target is None:
                target = _python_inf  # *REASSIGNMENT*
            partition_out = self._partition_out
            added_path_count = 0  # Initialize.
            cumulative_cost = 0.  # Initialize.
            includes_truncated = True  # Initialize.
            considered_path_count = 0  # Initialize.
            deleted_path_count = 0  # Initialize.
            while new_added_path_count and added_path_count < target:
                (new_added_path_count,
                 new_cumulative_cost,
                 new_includes_truncated,
                 new_considered_path_count,
                 new_deleted_path_count
                 ) = partition_out(**kwargs)
                added_path_count += new_added_path_count
                cumulative_cost += new_cumulative_cost
                if includes_truncated and not new_includes_truncated:
                    includes_truncated = False
                considered_path_count += new_considered_path_count
                deleted_path_count += new_deleted_path_count
            return (added_path_count, cumulative_cost, includes_truncated,
                    considered_path_count, deleted_path_count)

        # Check if any edges remain to be partitioned out.
        if not self._graph_edge_dict:
            return (0, 0., True, 0, 0)

        # Process disjoint_kind argument.
        if disjoint_kind is None:
            forbidden_nodes_arrays = None
        else:
            # Note: Although forbidden_nodes_arrays could be generated
            # once per recursive (target >1 or None) call in some cases,
            # this becomes complicated if disjoint_kind and kind are the
            # same, or if kind may be changed to disjoint_kind by
            # ._register_line(). Because disjoint_kind is considered an
            # advanced functionality anyway and regeneration (and the
            # consequent processing in
            # ._kind_to_partitioned_path_dict()) is not a performance
            # bottleneck), forbidden_nodes_arrays is simply regenerated
            # each time that it is needed.
            forbidden_nodes_arrays = [
                line.vor_vert_idxs_array
                for line in self._kind_to_partitioned_path_dict[disjoint_kind].itervalues()
                ]

        # Partition out.
        return self._find_longest_shortest_paths(
            kind, cost_func, from_nodes, to_nodes, node_kind1, node_kind2,
            test_func, opportunistic is None, bool(opportunistic),
            forbidden_nodes_arrays
            )

    def partition_out_trunk(self, target=None, cost_func=None, test_func=None,
                            from_nodes=None, to_nodes=None, opportunistic=None,
                            disjoint=True):
        """
        Partition out a main path (or paths) from the graph skeleton.

        The current method must be called first among .partition_out*()'s, as
        it partitions out (what is meant to be) the main trunk (or trunks) of
        the skeleton. Each path partitioned out by the current method must have
        each of its ends at a node that is both a stub in the graph
        skeleton and not yet present in the partitioned skeleton. Like all
        .partition_out*()'s, the current method:
            1) Identifies paths of consecutively linked edges in the graph
               skeleton between suitable nodes. Each path is the lowest-cost
               (e.g., shortest) path between the two nodes at either end, and
               the first path identified is the highest-cost (e.g., longest) of
               all such paths. Therefore, the first path effectively links the
               two suitable nodes that are most widely separated in terms of
               path cost (e.g., length), under the requirement that all paths
               must be optimized for cost.
            2) Removes the component edges of accepted paths from the graph
               skeleton.
            3) Merges each path's components edges together.
            4) Adds each path to the partitioned skeleton.
            5) Returns a tuple of the form (added_path_count, cumulative_cost,
               includes_truncated, considered_path_count, deleted_path_count),
               where:
            added_path_count:      Total number of paths partitioned out
                                   (including each truncated part, if any).
            cumulative_cost:       Total cost of all paths (including at least
                                   those full paths that were not truncated).
            includes_truncated:    Boolean (True/False) indicating whether
                                   cumulative_cost includes the cost of any
                                   truncated parts that were partitioned out.
            considered_path_count: Total number of full paths that were
                                   considered. If search was aborted, this count
                                   includes the path that was being considered
                                   when the abort was triggered.
            deleted_path_count:    Total number of paths (including each
                                   truncated part, if any) deleted from the
                                   graph skeleton.

        target is an integer that specifies the minimum number of paths that
        should be partitioned out, if possible. In effect, the current method is
        called repeatedly (with target=0), with each call corresponding to a
        single pass (see opportunistic argument), until either target is
        satisfied (but possibly exceeded) or a call partitions out no paths. If
        target is None, it is interpreted as infinity. See note further below
        for method-specific consequences.

        cost_func is a funcion that specifies how the cost of each edge in the
        graph skeleton is determined. See .make_cost_func(). If cost_func is not
        specified (None), it defaults to EasySkeleton.length2D_cost_func.

        test_func is a function that is called to determine whether a particular
        path should be accepted and whether path-processing should continue. It
        will only be called for paths that are "considered" (see opportunistic
        argument). See .make_test_func() for more details.

        from_nodes is a sequence of the the nodes (represented by their Voronoi
        vertex indices) that are accetable to the user as path start nodes.
        These will be subset, as necessary, to ensure that they meet the base
        criteria for start and/or end nodes within the current method (see
        further above). Use .get_nodes() to explore the available node options.

        to_nodes is a sequence of the the nodes (represented by their Voronoi
        vertex indices) that are accetable to the user as path end (final)
        nodes. These will be subset, as necessary, to ensure that they meet the
        base criteria for start and/or end nodes within the current method (see
        further above). Use .get_nodes() to explore the available node options.

        opportunistic is a boolean that specifies some details of the behavior
        of path processing. At the start of each pass (see target argument), the
        graph skeleton is solved in a graph traversal sense such that the least-
        cost path between every pair of nodes is computed. Then, each of these
        paths is considered in order of decreasing cost. Note that, if more than
        one path is partitioned out in a single pass, it is possible for a path
        under consideration to converge with a path partitioned out earlier on
        the same pass. With this background in mind, the behavior of path
        processing depends on the value of opportunistic in the following ways:
            True:    Any convergent path is truncated at the point of
                     convergence and treated like any other path. The path is
                     then walked from the opposite end, if the terminal edge at
                     that end has not yet been partitioned out, and truncated a
                     second time where convergence occurs on that walk.
                     Therefore, up to two truncated parts are partitioned out as
                     paths for each full path that is considered.
            False    Upon encountering the first convergent path on a given
                     pass, that pass is terminated.
            None     Exactly one path is partitioned out per pass. Unless
                     test_func is specified, this is guaranteed to be the
                     highest-cost of all optimized paths.
        The options are listed in order of decreasing opportunism. The less
        opportunistic that processing is, the more likely that the path(s)
        partitioned out will have a consistent logic, potentially enhancing the
        consistency of any later pruning. However, because each call of the
        current method incurs significant overhead, lower opportunism usually
        implies lower computational efficiency. See note further below for
        method-specific consequences.

        disjoint is a boolean that specifies whether each successive trunk must
        be disjoint from all other trunks. If disjoint is True, opportunistic
        must not be True or an error is raised.

        Note: For the current method, opportunistic and target have the
        following practical consequences, where "exhaustive" indicates that no
        more paths of the indicated kind could be partitioned out on any future
        call:
            opportunistic    target      Behavior and Results
            True             0           A single pass is executed, in which at
                                         least one trunk is partitioned out and
                                         both trunks and branches are exhausted.
                             >0, None    The same behavior and results as
                                         target=0, except that a second pass is
                                         begun then cancelled when it is
                                         recognized that trunks and branches are
                                         exhausted.
            False            0           A single pass is executed, in which at
                                         least one (and typically exactly one)
                                         trunk is partitioned out.
                             >0, None    Multiple passes are exeucted until
                                         target is satisfied or tunks are
                                         exhausted.
            None*            0*          A single pass is executed, in which
                                         exactly one trunk is partitioned out.
                             >0, None    Multiple passes are exeucted until
                                         target is satisfied or tunks are
                                         exhausted.
            *Default for the current method.
        Note that even if the qualitative description is the same for two
        different combinations of opportunistic and target values, the results
        may not be identical.

        See also:
            .add_loops_to_partitioned()
            .partition_out_branches()
            .partition_out_bridges()
        """
        # Process disjoint argument.
        if disjoint and (self._kind_to_partitioned_path_dict["trunk"]
                         or target is None
                         or target > 1):
            if opportunistic:
                raise TypeError(
                    "if disjoint is True, opportunistic must not be True"
                    )
            disjoint_kind = "trunk"
        else:
            disjoint_kind = None
        del disjoint  # Exclude disjoint from **locals() further below.

        # Honor "SAFE" isolation option.
        # Note: Each successive call of ._find_longest_shortest_paths()
        # has a smaller memory footprint, if any partitioning-out
        # occurs. Therefore, only apply this test for partitioning out
        # the trunk.
        self._apply_brute_and_safe_options(safe=True)

        # Partition out paths.
        # Note: An opportunistic partitioning of trunks should be
        # exhaustive with only one pass, but for sake of debugging,
        # allow target to be nonzero.
        # Note: Because ._register_line() recognizes trunks when kind is
        # "branch" but not branches when kind is "trunk", set kind to
        # "branch" if opportunistic is True.
        result = self._partition_out(
            kind=("branch" if opportunistic else "trunk"),
            node_kind1="next_trunk", node_kind2="next_trunk",
            **locals()
            )

        # If no trunk could be identified, and none yet exists, raise an
        # error.
        if not result[0] and not self._kind_to_partitioned_path_dict["trunk"]:
            raise TypeError("no trunk could be identified")

        # Return result.
        return result

    def partition_out_branches(self, target=None, cost_func=None,
                               test_func=None, from_nodes=None, to_nodes=None,
                               opportunistic=None):
        """
        Partition out branch paths from the graph skeleton.

        Each path partitioned out by the current method must have one end at a
        node that is both a stub in the graph skeleton and already present in
        the partitioned skeleton. See .partition_out_trunk() for a description
        of general functionality.

        All arguments have the same meaning as in .partition_out_trunk().

        Note: For the current method, opportunistic and target have the
        following practical consequences, where "exhaustive" indicates that no
        more paths of the indicated kind could be partitioned out on any future
        call:
            opportunistic    target      Behavior and Results
            True             0           A single pass is executed, in which
                                         branches are exhausted.
                             >0, None    The same behavior and results as
                                         target=0, except that a second pass is
                                         begun then cancelled when it is
                                         recognized that branches are exhausted.
            False            0           A single pass is executed, in which at
                                         least one branch (if any are possible)
                                         is partitioned out.
                             >0, None    Multiple passes are exeucted until
                                         target is satisfied or branches are
                                         exhausted.
            None*            0*          A single pass is executed, in which
                                         exactly one branch is partitioned out.
                             >0, None    Same as for opportunistic=False.
            *Default for the current method.
        Note that even if the qualitative description is the same for two
        different combinations of opportunistic and target values, the results
        may not be identical.

        See also:
            .add_loops_to_partitioned()
            .partition_out_bridges()
            .partition_out_trunk()
        """
        # Partition out.
        # Note: An opportunistic partitioning of trunks should be
        # exhaustive with only one pass, but for sake of debugging,
        # allow target to be nonzero.
        return self._partition_out(kind="branch",
                                   node_kind1="next_branch_stem",
                                   node_kind2="next_branch_stub",
                                   **locals())

    def partition_out_bridges(self, cost_func=None, opportunistic=True,
                              test_func=None, from_nodes=None, to_nodes=None,
                              target=None):
        """
        Partition out bridge paths from the graph skeleton.

        Each path partitioned out by the current method must have both ends at
        nodes already present in the partitioned skeleton. See
        .partition_out_trunk() for a description of general functionality.

        All arguments have the same meaning as in .partition_out_trunk().

        Note: For the current method, opportunistic and target have the
        following practical consequences, where "exhaustive" indicates that no
        more paths of the indicated kind could be partitioned out on any future
        call:
            opportunistic    target      Behavior and Results
            True             0           A single pass is executed, in which at
                                         least one bridge (but typically
                                         multiple bridges, if possible) is
                                         partitioned out.
                             >0, None    Multiple passes are executed until
                                         target is satisfied or bridges are
                                         exhausted for the current configuration
                                         of the partitioned skeleton only.
                                         (Partitioning out of other kinds might
                                         "un-exhaust" bridges.)
            False            0           A single pass is executed, in which at
                                         least one bridge (but typically
                                         multiple bridges, if possible) is
                                         partitioned out.
                             >0, None    Same as for opportunistic=True.
            None*            0*          A single pass is executed, in which
                                         exactly one bridge is partitioned out.
                             >0, None    Same as for opportunistic=True.
            *Default for the current method.
        Note that even if the qualitative description is the same for two
        different combinations of opportunistic and target values, the results
        may not be identical.

        See also:
            .add_loops_to_partitioned()
            .partition_out_branches()
            .partition_out_trunk()
        """
        # Partition out.
        return self._partition_out(kind="bridge",
                                   node_kind1="next_bridge",
                                   node_kind2="next_bridge",
                                   **locals())

    def prune(self, cutoff=None, get_cost=None, target=None, priority_func=None,
              gobble=True, kind="branch"):
        """
        Prune (remove and return) paths from the partitioned skeleton.

        Unless a filter is applied at some point, you'll likely end up with more
        paths in the partitioned skeleton than you want. You can apply a filter
        at any of the following points:
            1) After initialization, by deleting edges from the graph skeleton
               using .delete_lines().
            2) After any partitioning out, by deleting paths from the
               partitioned skeleton using the current function or (less
               preferred) .delete_lines().
            3) After writing out the partitioned skeleton, manually.
        The most common unwanted paths are branches that stem from a
        "legitimate" (wanted) trunk, branch, or bridge and head directly to the
        nearest boundary, effectively spanning half the local width. For
        brevity, we refer to these as "spurious branches." One of the major
        innovations of the current algorithm is the identification of the
        "normalized length" as a useful metric for estimating the relative
        importance of a branch. The normalized length simply scales a branch's
        length by half the approximate local width at its stem:
            2. * branch.length / branch.stem_width
        For spurious branches, the normalized length is typically very near 1.
        For branches of progressively greater importance, the normalized length
        is typically progressively larger, but note that normalized length is
        more reliable at finer sampling intervals. A major advantage of
        normalized length over length itself is its scaling to local geometry.
        For example, if the input polygon is the outline of a river, a 100 m
        tributary joining a stream that is locally 10 m wide would have the same
        normalized length as a 1 km tributary joining a stream that is locally
        100 m wide, even though its length is only one-tenth as great. The
        current function returns a list of lines that were pruned (but see
        gobble).

        cutoff is a float that specifies the critical cost (e.g., normalized
        length) for pruning. If a path has a cost (as calculated by get_cost())
        equal to or greater than cutoff, it is guaranteed not be pruned.
        (However, paths may be gobbled regardless of their cost (see gobble) and
        paths with costs smaller than cutoff may not be pruned if target is
        specified (not None).) For help choosing a normalied length cutoff, see
        note further below.

        get_cost is a function that is called to assign a cost to each path. See
        documentation for .make_cost_func(). If get_cost is not specified
        (None), its default depends on kind:
            kind           Default get_cost
            "branch"       EasySkeleton.get_normalized_length
            [otherwise]    EasySkeleton.length2D_cost_func.get_cost

        target is an integer that specifies a limit for the number of paths to
        be pruned. It is interpreted as follows:
            target < 0:  the negative of the maximum number of paths to be
                         pruned (i.e., the most extreme allowed change in path
                         count)
            target > 0:  the maximum number of paths of the specified kind that
                         should remain after pruning
            target == 0: disallowed (raises TypeError)
        If target is unspecified (None), no limit is applied.

        priority_func is a function that determines the order in which paths are
        pruned. It is called with a path (LineString) an its only argument and
        must return a sortable result. Therefore, any function that can be
        specified for path_list.sort()'s key argument, where path_list is a list
        of paths (LineString's), is suitable for priority_func. Paths for which
        the smallest values are returned by priority_func() will be pruned
        first. If target is not specified (None), priority_func is irrelevant
        and therefore ignored. If target is specified but priority_func is not
        specified (None), the default for priority_func depends on kind:
            kind           Default priority_func
            "branch"       EasySkeleton.get_normalized_length
            [otherwise]    EasySkeleton.length2D_cost_func.get_cost

        gobble is a boolean that specifies whether
            .gobble_paths(stub_indices, True, True)
        should be called immediately after pruning, where stub_indices is
        generated by calling
            .get_nodes("stub", False, True, coords=False)[0]
        prior to pruning. Therefore, any nodes that are newly stubs due to
        pruning will be gobbled. This prevents some otherwise commonly
        encountered but usually undesired consequences of pruning, such as
        branches that are no longer connected to the rest of the partitioned
        skeleton at either end and bridges that become disconnected from the
        partitioned skeleton at one or both ends. If gobble is True, a tuple is
        returned of the form
            (pruned_paths, gobbled_paths)
        where each item is a list of paths (LineString's) that were directly
        pruned or later gobbled, respectively. Note that gobbled_paths may
        include paths whose kind is not the specified kind.

        kind is a string that specifies the kind of partitioned-out path to be
        pruned. In the special case that kind is "trunk" and the call would
        prune all trunks, an error is raised and no pruning is executed.

        Warning: cutoff can be specified as None (the default), but in that
        case, every path of the specified kind is considered a candidate for
        pruning! This may be useful if target is also specified, but otherwise,
        cutoff=None should be used with care.

        Note: If cost_func is EasySkeleton.get_normalized_length, choosing a
        cutoff that suits your application can be a bit tricky. Commonly, a
        value between 1.1 and 5 is useful, but even if your ideal value lies
        within that interval (which is not guaranteed), there remains a lot of
        variability in behavior across that range. Here are a few strategies
        that you might consider:
            1) Write out the partitioned skeleton without pruning, being sure to
               include the normalized length of branches in the output. Then
               inspect the output in your preferred GIS program and manually
               determine what cutoff works well for you. You can then either
               manually prune the output or run the same code again with that
               cutoff specified.
            2) Output multiple partitioned skeletons at successively higher
               cutoff's and then retain the output that works best. It's
               important to note that you need not create a new Skeleton for
               each of these outputs, so the processing time should be small as
               it only represents the pruning itself. However, for large
               skeletons, the write-out times may still be significant.
            3) Reason what the cutoff should be for your application, given how
               the normalized length is calculated. For example, do you want to
               retain branches that are 10% longer than half the local width of
               the input polygon? If so, set cutoff to 1.1. If those branches
               sounds unimportant to you, do you instead want to retain branches
               that are twice as long as half the local width? Then set cutoff
               to 2.0. However, bear in mind that the normalized length only
               approximates the local width (see documentation for
               SkeletalLineString2D.segment_widths) and is calculated with
               better precision at finer sampling intervals. For these reasons,
               a cutoff slightly less than your reasoned ideal value may be
               preferable.
        """
        # Default cost_func, if necessary.
        if get_cost is None:
            if kind == "branch":
                # *REASSIGNMENT*
                get_cost = EasySkeleton.get_normalized_length
            else:
                # *REASSIGNMENT*
                get_cost = EasySkeleton.length2D_cost_func.get_cost

        # If target is specified, standardize a proxy to the number of
        # paths to be pruned, and assign a default priority function if
        # one was not specified.
        path_dict = self._kind_to_partitioned_path_dict[kind]
        if target is not None:
            if target == 0:
                raise TypeError("target cannot be 0")
            if target < 0:
                pos_target = -target
            else:
                pos_target = len(path_dict) - target
                if pos_target <= 0:
                    return 0
            if priority_func is None:
                if kind == "branch":
                    # *REASSIGNMENT*
                    priority_func = EasySkeleton.get_normalized_length
                else:
                    # *REASSIGNMENT*
                    priority_func = EasySkeleton.length2D_cost_func.get_cost

        # Identify those paths that are candidates for pruning.
        if cutoff is None:
            prunable_paths = path_dict.values()
        else:
            prunable_paths = [path for path in path_dict.itervalues()
                              if get_cost(path) < cutoff]

        # Subset candidates if the number identified exceeds the
        # positively-valued target proxy.
        if target is not None and len(prunable_paths) > pos_target:
            prunable_paths.sort(key=priority_func)
            # *REASSIGNMENT*
            prunable_paths = prunable_paths[:pos_target]

        # Optionally prepare for later gobbling.
        if gobble:
            stub_indices = self.get_nodes("stub", False, True, coords=False)[0]

        # Delete paths.
        self._delete_keys({prunable_path._aligned_key
                           for prunable_path in prunable_paths})

        # Optionally gobble and return.
        if gobble:
            return (prunable_paths, self.gobble_paths(stub_indices, True, True))
        return prunable_paths

    def sweep(self, branch_test_func=None, bridge_test_func=None,
              orient_func=None, delete_others=True):
        """
        Sweep isolated edges from the graph skeleton.

        After any partitioning-out, the graph skeleton may have "isolated" edges
        (see .describe_line()). For each such isolated edge, the only path that
        can include it is the path that is identical to the edge itself.
        Therefore, the path-finding that underlies .partition_out*()'s (though
        effective) is needlessly inefficient. The current function identifies
        all isolated edges and (by default) partitions them out from the graph
        skeleton if they pass the corresponding test, automatically categorizing
        them as branches or bridges. This can substantially improve performance
        without any deviation in the final results. A tuple of the form
            (swept_count, deleted_edges)
        is returned where swept_count is an integer that gives the number of
        edges swept to the partitioned skeleton and deleted_edges is a list of
        any edges that were deleted.

        branch_test_func is a function that specifies a test function to be
        applied only to those isolated edges that would be partitioned out as
        branches.

        bridge_test_func is a function that specifies a test function to be
        applied only to those isolated edges that would be partitioned out as
        bridgees.

        orient_func is a function that is passed an edge (LineString) as its
        first and only argument and must return either 1) that edge oriented in
        the desired direction or 2) None, to indicate that the edge should not
        be partitioned out. For consistency, the edge should be so oriented as
        to minimize its traversal cost from start to end, as this would have
        occurred if it had instead been partitioned out by a .partition_out*().
        (See the orient_func attribute of the function returned by
        .make_cost_func().) If orient_func is not specified (None), edges are
        partitioned out with arbitrary directionality.

        delete_others is a boolean that specifies whether isolated edges that
        are neither branch- nor bridge-like should be deleted. Such edges are
        "irreparably isolated", that is, they are isolated in the graph skeleton
        and would also be isolated in the partitioned skeleton if they were
        partitioned out. Note that irreparably isolated edges only exist if
        .delete_lines() was called.

        Note: branch_test_func and bridge_test_func serve the same role as
        test_func in .partition_out_branches/bridges() except that 1) only a
        Type 3 call is executed (see .make_test_func() and the
        .make_deep_test_func() of the function returned by that method) and 2)
        the abort value returned by *_test_func() is ignored, because
        isolated  edges are iterated over in arbitrary order. If the relevant
        *test_func is not specified (None), the edge is considered acceptable.
        """
        # Identify the Voronoi vertex indices of current stubs in the
        # graph skeleton.
        stub_indices_array, = (
            self._vor_vert_idx_to_graph_degree_array == 1
            ).nonzero()
        stub_indices_set = set(stub_indices_array.tolist())
        del stub_indices_array  # Release memory.

        # Iterate over each edge's key in the graph skeleton.
        graph_edge_dict = self._graph_edge_dict
        vor_vert_idx_to_partitioned_degree_array = self._vor_vert_idx_to_partitioned_degree_array
        lines_were_manually_deleted = self._lines_were_manually_deleted
        accept = True  # Initialize with default.
        edge_is_irreparably_isolated = False  # Initialize with default.
        deleted_edges = []
        orig_graph_edge_count = len(graph_edge_dict)
        for key in graph_edge_dict.keys():

            # If key indicates that the edge is isolated, retrieve the
            # edge.
            if key <= stub_indices_set:
                edge = graph_edge_dict[key]
                idx0, idxN = key.tuple

                # Determine the edge's potential (partitioned-out) kind
                # and the corresponding test function.
                if vor_vert_idx_to_partitioned_degree_array[idx0]:
                    if vor_vert_idx_to_partitioned_degree_array[idxN]:
                        kind = "bridge"
                        test_func = bridge_test_func
                    else:
                        kind = "branch"
                        test_func = branch_test_func
                # Note: In the special case that lines were manually
                # deleted and the edge would also be isolated in the
                # partitioned skeleton, discard it or ignore it, as
                # directed.
                elif (lines_were_manually_deleted and not
                      vor_vert_idx_to_partitioned_degree_array[idxN]):
                    test_func = None
                    if delete_others:
                        edge_is_irreparably_isolated = True
                    else:
                        accept = False
                else:
                    # Note: Because no lines were manually deleted,
                    # assume that edge would not be isolated in the
                    # partitioned skeleton.
                    kind = "branch"
                    test_func = branch_test_func

                # Apply test, if required.
                if test_func is not None:
                    accept, _ = test_func(edge, True, kind)

                # If edge failed its test or is irreparably isolated,
                # delete or skip it, as appropriate.
                if not accept or edge_is_irreparably_isolated:
                    if accept is None or edge_is_irreparably_isolated:
                        self._register_line(edge, False, key.tuple, idx0, idxN)
                        deleted_edges.append(edge)
                    # Note: Re-initialize default values.
                    if edge_is_irreparably_isolated:
                        edge_is_irreparably_isolated = False
                    else:
                        accept = True
                    continue

                # Optionally orient edge and partition it out.
                if orient_func is not None:
                    orient_func(edge)  # *REASSIGNMENT*
                    idx0 = idxN = None  # Note: Edge might be flipped.
                self._register_line(edge, kind, key.tuple, idx0, idxN)

        # Return deleted edges.
        return (
            orig_graph_edge_count - len(graph_edge_dict) - len(deleted_edges),
            deleted_edges
            )

    def orient_downslope(self, kind="partitioned"):
        for line in self.get_lines(kind, False, 0):
            if line.delta_z > 0.:
                line.flip(False)
                continue


class EasySkeleton(Skeleton):

    # Commonly used cost-related functions.
    get_normalized_length = _operator.attrgetter("normalized_length")
    length2D_cost_func = staticmethod(
        Skeleton.make_cost_func(".length", avoid_nan=None, infinite=False)
        )
    length2D_cost_func.__func__.is_length_equivalent = True
    length3D_cost_func = staticmethod(
        Skeleton.make_cost_func(".length3D", avoid_nan=None, infinite=False)
        )
    length3D_cost_func.__func__.is_length_equivalent = False

    # Defaults related to the template interval.
    _template_interval_applied = False
    template_interval = None
    template_normalized_length_cutoff = 0

    def __init__(self, polygon, interval, min_normalized_length,
                 max_branch_count=None, max_trunk_count=1, mode="PREFER_BRIDGE",
                 trunk_mode="DEFAULT", isolation_mode="SAFE", tails=True,
                 shortest_branch=0., shortest_bridge=0., shortest_loop=0.,
                 cost_func=None, template_interval=None, memory_option=0,
                 targ_GB=None, **kwargs):
        """
        Initiate skeleton derivation for a polygon. (Compare Skeleton.)

        In simplest terms, the skeleton of a polygon collapses an elongate
        polygon to a linear form. For example, the outline of a river is a
        polygon (with a nonzero area). However, we often think of a river in
        terms of its skeleton, such as when a river is drawn as a line on a map.
        Even though that line does not represent the river's width, it still
        captures a useful expression of the river's geometry. In the algorithm
        used by the current type, the skeleton is composed of paths of three
        different kinds. Approximately, these are the main path, called the
        "trunk"; paths that reach from any other path to the boundary of
        polygon, called "branches"; and paths that link from one path to
        another, called "bridges". You are strongly encouraged to take a look at
        the documentation for Skeleton.__init__() for a somewhat fuller
        description than is included here, as well as read the following paper
        for an even more detailed, albeit technical, description of this
        algorithm, including some very helpful figures:
            [placeholder]

        polygon is a Polygon for which skeleton derivation will be executed.

        interval is a float that specifies a sampling interval, which is
        analogous to the desired resolution of the skeleton. Ideally, interval
        should be no coarser than half the width of polygon's narrowest
        constriction (see "CUT" isolation_mode option), but finer intervals
        require larger memory footprints. See also template_interval for
        supported special values.

        min_normalized_length is a float that specifies the minimum normalized
        length permitted for a branch. Any branch with a lower value will be
        pruned by .prune(min_normalized_length). The normalized length is
        intended to approximately score the importance of a branch and starts at
        ~1. Practically, it approximates the ratio between the length of a
        branch and half the local width of polygon at the branch's stem. Specify
        0 to disable. See .prune() for additional guidance.

        max_branch_count is an integer that specifies the maximum number of
        branches that can be retained. If more branches are initially
        identified, they are subset by .prune(target=max_branch_count).

        max_trunk_count is an integer that specifies the maximum number of
        trunks that can be identified. In many cases, max_trunk_count is best
        set to 1 (the default). However, consider the two following scenarios:
            Scenario 1: polygon represents a divided highway with occasional
                        links betwen these two parallel roads.
            Scenario 2: polygon represents a braided river whose shape
                        approximates the shape of the polygon in Scenario 1.
        Even though polygon is similar in each scenario, you probably want to
        define two trunks in Scenario 1 and one trunk in Scenario 2.

        mode is a string that specifies how the skeleton is subdivided (and,
        more importantly, "partitioned out"). Depending on which two modes you
        are comparing and the geometry of polygon, the skeletons produced by two
        different modes may differ not at all, by very little, or dramatically.
        In addition, the range of speeds implied by the range of modes can be
        extremely large. The supported modes, with descriptions of typical (but
        not necessarily guaranteed) behavior, are:
            "PREFER_BRIDGE" *
                Paths within the skeleton will be preferentially categorized as
                bridges, resulting in a longer total length for that category.
                Moderately fast.
            "MAX_BRIDGE"
                Paths within the skeleton will be preferentially categorized as
                bridges, resulting in a longer total length for that category,
                and the lengths of the longest bridges will be maximized.
                Slow.
            "MAX_BRIDGE_MAX_BRANCH"
                Paths within the skeleton will be preferentially categorized as
                bridges, resulting in a longer total length for that category,
                and the lengths of the longest bridges, then the longest
                branches, will be approximately maximized. Slow to very slow.
            "PREFER_BRANCH"
                Paths within the skeleton will be preferentially categorized as
                branches, resulting in a longer total length for that category.
                Fast.
            "MAX_BRANCH"
                Paths within the skeleton will be very preferentially
                categorized as branches, resulting in a longer total length for
                that category, and the lengths of the longest branches will be
                maximized. Slow.
            "MAX_BRANCH_MAX_BRIDGE"
                Paths within the skeleton will be preferentially categorized as
                branches, resulting in a longer total length for that category,
                and the lengths of the longest branches, then the longest
                bridges, will be approximately maximized. Slow to very slow.
        where all lengths are measured in the x-y plane.
        *This mode is the default and will typically yield categorizations that
        most closely align with intuition. Consequently, branches will be
        defined in a way that may feel more logical, making normalized length a
        more useful criterion.

        trunk_mode is a string that specifies how the trunk is identified. For
        each supported option, the trunk(s) are defined as the first and (if
        max_trunk_count > 1) each subsequent possible, non-overlapping,
        length-optimized path that...
            "DEFAULT"
                ...has the longest 2D length.
            "EUCLIDEAN" [1]
                ...has the greatest straight-line end-to-end span in 2D.
            "EUCLIDEAN3D" [1] [3]
                ...has the greatest straight-line end-to-end span in 3D.
            "MIN_EUCLIDEAN_#" [2]
                ...has at least the specified straight-line end-to-end span in
                2D in map units (e.g., "MIN_EUCLIDEAN_123.45").
            "MIN_EUCLIDEAN3D_#" [2]
                ...has at least the specified straight-line end-to-end span in
                3D in map units (e.g., "MIN_EUCLIDEAN3D_123.45").
            "NEAR_#_#" [1]
                ...has the longest 2D length of those paths with one end as near
                to the specified coordinates as possible (e.g.,
                "NEAR_123.45_678.90").
            "NEAR_#_#_AND_#_#" [1]
                ...has each end as near to the specified coordinates as possible
                (e.g., "NEAR_123.45_678.90_AND_678.90_123.45").
            "HIGH_LOW" [1] [3]
                ...has each end at either the z-coordinate minimum or maximum
                (among all stub nodes).
            "MIN_RELIEF_#"[2] [3]
                ...has at least the specified end-to-end change in z-coordinate,
                where the change is unsigned (e.g., "MIN_RELIEF_123.45").
            "MIN_RELIEF_#%" [2] [3]
                ...has at least the specified end-to-end change in z-coordinate,
                where that change is specified as a percentage of the maximum z-
                coordinate difference between any two vertices of the input
                polygon (e.g., "MIN_RELIEF_90.5%").
            "MIN_MEAN_WIDTH_#" [2]
                ...has at least the specified mean width, approximated by
                path.area / path.length2D.
        [1] If max_trunk_count is greater than one, an error is raised.
        [2] This option is at least partially implemented as a brute-force
            search in which paths are considered in the same order as "DEFAULT".
            Therefore, it can be very slow.
        [3] polygon must be 3D. Additionally, this option can be very slow due
            to the required interpolation of z-coordinates.

        isolation_mode is a string that specifies several aspects of how the
        graph skeleton is isolated from its complement and general memory
        safeguards. It is therefore considered advanced functionality and should
        not be changed from its default value without understanding the
        consequences. It is documented with other arguments to
        Skeleton.__init__().

        tails is a boolean that specifies whether a segment ("tail") should be
        added to each trunk end and each outward-facing branch end to extend
        them to the boundary of polygon. More precisely, if tails is True, tails
        are added prior to implementation of the min_normalized_length and
        max_branch_count arguments and generally helps to align the behavior of
        these arguments to expectations, at some cost to processing time.
        Alternatively, tails can be added after skeleton generation by
        .add_tails().

        shortest_branch is a float that specifies the shortest 2D length
        permitted for a branch.

        shortest_bridge is a float that specifies the shortest 2D length
        permitted for a bridge.

        shortest_loop is a float that specifies the shortest 2D length permitted
        for an uninterrupted terminal loop (i.e., a loop at either end of a
        trunk or at the outward-facing end of a branch).

        cost_func is a function that specifies how the cost (usually length) of
        each candidate trunk, branch, or bridge is calculated, and therefore how
        it is routed (because routing always minimizes cost) and the order in
        which each candidate is considered (because paths are considered in
        order of decreasing cost). If cost_func is not specified (None), each
        path's 2D length is used, or equivalently, cost_func defaults to
        .length2D_cost_func. Alternatively, cost_func can be specified as
        .length3D_cost_func to use each path's 3D length, but this may be much
        slower.

        template_interval is a float that specifies the interval used to
        generate the internally-used template skeleton. Because interval is
        analogous to resolution, a finer interval is generally desirable but has
        an exponentially greater memory footprint. To *simulate* accommodation
        of a much finer interval than could otherwise be accommodated within
        available memory, a template skeleton can be generated first, at the
        coarser template_interval. Raw data can then be generated at the (finer)
        specified interval but subset based on the geometry of the template
        skeleton. The result is a skeleton that has local positional and
        directional accuracies identical to those of a skeleton fully processed
        at the specified interval but no more (topologically) complete than the
        template skeleton. The exact effects of this simulation depend on
        details of polygon's geometry but could result, for example, in the
        selection of less-than-ideal branches or even the loss of (usually less
        important) portions of the skeleton. Some special values are supported
        for both interval and template_interval:
        interval    template_interval    1st Attempt    2nd Attempt*
        >0          None                 [i1, t0]
        >0          >0                   [i1, t0]       [i1, t1]
        >0          0                    [i1, t0]       [i1, t2]
        >0          -0.                  [i1, t2]**
        >0          <0                   [i1, t1]**
        0           None                 [i2, t0]
        0           0                    [i3, t2]
        0           >0                   [i3, t1]
        0           -0.                  [i3, t2]**
        0           <0                   [i3, t1]**
         * Occurs if and when memory becomes exhausted, or if and when the first
           attempt is aborted because such exhaustion is predicted.
        ** If template_interval is negative, all internal memory safeguards are
           disabled, but the "SAFE" isolation_mode option is still honored if
           specified. If the algorithm completes successfully, the skeleton is
           typically very nearly as topologically complete as the template
           skeleton. Otherwise, for a non-negative template_interval, stubs are
           first subset to match those in the template skeleton (as for a
           negative template_interval) but may then be further subset if deemed
           necessary to ensure that all processing can be accommodated in
           available/specified memory. That second subsetting, if required, is
           based on the normalized length and therefore is intended to
           prioritize deletion of the least important stubs (and therefore, loss
           of the least important portions of the skeleton).
        [i1] Final skeleton is based on sampling at the specified interval.
        [i2] Final skeleton is based on sampling at
             self.estimate_safe_interval(..., include_partitioning=True)[0].
        [i3] Final skeleton is based on sampling at 1.1 *
             self.estimate_safe_interval(..., include_partitioning=False).
        [t0] No template skeleton is generated.
        [t1] A (sampling) interval of abs(template_interval) is used to generate
             a template skeleton.
        [t2] A (sampling) interval of self.estimate_safe_interval(...,
             include_partitioning=True)[0] is used to generate a template
             skeleton.
        If any special implicit (zero) values are used for interval or
        template_interval, you are strongly encouraged to read the corresponding
        note in the documentation of Skeleton.__init__().

        memory_option is an integer that specifies how the memory footprint is
        managed by the algorithm. The available options are:
            0   No special attempt is made to reduce the memory footprint.
            1   Some special attempts are made to reduce the memory footprint,
                but both the memory footprint and performance are typically only
                mildly impacted.
        More options may be added in the future.

        targ_GB is a float that specifies the target memory footprint, in
        gigabytes. targ_GB is unused unless the "SAFE" isolation_mode option or
        an implicit (zero) interval is specified. In those cases, if targ_GB is
        None, it is reset to 80% of the available physical memory as reported by
        the system, if possible, or an error is raised.

        Any additional keyword arguments are passed to Voronoi2D.__init__() when
        it is called internally.

        Note: If you are unable to generate a skeleton at the desired interval
        because processing is too slow and/or you run out of memory, consider
        any combination of the following options:
            1) smooth polygon's boundary
            2) remove unimportant (small?) holes in polygon
            3) specify template_interval
            4) specify memory_option > 0
            5) review documentation on the relative performance of your chosen
               mode and trunk_mode options
        """
        ## Double-check the descriptions of trunk_modes, including their
        ## relative speed.
        # Locally store calling arguments.
        calling_args = locals().copy()
        del calling_args["self"]

        # Validate mode.
        # *REASSIGNMENT*
        mode = _validate_string_option(
            mode, "mode", ("PREFER_BRANCH", "MAX_BRANCH",
                           "MAX_BRANCH_MAX_BRIDGE", "PREFER_BRIDGE",
                           "MAX_BRIDGE", "MAX_BRIDGE_MAX_BRANCH", "FAST")
            )
        mode_is_maxed = mode.startswith("MAX_")
        mode_is_maxed2 = mode.count("MAX_") > 1

        # Validate trunk mode.
        trunk_mode_form = _validate_string_option(
            trunk_mode, "trunk_mode",
            ("DEFAULT",
             "EUCLIDEAN", "EUCLIDEAN3D",
             "MIN_EUCLIDEAN_#", "MIN_EUCLIDEAN3D_#",
             "NEAR_#_#", "NEAR_#_#_AND_#_#",
             "HIGH_LOW",
             "MIN_RELIEF_#", "MIN_RELIEF_#%",
             "MIN_MEAN_WIDTH_#"),
            True
            )
        if (trunk_mode_form in ("EUCLIDEAN3D", "MIN_EUCLIDEAN3D"
                                "HIGH_LOW",
                                "MIN_RELIEF_#%", "MIN_RELIEF_#") and not
            polygon.is_3D):
            raise TypeError(
                "trunk_mode option requires input polygon to be 3D: {!r}".format(
                    trunk_mode
                    )
                )
        if (max_trunk_count > 1 and
            trunk_mode_form in ("EUCLIDEAN", "EUCLIDEAN3D",
                                "NEAR_#_#", "NEAR_#_#_AND_#_#", "HIGH_LOW")):
            raise TypeError(
                "trunk_mode option requires max_trunk_count to be 1: {!r}".format(
                    trunk_mode
                    )
                )

        # Default to using 2D length if no cost function is specified.
        if cost_func is None:
            cost_func = self.length2D_cost_func

        # Prepare for possible simulation of a sampling interval, if
        # appropriate.
        reset_attr_dict = self.__dict__.copy()
        may_simulate_interval = (template_interval is not None and
                                 not self._template_interval_applied)
        if may_simulate_interval:
            orig_isolation_mode = isolation_mode
            # Note: math.copysign() is used in case
            # template_interval is -0.
            template_interval_is_positive = _math.copysign(
                1., template_interval
                ) > 0.
            interval_is_implicit = interval == 0.
            # Unless template interval is negative or interval is
            # implicit, ensure that processing is memory safe.
            isolation_options_set = set(isolation_mode.upper().split("_"))
            isolation_options_set.discard("SAFE")
            unsafe_isolation_mode = "_".join(isolation_options_set)
            if template_interval_is_positive and not interval_is_implicit:
                isolation_options_set.add("SAFE")
                # *REASSIGNMENT*
                isolation_mode = "_".join(isolation_options_set)
            else:
                isolation_mode = orig_isolation_mode

        # Proceed as implied by interval-template_interval combination.
        try:
            if may_simulate_interval and (interval_is_implicit or
                                          not template_interval_is_positive):
                raise MemoryError
            if not hasattr(self, "polygon"):
                Skeleton.__init__(self, polygon, interval, isolation_mode,
                                  memory_option, targ_GB)
            self._generate_skeleton(**locals())
        except MemoryError:
            # Error if use of a template interval is not permitted or if
            # it is known that the non-partitioning memory footprint of
            # the (finer) simulated interval would itself exceed the
            # target memory.
            # Note: If "SAFE" is not specified as an isolation_mode
            # option and template_interval is negative, it is not tested
            # internally whether the simulated interval is safe.
            if (not may_simulate_interval or
                not getattr(self, "interval_is_nonpartitioning_safe", True)):
                raise
            self._apply_template(**locals())

    @staticmethod  # Note: self is passed when called.
    def _apply_template(self, orig_isolation_mode, reset_attr_dict,
                        calling_args, template_interval, tails, polygon,
                        interval, memory_option, targ_GB,
                        template_interval_is_positive, min_normalized_length,
                        unsafe_isolation_mode, **kwargs):

        # Clear any attributes assigned since .__init__() was first
        # called (e.g., in a skeleton initialization attempt).
        self.__dict__.clear()
        self.__dict__.update(reset_attr_dict)

        # Derive partitioned skeleton at either an explicit or brute-
        # search template interval.
        template_calling_args = calling_args.copy()        
        template_calling_args["interval"] = abs(template_interval)
        template_calling_args["template_interval"] = None
        template_calling_args["isolation_mode"] = orig_isolation_mode
        # Note: Because of the crudity of the template skeleton, do not
        # attempt to apply any constraints.
        ## Could add support so that negative values force the 
        ## application of these constraints to the template skeleton,
        ## especially for performance reasons.
        template_calling_args["min_normalized_length"] = 0.
        template_calling_args["max_branch_count"] = None
        template_calling_args["shortest_branch"] = 0.
        # Note: Reuse the memory footprint determined internally on the
        # first skeleton initialization attempt, if there were such an
        # attempt.
        if self.target_memory_footprint is not None:
            template_calling_args["targ_GB"] = self.target_memory_footprint / 2.**30.
        self.__init__(**template_calling_args)
        targ_GB = self.target_memory_footprint / 2.**30.

        # Add tails, if they were not already added, so that stubs
        # identified further below are more accurately identified.
        # Note: Tails are added to partially compensate for the coarser
        # template interval.
        if not tails:
            self.add_tails()

        # Preserve relevant results from template skeleton.
        if template_interval == 0.:
            reset_attr_dict["template_interval"] = self.interval
        else:
            reset_attr_dict["template_interval"] = template_interval
        reset_attr_dict["_cut_vor_vert_idxs_array"] = self._cut_vor_vert_idxs_array

        # Use partitioned skeleton at template interval to identify
        # the tail (or stub) coordinates that may be most important.
        # Note: The variable x is used to ensure that all tuples can
        # be sorted further below (even in the case of identical
        # normalized lengths) without relying on numpy arrays, which
        # cannot be sorted.
        coarse_stub_data = [
            (branch.normalized_length, x, branch.stub_coords_array)
            for x, branch in
            enumerate(self.get_lines("branch", hardness_level=0))
            ]
        trunks = self.get_lines("trunk", hardness_level=0)
        # Note: If trunks were forced to be disjoint (the default), the
        # first found and therefore primary trunk has the shortest
        # length, and each successive trunk is longer.
        trunks.sort(key=lambda line: line.length, reverse=True)
        for x, trunk in _izip(_itertools.count(-1, -2), trunks):
            coarse_stub_data.extend(
                ((_python_inf, x, trunk.coords_array[0]),
                 (_python_inf, x - 1, trunk.coords_array[-1]))
                )
        del branch, trunks, trunk  # Reduce local namespace.

        # Reset and re-initialize self, generating a graph skeleton at
        # the user-specified (fine) interval.
        # Note: Even if these results were generated earlier, they are
        # regenerated, rather than remembered and restored, because they
        # can be regenerated reasonably quickly compared to a write-read
        # cycle on disk, and storing to memory would coarsen the minimum
        # template interval that could be safely processed in full, thus
        # impairing the quality of the final results.
        self.__dict__.clear()
        self.__dict__.update(reset_attr_dict)
        if interval == 0.:
            # Note: 1.1 applies a safety margin.
            interval = self.estimate_safe_interval(polygon, orig_isolation_mode,
                                                   memory_option, targ_GB,
                                                   False,
                                                   **kwargs["kwargs"])[0] * 1.1
        # Note: Because a safety margin was already applied above, do 
        # not engage the internal memory safety option.
        Skeleton.__init__(self, polygon, interval, unsafe_isolation_mode,
                          memory_option, targ_GB)

        # Unless forbidden by the user, subset the (coarse) template
        # skeleton's stubs if necessary to ensure that processing can
        # complete within available/specified memory.
        (fine_stubs_vor_vert_idxs_array,
         fine_stubs_coords_array, _) = self.get_nodes("stub", loops=False)
        if template_interval_is_positive:
            # Determine whether, after subsetting by the template
            # skeleton's stubs, there would still be too many
            # (uncontracted) nodes in the (fine) graph skeleton.
            # Note: 0.9 applies the same safety margin as implemented
            # for the "SAFE" isolation_mode option.
            safe_node_count = int(0.9 * self.estimate_max_node_count())
            template_stub_count = len(coarse_stub_data)
            # Note: Each marginal edge is associated with exactly one
            # hub and exactly one stub. The final (uncontracted) node
            # count after applyg the template is equal to all hubs not
            # associated with marginal edges (i.e., hubs - stubs) +
            # double the new marginal edge count.
            fine_hub_count = self.get_node_count("hub", loops=False)
            fine_stub_count = len(fine_stubs_vor_vert_idxs_array)
            templated_node_count = (fine_hub_count
                                    - fine_stub_count
                                    + 2*template_stub_count)
            #templated_node_count = (self.get_node_count("hub", loops=False)
            #                        + template_stub_count)
            if templated_node_count > safe_node_count:
                # Determine how many stubs can be retained in the
                # template skeleton to ensure safe processing.
                cur_excess_node_count = templated_node_count - safe_node_count
                # Note: The gobbling of each marginal edge sequence
                # removes exactly two uncontracted nodes: the stub at
                # one end and the hub at the other end (which becomes a
                # binode that can be contracted through).
                safe_stub_count = template_stub_count - (
                    cur_excess_node_count // 2
                    )
                if safe_stub_count < 2:
                    raise MemoryError(
                        "at this template_interval, there are ~{:,} too many hubs (out of {:,} total) to even preserve the trunk and safely complete processing in memory: {}".format(
                            safe_node_count - (fine_hub_count
                                               - fine_stub_count
                                               + 2),
                            fine_hub_count
                            )
                        )
                # Subset the template skeleton's stubs, preferring to
                # retain those stubs that terminate higher normalized
                # length branches or the trunk.
                coarse_stub_data.sort(reverse=True)
                # *REASSIGNMENT*
                coarse_stub_data = coarse_stub_data[:safe_stub_count]
                min_retained_normalized_length = self.template_normalized_length_cutoff = coarse_stub_data[-1][0]
                if safe_stub_count == 2:
                    _warnings.warn_explicit(
                        "Because of memory constraints, only the template skeleton's (shortest) trunk was used.",
                        UserWarning, __file__, 0
                        )
                elif min_retained_normalized_length > min_normalized_length:
                    _warnings.warn_explicit(
                        "Because of memory constraints, only the template skeleton's branches with normalized length greater than the following value were used: {}".format(min_retained_normalized_length),
                        UserWarning, __file__, 0
                        )

        # For each tail (or stub) coordinate (retained) from the
        # (coarse) template skeleton, find the marginal edge with the
        # nearest stub coordinate (in 2D) from the (fine) graph
        # skeleton.
        # Note: Tails are not added for sake of performance and
        # because they would be unlikely to significantly change the
        # final result.
        cdist = _scipy.spatial.distance.cdist
        fine_stubs_vor_vert_idxs = fine_stubs_vor_vert_idxs_array.tolist()
        del fine_stubs_vor_vert_idxs_array
        matching_fine_stub_vor_vert_idxs = []
        matching_fine_stub_vor_vert_idxs_append = matching_fine_stub_vor_vert_idxs.append
        for coarse_norm_length, x, coarse_stub_coords_array in coarse_stub_data:
            # Note: Square Euclidean used instead of regular
            # Euclidean for computational efficiency.
            i = cdist(coarse_stub_coords_array[None], fine_stubs_coords_array,
                      "sqeuclidean").argmin()
            matching_fine_stub_vor_vert_idxs_append(fine_stubs_vor_vert_idxs[i])
        # Release memory.
        del fine_stubs_vor_vert_idxs, fine_stubs_coords_array

        # Conform the (fine) graph skeleton to the (coarse) template
        # skeleton (or its retained subset).
        self.gobble_edges(matching_fine_stub_vor_vert_idxs, True)

        # Partition the (gobbled, fine) graph skeleton.
        final_calling_args = calling_args.copy()
        final_calling_args["interval"] = interval        
        final_calling_args["isolation_mode"] = orig_isolation_mode                
        final_calling_args["targ_GB"] = targ_GB
        self._template_interval_applied = True
        # Note: Skeleton is not re-initialized.        
        self.__init__(**final_calling_args)

    @staticmethod  # Note: self is passed when called.
    def _generate_skeleton(self, trunk_mode_form, trunk_mode, polygon,
                           max_branch_count, shortest_branch, shortest_bridge,
                           shortest_loop, cost_func, max_trunk_count,
                           min_normalized_length, mode, mode_is_maxed,
                           mode_is_maxed2, tails, **kwargs):
        # Process trunk mode.
        # Defaults.
        trunk_from_nodes = None
        trunk_to_nodes = None
        trunk_path_test_func = None
        trunk_ends_test_func = None
        trunk_test_func = None
        if trunk_mode_form != "DEFAULT":
            trunk_from_node = trunk_to_node = None  # Defaults.
            if trunk_mode_form in ("EUCLIDEAN", "EUCLIDEAN3D"):
                stubs_vor_vert_idxs_array, stubs_coords_array, _ = self.get_nodes(
                    "stub", loops=False,
                    interpolate_3D=(trunk_mode_form == "EUCLIDEAN3D")
                    )
                # Note: Square Euclidean used instead of regular
                # Euclidean for computational efficiency.
                dists_array = _scipy.spatial.distance.pdist(stubs_coords_array,
                                                            "sqeuclidean")
                n = len(stubs_vor_vert_idxs_array)
                k = int(dists_array.argmax())
                i = int(
                    _math.ceil(
                        (0.5 * (2.*n - 1. - (4.*n**2. - 8.*k - 4.*n - 7.)**0.5)
                         - 1)
                        )
                    )
                j = int(
                    n - ((i + 1.) * (n - i - 2.) + ((i + 1.) * (i + 2.))/2.) + k
                    )
                trunk_from_node = stubs_vor_vert_idxs_array[i]
                trunk_to_node = stubs_vor_vert_idxs_array[j]
            elif trunk_mode_form in ("NEAR_#_#", "NEAR_#_#_AND_#_#"):
                str_vals = trunk_mode[5:].upper().replace("_AND_", "_").split(
                    "_"
                    )
                vals = map(float, str_vals)
                # Note: If trunk_mode_option is "NEAR_#_#", vals[2:] is
                # am empty list.
                xys = (vals[:2], vals[2:])
                trunk_from_and_to_nodes = []
                stubs_vor_vert_idxs_array, stubs_coords_array, _ = self.get_nodes(
                    "stub", loops=False,
                    interpolate_3D=(trunk_mode_form == "EUCLIDEAN3D")
                    )
                for xy in xys:
                    # Note: Square Euclidean used instead of regular
                    # Euclidean for computational efficiency.
                    dists_array = _scipy.spatial.distance.cdist(
                        _numpy_fromiter(xy, _numpy_float64, 2),
                        stubs_coords_array,
                        "sqeuclidean"
                        )
                    trunk_from_and_to_nodes.append(
                        stubs_vor_vert_idxs_array[dists_array.argmin()]
                        )
                    if trunk_mode_form == "NEAR_#_#":
                        trunk_from_and_to_nodes.append(None)
                        break
                trunk_from_node, trunk_to_node = trunk_from_and_to_nodes
            elif trunk_mode_form == "HIGH_LOW":
                stubs_vor_vert_idxs_array, stubs_coords_array, _ = self.get_nodes(
                    "stub", loops=False, interpolate_3D=True
                    )
                stubs_z_coords_array = stubs_coords_array[:,3]
                trunk_from_node = stubs_vor_vert_idxs_array[stubs_z_coords_array.argmax()]
                trunk_to_node = stubs_vor_vert_idxs_array[stubs_z_coords_array.argmin()]
            elif trunk_mode_form in ("MIN_EUCLIDEAN_#", "MIN_EUCLIDEAN3D_#"):
                min_euclidean = float(trunk_mode.rpartition("_")[2])
                trunk_mode_is_3D = trunk_mode_form == "MIN_EUCLIDEAN3D_#"
                # Note: As a first optimization, isolate those stubs
                # that are at least the required Euclidean distance from
                # any one other stub.
                stubs_vor_vert_idxs_array, stubs_coords_array, _ = self.get_nodes(
                    "stub", loops=False, interpolate_3D=trunk_mode_is_3D
                    )
                ## Note: Could derive node indices from the condensed
                ## matrix (i.e., avoid calling squareform()) for better
                ## performance and a smaller memory footprint, but that
                ## derivation is somewhat complicated (cf. "EUCLIDEAN"
                ## trunk mode).
                # Note: Square Euclidean used instead of regular
                # Euclidean for computational efficiency.
                dists_array = _scipy.spatial.distance.squareform(
                    _scipy.spatial.distance.pdist(stubs_coords_array,
                                                  "sqeuclidean")
                    )
                del stubs_coords_array  # Release memory.
                row_filter_array = dists_array.max(0) >= min_euclidean
                del dists_array  # Release memory.
                trunk_from_nodes = trunk_to_nodes = stubs_vor_vert_idxs_array[row_filter_array]
                # Release memory.
                del stubs_vor_vert_idxs_array, row_filter_array
                if trunk_mode_is_3D:
                    def trunk_ends_test_func(
                        cost, complete, start_array, end_array,
                        min_euclidean=min_euclidean,
                        calc_dist=_geom.Point3D.calculate_distance.__func__
                        ):
                        span = calc_dist(start_array.tolist(),
                                         end_array.tolist())
                        return (span >= min_euclidean, False)
                else:
                    def trunk_ends_test_func(
                        cost, complete, start_array, end_array,
                        min_euclidean=min_euclidean,
                        calc_dist=_geom.Point2D.calculate_distance.__func__
                        ):
                        span = calc_dist(start_array.tolist()[:2],
                                         end_array.tolist()[:2])
                        return (span >= min_euclidean, False)
            elif trunk_mode_form in ("MIN_RELIEF_#%", "MIN_RELIEF_#"):
                min_relief = abs(
                    float(trunk_mode.rpartition("_")[2].rstrip("%"))
                    )
                if trunk_mode_form[-1] =="%":
                    z_arrays = [ring.coords_array[:,2]
                                for ring in polygon.boundary]
                    high = max([z_array.max() for z_array in z_arrays])
                    low = min([z_array.min() for z_array in z_arrays])
                    min_relief *= 0.01 * (high - low)  # *REASSIGNMENT*
                # Note: As a first optimization, isolate those nodes
                # that are at least the required elevation above any one
                # other node.
                stubs_vor_vert_idxs_array, stubs_coords_array, _ = self.get_nodes(
                    "stub", loops=False, interpolate_3D=True
                    )
                stubs_z_array = stubs_coords_array[:,2]
                min_stub_z = stubs_z_array.min()
                max_stub_z = stubs_z_array.max()
                row_high_filter_array = stubs_z_array >= min_stub_z + min_relief
                row_low_filter_array = stubs_z_array <= max_stub_z - min_relief
                # Release memory.
                del stubs_z_array
                row_filter_array = _numpy.logical_or(row_high_filter_array,
                                                     row_low_filter_array,
                                                     row_low_filter_array)
                # Reduce namespace and release memory.
                del row_high_filter_array, row_low_filter_array
                trunk_from_nodes = trunk_to_nodes = stubs_vor_vert_idxs_array[row_filter_array]
                # Release memory.
                del stubs_vor_vert_idxs_array, row_filter_array
                def trunk_ends_test_func(path, complete, start_array, end_array,
                                         min_relief=min_relief):
                    return (abs(start_array[2] - end_array[2]) >= min_relief,
                            False)
            elif trunk_mode_form == "MIN_MEAN_WIDTH_#":
                min_mean_width = float(trunk_mode.rpartition("_")[2])
                def trunk_path_test_func(path, complete,
                                         min_mean_width=min_mean_width):
                    return (path.area / path.length >= min_mean_width, False)
            else:
                # Note: This error indicates that supporting code was
                # not added.
                raise NotImplementedError(
                    "trunk_mode form: {!r}".format(trunk_mode_form)
                    )
            # Note: trunk_from_node and trunk_to_node are float's (or
            # None's).
            if trunk_from_node is not None:
                trunk_from_nodes = (int(trunk_from_node),)
            if trunk_to_node is not None:
                trunk_to_nodes = (int(trunk_to_node),)

        # Determine what path types should be included.
        include_branches = (
            (max_branch_count is None or max_branch_count >= 1) and
             shortest_branch >= 0.
            )
        include_bridges = shortest_bridge >= 0.
        include_loops = shortest_loop >= 0.

        # Derive test functions, as necessary.
        # Note: These keyword arguments optimize performance.
        supplemental_make_test_func_kwargs = {"no_negative_costs": True,
                                              "force_accuracy": False}
        if (trunk_path_test_func is not None or
            trunk_ends_test_func is not None):
            trunk_test_func = self.make_test_func(
                path_test_func=trunk_path_test_func,
                ends_test_func=trunk_ends_test_func,
                **supplemental_make_test_func_kwargs
                )
        if include_branches:
            if shortest_branch == 0.:
                branch_test_func = branch_deep_test_func = None
            else:
                branch_test_func = self.make_test_func(
                    shortest_branch, min_trunc_weight=shortest_branch,
                    **supplemental_make_test_func_kwargs
                    )
                branch_deep_test_func = branch_test_func.make_deep_test_func(
                    cost_func
                    )
        if include_bridges:
            if shortest_bridge == 0.:
                bridge_test_func = bridge_deep_test_func = None
            else:
                bridge_test_func = self.make_test_func(
                    shortest_bridge, min_trunc_weight=shortest_bridge,
                    **supplemental_make_test_func_kwargs
                    )
                bridge_deep_test_func = bridge_test_func.make_deep_test_func(
                    cost_func
                    )
        if include_loops:
            if shortest_loop == 0.:
                loop_test_func = None
            else:
                loop_test_func = self.make_test_func(
                    shortest_loop,
                    **supplemental_make_test_func_kwargs
                    ).make_deep_test_func(cost_func)

        # Create some useful objects.
        partition_out_trunks_individually = (
            self.partition_out_trunk,
            {"cost_func": cost_func,
             "opportunistic": None,
             "test_func": trunk_test_func,
             "target": max_trunk_count,
             "from_nodes": trunk_from_nodes,
             "to_nodes": trunk_to_nodes}
             )
        if include_branches:
            partition_out_all_branches = (
                self.partition_out_branches,
                {"cost_func": cost_func,
                 "opportunistic": True,
                 "test_func": branch_test_func,
                 "target": 0}
                )
            partition_out_all_branches_individually = (
                self.partition_out_branches,
                {"cost_func": cost_func,
                 "opportunistic": None,
                 "test_func": branch_test_func,
                 "target": None}
                )
            partition_out_next_branch = (
                self.partition_out_branches,
                {"cost_func": cost_func,
                 "opportunistic": None,
                 "test_func": branch_test_func,
                 "target": 0}
                )
        if include_bridges:
            partition_out_all_bridges = (
                self.partition_out_bridges,
                {"cost_func": cost_func,
                 "opportunistic": True,
                 "test_func": bridge_test_func,
                 "target": None}
                )
            partition_out_all_bridges_individually = (
                self.partition_out_bridges,
                {"cost_func": cost_func,
                 "opportunistic": None,
                 "test_func": bridge_test_func,
                 "target": None}
                )
            partition_out_next_bridge = (
                self.partition_out_bridges,
                {"cost_func": cost_func,
                 "opportunistic": None,
                 "test_func": bridge_test_func,
                 "target": 0}
                )
            # Note: Suppress "unused variable" code warnings.
            partition_out_next_bridge
        if include_loops:
            partition_out_loops = (
                self.add_loops_to_partitioned,
                {"test_func": loop_test_func}
                )
        discard_always = lambda edge, complete: (None, False)
        sweep = self.sweep
        sweep_kwargs = {
            "branch_test_func": (branch_deep_test_func if include_branches
                                 else discard_always),
            "bridge_test_func": (bridge_deep_test_func if include_bridges
                                 else discard_always),
            "orient_func": (cost_func.orient_func if cost_func.is_directed
                            else None)
            }
        if min_normalized_length == 0. or not include_branches:
            prune_branches_to_min_norm_length = False
        else:
            prune_branches_to_min_norm_length = (
                self.prune,
                {"cutoff": min_normalized_length,
                 "get_cost": self.get_normalized_length,
                 "kind": "branch"}
                )
        if max_branch_count is None or not include_branches:
            prune_branches_to_max_count = False
        else:
            # Note: It is assumed that user wishes to prioritize for
            # retention those branches with greater normalized lengths.
            prune_branches_to_max_count = (
                self.prune,
                {"target": max_branch_count,
                 "priority_func": self.get_normalized_length,
                 "kind": "branch"}
                )

        # Prepare for partitioning out, according to mode.
        func_kwargs_pairs = []
        if mode in ("PREFER_BRANCH", "MAX_BRANCH", "MAX_BRANCH_MAX_BRIDGE"):
            # Partition out one or more trunks, as specified.
            func_kwargs_pairs.append(partition_out_trunks_individually)
            # If bridges are to be included, partition these out.
            if include_branches:
                if mode_is_maxed:
                    func_kwargs_pairs.append(
                        partition_out_all_branches_individually
                        )
                else:
                    func_kwargs_pairs.append(partition_out_all_branches)
                # If a minimum normalized length is specified, prune
                # branches accordingly.
                if prune_branches_to_min_norm_length:
                    func_kwargs_pairs.append(prune_branches_to_min_norm_length)
                # If a maximum branch count is specified, prune down to
                # that target, if necessary.
                if prune_branches_to_max_count:
                    func_kwargs_pairs.append(prune_branches_to_max_count)
            # If bridges are to be included, partition these out.
            if include_bridges:
                if mode_is_maxed2:
                    func_kwargs_pairs.append(
                        partition_out_all_bridges_individually
                        )
                else:
                    func_kwargs_pairs.append(partition_out_all_bridges)
        elif mode in ("PREFER_BRIDGE", "MAX_BRIDGE", "MAX_BRIDGE_MAX_BRANCH"):
            # Note: Also see comments for "prefer_branch" mode.
            func_kwargs_pairs.append(partition_out_trunks_individually)
            if include_bridges:
                if mode_is_maxed:
                    partition_out_all_bridges_somehow = partition_out_all_bridges_individually
                else:
                    partition_out_all_bridges_somehow = partition_out_all_bridges
                func_kwargs_pairs.append(partition_out_all_bridges_somehow)
                if include_branches:
                    # Add each branch one at a time, and then add all
                    # possible bridges. Repeat.
                    func_kwargs_pairs.append(
                        (partition_out_next_branch,
                         prune_branches_to_min_norm_length if prune_branches_to_min_norm_length else (None, None),
                         partition_out_all_bridges_somehow)
                        )
                    # If a minimum normalized length is specified, prune
                    # branches one last time, to catch any swept 
                    # branches.
                    if prune_branches_to_min_norm_length:
                        func_kwargs_pairs.append(
                            prune_branches_to_min_norm_length
                            )
                    if prune_branches_to_max_count:
                        func_kwargs_pairs.append(prune_branches_to_max_count)
                    # Make sure that all bridges are partitioned out
                    # even if no more branches could be partitioned out,
                    # ending the loop. (See further below.)
                    func_kwargs_pairs.append(partition_out_all_bridges_somehow)
            elif include_branches:
                # Add all branches, one at a time (for consistency with
                # the case where bridges are included).
                func_kwargs_pairs.append(
                    partition_out_all_branches_individually
                    )
                if prune_branches_to_min_norm_length:
                    func_kwargs_pairs.append(prune_branches_to_min_norm_length)
                if prune_branches_to_max_count:
                    func_kwargs_pairs.append(prune_branches_to_max_count)
        # If loops are to be included, partition these out.
        if include_loops:
            func_kwargs_pairs.append(partition_out_loops)

        # Partition out.
        if tails:
            add_tails = self.add_tails
            prune = self.prune
        partition_out_funcs = (self.partition_out_trunk,
                               self.partition_out_branches,
                               self.partition_out_bridges)
        prev_result = None
        for func_kwargs_pair in func_kwargs_pairs:
            # In the special case that an inner loop is required,
            # execute it.
            # Note: For example, such an inner loop might be used to add
            # one branch at a time, prune it away if criteria exclude
            # it, else add all new bridges that can be added because of
            # the new branch, and repeat until no new branch can be
            # added (even temporarily, before pruning).
            if isinstance(func_kwargs_pair[0], tuple):
                while True:
                    ((partition_out1, kwargs1),
                     (prune, kwargs2),
                     (partition_out2, kwargs3)) = func_kwargs_pair
                    (added_path_count, _,
                     _, _, deleted_path_count) = partition_out1(**kwargs1)
                    if deleted_path_count:
                        sweep(**sweep_kwargs)
                    if not added_path_count:
                        break
                    if prune is not None:
                        if tails:
                            add_tails()
                        pruned_path_count = len(prune(**kwargs2)[0])
                        added_path_count -= pruned_path_count
                        # If no new paths of the first kind (e.g.,
                        # branch) remain, no new paths of the second
                        # kind (e.g., bridge) can (presumably) be added.
                        if not added_path_count:
                            continue
                    deleted_path_count = partition_out2(**kwargs3)[-1]
                    if deleted_path_count:
                        sweep(**sweep_kwargs)
                continue
            func, kwargs = func_kwargs_pair
            if tails and func is prune:
                add_tails()
            prev_result = func(**kwargs)
            # If any edges were partitioned out or otherwise deleted
            # from the graph skeleton, sweep.
            if func in partition_out_funcs and prev_result[-1]:
                sweep(**sweep_kwargs)

        # Add tails (in case any partitioning out of trunks or branches
        # occurred after the final pruning, if any).
        if tails:
            add_tails()

    def add_tails(self, cut=True):
        """
        Add tails to the trunk(s) and all branches.

        If any trunk or branch already has a tail, it is not modified.

        See SkeletalLineString2D.add_tails() for documentation of the cut
        argument.
        """
        for kind in ("trunk", "branch"):
            for edge in self.get_lines(kind, hardness_level=0):
                edge.add_tails(cut, True)



###############################################################################
# TESTING SUPPORT                                                             #
###############################################################################

class BinaryTree(object):

    def __init__(self, trunk_length, length_factor, angle, generations,
                 trunk_base_coords=(0., 0.)):
        """
        Construct a symmetric binary (fractal) tree.

        The tree is stored as a list of LineSegment2D's to .segments. Each
        segment has a .generation that gives the branch generation, or 0 for the
        trunk.

        trunk_length is a float that specifies the length of the tree's trunk,
        which is oriented upward and vertical (i.e., in a +y direction).

        length_factor is a float that specifies the ratio of the length of a
        given branch to that of its child branch, or equivalently, of the trunk
        to that of the first branches.

        angle is a float that specifies the angle (in degrees) of deflection of
        each child branch relative to the direction of the parent branch.

        generations is an integer that specifies the number of branch
        generations to create.

        trunk_base_coords is a sequence of length 2 that specifies the (x, y)
        coordinates of the trunk at its base.

        Warning: Specifying progressively higher generations requires
        exponentially more processing time, due to the fractal nature of the
        output.

        Note: To choose a length_factor that produces a tree that just barely
        contacts itself after infinite generations, you can use the following
        code at www.wolframalpha.com, being sure to specify the variable a
        (right end of code) with the desired angle:
            For 30 <= angle < 45:
                solve r*sin(-a) + (r^3)*sin(a) + ((r^4)/(1-(r^2)))*sin(2a) +
                    ((r^5)/(1-(r^2)))*sin(3a) = 0 for a=radians(30 degrees)
            For 45 <= angle < 90:
                solve r*sin(-a) + ((r^3)/(1-(r^2)))*sin(a) +
                    ((r^4)/(1-(r^2)))*sin(2a) = 0 for a=radians(45 degrees)
            For 90:
                r = 2^(-1/2)
            For 90 < angle <= 135:
                solve r*sin(-a) + ((r^2)/(1-(r^2)))*sin(-2a) +
                    ((r^3)/(1-(r^2)))*sin(-3a) = 0 for a=radians(91 degrees)
            For 135 < angle < 180:
                solve (r/(1-(r^2)))*sin(-a) + ((r^2)/(1-(r^2))) = 0 for
                    a=radians(91 degrees)
        Credit: Mandelbrot and Frame (1999), "The Canopy and Shortest Path in
        a Self-Contacting Fractal Tree", The Mathematical Intelligencer.
        """
        # Localize some useful objects.
        _2_2_tuple = (2, 2)
        atan2 = _math.atan2
        cos = _math.cos
        sin = _math.sin
        LineSegment2D = _geom.LineSegment2D
        empty = _numpy_empty
        float64 = _numpy_float64

        # Prepare for recursive generation.
        trunk_length = float(trunk_length)
        trunk_coords_array = _numpy_empty(_2_2_tuple, float64)
        trunk_coords_array[0] = trunk_coords_array[1] = trunk_base_coords
        trunk_coords_array[1][1] += trunk_length
        trunk = LineSegment2D(trunk_coords_array)
        trunk.generation = 0
        prev_segs = [trunk]
        angle_radians = _math.radians(angle)

        # Recursively generate branches.
        cum_segs = self.segments = [trunk]
        for i in xrange(1, generations + 1):
            # Calculate branch length for the next generation.
            next_length = trunk_length * length_factor**i
            next_segs = []
            for seg in prev_segs:

                # For each branch (or trunk) from the previous
                # generation, calculate its azimuth.
                seg_coords_array = seg.coords_array
                (x0, y0), (x1, y1) = seg_coords_array.tolist()
                # Note: azimuth is measured from the +x-axis.
                azimuth = atan2(y1 - y0, x1 - x0)

                # Initially populate the coordinate array for each
                # branch of the next pair as though each were a 0-length
                # segment at the distal end of the branch (or trunk)
                # from the previous generation.
                new_seg_coords_array = empty(_2_2_tuple, float64)
                new_seg_coords_array[:] = seg_coords_array[1]
                for branch_coords_array in (new_seg_coords_array,
                                            new_seg_coords_array.copy()):

                    # Set azimuth for the left- or right-bearing branch,
                    # respectively.
                    if branch_coords_array is new_seg_coords_array:
                        next_azimuth = azimuth + angle_radians
                    else:
                        next_azimuth = azimuth - angle_radians

                    # Populate distal coordinates for this branch with
                    # the appropriate end coordinates.
                    next_seg_coords_array1 = branch_coords_array[1]
                    next_seg_coords_array1[0] += next_length * cos(next_azimuth)
                    next_seg_coords_array1[1] += next_length * sin(next_azimuth)

                    # Store branch that was just created.
                    next_seg = LineSegment2D(branch_coords_array)
                    next_seg.generation = i
                    next_segs.append(next_seg)

            # Add branches from this generation to cumulative record,
            # and prepare for the next generation.
            cum_segs.extend(next_segs)
            prev_segs = next_segs

    def add_noise(self, vertices=10, offset=0.05):
        """
        Add noise to the generated .segments. Result saved to .noisy_lines.

        vertices is an integer that specifies the number of vertices that each
        line, generated from a single segment in .segments, should have. This
        count includes the start and end vertices.

        offset is a float that specifies the maximum offset permitted for a
        vertex of a line, relative to the corresponding segment from .segments
        and scaled to that segment's length. For example, if a segment has a
        length of 100 and offset is 0.05, the maximum offset of any vertex in
        the generated line is 5. Note that the start and end vertices are never
        offset, to ensure continuity of the noisy tree.
        """
        norm_dists = _numpy.linspace(0., 1., vertices)
        rand = _numpy.random.rand
        rand_row_count = vertices - 2
        rand_col_count = 2
        double_offset = 2. * offset
        LineString2D = _geom.LineString2D
        noisy_lines = self.noisy_lines = []
        for seg in self.segments:
            line_coords_array = seg.interpolate(norm_dists, True)
            noise = rand(rand_row_count, rand_col_count)
            noise -= 0.5  # Make noise symmetric about 0.
            # Note: noise has range [-0.5, 0.5), so it should be
            # multiplied by twice the specified offset.
            noise *= (double_offset * seg.length)
            # Note: Leave end vertices untouched.
            line_coords_array[1:-1] += noise
            line = LineString2D(line_coords_array)
            line.generation = seg.generation
            if hasattr(seg, "spatial_reference"):
                line.spatial_reference = seg.spatial_reference
            noisy_lines.append(line)

    def set_spatial_reference(self, spatial_reference):
        """
        Specify a spatial reference for .segments and .noisy_lines.

        If .noisy_lines is defined prior to calling the current function, both
        .segments and .noisy_lines are assigned the specified spatial reference.
        Otherwise, .noisy_lines will inherit that spatial reference upon
        generation (if that ever occurs).

        spatial_reference may be a Geometry, Information, Definition, or WKT
        string.
        """
        import numgeo.vec_io
        spatial_reference = numgeo.vec_io._isolate_spatial_reference(
            (spatial_reference,), True
            )
        if not isinstance(spatial_reference, basestring):
            raise TypeError(
                "spatial_reference is not recognized: {!r}".format(
                    spatial_reference
                    )
                )
        for seg in self.segments:
            seg.spatial_reference = spatial_reference
        if hasattr(self, "noisy_lines"):
            for line in self.noisy_lines:
                line.spatial_reference = spatial_reference

    def show(self, prefer_noisy_lines=True):
        """
        Display tree in a plot.

        prefer_noisy_lines is a boolean that specifies whether .noisy_lines will
        be used if it has been assigned. If .noisy_lines has not been assigned
        or prefer_noisy_lines is False, .segments is used instead.

        Note: Requires pylab, matplotlib.collections modules.
        """
        import pylab
        import matplotlib.collections
        if prefer_noisy_lines and hasattr(self, "noisy_lines"):
            lines = self.noisy_lines
        else:
            lines = self.segments
        nested_coords = [line.coords_array.tolist() for line in lines]
        lc = matplotlib.collections.LineCollection(nested_coords, linewidths=1)
        fig, ax = pylab.subplots()
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
        matplotlib.pyplot.show()
