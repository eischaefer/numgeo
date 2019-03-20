"""
Incomplete geometric library primarily based on numpy arrays.
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
# USER SETTINGS                                                               #
###############################################################################



###############################################################################
# IMPORT                                                                      #
###############################################################################

# Import internal (intra-package).
from numgeo import opt as _opt
from numgeo import util as _util

# Import external.
import copy as _copy
import collections as _collections
import itertools as _itertools
import math as _math
import numpy as _numpy
import re as _re
import struct as _struct



###############################################################################
# LOCALIZATION                                                                #
###############################################################################

# Derived from built-ins.
_neg_1_tuple = (-1,)
_2_tuple = (2,)
_3_tuple = (3,)
_neg_1_1_tuple = (-1, 1)
_neg_1_2_tuple = (-1, 2)
_neg_1_3_tuple = (-1, 3)
_1_2_tuple = (1, 2)
_1_3_tuple = (1, 3)
_2_2_tuple = (2, 2)
_join = "".join
_comma_space_join = ", ".join
_marker = object() # Arbitrary unique value.

# Derived from internal.
_OPTIMIZE_SUM_CUTOFF = _opt.OPTIMIZE_SUM_CUTOFF
_OPTIMIZE_EXTREME_CUTOFF = _opt.OPTIMIZE_EXTREME_CUTOFF
_take2 = _util._take2
_Instantiable = _util.Instantiable
_InstantiableList = _util.InstantiableList

# Derived from external.
_defaultdict = _collections.defaultdict
_deepcopy = _copy.deepcopy
_imap = _itertools.imap
_math_sqrt = _math.sqrt
_numpy_abs = _numpy.abs
_numpy_add = _numpy.add
_numpy_arange = _numpy.arange
_numpy_array = _numpy.array
_numpy_bool8 = _numpy.bool8
_numpy_concatenate = _numpy.concatenate
_numpy_divide = _numpy.divide
_numpy_empty = _numpy.empty
_numpy_fromiter = _numpy.fromiter
_numpy_fromstring = _numpy.fromstring
_numpy_float64 = _numpy.dtype("<f8")
_numpy_float64_big_endian = _numpy.dtype(">f8")
_numpy_greater = _numpy.greater
_numpy_multiply = _numpy.multiply
_numpy_negative = _numpy.negative
_numpy_ndarray = _numpy.ndarray
_numpy_not_equal = _numpy.not_equal
_numpy_sqrt = _numpy.sqrt
_numpy_square = _numpy.square
_numpy_subtract = _numpy.subtract



###############################################################################
# GENERAL SUPPORT                                                             #
###############################################################################

# These numpy arrays support .make_example() for Geometry's.
_unit_square_coords_array3D = _numpy.array(
    ((0., 0., 0.), (1., 0., 1.), (1., 1., 2.), (0., 1., 3.), (0., 0., 0.)),
    _numpy_float64
    )
_unit_square_coords_array2D = _unit_square_coords_array3D[:,:2].copy()

# This regular expression supports .flip() for LineString's.
_flip_dict_re_match = _re.compile("_\w+?__flip_dict$").match

# This dictionary maps arcpy shape type names to the corresponding wkb
# type integer.
_arcpy_shape_type_to_wkb_type2D = {"Point": 1, "Polyline": 5, "Polygon": 6,
                                   "Multipoint": 4}

# These methods are used to pack numbers into binary data, and vice
# versa. They are used to support wkb reading and writing.
_struct_unsigned_integer = _struct.Struct("<I")
_pack_unsigned_integer = _struct_unsigned_integer.pack
_unpack_unsigned_integer = _struct_unsigned_integer.unpack

# Note: The following containers are populated by _flesh_out_geometry().
# This dictionary maps wkb type integers to the corresponding Geometry
# subclass.
_wkb_type_to_geom_type = {}
# This dictionary maps arcpy shape type names to the corresponding
# Geometry2D subclass.
_arcpy_shape_type_to_geom_type2D = {}
# This dictionary maps Geometry2D subclasses to the corresponding arcpy
# shape type names.
_geom_type2D_to_arcpy_shape_type = {}
# This dictionary maps Geometry subclasses to the corresponding
# Geometry2D subclasses.
_geom_type_to_2D = {}
# This dictionary maps Geometry subclasses to the corresponding
# Geometry3D subclasses.
_geom_type_to_3D = {}
# This dictionary maps Geometry subclasses to the corresponding
# MultiGeometry subclasses.
_geom_type_to_multi = {}
# This nested dict maps an is_3D boolean to the corresponding
# _geom_type_to_*D dict.
_is_3D_to_geom_type_to_XD = {False: _geom_type_to_2D, True: _geom_type_to_3D}

def _flesh_out_geometry():
    """
    Geometry-priming function called once during import.

    For each subclass X of SingleGeometry or MultiGeometry, and each subclass
    X2D/3D of X, assigns:
        X.wkb_type(2D/3D)
        X._wkb_prefix(2D/3D)
        X._ewkb_prefix(3D)
        X._wkt_prefix(2D/3D)
        X._match_wkt_prefix
        X2D/X3D.wkb_type
    And for MultiGeometry subclasses only:
        X.member_type
        X._member_wkt_trim_length
    And populates the global dict's:
        _wkb_type_to_geom_type
        _geom_type_to_2D/3D
        _geom_type_to_multi
        _arcpy_shape_type_to_geom_type2D
        _geom_type2D_to_arcpy_shape_type
    """
    # Note: This code is defined as a function merely to hide its
    # namespace from the module's namespace.
    pack_wkb_prefix = _struct.Struct("<BI").pack
    format_wkt_prefix2D = "{} (".format
    format_wkt_prefix3D = "{} Z (".format
    re_compile = _re.compile
    re_escape = _re.escape
    re_I = _re.I
    for cls, is_multi in ((SingleGeometry, False), (MultiGeometry, True)):
        for subcls in cls.__subclasses__():
            # Example subcls's: LineString, MultiLineString
            subcls.wkb_type2D = wkb_type2D = (1 + subcls.topological_dimension +
                                              3*is_multi)
            subcls.wkb_type3D = wkb_type3D = wkb_type2D + 1000
            subcls._wkb_prefix2D = wkb_prefix2D = pack_wkb_prefix(1, wkb_type2D)
            subcls._wkb_prefix3D = wkb_prefix3D = pack_wkb_prefix(1, wkb_type3D)
            subcls._ewkb_prefix3D = ewkb_prefix3D = wkb_prefix2D[:4] + "\x80"
            subcls._wkt_prefix2D = wkt_prefix2D = format_wkt_prefix2D(
                subcls.__name__
                )
            subcls._wkt_prefix3D = wkt_prefix3D = format_wkt_prefix3D(
                subcls.__name__
                )
            for subsubcls in subcls.__subclasses__():
                if not issubclass(subsubcls, _Instantiable):
                    # Example subsubcls: LineSegment (if it were defined
                    # before the current function is called).
                    continue
                # Example subsubcls's: LineString2D, MultiLineString2D
                # Note: "Fourth generation" classes (e.g.,
                # SingleGeometry -> LineString -> LineSegment ->
                # LineSegment2D) are subsubsubcls's and not seen by the
                # current function. One benefit is that these highly
                # derived classes are not accidentally set as the
                # default for a given wkb type.
                if subsubcls.is_3D:
                    subsubcls.wkb_type = wkb_type = wkb_type3D
                    subsubcls._wkb_prefix = wkb_prefix3D
                    subsubcls._ewkb_prefix = ewkb_prefix3D
                    subsubcls._wkt_prefix = wkt_prefix = wkt_prefix3D
                    # Note: Only the second converntion is currently
                    # used, but the second is included to potentially
                    # future-proof the code.
                    _wkb_type_to_geom_type[wkb_type3D] = subsubcls
                    _wkb_type_to_geom_type[wkb_type2D - 2147483648] = subsubcls
                else:
                    subsubcls.wkb_type = wkb_type = wkb_type2D
                    subsubcls._wkb_prefix = subsubcls._ewkb_prefix = wkb_prefix2D
                    subsubcls._wkt_prefix = wkt_prefix = wkt_prefix2D
                    _wkb_type_to_geom_type[wkb_type2D] = subsubcls
                # Note: Trailing "(" must be retained (otherwise, e.g.,
                # "Point " would match "Point Z (...") and escaped.
                subsubcls._match_wkt_prefix = re_compile(
                    re_escape(wkt_prefix), re_I
                    ).match
                if is_multi:
                    subsubcls.member_type = _wkb_type_to_geom_type[wkb_type - 3]
                    # Note: "Multi" not in member wkt's (-5), but also
                    # preserve "(" at end of member wkt's (-1).
                    subsubcls._member_wkt_trim_length = len(wkt_prefix) - 6
    for wkb_type, geom_type in _wkb_type_to_geom_type.iteritems():
        if wkb_type >= 1000:
            _geom_type_to_3D[geom_type] = geom_type
            geom_type2D = _wkb_type_to_geom_type[wkb_type - 1000]
            _geom_type_to_3D[geom_type2D] = geom_type
            _geom_type_to_2D[geom_type] = geom_type2D
        elif wkb_type >= 0:
            _geom_type_to_2D[geom_type] = geom_type
        else:
            # Note: ewkb entries can be ignored as redundant.
            continue
        if issubclass(geom_type, MultiGeometry):
            _geom_type_to_multi[geom_type] = geom_type
            _geom_type_to_multi[_wkb_type_to_geom_type[geom_type.member_type.wkb_type]] = geom_type
            continue
    for arcpy_shape_type, wkb_type2D in _arcpy_shape_type_to_wkb_type2D.iteritems():
        ## Note: The if clause below is only necessary until
        ## MultiGeometry's are fully implemented.
        if wkb_type2D not in _wkb_type_to_geom_type:
            continue
        _arcpy_shape_type_to_geom_type2D[arcpy_shape_type] = _wkb_type_to_geom_type[wkb_type2D]
    _geom_type2D_to_arcpy_shape_type.update(
        _util._reverse_dict(_arcpy_shape_type_to_geom_type2D)
        )
    for geom_type2D, arcpy_shape_type in _geom_type2D_to_arcpy_shape_type.items():
        if issubclass(geom_type2D, MultiGeometry):
            member_geom_type = geom_type2D.member_type
            if member_geom_type in _geom_type2D_to_arcpy_shape_type:
                continue
            _geom_type2D_to_arcpy_shape_type[member_geom_type] = arcpy_shape_type


###############################################################################
# GEOMETRY BASE CLASSES                                                       #
###############################################################################

class Geometry(_util.Lazy2):
    "Base class for all Geometry types."

    def __getitem__(self, key):
        """
        Support self[field_name] --> self.data[field_name].
        """
        if isinstance(key, basestring):
            return self.data[key]
        raise KeyError(key)

    @staticmethod
    def _get_data(self):
        """
        Create empty data dict (to hold field data, if any).
        """
        return {}
    # Note: .data cannot be automatically repopulated if it is deleted.
    _get_data.__func__.deletable = False

    def _process_array(self, seq, arg_name="argument", targ_shape=_neg_1_tuple,
                       init=False):
        """
        Convert sequence to a numpy array, if necessary, and validate its shape.

        seq is a numpy array or array-like sequence.

        arg_name is a string that specifies the name that will be used to refer
        to seq if any error is raised.

        targ_shape is the shape that the returned array should have. If seq does
        not intrinsically have that shape, an error is raised.

        init is a boolean that specifies whether self is being actively
        (re)initialized. If init is True and seq is None, .coords_array will be
        used instead, if specified. This permits both delayed initialization as
        well as re-initialization by another type (e.g., a LineSegment2D calling
        LineString2D.__init__()).
        """
        # If called from within .__init__() and seq is unspecified,
        # attempt to find a stored substitute.
        if init and seq is None:
            # If seq is not specified, resort to using .coords_array
            # (for re-initialization) or ._arg0 (for delayed
            # initialization).
            # Note: Although similar, re- and delayed initialization
            # must use different attribute names, so that a processed
            # .coords_array can be distinguished from an ._arg0 that may
            # yet require processing.
            # Note: The current function can be called by the
            # automatically generated ._get_coords_array(), which is
            # desirable if .coords_array is used prior to
            # initialization. To avoid infinite recursion, lazy
            # attribute machinery is therefore intentionally avoided in
            # the line below. However, this has the side-effect that no
            # manual ._get_coords_array() can be defined because if some
            # other initialization-time attribute (e.g., .length) is
            # used, it will trigger delayed initialization via the
            # current function, which cannot (as just explained) attempt
            # to use .coords_array directly. Alternatively, a manual
            # ._get__arg0() can be defined.
            # *REDEFINITION*
            seq = self.__dict__.get("coords_array", None)
            if seq is None:
                try:
                    seq, process = self._arg0 # *REDEFINITION*
                except AttributeError:
                    raise TypeError("{} cannot be None".format(arg_name))
                if not process:
                    return seq
            else:
                return seq

        # Convert seq to a numpy array, if necessary.
        if not isinstance(seq, _numpy_ndarray):
            seq = _numpy_array(seq) # *REDEFINITION*

        # Validate seq's dtype.
        try:
            seq = seq.astype(_numpy_float64, "K", "safe", True, False)
        except:
            raise TypeError(
                "it is not safe to cast {} to float64".format(arg_name)
                )

        # Validate seq's shape.
        seq_shape = seq.shape
        if len(seq_shape) == len(targ_shape):
            if targ_shape is _neg_1_tuple:
                return seq
            for i, targ_i in zip(seq_shape, targ_shape):
                if i != targ_i and targ_i != -1:
                    break
            else:
                return seq
        raise TypeError(
            "{} must have a shape of {}, but has shape: {}".format(
                arg_name, targ_shape, seq_shape
                )
            )

    @classmethod
    def from_wkb(cls, wkb, coerce_partedness=True, start=0):
        """
        Create a new instance from the specified wkb.

        If the current type is not instantiable, an instance of an instantiable
        subtype (or related type, if coerce_partedness is True) is returned.

        wkb is a string that specifies the well-known binary representation for
        the returned instance. It must be of the correct geometric type
        (partedness, topological dimension, and spatial dimension).

        coerce_partedness is a boolean that specifies whether wkb, if it
        represents an instance of a counterpart Geometry with different
        partedness, should be processed and coerced, if possible, to the current
        type. For example, if the current type is MultiPolygon and wkb
        represents a Polygon2D, a MultiPolygon2D with that Polygon2D as its sole
        member would be returned if coerce_partedness is True. If
        coerce_partedness is False and the partedness of wkb does not match that
        of the current type, an error is raised. Note that a Geometry can always
        be coerced from a SingleGeometry to a MultiGeometry (as in the foregoing
        example) but can only be coerced from a MultiGeometry to a
        SingleGeometry if the MultiGeometry has a single member. Otherwise, the
        MultiGeometry is returned. If the current type has no implied partedness
        (e.g., Geometry), coerce_partedness is ignored.

        start is an integer that specifies the index at which wkb should start
        being read. If start is not 0, a tuple is returned of the form (geom,
        next_start), where geom is the new Geometry and next_start is the index
        in wkb that immediately follows that Geometry's definition. (Used
        internally.)
        """
        if coerce_partedness:
            related_classes, base = cls._fetch_related_classes(True)
        else:
            related_classes = cls._fetch_related_classes(False)
        wkb_prefix = wkb[start:start+5]
        ## Note: Could add ability to read big endian.
        for related_class in related_classes:
            if (wkb_prefix == related_class._wkb_prefix or
                wkb_prefix == related_class._ewkb_prefix):
                break
        else:
            cls._raise_incompatible_wkx(related_classes, "wkb")
        geom = related_class._from_wkb(wkb, start)
        if coerce_partedness:
            return cls._coerce_partedness(geom, related_class, base)
        return geom

    @classmethod
    def from_wkt(cls, wkt, coerce_partedness=True):
        if coerce_partedness:
            related_classes, base = cls._fetch_related_classes(True)
        else:
            related_classes = cls._fetch_related_classes(False)
        for related_class in related_classes:
            if related_class._match_wkt_prefix(wkt) is not None:
                break
        else:
            cls._raise_incompatible_wkx(related_classes, "wkt")
        geom = related_class._from_wkt(wkt)
        if coerce_partedness:
            return cls._coerce_partedness(geom, related_class, base)
        return geom
    # Note: from_wkt()'s documentation is nearly identical to that of
    # from_wkb(), except that from_wkt() lacks a start argument.
    from_wkt.__func__.__doc__ = from_wkb.__func__.__doc__.rsplit(
        "\n\n", 1
        )[0].replace("binary", "text").replace("wkb", "wkt") + "\n"

    @classmethod
    def _fetch_related_classes(cls, any_partedness):
        if issubclass(cls, _Instantiable):
            related_classes = [cls]
        else:
            related_classes = [geom_type for geom_type in
                               set(_wkb_type_to_geom_type.itervalues())
                               if issubclass(geom_type, cls)]
        if any_partedness:
            if issubclass(cls, SingleGeometry):
                base = SingleGeometry
                for related_class in tuple(related_classes):
                    related_classes.append(_geom_type_to_multi[related_class])
            elif issubclass(cls, MultiGeometry):
                base = MultiGeometry
                for related_class in tuple(related_classes):
                    related_classes.append(related_class.member_type)
            else:
                # Note: cls could be more abstract than Single/
                # MultiGeometry (e.g., Geometry), in which case
                # related_classes already includes both types of
                # partedness.
                base = cls
            return (related_classes, base)
        return related_classes

    @staticmethod
    def _coerce_partedness(geom, geom_type, base):
        if not isinstance(geom, base):
            if base is MultiGeometry:
                return _geom_type_to_multi[geom_type]((geom,))
            elif len(geom) == 1:
                geom, = geom # *REDEFINITION*
                return geom
        return geom

    @staticmethod
    def _raise_incompatible_wkx(related_classes, wkx_str):
        raise TypeError(
            "{} is not compatible with {}: {}".format(
                wkx_str,
                "this type" if len(related_classes) == 1 else "any of these types",
                _comma_space_join(
                    sorted([related_class.__name__
                            for related_class in related_classes])
                    )
                )
            )


class Geometry2D(Geometry):
    "Base class for all 2D single-part Geometry types."
    is_3D = False

    @staticmethod
    def _get__2D(self):
        return self


class Geometry3D(Geometry):
    "Base class for all 3D single-part Geometry types."
    is_3D = True

    @staticmethod
    def _get__2D(self):
        """
        Make corresponding Geometry2D.
        """
        return _geom_type_to_2D[type(self)](self.coords_array[:,:2])



###############################################################################
# SINGLE-GEOMETRY BASE CLASSES                                                #
###############################################################################

class SingleGeometry(Geometry):
    "Base class for all single-part Geometry types."

    @staticmethod
    def _get_ewkb(self):
        """
        Generate "extended" well-known binary representation for the instance.

        Extended WKBs are a PostGIS convention and differ from the ISO standard
        for Geometry3D's (but not for Geometry2D's).
        """
        if self.is_3D:
            return self._ewkb_prefix3D + self.wkb[5:]
        return self.wkb

    ## Note: It would probably be best to add some rotation and
    ## translation options to this function.
    @classmethod
    def make_example(cls):
        """
        Create an arbitrary instance of the current type.
        """
        if cls.is_3D:
            unit_square_coords_array = _unit_square_coords_array3D
        else:
            unit_square_coords_array = _unit_square_coords_array2D
        return cls(unit_square_coords_array[:cls.topological_dimension**2 + 1])

    @classmethod
    def make_fast(cls, arg0, process=True):
        """
        Create an instance very quickly.

        The current method is an approximate substitute for the initialization
        method of the current type, and the first argument for both functions is
        interpreted in the same way (despite having different names). However,
        even though an instance is returned, initialization of that instance is
        delayed until necessary (as much as possible), and no validation may
        occur prior to that time. After initialization, the instance acts like
        any other instance of the current type. The primary reason for this
        functionality is to support use cases where many instances may be
        created but few of them ever used, so that "placeholder" instances that
        are only lazily initialized could significantly increase performance.

        arg0 is the first argument taken at initialization (e.g., coords for
        LineString2D, boundary for Polygon2D).

        process is a boolean that specifies whether the value for arg0 should be
        "processed" before it is first used, to ensure that it is the exact type
        and format expected. This processing is performed automatically upon
        normal initialization (i.e., when the current function is not used). For
        example, if arg0 expects an array and process is False, arg0 must be an
        array of the right shape, dtype, and byte order. Even if process is
        True, arg0's value will only be processed "just in time", so the
        instance is still returned quickly.

        Warning: process is an advanced functionality. If process is False but
        arg0's value is not exactly as expected, errors and/or corrupted results
        may occur. To test whether arg0 requires processing, you can try
        something like:
            coords_array is LineString2D(coords_array).coords_array --> True
        """
        self = cls.__new__(cls, arg0)
        self._arg0 = (arg0, process)
        return self

    def minimize_memory(self, deep=False):
        """
        Minimize the memory footprint of the current instance.

        The current method primarily deletes objects (stored as possibly hidden
        attributes) that optimize performance but do so at the cost of increased
        memory use. In these cases, the relevant object is automatically
        ("lazily") re-generated at the moment it is next needed (if ever), but
        this re-generation can be computationally expensive. It is therefore not
        advisable to call the current method unless memory availability is an
        issue.

        deep is a boolean that specifies whether a deeper memory release should
        be performed. If deep is False, only objects that are almost certainly
        capable of lazy re-generation are deleted. If deep is instead True,
        objects that seem likely capable of lazy re-generation are also deleted.
        """
        # If deep is False, resort to clearing only those attributes
        # that are supported by the related "base" class (e.g.,
        # LineString2D rather than LineSegment2D). This is primarily
        # useful in case custom (e.g., user-derived) subclasses
        # implement complicated dependencies between lazy attributes
        # that render them unsafe to be deleted in batch. The assumption
        # is that "basic" attributes can be reliably rederived using the
        # lazy machinery in the "base" class.
        if deep:
            allow = None
        else:
            base_cls = _wkb_type_to_geom_type[self.wkb_type]
            allow = base_cls.__dir__.__func__(base_cls, True, True)
        self._Lazy__clear_lazy(allow)


class Point(SingleGeometry):
    "Base class for all single-part point-like Geometry types."
    topological_dimension = 0

    def __contains__(self, item):
        return item in self.tuple

    def __eq__(self, other):
        try:
            return hash(self.tuple) == hash(other.tuple)
        except:
            return False

    def __getitem__(self, key):
        return self.tuple[key]

    def __hash__(self):
        return hash(self.tuple)

    def __iter__(self):
        return iter(self.tuple)

    def __len__(self):
        return len(self.tuple)

    def __reversed__(self):
        return reversed(self.tuple)

    def __repr__(self):
        if "ID" in self.__dict__:
            ID_str = " (ID={})".format(self.ID)
        else:
            ID_str = ""
        return "<{}{}: x={}, y={}{} at {}>".format(
            type(self).__name__, ID_str, self[0], self[1],
            "" if not self.is_3D else ", z={}".format(self[2]), hex(id(self))
            )

    def __init__(self, coords=None):
        """
        Make an object representing a point-like geometry.

        coords is a numpy array or array-like sequence that specifies the
        Points's coordinates.
        """
        # Prefer coords formatted like (1, 2), but also accept coords
        # formatted like [(1, 2)].
        try:
            self.coords_array = self._process_array(
                coords, "coords", _3_tuple if self.is_3D else _2_tuple, True
                )
        except:
            self.coords_array = self._process_array(
                coords, "coords", _1_3_tuple if self.is_3D else _1_2_tuple,
                True
                ).ravel()
        self.tuple = tuple(self.coords_array.tolist())

    @classmethod
    def _from_wkb(cls, wkb, start=0):
        """
        Create a new instance from the specified wkb.

        wkb is a string that specifies the well-known binary representation for
        the returned instance. It must be of the correct geometric type
        (partedness, topological dimension, and spatial dimension).

        start is an integer that specifies the index at which wkb should start
        being read. If start is not 0, a tuple is returned of the form (geom,
        next_start), where geom is the new Geometry and next_start is the index
        in wkb that immediately follows that Geometry's definition.
        """
        ## Is this optimized for the special case of points?
        if start != 0:
            raise NotImplementedError("nonzero start")
        # Extract coordinates from wkb and convert them to a numpy array.
        coords_array = _numpy_fromstring(wkb[5:], _numpy_float64,
                                         3 if cls.is_3D else 2)

        # Return new instance.
        geom = cls._make_fast1(coords_array)
        geom.wkb = wkb
        return geom

    @staticmethod
    def _get_tuple(self):
        return tuple(self.coords_array.tolist())

    @staticmethod
    def _get_wkb(self):
        """
        Generate well-known binary representation for the instance.
        """
        coords_array = self.coords_array
        swap_bytes = coords_array.dtype.str[0] != "<"
        if swap_bytes:
            coords_array.byteswap(True)
        try:
            wkb = self._wkb_prefix + coords_array.tostring()
        finally:
            if swap_bytes:
                coords_array.byteswap(True)
        return wkb

    @staticmethod
    def _get_wkt(self, format2D="Point ({:.17G} {:.17G})".format,
                 format3D="Point Z ({:.17G} {:.17G} {:.17G})".format):
        """
        Generate well-known text representation for the instance.
        """
        if self.is_3D:
            return format3D(*self.tuple)
        return format2D(*self.tuple)

    @classmethod
    def _make_fast1(cls, coords_array):
        """
        Create a Point instance very quickly.

        coords_array must be flat numpy array such that
            coords_array is Point2D/3D(coords_array).coords_array --> True

        Warning: coords_array is not validated!
        """
        self = cls.__new__(cls, coords_array)
        self.coords_array = coords_array
        self.tuple = tuple(coords_array.tolist())
        return self

    @classmethod
    def _make_fast2(cls, coords_tuple):
        """
        Create a Point instance very quickly.

        coords_tuple must be flat tuple of Python float's such that
            Point2D/3D(coords_tuple)
        would not raise an error.

        Warning: coords_tuple is not validated!
        """
        self = cls.__new__(cls, coords_tuple)
        self.tuple = coords_tuple
        self.coords_array = _numpy_fromiter(coords_tuple, _numpy_float64)
        return self

    @classmethod
    def _make_fast3(cls, coords_array, coords_tuple):
        """
        Create a Point instance very quickly.

        coords_array must be flat numpy array such that
            coords_array is Point2D/3D(coords_array).coords_array --> True

        coords_tuple must be flat tuple of Python float's such that
            Point2D/3D(coords_tuple)
        would not raise an error.

        Warning: Neither coords_array nor coords_tuple is validated!
        """
        self = cls.__new__(cls, coords_array)
        self.coords_array = coords_array
        self.tuple = coords_tuple
        return self

    def calculate_distance(self, geom):
        """
        Calculate distance from self to a Point or point-like sequence.

        Note: The unbound method may be used with any sequence. For example:
            Point1 = Point2D((0, 0))
            Point2 = Point2D((3, 4))
            Point1.calculate_distance(Point2) --> 5
            Point.calculate_distance.__func__(Point1.tuple, Point2.tuple) --> 5
        """
        try:
            if len(self) == 3:
                self_x, self_y, self_z = self
                geom_x, geom_y, geom_z = geom
                dx = self_x - geom_x
                dy = self_y - geom_y
                dz = self_z - geom_z
                return _math_sqrt(dx*dx + dy*dy + dz*dz)
            self_x, self_y = self
            geom_x, geom_y = geom
            dx = self_x - geom_x
            dy = self_y - geom_y
            return _math_sqrt(dx*dx + dy*dy)
        except TypeError:
            raise TypeError(
                "geom must be a Point or other sequence with the same (implied) spatial dimension"
                )


class Point2D(_Instantiable, Point, Geometry2D):
    pass


class Point3D(_Instantiable, Point, Geometry3D):
    pass


class LineString(SingleGeometry):
    "Base class for all single-part line-like Geometry types."
    topological_dimension = 1

    # Support .flip(). .__flip_dict must include the "defining"
    # attribute (e.g., .coords_array). See .flip().
    __flip_dict = {
        None: ("data", "length"),
        lambda a: a[::-1]: ("_base_coords_array", "coords_array",
                             "segment_lengths"),
        lambda a: _numpy_negative(a, a)[::-1]: ("components",
                                                "unit_components"),
        lambda b: -b: ("_enclosed_area_signed",)
        }

    def __init__(self, coords=None):
        """
        Make an object representing a line-like geometry.

        coords is a numpy array or array-like sequence that specifies the
        LineString's coordinates.
        """
        self.coords_array = coords_array = self._process_array(
            coords, "coords", _neg_1_3_tuple if self.is_3D else _neg_1_2_tuple,
            True
            )
        self.components = components = coords_array[1:] - coords_array[:-1]
        # Note: segment_lengths is temporarily a misnomer.
        self.segment_lengths = segment_lengths = _numpy_square(components[:,0])
        y_comp_squared = _numpy_square(components[:,1])
        _numpy_add(segment_lengths, y_comp_squared, segment_lengths)
        if self.is_3D:
            _numpy_add(segment_lengths,
                       _numpy_square(components[:,2], y_comp_squared),
                       segment_lengths)
        _numpy_sqrt(segment_lengths, segment_lengths)
        if len(segment_lengths) < _OPTIMIZE_SUM_CUTOFF:
            self.length = sum(segment_lengths.tolist())
        else:
            self.length = float(segment_lengths.sum())
        self.unit_components = _numpy_divide(components,
                                             segment_lengths.reshape(-1, 1))

    def __repr__(self):
        if "ID" in self.__dict__:
            ID_str = " (ID={})".format(self.ID)
        else:
            ID_str = ""
        if "coords_array" not in self.__dict__:
            return "<{}{} at {}>".format(type(self).__name__, ID_str,
                                         hex(id(self)))
        return "<{}{}: {:n} coords, length={:n} at {}>".format(
            type(self).__name__, ID_str, len(self.coords_array), self.length,
            hex(id(self))
            )

    @classmethod
    def _fetch_flip_data(cls):
        # If flip data was already generated for this class (strictly,
        # rather than inherited), retun it immediately.
        flip_data = cls.__dict__.get("__flip_data", None)
        if flip_data is not None:
            return flip_data

        # Compiled flip dict by finding all class attribute names that
        # look like mangled .__flip_dict's and combining them.
        # Note: This is the "magic" wherein .__flip_dict's are
        # inherited.
        flip_dict = _defaultdict(list) # Initialize.
        for class_attr_name in dir(cls):
            if _flip_dict_re_match(class_attr_name):
                for k, v in getattr(cls, class_attr_name).iteritems():
                    flip_dict[k].extend(v)

        # Derve, store, and return flip data.
        flip_data = cls.__flip_data = flip_dict.items()
        return flip_data

    @staticmethod
    def _get_cumulative_lengths(self):
        return self.segment_lengths.cumsum()

    # Note: Overridden by Polygon.
    @staticmethod
    def _get_envelope_coords(self):
        """
        Get the coordinates of the bounding envelope.

        The coordinates are returned as a tuple of the form
            (min_x, min_y, [min_z], max_x, max_y, [max_z])
        """
        coords_array = self.coords_array
        if len(coords_array) >= _OPTIMIZE_EXTREME_CUTOFF:
            return tuple(
                coords_array.min(0).tolist() + coords_array.max(0).tolist()
                )
        x = coords_array[:,0].tolist()
        min_x = min(x)
        max_x = max(x)
        y = coords_array[:,1].tolist()
        min_y = min(y)
        max_y = max(y)
        if self.is_3D:
            z = coords_array[:,2].tolist()
            return (min_x, min_y, min(z), max_x, max_y, max(z))
        return (min_x, min_y, max_x, max_y)

    @staticmethod
    def _get_start_point(self, return_end=False):
        # Note: Create both start and end points, since they are
        # typically used together.
        PointXD_make_fast1 = _is_3D_to_geom_type_to_XD[self.is_3D][Point2D]._make_fast1
        start = PointXD_make_fast1(self.coords_array[0])
        end = PointXD_make_fast1(self.coords_array[-1])
        if return_end:
            self.start_point
            return end
        return start

    @staticmethod
    def _get_end_point(self):
        return self._get_start_point(self, True)

    @staticmethod
    def _get_wkb(self):
        """
        Generate well-known binary representation for the instance.
        """
        coords_array = self.coords_array
        swap_bytes = coords_array.dtype.str[0] != "<"
        if swap_bytes:
            coords_array.byteswap(True)
        try:
            wkb = (self._wkb_prefix +
                   _pack_unsigned_integer(len(coords_array)) +
                   coords_array.tostring())
        finally:
            if swap_bytes:
                coords_array.byteswap(True)
        return wkb

    @staticmethod
    def _get_wkt(self, _format_row2D = "{:.17G} {:.17G}, ".format,
                 _format_row3D = "{:.17G} {:.17G} {:.17G}, ".format):
        """
        Generate well-known text representation for the instance.
        """
        if self.is_3D:
            wkt_parts = [_format_row3D(x, y, z) for x, y, z in
                         self.coords_array.tolist()]
        else:
            wkt_parts = [_format_row2D(x, y) for x, y in
                         self.coords_array.tolist()]
        wkt_parts[0] = self._wkt_prefix + wkt_parts[0]
        wkt_parts[-1] = wkt_parts[-1][:-2] + ")"
        return _join(wkt_parts)

    def flip(self, copy=True, copy_all=True):
        """
        Flip and return a copy of self (or self itself).

        For a LineString2D (for example), the current function is similar to
            new = LineString2D(self.coords_array[::-1])
        but reduces recalculation. For example, if self.length is already
        calculated, it will be copied to new.length rather than be recalculated
        for new.

        copy is a boolean that specifies whether a new, deep-copied object
        should be returned. If copy is False, self is returned. Consider:
            self = LineString2D.make_example()
            test1 = self.flip()
            self is test1 --> True
            self.coords_array is test1.coords_array --> True
            test2 = self.flip(True)
            self is test2 --> False
            self.coords_array is test2.coords_array --> False

        copy_all is a boolean that specifies whether attributes that appear to
        be added by the user should also be copied. More precisely, any
        attribute that is not self-derived (typically ultimately from
        .coords_array) will be deep copied, including .data, if present. If copy
        is False, copy_all is ignored.
        """
        # Get the "flip data" for the class, which specifies how
        # attributes should be processed during the flip.
        # Note: This flip data is a class-level, nested list record,
        # computed once per class. It supports inheritance.
        flip_data = self._fetch_flip_data()

        # If copy_all is True (and relevant), register all non-lazy
        # attribute names in a copy of flip data.
        self_dict = self.__dict__
        if copy and copy_all:
            lazy_attr_names_set = self.__dir__(True, True)
            flip_data = flip_data[:] # *REDEFINITION*
            # Note: The new entry is left-appended so that, in case of
            # any conflict with the original flip_data, the result of
            # simply (deep) copying will be overwritten. However, such a
            # conflict is not expected, as the original flip_data should
            # only reference lazy attributes and the added entry should
            # only support non-lazy attributes.
            flip_data.insert(0,
                             (None, [attr_name for attr_name in self_dict
                                     if attr_name not in lazy_attr_names_set]))

        # Iterate over flip data, processing attributes as specified.
        restore_dict = {}
        for func, attr_names in flip_data:
            for attr_name in attr_names:
                # Only process an attribute if it already exists. (Do
                # not trigger calculation of lazy attributes.)
                if attr_name in self_dict:
                    attr_val = self_dict[attr_name]
                    if copy:
                        # Note: Deep copying is supported by arrays and
                        # dict's (e.g., .data), among other types.
                        # *REDEFINITION*
                        attr_val = _deepcopy(self_dict[attr_name])
                    # Interpret a func that is None to indicate that
                    # nothing should be done to process the attribute
                    # value.
                    if func is not None:
                        restore_dict[attr_name] = func(attr_val)
                        continue
                    restore_dict[attr_name] = attr_val
                    continue

        # Create new object or clear self, as appropriate. Then restore
        # processed attributes to the salient object and return it.
        if copy:
            # Note: new would be corrupted if it doesn't include the
            # defining attribute for the type, usually .coords_array.
            # For this reason, such a defining attribute must always be
            # registered in .__flip_dict for the class.
            new = object.__new__(type(self))
            new.__dict__.update(restore_dict)
            return new
        self._Lazy__clear_lazy()
        self_dict.update(restore_dict)
        return self

    @classmethod
    def _from_wkb(cls, wkb, start=0):
        """
        Create a new instance from the specified wkb.

        wkb is a string that specifies the well-known binary representation for
        the returned instance. It must be of the correct geometric type
        (partedness, topological dimension, and spatial dimension).

        start is an integer that specifies the index at which wkb should start
        being read. If start is not 0, a tuple is returned of the form (geom,
        next_start), where geom is the new Geometry and next_start is the index
        in wkb that immediately follows that Geometry's definition.
        """
        # Extract coordinates from wkb and convert them to a numpy array.
        coord_count, = _unpack_unsigned_integer(wkb[start+5:start+9])
        if cls.is_3D:
            values_per_coord = 3
            coords_array_shape = _neg_1_3_tuple
        else:
            values_per_coord = 2
            coords_array_shape = _neg_1_2_tuple
        coord_start_idx = start + 9
        value_count = values_per_coord * coord_count
        coords_array = _numpy_fromstring(wkb[coord_start_idx:], _numpy_float64,
                                         value_count)
        coords_array.shape = coords_array_shape

        # Return new instance.
        geom = cls(coords_array)
        # Note: If a start was specified, the wkb string cannot simply
        # be assigned to geom.wkb. Instead, allow it to be re-created
        # lazily. This also addresses the issue that, when a start is
        # specified, wkb may not be consumed to its end.
        if start:
            return (geom, coord_start_idx + value_count*8)
        geom.wkb = wkb
        return geom

    @classmethod
    def _from_wkt(cls, wkt, idx0=12, idxN=-1):
        """
        Create a new LineString from the specified wkt.

        wkt is a string that specifies the well-known text representation for
        the returned instance, or at least all numeric values therein (e.g., the
        portion of a Polygon's wkt representing a hole).

        idx0 is an integer that specifies the index at which wkt should start
        being read so as to skip any leading labels (e.g., "LineString") and
        parentheses. More specifically, wkt[idx0:idxN] should contain only
        numbers (including decimal points), commas (which are ignored), and
        spaces.

        idxN is an integer that specifies the index at which wkt should stop
        being read so as to avoid any trailing parentheses.

        Note: In practice, idxN is only -1 if wkt represents the complete wkt
        for a LineString. In that case only:
            1) wkt is assigned to .wkt of the returned instance
            2) if cls is 3D, idx0 is incremented by 2 to account for the " Z"
               that trails the label (i.e., in "LineString Z").
        """
        # Note: The approach below was faster than regex approaches in
        # testing.
        if idxN == -1 and cls.is_3D:
            idx0 += 2 # *REDEFINITION*
        coords_array = _numpy_fromiter(
            _imap(float, wkt[idx0:idxN].replace(",", "").split()),
            _numpy_float64
            )
        if cls.is_3D:
            coords_array.shape = _neg_1_3_tuple
        else:
            coords_array.shape = _neg_1_2_tuple
        geom = cls(coords_array)
        if idxN == -1:
            geom.wkt = wkt
        return geom

    def interpolate(self, distances, normalized=False, copy=True,
                    segment_indices=False):
        """
        Interpolate coordinates at specified distances along the length.

        The interpolated coordinates are returned as a numpy array in the same
        order as specified by distances.

        distances is a numpy array or similar sequence of distances represented
        as numbers.

        normalized is a boolean that specifies whether distances is normalized
        to the length. For example, a normalized length of 0.5 would be
        interpolated at the midpoint.

        copy is a boolean that specifies whether a copy of distances should be
        made internally if any modification of its values is required. If copy
        is False, its values and shape may be modified in place.

        segment_indices is a boolean that specifies whether to also return a
        (flat) array storing the index of the segment on which each coordinate
        was interpolated. Because this array is an internal byproduct, it
        requires no further processing. If segment_indices is True, a tuple is
        returned of the form (interpolated_coords, segment_indices). For
        example, if segment_indices[0] is n, interpolated_coords[0] is located
        between the vertices .coords_array[n] and .coords_array[n+1], that is,
        on the nth segment.

        See also: sample_regularly().
        """
        # Convert distances to a numpy array, if necessary.
        distances_array = self._process_array(distances, "distances")
        if copy and distances_array is not distances:
            copy = False # *REDEFINITION*

        # Validate and de-normalize, as necessary, the values from
        # distances. If this requires copying that array, turn off
        # future copying.
        if distances_array.min() < 0.:
            raise TypeError("no distance can be negative")
        length = self.length
        if normalized:
            if distances_array.max() > 1.:
                raise TypeError("no distance can be > 1 if normalized is True")
            # *REDEFINITION*
            distances_array = _numpy_multiply(distances_array, length,
                                              None if copy else distances_array)
            copy = False # *REDEFINITION*
        else:
            if distances_array.max() > length:
                raise TypeError(
                    "no distance can be > .length if normalized is False"
                    )

        # Identify the segment index for each distance and the
        # components (x, y, and possibly z) that must be added to that
        # segment's end vertex to interpolate back along that segment to
        # the desired cumulative distance.
        cumulative_lengths = self.cumulative_lengths
        seg_idxs = cumulative_lengths.searchsorted(distances_array)
        # *REDEFINITION*
        distances_array = _numpy_subtract(distances_array,
                                          cumulative_lengths[seg_idxs],
                                          None if copy else distances_array)
        distances_array.shape = _neg_1_1_tuple
        # Note: This is the only place in this module where
        # unit_components is presently used. Because unit_components
        # will contain infinity for any 0-length segment, it must be
        # handled with care. seg_idxs, generated furhter above, can only
        # point to a 0-length segment if that segment lies exactly at
        # the desired distance and is the first segment whose cumulative
        # length is that distance. However, because 0-length segments
        # have no length, a 0-length segment could only be targeted if
        # it is the first segment and a distance of 0 were targeted.
        ## This rare case should ultimately be resolved. One option: use
        ## numpy.seterr() to catch 0-length segments when 
        ## .unit_components is created and change from inf to 0, and 
        ## perhaps move this slightly more expensive operation to lazy 
        ## calculation. Although that may be the better general 
        ## solution, also consider case where user expects certain
        ## numpy.seterr() behavior.
        residual_segment_components = (self.unit_components[seg_idxs] *
                                       distances_array)

        # Add the components and segment end vertices just identified,
        # and return the result(s).
        interpolated_coords = _numpy_add(_take2(self.coords_array[1:],
                                                seg_idxs),
                                         residual_segment_components,
                                         residual_segment_components)
        if segment_indices:
            return (interpolated_coords, seg_idxs)
        return interpolated_coords


    def sample_regularly(self, interval, normalized=False, vertices=False,
                         segment_indices=False):
        """
        Sample coordinates at a regular interval along the length.

        The sampled coordinates are returned as a numpy array, and their order
        is as though all coordinates were sampled as the length was walked in a
        single pass (even if vertices is True). The starting vertex is always
        the first returned coordinate.

        interval is a float that specifies the length interval at which
        coordinates will be sampled along the length.

        normalized is a boolean that specifies whether interval is normalized to
        the length. For example, a normalized length of 0.4 would be sampled at
        0%, 40%, and 80% of the length (if vertices is False).

        vertices is a boolean that specifies whether the existing vertices
        should be included in the returned coordinates.

        segment_indices is a boolean that specifies whether to also return a
        (flat) array storing the index of the segment on which each coordinate
        was interpolated. Because this array is an internal byproduct, it
        requires no further processing. If segment_indices is True, a tuple is
        returned of the form (interpolated_coords, segment_indices). For
        example, if segment_indices[0] is n, interpolated_coords[0] is located
        between the vertices .coords_array[n] and .coords_array[n+1], that is,
        on the nth segment.

        See also: interpolate().
        """
        length = self.length
        if normalized:
            interval = interval * length # *REDEFINITION*
        if vertices:
            distances = _numpy_concatenate((_numpy_arange(0., length, interval),
                                            self.cumulative_lengths))
            distances.sort()
        else:
            distances = _numpy_arange(0., length, interval)
        return self.interpolate(distances, False, False, segment_indices)


class LineString2D(_Instantiable, LineString, Geometry2D):

    @staticmethod
    def _get_base_coords_array(self):
        base_coords_array = self.coords_array.copy()
        MultiPoint2D.make_fast(base_coords_array, False).translate(
            base_coords_array.mean(0).tolist(), True
            )
        return base_coords_array

    @staticmethod
    def _get__enclosed_area_signed_addends(self):
        self._get__enclosed_area(self)
        return self._enclosed_area_signed_addends

    @staticmethod
    def _get__enclosed_area_signed(self):
        base_coords_array = self.base_coords_array
        x = base_coords_array[:,0]
        y = base_coords_array[:,1]
        # Store _enclosed_area_signed_addends, which may be used later
        # for finding the centroid. (Note:
        # self._enclosed_area_signed_addends is temporarily a misnomer.)
        self._enclosed_area_signed_addends = enclosed_area_signed_addends = _numpy.multiply(
            x[1:], y[:-1]
            )
        _numpy_subtract(_numpy_multiply(x[:-1], y[1:]),
                        enclosed_area_signed_addends,
                        enclosed_area_signed_addends)
        if len(enclosed_area_signed_addends) < _OPTIMIZE_SUM_CUTOFF:
            return 0.5 * sum(enclosed_area_signed_addends.tolist())
        return 0.5 * float(enclosed_area_signed_addends.sum())

    @staticmethod
    def _get__encloses_point_data(self):
        """
        Generate arrays that are (re)used by ._encloses_point().
        """
        components = self.components
        dx = components[:,0]
        dy = components[:,1]
        coords_array = self.coords_array
        y = coords_array[:,1]
        seg_count = len(dx)
        # Note: Although the first array in the returned tuple will
        # contain inf's for horizontal segments, these are
        # invariably excluded when that array is subset in
        # ._encloses_point().
        return (_util._force_divide(dx, dy),
                 y[:-1], y[1:], coords_array[:,0][:-1],
                 _numpy_empty((seg_count,), _numpy_bool8),
                 _numpy_empty((seg_count,), _numpy_bool8))

    def _encloses_point(self, point):
        """
        Test whether the instance (assumed to be closed) encloses point.

        point is a Point2D, 2-tuple, or similar that specified the (x, y)
        coordinates.

        Warning: If point lies exactly on a segment of the instance,
        either True or False may be returned.

        Credit: This function's code is based on the algorithm PNPOLY, which has
        the following license:

        Copyright (c) 1970-2003, Wm. Randolph Franklin

        Permission is hereby granted, free of charge, to any person obtaining a
        copy of this software and associated documentation files (the
        "Software"), to deal in the Software without restriction, including
        without limitation the rights to use, copy, modify, merge, publish,
        distribute, sublicense, and/or sell copies of the Software, and to
        permit persons to whom the Software is furnished to do so, subject to
        the following conditions:

        Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimers. Redistributions
        in binary form must reproduce the above copyright notice in the
        documentation and/or other materials provided with the distribution.
        The name of W. Randolph Franklin may not be used to endorse or promote
        products derived from this Software without specific prior written permission.
        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
        OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
        MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
        IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
        CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
        TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
        SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
        """
        # Extract point's x- and y-coordinates.
        try:
            point_x, point_y = point
        except:
            raise TypeError("point must be a Point2D, 2-tuple, or similar")

        # Get predefined arrays, including boolean workspaces.
        # Note: Storing these arrays for reuse makes the second and
        # later calls of the current function faster. Leveraging a lazy
        # attribute for this storage additionally permits the user to
        # reduce the memory footprint, if necessary, without loss of
        # functionality.
        dx_div_dy, y0, y1, x0, b1, b2 = self._encloses_point_data

        # Identify each segment for which the point lies in the
        # segment's y-range.
        _numpy_greater(y0, point_y, b1)
        _numpy_greater(y1, point_y, b2)
        _numpy_not_equal(b1, b2, b1)

        # Subset those predefind arrays that will be used further below
        # to include only the identified segments.
        # Note: Because multiple arrays must be sliced, it is faster to
        # generate indexes than to use a boolean mask directly.
        idxs, = b1.nonzero()
        # Note: _util._take2 was not typically faster in testing,
        # perhaps because len(idxs) is typically small.
        dx_div_dy_subset = dx_div_dy[idxs]
        y0_subset = y0[idxs]
        x0_subset = x0[idxs]
        # Note: Because b2(_subset) is only a boolean workspace, it need
        # only have an appropriate shape rather than subset like the
        # other data-specific arrays.
        subset_count = len(y0_subset)
        b2_subset = b2[:subset_count]

        # Count the number of times that a semi-infinite ray,
        # originating at point and extending horizontally in the +x
        # direction, crosses an identified segment. Return a boolean
        # indicating whether this happens an odd number of times, in
        # which case the point is enclosed.
        point_y_delta = _numpy_subtract(point_y, y0_subset, y0_subset)
        point_y_delta_times_dx_div_dy = _numpy_multiply(
            dx_div_dy_subset, point_y_delta, point_y_delta
            )
        test = _numpy_add(point_y_delta_times_dx_div_dy, x0_subset,
                          point_y_delta_times_dx_div_dy)
        crosses = _numpy_greater(test, point_x, b2_subset)
        if subset_count < _OPTIMIZE_SUM_CUTOFF:
            return bool(sum(crosses.tolist()) % 2)
        return bool(crosses.sum() % 2)


    def _optimize(self):
        """
        Optimize for enclosement tests.
        """
        # Avoid re-optimizing.
        if self._encloses_point is self._encloses_point_optimized:
            return

        # Define some useful arrays.
        coords_array = self.coords_array
        x = coords_array[:,0]
        x0 = x[:-1]
        x1 = x[1:]
        coords_array = self.coords_array
        y = coords_array[:,1]
        y0 = y[:-1]
        y1 = y[1:]
        dx = x1 - x0
        dy = y1 - y0
        quot = _util._force_divide(dx, dy, dx)

        # Create and populate (unsorted) optimization array.
        x0_y0_quot_ymax = _numpy_empty((len(x0), 4), _numpy_float64)
        x0_y0_quot_ymax[:,0] = x0
        x0_y0_quot_ymax[:,1] = y0
        x0_y0_quot_ymax[:,2] = quot
        y0_y1 = _util._slide_flat(y)
        y0_y1.max(1, out=x0_y0_quot_ymax[:,3])

        # Sort optimization array.
        ymin = y0_y1.min(1, out=quot)
        ymin_sort_idxs = ymin.argsort()
        x0_y0_quot_ymax_sorted = _take2(x0_y0_quot_ymax, ymin_sort_idxs)
        del x0_y0_quot_ymax  # Release memory.

        # Support useful search functions.
        ymin_sorted = _take2(ymin, ymin_sort_idxs, out=dy)
        ymax_cum_max = _numpy.maximum.accumulate(x0_y0_quot_ymax_sorted[:,3],
                                                 out=ymin)

        # Set attributes.
        self._optimized_encloses_point_data = (
            x0_y0_quot_ymax_sorted,
            ymax_cum_max.searchsorted, ymin_sorted.searchsorted
            )
        self._encloses_point_unoptimized = self._encloses_point
        self._encloses_point = self._encloses_point_optimized

    def _encloses_point_optimized(self, point):
        # Extract point's x- and y-coordinates.
        try:
            point_x, point_y = point
        except:
            raise TypeError("point must be a Point2D, 2-tuple, or similar")

        # self cannot contain point if point lies outside self's
        # envelope.
        min_x, min_y, max_x, max_y = self.envelope_coords
        if (point_x < min_x or point_x > max_x or
            point_y < min_y or point_y > max_y):
            return False

        # Get predefined arrays.
        try:
            (x0_y0_quot_ymax,
             search_for_idx0,
             search_for_idxN) = self._optimized_encloses_point_data
        except AttributeError:
            # User has (presumably) called .minimize_memory(), so resort
            # to un-optimized form for this and future calls.
            self._encloses_point = self._encloses_point_unoptimized
            return self._encloses_point(point)

        # Find indices to slice.
        # Note: The ability to slice, rather than test the full arrays,
        # is the main benefit of optimization.
        idx0 = search_for_idx0(point_y, "right")
        idxN = search_for_idxN(point_y, "right")

        # Test by PNPOLY.
        # Note: See LineString2D._encloses_point() for details on
        # PNPOLY.
        # Note: In typical cases, the number of loops are so few that
        # standard Python (as opposed to numpy operations) are
        # sufficient (and probably faster).
        inside = False # Initialize.
        for x0, y0, dx_div_dy, ymax in x0_y0_quot_ymax[idx0:idxN].tolist():
            # Note: Because x0_y0_quot_ymax is sorted by increasing
            # ymin, every record in the slice is known to have a minimum
            # y <= point_y.
            if ymax > point_y and point_x < dx_div_dy*(point_y - y0) + x0:
                inside = not inside
        return inside


class LineString3D(_Instantiable, LineString, Geometry3D):
    pass


class Polygon(SingleGeometry):
    "Base class for all single-part polygon-like Geometry types."
    topological_dimension = 2

    def _process_boundary(self, boundary, init=True):
        # If called from within .__init__() and boundary is unspecified,
        # attempt to find a stored substitute.
        if init and boundary is None:
            # If boundary is not specified, resort to using .boundary
            # (for re-initialization) or ._arg0 (for delayed
            # initialization).
            # Note: Although similar, re- and delayed initialization
            # must use different attribute names, so that a processed
            # .boundary can be distinguished from an ._arg0 that may yet
            # require processing.
            # Note: No manual ._get_boundary() can be defined, for the
            # same reason that no manual ._get_coords_array() can be
            # defined. (See note in Geometry._process_array().)
            # *REDEFINITION*
            boundary = self.__dict__.get("boundary", None)
            if boundary is None:
                try:
                    boundary, process = self._arg0 # *REDEFINITION*
                except AttributeError:
                    raise TypeError("boundary cannot be None")
                if not process:
                    return boundary

        # Test that boundary is a non-empty sequence of LineString's of
        # the same spatial dimension.
        try:
            boundary = tuple(boundary) # *REDEFINITION*
            self_is_3D = self.is_3D
            for ring in boundary:
                assert ring.is_3D == self_is_3D
                assert isinstance(ring, LineString)
        except:
            raise TypeError(
                "boundary must be a non-empty sequence of LineString's of the same spatial dimension"
                )
        return boundary


    def __init__(self, boundary):
        """
        boundary is a sequence of LineString's that specifies the instance's
        boundary. It cannot be empty, and all LineString's must have the same
        spatial dimension as the current class.
        """
        self.boundary = self._process_boundary(boundary)

    def __repr__(self):
        if "ID" in self.__dict__:
            ID_str = " (ID={})".format(self.ID)
        else:
            ID_str = ""
        if "boundary" not in self.__dict__:
            return "<{}{} at {}>".format(type(self).__name__, ID_str,
                                         hex(id(self)))
        return "<{}{}: {:n} coords, {:n} ring{} at {}>".format(
            type(self).__name__, ID_str,
            self.vertex_count,
            len(self.boundary), "s" if len(self.boundary) != 1 else "",
            hex(id(self))
            )
        ## When .area is implemented, probably add to representation.

    @staticmethod
    def _get_area(self):
        return sum([abs(ring._enclosed_area_signed) for ring in self.boundary])

    @staticmethod
    def _get_envelope_coords(self):
        return self.boundary[0].envelope_coords
    _get_envelope_coords.__func__.__doc__ = LineString._get_envelope_coords.__doc__

    @staticmethod
    def _get_length(self):
        return sum([ring.length for ring in self.boundary])

    @staticmethod
    def _get_wkb(self):
        """
        Generate well-known binary representation for the instance.
        """
        boundary = self.boundary
        wkb_parts = [self._wkb_prefix, _pack_unsigned_integer(len(boundary))]
        swap_bytes = boundary[0].coords_array.dtype.str[0] != "<"
        for ring in boundary:
            coords_array = ring.coords_array
            wkb_parts.append(_pack_unsigned_integer(len(coords_array)))
            if swap_bytes:
                coords_array.byteswap(True)
            try:
                wkb_parts.append(coords_array.tostring())
            finally:
                if swap_bytes:
                    coords_array.byteswap(True)
        return _join(wkb_parts)

    @staticmethod
    def _get_wkt(self):
        """
        Generate well-known text representation for the instance.
        """
        # Note: This code is very similar to that used for
        # MultiGeometry._get_wkt().
        ring_wkt_trim_length = 11 + 2*self.is_3D
        wkt_parts = [ring.wkt[ring_wkt_trim_length:] for ring in self.boundary]
        wkt_parts[0] = self._wkt_prefix + wkt_parts[0]
        wkt_parts[-1] += ")"
        return _comma_space_join(wkt_parts)

    @classmethod
    def _from_wkb(cls, wkb, start=0):
        """
        Create a new instance from the specified wkb.

        wkb is a string that specifies the well-known binary representation for
        the returned instance. It must be of the correct geometric type
        (partedness, topological dimension, and spatial dimension).

        start is an integer that specifies the index at which wkb should start
        being read. If start is not 0, a tuple is returned of the form (geom,
        next_start), where geom is the new Geometry and next_start is the index
        in wkb that immediately follows that Geometry's definition. (Used
        internally.)
        """
        # Create a LineString for each boundary component ("ring").
        ring_count, = _unpack_unsigned_integer(wkb[start+5:start+9])
        # Note: Skip byte order and type (both of which were vetted
        # further above), and also skip ring count (which was read
        # above).
        processed = start + 9
        if cls.is_3D:
            bytes_per_coord = 24
            values_per_coord = 3
            coords_array_shape = _neg_1_3_tuple
            ring_geom_type = LineString3D
        else:
            bytes_per_coord = 16
            values_per_coord = 2
            coords_array_shape = _neg_1_2_tuple
            ring_geom_type = LineString2D
        boundary = []
        boundary_append = boundary.append
        # Process only the number of rings specified.
        for _ in xrange(ring_count):
            # Extract coordinates and convert them to a numpy array.
            processed_plus_4 = processed + 4
            ring_coords_count, = _unpack_unsigned_integer(
                wkb[processed:processed_plus_4]
                )
            # Increment processed to the next ring.
            # *REDEFINITION*
            processed = processed_plus_4 + ring_coords_count*bytes_per_coord
            ring_coords = _numpy_fromstring(
                wkb[processed_plus_4:processed], _numpy_float64,
                ring_coords_count * values_per_coord
                )
            ring_coords.shape = coords_array_shape
            # Create LineString2D for the ring and add it to the growing
            # list boundary.
            boundary_append(ring_geom_type(ring_coords))

        # Return new instance.
        geom = cls(boundary)
        # Note: If a start was specified, the wkb string cannot simply
        # be assigned to geom.wkb. Instead, allow it to be re-created
        # lazily. This also addresses the issue that, when a start is
        # specified, wkb may not be consumed to its end.
        if start:
            return (geom, processed)
        geom.wkb = wkb
        return geom

    @classmethod
    def _from_wkt(cls, wkt, first_idx0=10, last_idxN=-2, return_rings=False):
        """
        Create a new Polygon or sequence of LineString's from the specified wkt.

        wkt is a string that specifies the well-known text representation for
        the returned instance(s), or at least all numeric values therein (e.g.,
        the portion of a MultiPolygon's wkt representing a single Polygon).

        first_idx0 is an integer that specifies the index at which the first wkt
        part (after the wkt is split by "), (") should start being read so as to
        skip any leading labels (e.g., "Polygon") and parentheses. More
        specifically:
            1) wkt.split("), (")[0][first_idx0:],
            2) wkt.split("), (")[1],
            ...
            3) wkt.split("), (")[-2], and
            4) wkt.split("), (")[-1][:last_idxN]
        should each contain only numbers (including decimal points), commas
        (which are ignored), and spaces. In the special case that "), (" does
        not occur in wkt (e.g., for a single-part Polygon), this requirement
        instead applies to wkt[first_idx0:last_idxN].

        last_idxN is an integer that specifies the index at which the last wkt
        part should stop being read so as to avoid any trailing parentheses.

        return_rings is a boolean that specifies whether a sequence of
        LineString's should be returned instead of a Polygon (for which that
        sequence would represent the rings).

        Note: In practice, last_idxN is only -2 if wkt represents the complete
        wkt for a Polygon. In that case only, and only if return_rings is False,
        wkt is assigned to .wkt of the returned instance. Also, in practice,
        first_idx0 is only nonzero if a leading label (namely "Polygon" or "MultiLineString") is present. In
        that case only, and only if cls is 3D, first_idx0 is incremented by 2 to
        account for the " Z" that trails the label (e.g., in "Polygon Z").
        """
        if cls.is_3D:
            ring_from_wkt = LineString3D._from_wkt
            if first_idx0:
                first_idx0 += 2
        else:
            ring_from_wkt = LineString2D._from_wkt
        wkt_parts = wkt.split("), (")
        last_wkt_part = wkt_parts.pop()
        if wkt_parts:
            wkt_parts_iter = iter(wkt_parts)
            rings = [ring_from_wkt(next(wkt_parts_iter), first_idx0, None)]
            for wkt_part in wkt_parts_iter:
                rings.append(ring_from_wkt(wkt_part, 0, None))
            rings.append(ring_from_wkt(last_wkt_part, 0, last_idxN))
        else:
            rings = (ring_from_wkt(last_wkt_part, first_idx0, last_idxN),)
        if return_rings:
            return rings
        poly = cls(rings)
        if last_idxN == -2:
            poly.wkt = wkt
        return poly

    @classmethod
    def make_example(cls):
        """
        Create an arbitrary instance of the current type.
        """
        if cls.is_3D:
            return cls((LineString3D(_unit_square_coords_array3D),))
        return cls((LineString2D(_unit_square_coords_array2D),))


class Polygon2D(_Instantiable, Polygon, Geometry2D):

    def _contains_point(self, point):
        # Extract point's x- and y-coordinates.
        try:
            point_x, point_y = point
        except:
            raise TypeError("point must be a Point2D, 2-tuple, or similar")

        # self cannot contain point if self's exterior does not enclose
        # point.
        boundary = self.boundary
        if not boundary[0]._encloses_point(point):
            return False

        # self does not contain point if any of self's holes enclose
        # point.
        for ring in boundary[1:]:
            if ring._encloses_point(point):
                return False
        return True

    def optimize(self):
        """
        Optimize Polygon for containment tests.
        """
        for ring in self.boundary:
            ring._optimize()

    @staticmethod
    def _get_vertex_count(self):
        return sum([len(ring.coords_array) for ring in self.boundary])


class Polygon3D(_Instantiable, Polygon, Geometry3D):

    @staticmethod
    def _get__2D(self):
        """
        Make corresponding Polygon2D.
        """
        return Polygon2D([ring._2D for ring in self.boundary])



###############################################################################
# MULTI-GEOMETRY CLASSES                                                      #
###############################################################################

class MultiGeometry(Geometry, list):

    def __init__(self, geoms, test=True):
        """
        Make a list-like object representing a collection of SingleGeometry's.

        geoms is an iterable of SingleGeometry's. Each object in geoms must be
        of the correct geometric type (topological and spatial dimensions).

        test is a boolean that specifies whether it should be tested that each
        object in geoms is the correct geometric type.
        """
        list.__init__(self, geoms)
        if test:
            self._test(self)

    def __add__(self, other, add=list.__add__):
        if isinstance(other, type(self)):
            return type(self)(tuple(self) + tuple(other), False)
        raise TypeError(
            'can only concatenate {0} (not "{1}") to {0}'.format(
                type(self).__name__, type(other).__name__
                )
            )

    def __iadd__(self, other, iadd=list.__iadd__):
        # Note: This is equivalent to .extend().
        iadd(self, self._test(other))
        self._Lazy__clear_lazy()

    def __imul__(self, other):
        raise TypeError("unsupported operand: *=")

    def __mul__(self, other):
        raise TypeError("unsupported operand: *")

    def __radd__(self, other):
        if isinstance(other, type(self)):
            return type(self)(tuple(other) + tuple(self), False)
        raise TypeError(
            'can only concatenate {0} (not "{1}") to {0}'.format(
                type(self).__name__, type(other).__name__
                )
            )

    def __rmul__(self, other):
        raise TypeError("unsupported operand: *")

    def __delitem__(self, index_or_slice, delitem=list.__delitem__):
        delitem(self, index_or_slice)
        self._Lazy__clear_lazy()

    def __delslice__(self, i, j, delslice=list.__delslice__):
        delslice(self, i, j)
        self._Lazy__clear_lazy()

    def __getitem__(self, index_or_slice, getitem=list.__getitem__):
        if isinstance(index_or_slice, slice):
            return type(self)(getitem(self, index_or_slice), False)
        return getitem(self, index_or_slice)

    def __getslice__(self, i, j, getslice=list.__getslice__):
        return type(self)(getslice(self, i, j), False)

    def __repr__(self, list_repr=list.__repr__):
        return "{}({})".format(type(self).__name__, list_repr(self))

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name == "spatial_reference":
            for geom in self:
                geom.spatial_reference = value

    def __setitem__(self, index_or_slice, geom, setitem=list.__setitem__):
        if isinstance(index_or_slice, slice):
            setitem(self, index_or_slice, self._test(geom))
        else:
            self._test((geom,))
            setitem(self, index_or_slice, geom)
        self._Lazy__clear_lazy()

    def __setslice__(self, i, j, sequence, setslice=list.__setslice__):
        setslice(self, i, j, self._test(sequence))
        self._Lazy__clear_lazy()

    @staticmethod
    def _get_ewkb(self):
        """
        Generate "extended" well-known binary representation for the instance.

        Extended WKBs are a PostGIS convention and differ from the ISO standard
        for Geometry3D's (but not for Geometry2D's).
        """
        if self.is_3D:
            ewkb_parts = [geom.ewkb for geom in self]
            ewkb_parts[0] = (self._ewkb_prefix +
                             _pack_unsigned_integer(len(self)) +
                             ewkb_parts[0])
            return _join(ewkb_parts)
        return self.wkb

    @staticmethod
    def _get_wkb(self):
        wkb_parts = [geom.wkb for geom in self]
        wkb_parts[0] = (self._wkb_prefix + _pack_unsigned_integer(len(self)) +
                        wkb_parts[0])
        return _join(wkb_parts)

    @staticmethod
    def _get_wkt(self):
        """
        Generate well-known text representation for the instance.
        """
        # Note: This code is very similar to that used for
        # Polygon._get_wkt().
        member_wkt_trim_length = self._member_wkt_trim_length
        wkt_parts = [geom.wkt[member_wkt_trim_length:] for geom in self]
        wkt_parts[0] = self._wkt_prefix + wkt_parts[0]
        wkt_parts[-1] += ")"
        return _comma_space_join(wkt_parts)

    @classmethod
    def _test(cls, geoms):
        # Note: Convert geoms to tuple, in case geoms is not a
        # container.
        geoms = tuple(geoms) # *REDEFINITION*
        member_wkb_type = cls.member_type.wkb_type
        try:
            for geom in geoms:
                assert geom.wkb_type == member_wkb_type
        except (AssertionError, AttributeError):
            raise TypeError(
                "members must have {}'s (not {})".format(
                    _wkb_type_to_geom_type[member_wkb_type].__name__,
                    type(geom).__name__
                    )
                )
        return geoms

    def append(self, geom, test=True):
        if test:
            self._test((geom,))
        list.append(self, geom)
        self._Lazy__clear_lazy()

    def _cascade(self, ID=False, data=True, attrs=False):
        ## Note: Keep this "private", for now, but consider whether
        ## exposing it publically in the future might be worthwhile.
        if attrs:
            # Note: The only attribute names considered safe to assign
            # to members are those that are neither lazily generated by
            # the current type or its member type.
            safe_names_set = set(self.__dict__)
            safe_names_set -= self.__dir__(True, True)
            safe_names_set -= self.member_type.__dir__.__func__(
                self.member_type, True, True
                )
            if safe_names_set:
                # Note: Individually assigning ID would be redundant.
                ID = False
                safe_name_to_attr = {name: attr for name, attr in
                                     self.__dict__.iteritems()
                                     if name in safe_names_set}
            else:
                attrs = False # *REDEFINITION*
        assign_ID = ID and "ID"in self.__dict__
        if assign_ID:
            ID = self.ID # *REDEFINITION*
        assign_data = data and "data" in self.__dict__
        if assign_data:
            data = self.data # *REDEFINITION*
        for geom in self:
            if assign_ID:
                geom.ID = ID
            if assign_data:
                geom.data.update(data)
            if attrs:
                geom.__dict__.update(safe_name_to_attr)

    def extend(self, geoms, test=True):
        if test:
            list.extend(self, self._test(geoms))
        else:
            list.extend(self, geoms)
        self._Lazy__clear_lazy()

    @classmethod
    def _from_wkb(cls, wkb, start=0):
        """
        Create a new instance from the specified wkb.

        wkb is a string that specifies the well-known binary representation for
        the returned instance. It must be of the correct geometric type
        (partedness, topological dimension, and spatial dimension).

        start is an integer that specifies the index at which wkb should start
        being read. If start is not 0, a tuple is returned of the form (geom,
        next_start), where geom is the new Geometry and next_start is the index
        in wkb that immediately follows that Geometry's definition. (Used
        internally.)
        """
        # Create members and populate an instance of the current class.
        member_count, = _unpack_unsigned_integer(wkb[start+5:start+9])
        member_from_wkb = cls.member_type._from_wkb
        next_start = start + 9
        members = []
        members_append = members.append
        for _ in xrange(member_count):
            member, next_start = member_from_wkb(wkb, next_start)
            members_append(member)

        # Return new instance.
        multi = cls(members, False)
        # Note: If a start was specified, the wkb string cannot simply
        # be assigned to multi.wkb. Instead, allow it to be re-created
        # lazily. This also addresses the issue that, when a start is
        # specified, wkb may not be consumed to its end.
        if start:
            return (multi, next_start)
        multi.wkb = wkb
        return multi

    def insert(self, index, geom, test=True):
        if test:
            self._test((geom,))
        list.insert(self, index, geom)
        self._Lazy__clear_lazy()

    ## Note: It would probably be best to add some rotation and
    ## translation options to this function.
    @classmethod
    def make_example(cls):
        """
        Create an arbitrary instance of the current type.
        """
        return cls((cls.member_type.make_example(),))

    def pop(self, index=-1):
        list.pop(self, index)
        self._Lazy__clear_lazy()

    def remove(self, geom):
        list.remove(self, geom)
        self._Lazy__clear_lazy()

    def reverse(self):
        list.reverse(self)
        self._Lazy__clear_lazy()

    def sort(self, cmp=None, key=None, reverse=False):
        list.sort(self, cmp, key, reverse)
        self._Lazy__clear_lazy()

    def test(self):
        self._test(self)
        return True


class MultiPoint(MultiGeometry):
    "Base class for all multi-part point-like Geometry types."
    topological_dimension = 0
    _not_initialized = False # This is the default.

    def __init__(self, geoms, test=True):
        """
        Make a list-like object representing a collection of Point's.

        geoms is a numpy array, array-like sequence, or iterable of Point's.
        Each row or object in geoms must have the correct spatial dimension.

        test is a boolean that specifies whether it should be tested that each
        object in geoms is the correct geometric type (i.e., has the correct
        topological and spatial dimensions). If geoms is not an iterable of
        Point's, it is always tested (i.e., test is ignored).
        """
        if not isinstance(geoms, (_numpy_ndarray, _collections.Sequence)):
            geoms = tuple(geoms) # *REDEFINITION*
        try:
            assert (not geoms or isinstance(geoms[0], Point))
        except (AssertionError, ValueError):
            self.coords_array = self._process_array(
                geoms, "geoms",
                _neg_1_3_tuple if self.is_3D else _neg_1_2_tuple, True
                )
            self._not_initialized = True
        else:
            list.__init__(self, geoms)
            if test:
                self._test(self)

    # Note: The behavior of the current class is special cased in some
    # places to use numpy arrays rather than generate Point's. However,
    # this behavior is only applied in the narrowest cases to ensure
    # that user expectations are not violated. For example, if a Point2D
    # from a slice of a MultiPoint2D is modified (e.g., some key of
    # .data is set), that modification should also be reflected in the
    # corresponding Point2D in the original MultiPoint2D, which cannot
    # (easily) be honored unless Point2D's are generated upon slicing.
    # More generally, those two Point2D's should evaluate as identical
    # (not merely equivalent).

    def __add__(self, other, add=MultiGeometry.__add__):
        if self._not_initialized:
            self._init()
        return add(self, other)

    def __iadd__(self, other, iadd=MultiGeometry.__iadd__):
        if self._not_initialized:
            self._init()
        return iadd(self, other)

    def __contains__(self, other, contains=list.__contains__):
        if self._not_initialized:
            self._init()
        return contains(self, other)

    def __delitem__(self, index_or_slice, delitem=MultiGeometry.__delitem__):
        if self._not_initialized:
            self._init()
        return delitem(self, index_or_slice)

    def __delslice__(self, i, j, delslice=MultiGeometry.__delslice__):
        if self._not_initialized:
            self._init()
        return delslice(self, i, j)

    def __getitem__(self, index_or_slice, getitem=MultiGeometry.__getitem__):
        if self._not_initialized:
            self._init()
        return getitem(self, index_or_slice)

    def __getslice__(self, i, j, getslice=MultiGeometry.__getslice__):
        if self._not_initialized:
            self._init()
        return getslice(self, i, j)

    def __iter__(self, list_iter=list.__iter__):
        if self._not_initialized:
            self._init()
        return list_iter(self)

    def __len__(self, list_len=list.__len__):
        if self._not_initialized:
            return len(self.coords_array)
        return list_len(self)

    def __reversed__(self, list_rev=list.__reversed__):
        if self._not_initialized:
            self._init()
        return list_rev(self)

    def __setitem__(self, index_or_slice, geom,
                    setitem=MultiGeometry.__setitem__):
        if self._not_initialized:
            self._init()
        return setitem(self, index_or_slice, geom)

    def __setslice__(self, i, j, sequence, setslice=MultiGeometry.__setslice__):
        if self._not_initialized:
            self._init()
        return setslice(self, i, j, sequence)

    @staticmethod
    def _get_coords_array(self):
        coords_array = _numpy_concatenate(
            [point.coords_array for point in self]
            )
        coords_array.shape = _neg_1_3_tuple if self.is_3D else _neg_1_2_tuple
        return (coords_array, False)

    _get_envelope_coords = staticmethod(LineString._get_envelope_coords)

    def _init(self):
        if self.is_3D:
            make_fast3 = Point3D._make_fast3
        else:
            make_fast3 = Point2D._make_fast3
        self.__init__(_imap(make_fast3, self.coords_array,
                            self.coords_array.tolist()), False)
        del self._not_initialized

    def append(self, geom, test=True):
        if self._not_initialized:
            self._init()
        return MultiGeometry.append(self, geom, test)

    def count(self, geom):
        if self._not_initialized:
            self._init()
        return MultiGeometry.count(self, geom)

    def extend(self, geoms, test=True):
        if self._not_initialized:
            self._init()
        return MultiGeometry.extend(self, geoms, test)

    def index(self, value, start=_marker, stop=_marker):
        if self._not_initialized:
            self._init()
        if start is _marker:
            if stop is _marker:
                return MultiGeometry.extend(self, value)
            return MultiGeometry.extend(self, value, stop=stop)
        if stop is _marker:
            return MultiGeometry.extend(self, value, start)
        return MultiGeometry.extend(self, value, start, stop)

    def insert(self, index, geom, test=True):
        if self._not_initialized:
            self._init()
        return MultiGeometry.insert(self, index, geom, test)

    def iterate_only(self, reuse=True):
        if reuse and not self._not_initialized:
            for point in self:
                yield point
        else:
            if self.is_3D:
                make = Point3D._make_fast1
            else:
                make = Point2D._make_fast1
            for point_coords_array in self.coords_array:
                yield make(point_coords_array)

    @classmethod
    def make_fast(cls, arg0, process=True):
        if not process and isinstance(arg0, _numpy_ndarray):
            self = cls.__new__(cls, arg0)
            self.coords_array = arg0
            self._not_initialized = True
            return self
        # Note: Though this is not the typical behavior of make_fast(),
        # it seems sufficient here.
        return cls(arg0, process)
    make_fast.__func__.__doc__ = SingleGeometry.make_fast.__doc__

    def pop(self, index=-1):
        if self._not_initialized:
            self._init()
        return MultiGeometry.pop(self, index)

    def remove(self, geom):
        if self._not_initialized:
            self._init()
        return MultiGeometry.remove(self, geom)

    def reverse(self):
        if self._not_initialized:
            self._init()
        return MultiGeometry.reverse(self)

    def sort(self, cmp=None, key=None, reverse=False):
        if self._not_initialized:
            self._init()
        return MultiGeometry.sort(self, cmp, key, reverse)

    def translate(self, deltas, negative=False):
        """
        Translate members of the current instance *in place*.

        deltas is a sequence of the form (delta_x, delta_y, [delta_z]) that
        specifies the shift in each relevant dimension.

        negative is a boolean that specifies whether the values in deltas should
        be subtracted (rather than added). For example, if deltas[0] > 0,
        negative=False (the default) would result in a rightward shift but
        negative=True would result in a leftward shift (of equal magnitude).
        """
        if not self._not_initialized or len(deltas) != 2 + self.is_3D:
            # Note: Raise descriptive error if appropriate.
            deltas_array = self._process_array(
                deltas, "deltas", _3_tuple if self.is_3D else _2_tuple
                )
        if negative:
            shift = _numpy_subtract
        else:
            shift = _numpy_add
        if self._not_initialized:
            coords_array = self.coords_array
            for i, delta in enumerate(deltas):
                # Note: If delta is 0, do nothing.
                if delta:
                    column = coords_array[:,i]
                    shift(column, delta, column)
        else:
            for geom in self:
                shift(geom.coords_array, deltas_array, geom.coords_array)
                geom._Lazy__clear_lazy()


class MultiPoint2D(_InstantiableList, MultiPoint, Geometry2D):
    pass


class MultiPoint3D(_InstantiableList, MultiPoint, Geometry3D):
    pass


class MultiLineString(MultiGeometry):
    "Base class for all multi-part line-like Geometry types."
    topological_dimension = 1

    @classmethod
    def _from_wkt(cls, wkt):
        """
        Create a new MultiLineString from the specified wkt.

        wkt is a string that specifies the well-known text representation for
        the returned instance.
        """
        multi = cls(Polygon._from_wkt.__func__(cls, wkt, 18, -2, True), False)
        multi.wkt = wkt
        return multi


class MultiLineString2D(_InstantiableList, MultiLineString, Geometry2D):
    pass


class MultiLineString3D(_InstantiableList, MultiLineString, Geometry3D):
    pass


class MultiPolygon(MultiGeometry):
    "Base class for all multi-part polygon-like Geometry types."
    topological_dimension = 2

    @classmethod
    def _from_wkt(cls, wkt):
        """
        Create a new MultiPolygon from the specified wkt.

        wkt is a string that specifies the well-known text representation for
        the returned instance.
        """
        # Note: Code is somewhat similar to Polygon._from_wkt().
        if cls.is_3D:
            Polygon_from_wkt = Polygon3D._from_wkt
        else:
            Polygon_from_wkt = Polygon2D._from_wkt
        wkt_parts = wkt.split(")), ((")
        last_wkt_part = wkt_parts.pop()
        if wkt_parts:
            wkt_parts_iter = iter(wkt_parts)
            polys = [Polygon_from_wkt(next(wkt_parts_iter), 16, None, False)]
            for wkt_part in wkt_parts_iter:
                polys.append(Polygon_from_wkt(wkt_part, 0, None, False))
            polys.append(Polygon_from_wkt(last_wkt_part, 0, -3, False))
        else:
            polys = (Polygon_from_wkt(last_wkt_part, 16, -3, False),)
        multi = cls(polys, False)
        multi.wkt = wkt
        return multi


class MultiPolygon2D(_InstantiableList, MultiPolygon, Geometry2D):
    pass


class MultiPolygon3D(_InstantiableList, MultiPolygon, Geometry3D):
    pass



###############################################################################
# FLESH OUT GEOMETRY BASE CLASSES                                             #
###############################################################################

# Perform automated data attribute generation and assignment for
# Geometry subclasses, and population of relevant module-level dict's.
_flesh_out_geometry()



###############################################################################
# SINGLE-GEOMETRY DERIVED (SPECIAL CASE) CLASSES                              #
###############################################################################

class LineSegment(LineString):
    "Base class for line-segment-like Geometry types."


class LineSegment2D(_Instantiable, LineSegment, Geometry2D):

    def __init__(self, coords=None):
        """
        Make an object representing a line-segment-like geometry.

        coords is a numpy array or array-like sequence that specifies the
        LineSegment's coordinates.
        """
        self.coords_array = coords_array = self._process_array(
            coords, "coords", _2_2_tuple, True
            )
        (x0, y0), (x1, y1) = coords_array.tolist()
        dx = x1 - x0
        dy = y1 - y0
        self.length = _math_sqrt(dx*dx + dy*dy)
