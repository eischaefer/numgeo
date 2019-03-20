"""
Support for reading and writing vector data, including interoperability.
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

# These variables define default parameters for fields.
DEFAULT_TEXT_FIELD_WIDTH = 254

# If this variable is True, only one module will ever be attempted for
# any operation. If this variable is instead False and a module cannot
# complete an operation, the operation will be attempted again using the
# next highest priority module.
FIRST_MODULE_ONLY = False  ## Should be set to False for distribution.



###############################################################################
# IMPORT                                                                      #
###############################################################################

# Import internal (intra-package).
from numgeo import geom as _geom
from numgeo import util as _util

# Import external.
import collections as _collections
import datetime as _datetime
import importlib as _importlib
import inspect as _inspect
import itertools as _itertools
import numbers as _numbers
import numpy as _numpy
import os as _os
import shutil as _shutil
import sys as _sys
import warnings as _warnings

# Set import placeholders.
_arcpy = None # Placeholder. Replaced lazily by is_available.
_gdal = None # Placeholder. Replaced lazily by is_available.



###############################################################################
# LOCALIZATION                                                                #
###############################################################################

# Derived from built-ins.
__marker = object() # Arbitrary unique value.

# Derived from internal.
_validate_string_option = _util.validate_string_option

# Derived from external.
_deque = _collections.deque
_imap = _itertools.imap
_izip = _itertools.izip
_repeat = _itertools.repeat
_numpy_integer = _numpy.integer
_numpy_floating = _numpy.floating
_numpy_bool8 = _numpy.dtype("<?")
_numpy_int8 = _numpy.dtype("<i1")
_numpy_int16 = _numpy.dtype("<i2")
_numpy_int32 = _numpy.dtype("<i4")
_numpy_int64 = _numpy.dtype("<i8")
_numpy_uint8 = _numpy.dtype("<u1")
_numpy_uint16 = _numpy.dtype("<u2")
_numpy_uint32 = _numpy.dtype("<u4")
_numpy_uint64 = _numpy.dtype("<u8")
_numpy_float16 = _numpy.dtype("<f2")
_numpy_float32 = _numpy.dtype("<f4")
_numpy_float64 = _numpy.dtype("<f8")



###############################################################################
# DEPENDENCY SUPPORT                                                          #
###############################################################################

def _auto_call(calling_func_name=None, module=None):
    """
    Called within a method, automates calls to other modules (e.g., gdal).

    For example, if MODULE_RESOLUTION_ORDER.active is ["gdal", "arcpy"], and
    the current function is called with a method that was called as
    self.function(...), the current function will attempt to call
    self._function_gdal(...) and then, upon failure,
    self._function_arcpy(...).

    calling_func_name is a string that specifies a method name to use instead
    of the actual one. This is useful if the method is a function used for lazy
    attribute generation (i.e., _get_*()), or if the method is special (e.g.,
    __iter__()), in which case the name of its module-specific counterpart
    would be mangled.

    module is a string that specifies the only module allowed to be auto-
    called, thereby overriding MODULE_RESOLUTION_ORDER.active.

    Warning: The counterpart functions (e.g., self._function_gdal()) are
    effectively called with **locals() from wherever _auto_call() is called.
    """
    calling_frame = _inspect.stack()[1]
    # Note: kwargs are the keyword equivalent of the original call only
    # if no other values have been (re)assigned within the scope in
    # which _auto_call() is called.
    kwargs = calling_frame[0].f_locals
    self = kwargs.pop("self")
    if calling_func_name is None:
        calling_func_name = calling_frame[3] # *REASSIGNMENT*
    # Note: If calling function's name already starts with "_", do not
    # prepend an additional "_", as this would cause name mangling.
    basename = "_" + "{}_".format(calling_func_name).lstrip("_")
    unavailable_modules = []
    attempted_module_errors_strs = [""]
    unsupported_modules = []
    if module is None:
        module_names = MODULE_RESOLUTION_ORDER
    else:
        module_names = (module,)
    for module_name in module_names:
        if not getattr(is_available, module_name)():
            unavailable_modules.append(module_name)
            continue
        try:
            module_specific_func = getattr(self, basename + module_name)
        except AttributeError:
            unsupported_modules.append(module_name)
            continue
        try:
            result = module_specific_func(**kwargs)
        except Exception as e:
            if FIRST_MODULE_ONLY:
                raise
            attempted_module_errors_strs.append(
                "{}: {}: {} <line {}>".format(
                    module_name, e.__class__.__name__,
                    e.args[0] if len(e.args) == 1 else e.args,
                    _sys.exc_info()[2].tb_next.tb_lineno
                    )
                )
            continue
        return result
    text = """
unavailable modules: {}
unsupported modules: {}
errors from attempted modules:{}""".format(
        ", ".join(unavailable_modules) if unavailable_modules else "<None>",
        ", ".join(unsupported_modules) if unsupported_modules else "<None>",
        "\n    ".join(attempted_module_errors_strs)
        )
    raise TypeError(text)


class is_available(object):
    "A one-use class to support lazy import of supporting modules."

    def __getattr__(self, name):
        "Raise a helpful error if the named module is not registered."
        raise TypeError("module {!r} not recognized".format(name))

    def register(self, module_name, global_name=None, user_alias=None,
                 post_import=None):
        """
        Register a module.

        After registration, calling .user_alias() (where user_alias is replaced
        with that argument's specified or defaulted value) returns a boolean
        indicating whether that module is available. If it has not previously
        been imported, it is imported at that time.

        module_name is a string that specifies the name of the module to be
        registered for lazy import.

        global_name is a string that specifies the name of the global variable
        by which the registered module may be accessed from within the current
        module after import. If unspecified, it defaults to: "_" + module_name.

        user_alias is a string that specifies the attribute by which the
        availability of the registered module may be assessed. If unspecified,
        it defaults to module_name.

        post_import is a function that will be called, without argument,
        immediately after the (first and only) attempted import of the
        registered module, whether or not this attempt is successful.
        """
        # Generate default values, as necessary.
        if global_name is None:
            global_name = "_" + module_name
        if user_alias is None:
            user_alias = module_name

        # Define function to be called on lazy import attempt.
        def on_first_import_attempt(self=self, module_name=module_name,
                                    global_name=global_name,
                                    user_alias=user_alias,
                                    post_import=post_import):
            # Attempt to import module by its name. If successful,
            # register the module in the module-level (global)
            # namespace.
            try:
                module = _importlib.import_module(module_name)
            except ImportError:
                permanent_result = False
            else:
                permanent_result = True
                globals()[global_name] = module

            # Replace method with one that quickly returns the same
            # boolean on each call.
            # Note: The method is set here so that post_import(), if
            # specified, can easily test for successful import.
            setattr(self, user_alias, _repeat(permanent_result).next)

            # Execute post_import(), if it was specified.
            if post_import is not None:
                post_import()

            # Return permanent result.
            return permanent_result

        # Register module for lazy import.
        setattr(self, user_alias, on_first_import_attempt)

    @staticmethod
    def _gdal_post_import():
        # If import was unsuccessful, return immediately.
        if not is_available.gdal():
            return

        # Request gdal to use Python exceptions wherever possible.
        _gdal.UseExceptions()

        # Avoid a type of write problem encountered only when writing
        # many features to a file geodatabase. (See
        # https://trac.osgeo.org/gdal/ticket/4420.)
        _gdal.SetConfigOption("FGDB_BULK_LOAD", "YES")

        # Populate two dictionaries, each of which maps the field types
        # used in the current module to get()'s. See additional comments
        # where _field_type_to_get_gdal and
        # _field_type_to_get_strict_gdal are created.
        # Note: The dictionaries below support behavior similar to
        # gdal.ogr.Feature.GetField, but explicitly supports boolean and
        # datetime values.
        def make_get(gdal_func, coerce_func=None):
            def get(feature, fieldno, is_nullable,
                    is_not_null=_gdal.ogr.Feature.IsFieldSet.im_func,
                    gdal_func=gdal_func, coerce_func=coerce_func):
                if is_nullable and not is_not_null(feature, fieldno):
                    return None
                if coerce_func is not None:
                    return coerce_func(gdal_func(feature, fieldno))
                return gdal_func(feature, fieldno)
            get.gdal_func = gdal_func
            get.coerce_func = coerce_func
            return get
        field_type_to_get_gdal = {
            bool: make_get(
                _gdal.ogr.Feature.GetFieldAsInteger.im_func, bool
                ),
            _numpy_int8.type: make_get(
                _gdal.ogr.Feature.GetFieldAsInteger.im_func
                ),
            _numpy_int16.type: make_get(
                _gdal.ogr.Feature.GetFieldAsInteger.im_func
                ),
            _numpy_int32.type: make_get(
                _gdal.ogr.Feature.GetFieldAsInteger.im_func
                ),
            _numpy_uint8.type: make_get(
                _gdal.ogr.Feature.GetFieldAsInteger.im_func
                ),
            _numpy_uint16.type: make_get(
                _gdal.ogr.Feature.GetFieldAsInteger.im_func
                ),
            _numpy_uint32.type: make_get(
                _gdal.ogr.Feature.GetFieldAsInteger.im_func
                ),
            _numpy_float16.type: make_get(
                _gdal.ogr.Feature.GetFieldAsDouble.im_func
                ),
            _numpy_float32.type: make_get(
                _gdal.ogr.Feature.GetFieldAsDouble.im_func
                ),
            _numpy_float64.type: make_get(
                _gdal.ogr.Feature.GetFieldAsDouble.im_func
                ),
            unicode: make_get(
                _gdal.ogr.Feature.GetFieldAsString.im_func
                ),
            _datetime.date: make_get(
                _gdal.ogr.Feature.GetFieldAsDateTime.im_func,
                lambda value: _datetime.date(*value[:3])
                ),
            _datetime.time: make_get(
                _gdal.ogr.Feature.GetFieldAsDateTime.im_func,
                lambda value: _datetime.time(*value[3:])
                ),
            _datetime.datetime: make_get(
                _gdal.ogr.Feature.GetFieldAsDateTime.im_func,
                lambda value: _datetime.time(*value)
                ),
            _numpy_int64.type: make_get(
                _gdal.ogr.Feature.GetFieldAsInteger64.im_func
                )
            }
        # Test that all field types represented by examples in
        # _gdal_type_subtype_tuple_to_example's values are also
        # represented in field_type_to_get_gdal keys.
        for example in _gdal_type_subtype_tuple_to_example.itervalues():
            assert type(example) in field_type_to_get_gdal
        # Populate the global-level _field_type_to_get_gdal.
        _field_type_to_get_gdal.update(field_type_to_get_gdal)
        # Populate the global-level _field_type_to_get_strict_gdal by
        # using each get() value from field_type_to_get_gdal or a
        # substitute that coerces the default type to that of the key.
        for field_type, get in field_type_to_get_gdal.iteritems():
            if get.coerce_func is None:
                _field_type_to_get_strict_gdal[field_type] = make_get(
                    get.gdal_func, field_type
                    )
                continue
            _field_type_to_get_strict_gdal[field_type] = get

        # Populate a dictionary that maps the field types used in the
        # current module to set()'s. See additional comments where
        # _field_type_to_set_gdal is created.
        # Note: The dictionary below supports behavior similar to
        # gdal.ogr.Feature.SetField, but explicitly supports boolean and
        # datetime values as well as conversion from numpy values.
        def make_set(gdal_func, coerce_func=None, expand_after_coercion=False):
            if coerce_func is None:
                return gdal_func
            if expand_after_coercion:
                def set_(feature, fieldno, value, gdal_func=gdal_func,
                         coerce_func=coerce_func):
                    return gdal_func(feature, fieldno, *coerce_func(value))
            else:
                def set_(feature, fieldno, value, gdal_func=gdal_func,
                         coerce_func=coerce_func):
                    return gdal_func(feature, fieldno, coerce_func(value))
            return set_
        field_type_to_set_gdal = {
            _datetime.datetime: make_set(
                _gdal.ogr.Feature.SetField.im_func,
                lambda d: (d.year, d.month, d.day, d.hour, d.minute,
                           d.second + 1000. * d.microsecond, 0),
                True
                ),
            _datetime.time: make_set(
                _gdal.ogr.Feature.SetField.im_func,
                lambda t: (1999, 12, 31, t.hour, t.minute,
                           t.second + 1000. * t.microsecond, 0),
                True
                ),
            _datetime.date: make_set(
                _gdal.ogr.Feature.SetField.im_func,
                lambda d: (d.year, d.month, d.day, 0, 0, 0., 0),
                True
                )
            }
        # In addition to the exceptions already entered in
        # field_type_to_set_gdal, insert all other required entries into
        # that dictionary.
        for field_type in field_type_to_get_gdal:
            # Skip exceptions that are already entered.
            if field_type in field_type_to_set_gdal:
                continue
            # For numeric types, convert from the numpy type to its
            # standard Python equivalent.
            if field_type.__module__ == "numpy":
                py_type = type(field_type(0).item())
                field_type_to_set_gdal[field_type] = make_set(
                    _gdal.ogr.Feature.SetField.im_func, py_type
                    )
                continue
            # For all other types, default to simply using
            # Feature.SetField.
            field_type_to_set_gdal[field_type] = make_set(
                _gdal.ogr.Feature.SetField.im_func
                )
        # Populate the global-level _field_type_to_set_gdal.
        _field_type_to_set_gdal.update(field_type_to_set_gdal)


# Replace the is_available class with an instance thereof, and register
# arcpy and gdal for lazy import.
is_available = is_available() # *REDEFINITION*
is_available.register("arcpy")
is_available.register("gdal", post_import=is_available._gdal_post_import)


class MODULE_RESOLUTION_ORDER(object):

    def __iter__(self):
        return iter(self.active)

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self.active)

    def __init__(self, iterable):
        """
        Specify the dependent modules available in the current module.

        iterable is an iterable of strings that specifies the available
        dependent modules by name.

        Note: .active is a list of strings representing the names of those
        available modules whose use is currently permitted.

        Warning: .active should never be modified directly! Use the current
        instance's methods instead.
        """
        self.available = iterable
        self.active = list(iterable)

    def activate(self, module_name):
        """
        Activate a module if it is available.

        If the specified module is available and already active, this method
        does nothing. If the specified module is instead available but inactive,
        it is made active and assigned the lowest priority. If the specified
        module is not available, a TypeError is raised.

        module_name is a string that specifies the name of the module to be
        activated.
        """
        if module_name in self:
            return
        if module_name not in self.available:
            raise TypeError(
                "module is not recognized: {!r}".format(module_name)
                )
        self.active.append(module_name)

    def deactivate(self, module_name):
        """
        Deactivate a module.

        If the specified module is active, it is deactivated. If the specified
        module is already inactive or completely unavailable, the current method
        does nothing (and no error is raised).

        module_name is a string that specifies the name of the module to be
        deactivated.
        """
        try:
            self.active.remove(module_name)
        except ValueError:
            pass

    def promote(self, module_name):
        """
        Promote a module to maximum priority.

        After this method is called, any operation supported by the specified
        module will be attempted with that module first, before the operation is
        attempted with any other module.

        module_name is a string that specifies the name of the module to be
        promoted.
        """
        self.active.insert(0, self.active.pop(self.active.index(module_name)))

    def demote(self, module_name):
        """
        Demote a module to minimum priority.

        After this method is called, any operation supported by the specified
        module will be attempted with all other active modules first, before the
        operation is attempted with the specified module.

        module_name is a string that specifies the name of the module to be
        demoted.
        """
        self.active.append(self.active.pop(self.active.index(module_name)))
MODULE_RESOLUTION_ORDER = MODULE_RESOLUTION_ORDER(("gdal", "arcpy"))



###############################################################################
# SPATIAL REFERENCE COMPARISON                                                #
###############################################################################

def _isolate_spatial_reference(objects, test_unspecified=False,
                               return_bool=False):
    """
    Isolate a compatible spatial reference wkt or error if impossible.

    return_bool is a boolean that specifies whether a boolean should be
    returned (instead of a spatial reference wkt). If return_bool is True, the
    boolean indicates whether all objects are compatible.

    See documentation for test_spatial_reference_equivalence().
    """
    # Ensure that that at least one object was passed and gdal is
    # available.
    if not objects:
        raise TypeError("at least 1 object is required")
    if not is_available.gdal():
        raise TypeError("gdal is required for spatial reference comparison")

    # Extract a spatial reference wkt from each object.
    wkts_set = {obj if isinstance(obj, basestring)
                else getattr(obj, "spatial_reference", None)
                for obj in objects}

    # Ignore unspecified spatial references, if appropriate.
    if not test_unspecified:
        wkts_set.discard(None)
        if not wkts_set:
            # Note: No spatial reference is implied.
            if return_bool:
                return True
            return None

    # If only one viable wkt was found, return immediately.
    if len(wkts_set) == 1:
        if return_bool:
            return True
        return wkts_set.pop()

    # If unspecified spatial references are not ignored and both
    # specified and unspecified spatial references were found, error
    # immediately.
    error_message = "multiple spatial references are specified or implied (e.g., by .spatial_reference of Geometry's)"
    if test_unspecified and None in wkts_set:
        if return_bool:
            return False
        raise TypeError(error_message)

    # Convert each unique spatial reference wkt into a gdal spatial
    # reference object and test compatibility.
    test_wkt = wkts_set.pop()
    test_spat_ref = _gdal.osr.SpatialReference()
    test_spat_ref.ImportFromWkt(test_wkt)
    test_equal = test_spat_ref.IsSame
    spat_ref = _gdal.osr.SpatialReference()
    for wkt in wkts_set:
        spat_ref.ImportFromWkt(wkt)
        if not test_equal(spat_ref):
            if return_bool:
                return False
            raise TypeError(error_message)

    # Return compatible spatial reference wkt.
    if return_bool:
        return True
    return test_wkt


def test_spatial_reference_equivalence(*objects, **kwargs):
    """
    Test equivalence of spatial-reference-related objects.

    objects may be any combination of spatial reference wkt's and objects
    with a .spatial_reference, including Geometry's, Information's, and
    Definition's.

    test_unspecified (must be specified by keyword) is a boolean that
    specifies whether objects that are not a spatial reference wkt and
    either lack a .spatial_reference or for which that attribute is None
    should be tested, that is, not treated as compatible with any other
    spatial reference. test_unspecified is False by default.
    """
    if len(objects) < 2:
        raise TypeError("at least 2 objects are required")
    return _isolate_spatial_reference(objects, return_bool=True, **kwargs)


###############################################################################
# SCHEMA                                                                      #
###############################################################################

# This tuple includes all field types used in the current module.
_OK_field_types = (unicode, _numpy_integer, _numpy_floating, _datetime.datetime,
                   _datetime.date, _datetime.time, bool)

# This dictionary maps commonly encountered types to corresponding field
# types used in the current module.
_type_to_field_type = {int: _numpy_int32.type, long: _numpy_int64.type,
                       float: _numpy_float64.type, str: unicode,
                       _numpy_bool8.type: bool}

# This dictionary maps the type and subtype integers used in gdal to
# (suggestive) examples of the corresponding field types used in the
# current module.
_gdal_type_subtype_tuple_to_example = {
    (0, 0): _numpy_int32.type(32),
    (0, 1): True,
    (0, 2): _numpy_int16.type(16),
    (2, 0): _numpy_float64.type(64.),
    (2, 3): _numpy_float32.type(32.),
    (4, 0): unicode("x"),
    (9, 0): _datetime.date(1999, 12, 31),
    (10, 0): _datetime.time(12, 59, 59),
    (11, 0): _datetime.datetime(1999, 12, 31, 12, 59, 59),
    (12, 0): _numpy_int64.type(64)
    }

# This dictionary maps the field types used in the current module to the
# corresponding type and subtype integers used in gdal.
_field_type_to_gdal_type_subtype_tuple = _util._NumpyTypeDict()
for (_gdal_type_subtype, 
     _example) in _gdal_type_subtype_tuple_to_example.iteritems():
    _field_type_to_gdal_type_subtype_tuple[type(_example)] = _gdal_type_subtype

# Note: The containers in the below block are populated by is_available
# upon import of gdal.
# This dictionary maps the field types used in the current module to
# get()'s. get(feature, fieldno, is_nullable) returns a value of the
# type to which it is keyed, or a similar standard type (or None), where
# feature is a gdal.ogr.Feature, fieldno is the field index (integer)
# that specifies the field to be read, and is_nullable is a boolean that
# specifies whether null values are supported by that field.
_field_type_to_get_gdal = _util._NumpyTypeDict()
# This dictionary is identical to the one immediately above, except that
# its get()'s invariably return a value of the type to which that get()
# is keyed (or None). Therefore, only instances of field types used in
# the current module (or None) are ever returned.
_field_type_to_get_strict_gdal = _util._NumpyTypeDict()
# This dictionary maps the field types used in the current module to
# set()'s. set(feature, fieldno, value) sets the relevant field to the
# specified value for the specified feature, where feature is a
# gdal.ogr.Feature and fieldno is the field index (integer) that
# specifies the relevant field.
_field_type_to_set_gdal = _util._NumpyTypeDict()

# This function is specified for _arcpy_type_to_example["Date"] to treat
# the special case for shapefiles, in which "Date" Field's specify
# datetime.date's rather than datetime.datetime's.
def _arcpy_type_to_example_date(i):
    if i.path[-4:].lower() == ".shp":
        return _datetime.date(1999, 12, 31)
    return _datetime.datetime(1999, 12, 31, 12, 59, 59)

# Deprecated. This function was specified for
# _arcpy_type_to_example["Geometry"] before enforcing consistency with
# gdal (in which geometry Field's are not part of the Schema).
def _arcpy_type_to_example_geometry(i):
    return i.geom_type.example

# This dictionary maps the field type names ("keywords") used by
# arcpy.Field's to (suggestive) examples of the corresponding field
# types used in the current module (with some exceptions).
_arcpy_type_to_example = {
    # Note: Special case for date in shapefiles is hardcoded.
    "Date": _arcpy_type_to_example_date,
    "Double": _numpy_float64.type(64.),
    "Geometry": None, # Exclude geometry field from Schema's, like gdal.
    "Integer": _numpy_int32.type(32),
    "OID": None, # Exclude feature index from Schema's, like gdal.
    "Single": _numpy_float32.type(32.),
    "SmallInteger": _numpy_int16.type(16),
    "String": unicode("x")
    }

# This dictionary maps the field types used in the current module to the
# field type names ("keywords") used by arcpy.AddField_management().
_field_type_to_arcpy_type = _util._NumpyTypeDict()
for _arcpy_type, _example in _arcpy_type_to_example.iteritems():
    if _example is None or _arcpy_type == "Date":
        continue
    _field_type_to_arcpy_type[type(_example)] = _arcpy_type
_field_type_to_arcpy_type[_datetime.date] = "Date"
_field_type_to_arcpy_type[_datetime.datetime] = "Date"
_field_type_to_arcpy_type[_datetime.time] = "Date"


class Field(object):
    # Note: Class-level default values avoid the need for some attribute
    # assignment (in some cases) at instantiation.
    default = None
    precision = None
    width = None

    def __eq__(self, other):
        """
        Field's are equal iff initial arguments are equivalent.
        """
        if not isinstance(other, type(self)):
            return NotImplemented
        return (type(self.example) is type(other.example),
                self.default == other.default, self.width == other.width,
                self.is_nullable == other.is_nullable)

    def __ne__(self, other):
        """
        Field's are unequal iff initial arguments are not equivalent.
        """
        return not (self == other)

    def __repr__(self):
        return "{}({!r} {!r}, {!r}, {!r}, {!r}, {!r})".format(
            type(self).__name__, self.example, type(self.example), self.default,
            self.width, self.precision, self.is_nullable
            )

    def __init__(self, example, default=None, width=None, precision=None,
                 nullable=True, minimize=False):
        """
        Define a field (the building block of Schema).

        A TypeError is raised if example's type is not supported and no
        substitute type can be identified, if example and default are of
        fundamentally different types (e.g., string and int), or if example and
        default are strings but either does not fit in the specified width.

        example is an instance whose type specifies the Field's type (but see
        default and minimize). If example is not from a directly supported type
        (e.g., int), a similar type (e.g., numpy.int32) will be used.

        default is a value that specifies the default for the Field, if any. If
        default is specified and minimize is False, it is guaranteed (insofar as
        is possible) that the Field's type will fit all values supported by the
        types of example and default (combined), which could result in a
        promotion of example's type. For instance, if example is an int and
        default is a long, Field's type will be numpy.int64 (equivalent to
        long).

        width is an integer that specifies the width of the field.

        nullable is a boolean that specifies whether null's are permitted.

        minimize is a boolean that specifies whether the type of the Field
        should be minimized so as to "just" fit the *values* (not types) of
        example and default (if specified), including the width of strings and
        the bit-depths of integers. For example, if width is not specified for a
        string Field, its width will default to the larger of (1) the minimum
        value that can fit example and default or (2) (the global variable)
        DEFAULT_TEXT_FIELD_WIDTH. Similarly, if example and default are both -1
        represented by long's, the Field's type will be numpy.int8, since no
        smaller type can fit -1. See warning further below for additional
        discussion.

        Warning: The current module invariably sets Field types implicitly by
        one or more examples. This is a conscious design choice intended to make
        setting the type of a Field less tedious. However, all type
        interpretation approaches have potential pitfalls, and the implicit
        approach used in the current module may further obscure these pitfalls.
        For example, consider the following cases:
            1) If the example is a string, it is not clear what Field.width
               should be, except that it should be at least equal to
               len(example). This ambiguity is a major reason for the global
               variable DEFAULT_TEXT_FIELD_WIDTH. Alternatively, note that you
               can explicitly specify the width that you wish the field to
               accommodate.
            2) If the example is a (Python) int, the required bit-depth is
               somewhat ambiguous. For instance, if all the values in the field
               will be small, a 32-bit depth (which supports values up to ~2
               billion) is not necessary. Perhaps a 16-bit depth (which supports
               values up ~30,000) would be perfectly sufficient. However,
               because the Field is based on a single example, it is assumed
               that a 32-bit depth (the same as Python int itself) is required
               (unless minimize is True). There are also issues at the other
               end. For instance, if the example is 2 billion but the field will
               also need to hold values around 3 billion, even a 32-bit depth
               would not be sufficient. If you wish to force a specific bit
               depth for a numeric value, consider using the corresponding numpy
               type. In the two aforementioned cases, suitable examples might be
               numpy.int16(0) and numpy.int64(0), respectively. Note that
               similar considerations also exist for float examples.
            3) To overcome some of the potential pitfalls just described, and
               also save the user from having to worry about field
               specifications at all in many common cases, the functions
               Schema.infer_by_union() and infer_by_intersection() use a
               different (default) approach. That approach can be called a
               "minimizing" approach, whereas the default behavior for direct
               Field specification (i.e., if minimize is False), described
               above, may be called a "naive" approach. See documentation for
               WriteCursor (which leverages the minimizing approach by default)
               for the details of that approach.
        """
        # Standardize example.
        self.example = std_example = self._standardize_value(example, "example")

        # Standardize default (prelimininarily, at least).
        # Note: .default defaults to None at the class level.
        # Note: If Field is numeric and minimize is True, the
        # standardization here is preliminary. Nonetheless, it is
        # useful, as ._standardize_value() will raise an error if the
        # values for example and default are of fundamentally different
        # types (e.g., a string and a number).
        if default is not None:
            # *REDEFINITION*
            self.default, std_example = self._standardize_value(
                default, "default", std_example, "example"
                )
            self.example = std_example

        # Minimize if minimize is True and Field is integer.
        if minimize and isinstance(std_example, _numpy.integer):
            example_min_std_type = _numpy.min_scalar_type(std_example).type
            if default is None:
                # *REDEFINITION*
                self.example = std_example = example_min_std_type(std_example)
            else:
                # Note: If default is not numeric, ._standardize_value()
                # would have raised an error above.
                default_min_std_type = _numpy.min_scalar_type(self.default)
                # Note: The for-loop below is necessary to ensure
                # optimal minimization. For example, consider an example
                # of 100 and a default of -1, for which
                # numpy.min_scalar_type().type would return numpy.uint8
                # and numpy.int8, respectively. Passing these types to
                # numpy.promote_types() would return numpy.int16 even
                # though numpy.int8 could fit both -1 and 100.
                for std_type in (example_min_std_type, default_min_std_type):
                    for value in (std_example, default):
                        if _numpy.can_cast(value, std_type):
                            continue
                        # The value (default or example) cannot be
                        # safely cast to the other's minimum standard
                        # type, so try the other's minimum standard type
                        # (if not yet checked).
                        break
                    else:
                        # Both values can fit in the current minimum
                        # standard type, so use that type and exit the
                        # (uppermost) for-loop.
                        break
                else:
                    # Neither the minimum standard type identified for
                    # example nor the one identified for default can fit
                    # both example and default values, so resort to
                    # promoting those types to a higher standard type.
                    # *REDEFINITION*
                    std_type = _numpy.promote_types(example_min_std_type,
                                                    default_min_std_type)
                # *REDEFINITION*
                self.example = std_example = std_type(std_type)
                self.default = std_type(default)

        # Standardize width and precision.
        ## Note: The choice to assign "effective" .width and .precision
        ## values for numeric Field's, as opposed to special-casing  the
        ## shapefile requirement that these values be explicit, could be
        ## revisited. For example, this choice may complicate appending
        ## to preexisting data.
        if width:
            self.width = int(width)
        elif isinstance(std_example, _numbers.Number):
            # Note: Assume that a width of 0 implies an implicit width
            # rather than a true width of 0. (OGR makes the same
            # assumption according to the documentation for
            # FieldDefn.GetWidth().)
            self.width, self.precision = self._estimate_width_and_precision_if_numeric(
                std_example
                )
        if precision:
            self.precision = int(precision)
        # Note: .precision may have already been assigned above.
        # Note: Assume precision should only be specified as 0 for
        # integers, for which .precision will be assigned 0 below.
        elif (self.precision is None and
              isinstance(std_example, _numbers.Number)):
            self.precision = min(
                width,
                self._estimate_width_and_precision_if_numeric(std_example)[1]
                )

        # Set or validate width if Field is a string.
        if isinstance(std_example, basestring):
            if width is None:
                self.width = max(
                    len(std_example),
                    0 if default is None else len(self.default),
                    0 if minimize else DEFAULT_TEXT_FIELD_WIDTH
                    )
            elif len(std_example) > self.width:
                raise TypeError(
                    "example {!r} is longer than width={}".format(std_example,
                                                                  self.width)
                    )
            elif default is not None and len(self.default) > self.width:
                raise TypeError(
                    "default {!r} is longer than width={}".format(self.default,
                                                                  self.width)
                    )

        # Standardize nullable.
        self.is_nullable = bool(nullable)

    def can_fit(self, other):
        """
        Test whether another Field or value can fit in self.

        Equivalent to (but faster than):
            def can_fit(self, other):
                try:
                    return self == self.promote(other, False)
                except TypeError:
                    return False
        """
        return not self._promote(other)

    @staticmethod
    def _estimate_width_and_precision_if_numeric(example):
        """
        Estimate the width and precision required by a numeric example.

        If example is not numeric, a TypeError is raised.

        Note: For a 64-bit float example, width and precision are returned as 24
        and 15, consistent with the defaults for OGR's ESRI Shapefile / DBF
        driver.
        """
        if isinstance(example, _numbers.Integral):
            # *REDEFINITION*
            width = len(repr(_numpy.iinfo(example).min))
            precision = 0
        elif isinstance(example, _numbers.Real):
            ## Note: It is not clear what values should be assigned for
            ## width and precision, but at least the approach below
            ## reproduces the same default behavior as OGR's ESRI
            ## Shapefile / DBF driver for a 64-bit float.
            finfo = _numpy.finfo(example)
            width = len(repr(-finfo.tiny))
            precision = finfo.precision
        else:
            raise TypeError(
                "example is not numeric: {!r} {!r}".format(example,
                                                           type(example))
                )
        return (width, precision)

    def promote(self, other, in_place=True):
        """
        Promote self's type to fit other. By default, promotion is in-place.

        A TypeError is raised if no permitted promotion of self's type could
        accommodate other. Only the following promotions are permitted:
            increase to .width
            .is_nullable --> True
            any numeric type (excluding bool) --> any numeric type

        other is either another Field or any instance that the promoted Field
        should accommodate. If other is a Field, any value that can fit in that
        field (or in self) can also fit in the promoted Field. If other is
        instead an instance, its value (but not necessarily its type) is
        guaranteed to fit in the promoted Field (insofar as is possible). For
        example, numpy.int64(2) can fit in a Field of type numpy.int16.

        in_place is a boolean that specifies whether self should be promoted in-
        place. Otherwise, a new Field is returned.
        """
        return self._copy(self._promote(other), not in_place)

    def _promote(self, other):
        """
        Helper method for promote().

        Raises TypeError if promotion is insufficient. Otherwise, returns a
        dictionary including only the initialization arguments that need to be
        overridden.

        Note: If no promotion is required, an empty dictionary (or False) must
        be returned, to support can_fit(). (This also has some performance gains
        in _copy().)
        """
        # Identify representative examples for self and other, and if
        # other is a Field, remember those initialization arguments that
        # may need to be overridden.
        self_example = self.example
        other_is_field = isinstance(other, Field)
        make_args = {}
        if other_is_field:
            other_example = other.example
            if other.width is not None and self.width < other.width:
                make_args["width"] = other.width
            if other.precision is not None and self.precision < other.precision:
                make_args["precision"] = other.precision
            if other.is_nullable and not self.is_nullable:
                make_args["is_nullable"] = True
        else:
            other_example = other

        # Format a standard error.
        err = TypeError(
            "cannot promote type {!r} to type {!r}".format(
                type(other_example).__name__, type(self_example).__name__
                )
            )

        # Identify all cases where promotion is impossible or no
        # promotion is required.
        # Error if a string and non-string are mixed.
        self_example_is_basestring = isinstance(self_example, basestring)
        if self_example_is_basestring != isinstance(other_example, basestring):
            raise err
        # Return immediately if other's example is an instance of the
        # type of self's example (excluding strings, for which widths
        # could still vary).
        elif isinstance(other_example, type(self_example)):
            return make_args
        # Error if the type of self's example belongs to the datetime
        # module (because identical types were already returned above).
        if type(self_example).__module__ == "datetime":
            raise err
        # Return immediately if self's and other's examples are both
        # booleans, but error if only one of these examples is a
        # boolean.
        if isinstance(self_example, bool):
            if isinstance(other_example, (bool, _numpy.bool8)):
                return make_args
            raise err

        # If self's and other's examples are both strings, the only
        # remaining change that could be required is an increase to
        # self.width, if other is longer (and not a Field, in which case
        # the longer width was already set further above).
        if self_example_is_basestring:
            if not other_is_field and len(other) > self.width:
                make_args["width"] = len(other)
            return make_args

        # Self's and other's examples are both numbers. Find the
        # (potentially) promoted dtype that can accommodate both
        # examples and (potentially) convert self's example to that
        # dtype.
        self_dtype = type(self.example)
        if other_is_field:
            other_dtype = type(other_example)
        elif _numpy.can_cast(other, self_dtype):
            other_dtype = self_dtype
        else:
            other_dtype = _numpy.min_scalar_type(other)
        if other_dtype is not self_dtype:
            promoted_dtype = _numpy.promote_types(self_dtype, other_dtype)
            if not _numpy.can_cast(promoted_dtype, self_dtype, "equiv"):
                make_args["example"] = promoted_dtype.type(self.example)
        return make_args

    def copy(self):
        """
        Return a shallow copy of self.
        """
        return self._copy()

    def _copy(self, override_args=False, copy=True):
        """
        Helper method for copy(), with some advanced options.

        override_args is a dictionary that specifies what, if any,
        initialization arguments should be overridden.

        copy is a boolean that specifies whether a (possibly modified) copy of
        self should be returned. Otherwise, self is re-initialized (if
        necessary).
        """
        # In the special case that modification should be performed in-
        # place and there are no modifications, return immediately.
        if not copy and not override_args:
            return

        # Build a dictionary of the current initialization arguments,
        # and update it with any overriding arguments.
        args = {"example": self.example, "default": self.default,
                "width": self.width, "precision": self.precision,
                "nullable": self.is_nullable}
        if override_args:
            args.update(override_args)

        # Create a copy and return it, or else re-initialize in-place.
        if copy:
            return type(self)(**args)
        self.__init__(**args)

    @staticmethod
    def _standardize_value(value, value_label="field value", value2=None,
                           value2_label="field value"):
        """
        Coerce a value (as necessary) to be of a permitted field type.

        If value2 is not specified, the type of the returned value is guaranteed
        to be a standard field type that can fit all values that the original
        value's type can fit. If value2 is specified, a 2-tuple (new_value,
        new_value2) is instead returned. Both new_value and new_value2 are of
        the same type, which is a standard field type that can fit all values
        that the original value's type and value2's type could fit (combined).
        In either case, if a suitable target type cannot be identified, a
        TypeError is raised.

        value is an instance that specifies the value to (possibly) be coerced.

        value_label is a string that specifies how value will be referred to in
        any error that is raised.

        value2 is an instance whose type specifies the minimum type to which
        value should be coerced. If value2 is not specified, the permitted type
        most similar to value's type is used instead.

        value_label2 is a string that specifies how value2 will be referred to
        in any error that is raised.
        """
        if not isinstance(value, _OK_field_types):
            try:
                # *REDEFINITION*
                value = _type_to_field_type[type(value)](value)
            except KeyError:
                raise TypeError(
                    "{} {!r} (type {}) cannot be coerced to a supported type: {}".format(
                        value_label, value, type(value).__name__,
                        ", ".join([type_.__name__ for type_ in _OK_field_types])
                        )
                    )
        if value2 is not None:
            if  isinstance(value, type(value2)):
                return (value, value2)
            if (isinstance(value2, _numpy.number) and
                isinstance(value, _numpy.number)):
                final_type = _numpy.promote_types(type(value), type(value2))
                return (final_type(value), final_type(value2))
            raise TypeError(
                "{} {!r} (type {}) is not of {}'s type: {}".format(
                    value_label, value, type(value).__name__, value2_label,
                    type(value2).__name__
                    )
                )
        return value


class OrderedStringDict(dict):
    __marker = object() # Arbitrary unique value.

    def __delitem__(self, key):
        if not isinstance(key, basestring):
            dict.__delitem__(self, self._keys.pop(key))
            return
        dict.__delitem__(self, key)
        self._keys.remove(key)

    def __iter__(self):
        return iter(self._keys)

    def __missing__(self, key):
        try:
            # Note: The line below supports the alternative index-based
            # mapping.
            return self[self._keys[key]]
        except:
            raise KeyError(key)

    def __repr__(self):
        return "{}({!r}, {!r})".format(type(self).__name__, self._keys,
                                       self.values())

    def __setitem__(self, key, value):
        if not isinstance(key, basestring):
            raise TypeError("key must be a string")
        if key not in self:
            self._keys.append(key)
        dict.__setitem__(self, key, value)

    def __init__(self, keys=None, values=None, copy_keys=True):
        """
        Ordered dictionary dually keyed to strings and indices.

        For convenience, each value is dually keyed to both a string and its
        numeric index:
            d["key"] --> value keyed to "key" in d
            d[n] --> the nth value (where n is an integer)
        This dual mapping applies to getting and deleting values, but not to
        setting them, as the string key must be specified, nor to testing
        membership (e.g., n in self --> False).

        keys is an iterable of strings that specifies the (initial) keys.

        values is a value or iterable of values that specifies the (initial)
        values. If values is not iterable, it is assigned as the (default) value
        for all keys.

        copy_keys is a boolean. If copy_keys is False, keys should be a list and
        will be used internally.

        Warning: Modifying a list after it has been passed to the keys argument
        will cause the instance to become corrupted if copy_keys is False!

        Note: The current class is derived from collections.MutableMapping. See
        documentation for that abstract base class for more implementation
        details.
        """
        if keys is not None:
            if copy_keys:
                # Note: keys is reassigned in case the original is not
                # re-iterable.
                self._keys = keys = list(keys)
            else:
                self._keys = keys
            if hasattr(values, "__iter__"):
                dict.__init__(self, _izip(keys, values))
            else:
                dict.__init__(self, _izip(keys, _repeat(values)))
            return
        dict.__init__(self)
        self._keys = []

    def clear(self):
        dict.clear(self)
        del self._keys[:]

    def copy(self, type_=None):
        """
        Make a shallow copy of self.

        type_ is OrderedStringDict or a subclass thereof that specifies the type
        of the returned copy.
        """
        if type_ is None:
            type_ = type(self)
        elif not isinstance(type_, OrderedStringDict):
            raise TypeError("cls must be subclass of OrderedStringDict")
        return type_(self._keys, self.values())

    @classmethod
    def fromkeys(cls, S, v=None):
        return cls(S, v)

    @classmethod
    def fromdict(cls, d):
        return cls(d, d.itervalues())

    def get(self, k, d=None):
        try:
            return self[k]
        except KeyError:
            return d

    def index(self, k):
        return self._keys.index(k)

    def items(self):
        return [(k, self[k]) for k in self._keys]

    def iteritems(self):
        return ((k, self[k]) for k in self._keys)

    iterkeys = __iter__

    def itervalues(self, unordered=False):
        if unordered:
            return dict.itervalues(self)
        return _imap(dict.__getitem__.__get__(self), self._keys)

    def keys(self):
        return list(self._keys)

    def pop(self, k, d=__marker):
        if k not in self:
            if d is not self.__marker:
                return d
            return dict.pop(self, k) # Raise expected error.
        self._keys.remove(k)
        return dict.pop(self, k)

    def popitem(self):
        item = k, v = dict.popitem(self)
        self._keys.remove(k)
        return item

    def setdefault(self, k, d=None):
        if k in self:
            return self[k]
        self[k] = d
        return d

    def update(self, keys, values):
        for k, v in _izip(keys, values):
            self[k] = v

    def values(self, unordered=False):
        if unordered:
            return dict.values(self)
        return [self[key] for key in self._keys]


class Schema(OrderedStringDict):

    def __setitem__(self, key, value):
        """
        Set key to value, converting value to a Field if necessary.
        """
        if not isinstance(value, Field):
            value = Field(value) # *REDEFINITION*
        OrderedStringDict.__setitem__(self, key, value)

    def __init__(self, keys=None, values=None, copy_keys=True):
        """
        Ordered dictionary of Field's, dually keyed to field names and indices.

        For convenience, each Field is dually keyed to both a field name and its
        numeric index:
            schema["field_name"] --> Field assigned to "field_name"
            schema[n] --> the nth Field (where n is an integer)
        This dual mapping applies to getting and deleting Field's, but not to
        setting them, as the field name must be specified.

        For further convenience, an entry may be set explicitly by a Field or
        implicitly by an example. Therefore, the following are equivalent:
            schema["field_name"] = Field(5)
            schema["field_name"] = 5

        The current class is derived from OrderedStringDict (current module) and
        collections.MutableMapping. See documentation for those classes for more
        details.
        """
        if values is not None:
            # *REDEFINITION*
            values = [value if isinstance(value, Field)
                      else Field(value) for value in values]
        OrderedStringDict.__init__(self, keys, values, copy_keys)

    def _make_field_data_getter_or_setter_gdal(self, field_names, get=True,
                                               strict=True):
        """
        Make a function to get or set values in batch in gdal.

        If get is True, the returned function takes a gdal.ogr.Feature as its
        only (non-default) argument. If set is False, the returned function
        takes a gdal.ogr.Feature and a sequence of values as its (non-default)
        arguments.

        field_names is a sequence of strings that specifies the fields from the
        current Schema that should be got or set by the returned function. Order
        is honored.

        get is a boolean that specifies whether the returned function should get
        (rather than set) the specified fields.

        strict is a boolean that specifies whether the returned function must
        return values with the same type as the corresponding Field (e.g.,
        numpy.float32) rather than possibly returning a value of a related type
        (e.g., Python float). If get is False, strict is ignored.
        """
        if get:
            if strict:
                _field_type_to_func = _field_type_to_get_strict_gdal
            else:
                _field_type_to_func = _field_type_to_get_gdal
        else:
            _field_type_to_func = _field_type_to_set_gdal
        # Note: x is is_nullable if get else unused.
        func_fieldno_x_tuples = []
        for field_name in field_names:
            func = _field_type_to_func[type(self[field_name].example)]
            fieldno = self.index(field_name)
            if get:
                func_fieldno_x_tuples.append(
                    (func, fieldno, self[field_name].is_nullable)
                    )
                continue
            func_fieldno_x_tuples.append((func, fieldno))
        if get:
            def get_field_data(
                feature, get_fieldno_nullable_tuples=func_fieldno_x_tuples
                ):
                return [get(feature, fieldno, nullable)
                        for get, fieldno, nullable in
                        get_fieldno_nullable_tuples]
            return get_field_data
        def set_field_data(
            feature, values, set_fieldno_tuples=func_fieldno_x_tuples
            ):
            for value, (set_, fieldno) in _izip(values, set_fieldno_tuples):
                if value is None:
                    continue
                set_(feature, fieldno, value)
        return set_field_data

    def apply(self, dictionary):
        """
        Apply schema to a dict (typically .data of a Geometry).

        Each key in dictionary that corresponds to a field name in the schema
        will have its paired value coerced to the exact type specified by the
        schema. For example, a (Python) int might be converted to a numpy.uint8
        if so specified in the schema. However, if the paired value is None, it
        is not coerced. Any key in dictionary that does not correspond to a
        field name in the schema is ignored, as is any field in schema that is
        not represented in dictionary.

        dictionary is a dict to which the schema is applied.

        Warning: dictionary itself is modified by the current function.
        """
        for field_name in self._keys:
            try:
                value = dictionary[field_name]
            except KeyError:
                continue
            if value is None:
                continue
            target_type = type(self[field_name].example)
            if not isinstance(value, target_type):
                dictionary[field_name] = target_type(value)

    def copy(self, deep=True):
        """
        Create copy of self.

        deep is a boolean that specifies whether copies of self's Field's should
        be used (instead of the original Field's themselves).
        """
        if deep:
            return type(self)(self._keys,
                              [field.copy() for field in self.itervalues()])
        return type(self)(self._keys, self.values())

    def _set_operate(self, seed, schema, promote, add_uncommon, drop_common):
        """
        Helper method for all set- and list-like operations.

        seed is the Schema that specifies the target of the operation. It should
        be self or self.copy(). If seed is self, it will not be modified until
        it is known that no error will be encountered across the operation (to
        avoid interrupted operations effectively corrupting self).

        schema is a sequence of strings that specifies the field names involved
        in the operation. In most scenarios, schema is a Schema (or other
        dictionary of Field's keys to field names). However, if add_uncommon is
        False and promote is None, schema may be any sequence of field names.

        promote is a boolean that specifies whether a Field in seed may be
        promoted to fit a corresponding Field in schema. If promote is False but
        a Field in seed cannot fit a corresponding Field in schema, a TypeError
        is raised. In the special case that promote is None, it is never tested
        whether a Field in seed can fit a Field in schema.
        """
        # If an error may arise (because promotion is not None) and the
        # target of the operation (seed) is self, delay modifying self
        # until the end by operating instead on a temporary copy.
        update_self = promote is not None and seed is self
        if update_self:
            seed = self.copy() # *REDEFINITION*

        # For each field in schema, add uncommon fields, drop common
        # fields, promote, and/or error if promotion is required, as
        # appropriate.
        for field_name in schema:
            if field_name not in seed:
                if add_uncommon:
                    seed[field_name] = schema[field_name]
                continue
            elif drop_common:
                del seed[field_name]
                continue
            # Note: If promote is None, do not test whether seed's field
            # can fit schema's field.
            if promote is None:
                continue
            if promote:
                # Note: Attempting promotion may raise an error.
                seed[field_name].promote(schema[field_name])
                continue
            if not seed[field_name].can_fit(schema[field_name]):
                raise TypeError(
                    "fields are incompatible for field named: {!r}".format(
                        field_name
                        )
                    )

        # Apply modifications to self now, if they were delayed.
        if update_self:
            self.update(seed._keys, seed.itervalues())

        # Return target (seed).
        return seed

    def difference(self, field_names):
        """
        Return a copy of self in which the specified field names are removed.

        The order of self's Field's is preserved in the returned Schema.

        field_names is a sequence of strings that specifies the field names to
        be dropped.
        """
        return self._set_operate(self.copy(), field_names, None, False, True)

    def difference_update(self, field_names):
        """
        Drop the specified field names from the schema (in place).

        The order of self's Field's is preserved.

        field_names is a sequence of strings that specifies the field names to
        be dropped.
        """
        self._set_operate(self, field_names, None, False, True)

    def extend(self, schema):
        """
        Add any missing specified Field's to the schema (in place).

        The order of self's Field's is preserved, and Field's unique to schema
        are added in the order in which they appear in schema.

        schema is a Schema (or other dictionary of Field's keyed to strings)
        that specifies the Field's to be added if absent.

        Warning: If a field name in schema also exists in self, the
        corresponding Field in schema is ignored. For example, it is not tested
        whether the corresponding Field in self can fit that Field.
        """
        self._set_operate(self, schema, None, True, False)

    def intersection(self, schema, promote=False):
        """
        Return a Schema containing Field's common to self and another Schema.

        The respective orders of self's and schema's Field's are preserved in
        the return Schema, with those retained from self coming first.

        schema is a Schema (or other dictionary of Field's keyed to strings)
        that specifies the Field's to be compared to those of self.

        promote is a boolean that specifies whether Field's in self may be
        promoted so that they fit the corresponding Field's (matched by field
        name) in schema. If promote is False (the default) but a Field in self
        cannot fit the corresponding field in schema, a TypeError is raised.
        """
        return self._set_operate(self.copy(), schema, bool(promote), False,
                                 False)

    def subset(self, field_names):
        """
        Subset the schema to only the specified field names (in place).

        The order of self's Field's is preserved.

        field_names is a sequence of strings that specifies the field names to
        be isolated.
        """
        self._set_operate(self, field_names, None, False, False)

    def symmetric_difference(self, schema):
        """
        Return a Schema containing Field's unique to self or another Schema.

        The respective orders of self's and schema's Field's are preserved in
        the returned Schema, with self's Fields coming first.

        schema is a Schema (or other dictionary of Field's keyed to strings)
        that specifies the Field's to be compared to those of self to evaluate
        uniqueness.
        """
        return self._set_operate(self.copy(), schema, None, True, True)

    def symmetric_difference_update(self, schema):
        """
        Repopulate self to contain Field's originally unique to self or schema.

        The respective orders of self's and schema's Field's are preserved, with
        self's Fields coming first.

        schema is a Schema (or other dictionary of Field's keyed to strings)
        that specifies the Field's to be compared to those of self to evaluate
        uniqueness.
        """
        self._set_operate(self, schema, None, True, True)

    def union(self, schema, promote=False):
        """
        Return a Schema containing the union of Field's from self and schema.

        The order of self's Field's is preserved in the returned Schema, and
        Field's unique to schema are added in the order in which they appear in
        schema.

        schema is a Schema (or other dictionary of Field's keyed to strings)
        that specifies the Field's to be added to those from self.

        promote is a boolean that specifies whether Field's in self may be
        promoted so that they fit the corresponding Field's (matched by field
        name) in schema. If promote is False (the default) but a Field in self
        cannot fit the corresponding field in schema, a TypeError is raised.
        """
        return self._set_operate(self.copy(), schema, bool(promote), True,
                                 False)

    @classmethod
    def _build_schema(cls, field_datas, field_names_set=None, minimize=True,
                      promote=True):
        """
        Helper class method for infer_*()'s that builds a Schema.

        A tuple (schema, nulled_field_names_set) is returned, in which schema is
        a Schema that includes all field names from field_datas that are in
        neither field_names_set (if specified) nor nulled_field_names_set
        (returned), paired to Field's that can fit all the corresponding values.
        Those Field's may be promoted above the type of any of the supplied
        values. For example, a mix of int32 and float32 values could result in a
        Field with type float64, if the values require it. The returned
        nulled_field_names_set is a set of any field names for which all values
        are None's, which renders inferring a suitable Field for these field
        names impossible. nulled_field_names_set may be empty.

        field_datas is a sequence of mappings of field values keyed to the
        corresponding field names. Typically, each item is a .data dict from a
        Geometry.

        field_names_set is a set (or other container that supports membership
        testing) that specifies that subset of field names that should be
        permitted in the returned Schema. If field_names_set is not specified,
        no field names are excluded.

        minimize is a boolean that specifies whether the types of the Field's in
        the returned Schema should be minimized to "just" fit the corresponding
        values, including the width of string fields and bit-depths of numeric
        fields. (See documentation for WriteCursor for additional details.) If
        minimize is False, the "naive" behavior described in the documentation
        for Field is used instead, though optionally enhanced by promote.

        promote is a boolean that specifies whether promotion is permitted. If
        promote is False, the Field's in the returned Schema may *not* fit all
        corresponding values. This functionality is provided primarily so that
        set operations on the returned field names (in schema) can be performed
        without the expense of promoting the corresponding Field's during
        schema's construction.
        """
        # Determine whether any field names should be excluded and
        # initialize both Schema to be returned and set of all-nulls
        # field names to be returned.
        subset_field_names = field_names_set is not None
        schema = cls()
        nulled_field_names_set = set()
        nulled_field_names_set_add = nulled_field_names_set.add

        # If minimizing, special attention is required, as explained
        # further below.
        if minimize:
            field_name_to_type_to_values = _collections.defaultdict(
                lambda: _util._NumpyTypeDict(list)
                )
                
        # Iterate through each field_data.
        for field_data in field_datas:
            for field_name, value in field_data.iteritems():
                # Completely ignore field name if so specified.
                if subset_field_names and field_name not in field_names_set:
                    continue
                # If value is None (null), remember the field name.
                if value is None:
                    nulled_field_names_set_add(field_name)
                    continue
                # If minimizing, simply "remember" the value for now,
                # and continue.
                if minimize:
                    field_name_to_type_to_values[field_name][type(value)].append(
                        value
                        )
                    continue
                # Add new Field to Schema if field name (at least,
                # paired to a non-null value) was not previously
                # encountered.
                if field_name not in schema:
                    schema[field_name] = Field(value)
                    continue
                # If corresponding Field already exists in Schema,
                # promote it in-place (if necessary) to fit the value.
                if promote:
                    schema[field_name].promote(value)
                    continue

        # If minimizing, carefully populate schema with optimally
        # minimized Field's.
        # Note: The approach below is similar to simply initializing
        # each Field with the first non-null value and then calling
        # .promote() on that Field for each subsequent value, except
        # that it is faster (e.g., minimizes calls to .promote()) and
        # is optimized for some numeric cases (see comment later in this
        # block).
        if minimize:
            Integer = _numbers.Integral
            Real = _numbers.Real
            for field_name, type_to_values in field_name_to_type_to_values.iteritems():
                min_int = max_int = min_float = max_float = max_string = None
                extremes = []
                for type_, values in type_to_values.iteritems():
                    if issubclass(type_, Integer):
                        if min_int is None:
                            min_int = min(values)
                            max_int = max(values)
                            continue
                        min_int = min(min_int, min(values))
                        max_int = min(max_int, max(values))
                        continue
                    if issubclass(type_, Real):
                        if min_float is None:
                            min_float = min(values)
                            max_float = max(values)
                            continue
                        min_float = min(min_float, min(values))
                        max_float = min(max_float, max(values))
                        continue
                    if issubclass(type_, basestring):
                        if max_string is None:
                            max_string = max(values, key=len)
                            continue
                        max_string = max(max_string, max(values, key=len))
                        continue
                    extremes.extend((min(values), max(values)))

                # Note: Initializing the Field with the minimum integer
                # rather than the maximum integer ensures that the
                # lowest-bit integer type possible is used. (See comment
                # in code for Field.__init__() for an example.)
                # Similarly, initializing with floats before promoting
                # by integers ensures that integers that can fit within
                # a given float type will not trigger unnecessary
                # promotion, as would occur if the Field were instead
                # initialized with integers and then promoted by floats.
                # (In both cases, the underlying issue is that once a
                # Field is instantiated, promotion compares a sample
                # value to the *type* of the Field rather than to a
                # remembered history of values.) Fundamentally mixed
                # values (e.g., strings and numbers) are disallowed, so
                # that restrictions also guides prioritization in the
                # list below.
                extremes.extend((max_string, min_float, max_float,
                                 min_int, max_int))
                field = None
                for extreme in extremes:
                    if extreme is None:
                        continue
                    if field is None:
                        field = Field(extreme, minimize=True)
                        continue
                    field.promote(extreme)
                schema[field_name] = field

        # Drop from nulled_field_names_set those field names for which a
        # non-null field_name was ever encountered.
        nulled_field_names_set.difference_update(schema)

        # Return Schema and nulled_field_names_set.
        return (schema, nulled_field_names_set)

    @staticmethod
    def _extract_field_datas(field_datas):
        """
        Simple helper function to extract "field data" dict's.

        If field_datas includes only Geometry's, a list of the .data's of those
        Geometry's is returned. Otherwise, field_datas is returned.

        field_datas is a sequence of Geometry's or mappings of field values
        keyed to the corresponding field names (most commonly, the .data's of
        Geometry's). field_datas must be either all Geometry's or all mappings,
        not mixed.
        """
        if  isinstance(field_datas[0], _geom.Geometry):
            return [field_data.data for field_data in field_datas]
        return field_datas

    @classmethod
    def infer_from_geom(cls, geom, minimize=False):
        """
        Infer and return a Schema from a single Geometry.

        A tuple (schema, nulled_field_names_set) is returned, in which schema is
        a Schema that includes all field names from geom.data that are not None,
        paired to Field's that can fit the corresponding values. The returned
        nulled_field_names_set is a set of any field names for which the paired
        value is None, which renders inferring a suitable Field for these field
        names impossible. nulled_field_names_set may be empty.

        geom is a Geometry whose .data specifies the basis for inferring the
        returned Schema.

        minimize is a boolean that specifies whether the types of the Field's in
        the returned Schema should be minimized to "just" fit the corresponding
        values, including the width of string fields and bit-depths of numeric
        fields. (See documentation for WriteCursor for additional details.) If
        minimize is False, the "naive" behavior described in the documentation
        for Field is used instead, though optionally enhanced by promote.

        Warning: It is strongly advised that minimize be False, since each
        Field in the returned Schema is based on a single value.
        """
        return cls.infer_from_first((geom.data,), minimize)

    @classmethod
    def infer_from_first(cls, field_datas, minimize=False):
        """
        Infer and return a Schema from the first "field data" (field_datas[0]).

        A tuple (schema, nulled_field_names_set) is returned, in which schema is
        a Schema that includes all field names from field_datas[0] that are not
        None, paired to Field's that can fit the corresponding values. The
        returned nulled_field_names_set is a set of any field names for which
        the paired value is None, which renders inferring a suitable Field for
        these field names impossible. nulled_field_names_set may be empty.

        field_datas is a sequence of Geometry's or mappings of field values
        keyed to the corresponding field names (most commonly, the .data's of
        Geometry's). field_datas must be either all Geometry's or all mappings,
        not mixed.

        minimize is a boolean that specifies whether the types of the Field's in
        the returned Schema should be minimized to "just" fit the corresponding
        values, including the width of string fields and bit-depths of numeric
        fields. (See documentation for WriteCursor for additional details.) If
        minimize is False, the "naive" behavior described in the documentation
        for Field is used instead, though optionally enhanced by promote.

        Warning: It is strongly advised that minimize be False, since each
        Field in the returned Schema is based on a single value.
        """
        return cls.infer_by_union((field_datas[0],), minimize)

    @classmethod
    def infer_by_union(cls, field_datas, minimize=True, promote=True):
        """
        Infer and return a Schema from the union of field_datas.

        A tuple (schema, nulled_field_names_set) is returned, in which schema is
        a Schema that includes all field names from field_datas that are not in
        the (also returned) nulled_field_names_set, paired to Field's that can
        fit all the corresponding values. Those Field's may be promoted above
        the type of any of the supplied values. For example, a mix of int32 and
        float32 values could result in a Field with type float64, if the values
        require it. The returned nulled_field_names_set is a set of any field
        names for which all values are None's, which renders inferring a
        suitable Field for these field names impossible. nulled_field_names_set
        may be empty.

        field_datas is a sequence of Geometry's or mappings of field values
        keyed to the corresponding field names (most commonly, the .data's of
        Geometry's). field_datas must be either all Geometry's or all mappings,
        not mixed.

        minimize is a boolean that specifies whether the types of the Field's in
        the returned Schema should be minimized to "just" fit the corresponding
        values, including the width of string fields and bit-depths of numeric
        fields. (See documentation for WriteCursor for additional details.) If
        minimize is False, the "naive" behavior described in the documentation
        for Field is used instead, though optionally enhanced by promote.

        promote is a boolean that specifies whether promotion is permitted. If
        promote is False, the Field's in the returned Schema may *not* fit all
        corresponding values. This functionality is provided primarily so that
        set operations on the returned field names (in schema) can be performed
        without the expense of promoting the corresponding Field's during
        schema's construction. For that use, minimize should also be False for
        maximum performance.
        """
        # *REDEFINITION*
        field_datas = cls._extract_field_datas(field_datas)
        return cls._build_schema(field_datas, None, minimize, promote)

    @classmethod
    def infer_by_intersection(cls, field_datas, minimize=True, promote=True):
        """
        Infer and return a Schema from the intersection of field_datas.

        A tuple (schema, nulled_field_names_set) is returned, in which schema is
        a Schema that includes those field names common to all field_datas that
        are not in the (also returned) field_names_set, paired to Field's that
        can fit all the corresponding values. Those Field's may be promoted
        above the type of any of the supplied values. For example, a mix of
        int32 and float32 values could result in a Field with type float64, if
        the values require it. The returned nulled_field_names_set is a set of
        any field names common to all field_datas for which all values are
        None's, which renders inferring a suitable Field for these field names
        impossible. nulled_field_names_set may be empty.

        field_datas is a sequence of Geometry's or mappings of field values
        keyed to the corresponding field names (most commonly, the .data's of
        Geometry's). field_datas must be either all Geometry's or all mappings,
        not mixed.

        minimize is a boolean that specifies whether the types of the Field's in
        the returned Schema should be minimized to "just" fit the corresponding
        values, including the width of string fields and bit-depths of numeric
        fields. (See documentation for WriteCursor for additional details.) If
        minimize is False, the "naive" behavior described in the documentation
        for Field is used instead, though optionally enhanced by promote.

        promote is a boolean that specifies whether promotion is permitted. If
        promote is False, the Field's in the returned Schema may *not* fit all
        corresponding values. This functionality is provided primarily so that
        set operations on the returned field names (in schema) can be performed
        without the expense of promoting the corresponding Field's during
        schema's construction. For that use, minimize should also be False for
        maximum performance.
        """
        # *REDEFINITION*
        field_datas = cls._extract_field_datas(field_datas)
        common_fieldnames_set = set(field_datas[0])
        common_fieldnames_set_intersection_update = common_fieldnames_set.intersection_update
        for field_data in field_datas:
            common_fieldnames_set_intersection_update(field_data._keys)
        return cls._build_schema(field_datas, common_fieldnames_set, minimize,
                                 promote)

    def order_by(self, keys, ignore_unrecognized=False, ignore_missing=False):
        """
        Order the Schema as specified by keys (in place).

        keys is a sequence of strings that specifies the order of field_names in
        the Schema after the operation.

        ignore_unrecognized is a boolean that specifies whether to silently
        ignore any item in keys that does not correspond to a field name in the
        Schema. If ignore_unrecognized is False (the default) and such an item
        is encountered, a ValueError is raised.

        ignore_missing is a boolean that specifies whether to silently ignore
        any field name in the Schema that is missing from keys. If
        ignore_missing is False (the default) and such an absence is
        encountered, a TypeError is raised. If ignore_missing is instead True,
        field names in the Schema are ordered as specified by keys followed by
        those field names not included in keys, in the same order in which they
        occurred prior to the operation.

        Note: No changes are made to the Schema until it is known that no error
        will be raised during the operation (to avoid interrupted operations
        effectively corrupting the Schema).
        """
        # Make a copy of the current ("old") keys and initialize a list
        # (new_keys) to hold the new key order.
        old_keys = list(self._keys)
        new_keys = []

        # Iterate over each key in keys, appending it to new_keys or
        # optionally raising a ValueError if the key is not an existing
        # field name (or duplicated).
        for key in keys:
            try:
                old_keys.remove(key)
            except ValueError:
                if ignore_unrecognized:
                    continue
                if key in self:
                    raise ValueError(
                        "{!r} is doubly represented in keys)".format(key)
                        )
                raise ValueError(
                    "{!r} is not an existing field name".format(key)
                    )
            new_keys.append(key)

        # Append any field names not in keys to the final new key order
        # or raise a TypeError if this is not allowed.
        if not ignore_missing and old_keys:
            raise TypeError(
                "the following field names are missing from keys: {}".format(
                    ", ".join(old_keys)
                    )
                )
        new_keys.extend(old_keys)

        # Replace pre-operation key order with new key order.
        self._keys = new_keys



###############################################################################
# DATA ACCESS                                                                 #
###############################################################################

# This dictionary maps a 3-tuple of consecutive (but reverse order) path
# level suffixes (that involve a file geodatabase) to a dictionary whose
# keys are boolean-like and represent whether it is preferred (True) or
# not preferred (False) to interpret the target as a subpackage (e.g.,
# feature dataset), or whether this preference is unknown or irrelevant
# (None). The final values for these nested dictionaries are strings
# that give descriptive names for the target's type. See
# Description._process_path().
_gdb_suffix_pattern_to_format = {
    (".gdb", "", ""): "ESRI FileGDB",
    ("", ".gdb", ""): "ESRI FileGDB",
    ("", "", ".gdb"): "ESRI FileGDB"
    }

# This dictionary is an extension of the one immediately above that adds
# a key corresponding to a shapefile.
_supported_suffix_pattern_to_format = _gdb_suffix_pattern_to_format.copy()
_supported_suffix_pattern_to_format[(".shp", "", "")] = "ESRI Shapefile"

# This sets contains all filename extensions (non-empty suffixes from
# the dictionary immediately above) for which writing is supported by
# the current module.
_supported_writable_ext_set = {
    exts[0] for exts in _supported_suffix_pattern_to_format if exts[0]
    }

# This nested dictionary maps format type to reported geometry type to
# accepted geometry types.
_format_type_to_reported_geom_type_to_geom_types = {
    "ESRI FileGDB": {_geom.Point2D: (_geom.Point2D,),
                     _geom.MultiLineString2D: (_geom.LineString2D,
                                               _geom.MultiLineString2D),
                     _geom.MultiPolygon2D: (_geom.Polygon2D,
                                            _geom.MultiPolygon2D)}
    }
# Note: Derive 3D counterparts for 2D records.
for format_type, reported_geom_type_to_geom_types in _format_type_to_reported_geom_type_to_geom_types.iteritems():
    for reported_geom_type, geom_types in reported_geom_type_to_geom_types.items():
        reported_geom_type_to_geom_types[_geom._geom_type_to_3D[reported_geom_type]] = [
            _geom._geom_type_to_3D[geom_type] for geom_type in geom_types
            ]
# Note: Re-use ESRI FileGDB dictionary for ESRI Shapefile record.
_format_type_to_reported_geom_type_to_geom_types["ESRI Shapefile"] = _format_type_to_reported_geom_type_to_geom_types["ESRI FileGDB"]

class DataAccess(_util.Lazy2):
    "Base class for data access."
    _driver = None # This is the default.

    def __init__(self):
        raise TypeError(
            "instantiate one of the following types (or their subtypes) instead: " +
            ", ".join([subcls.name for subcls in type(self).__subclasses__()])
            )

    def _find_gdal_driver_from_path(self, read_only=True, raster=True,
                                    vector=True):
        ext = _os.path.splitext(self.path)[1][1:]
        candidate_drivers = []
        for i in xrange(_gdal.GetDriverCount()):
            driver = _gdal.GetDriver(i)
            is_raster_driver = driver.GetMetadataItem(_gdal.DCAP_RASTER)
            is_vector_driver = driver.GetMetadataItem(_gdal.DCAP_VECTOR)
            if not raster and is_raster_driver:
                continue
            if not vector and is_vector_driver:
                continue
            if not is_raster_driver and not is_vector_driver:
                continue
            driver_exts = driver.GetMetadataItem(_gdal.DMD_EXTENSIONS)
            if driver_exts is None:
                if ext == "":
                    candidate_drivers.append(driver)
            elif ext in driver_exts.split(" "):
                candidate_drivers.append(driver)
        assoc_str = "associated with extension {!r}".format("." + ext)
        if len(candidate_drivers) != 1:
            if candidate_drivers:
                raise TypeError(">1 driver is {}".format(assoc_str))
            raise TypeError("no driver is {}".format(assoc_str))
        driver, = candidate_drivers # *REDEFINITION*
        if not read_only and driver.GetMetadataItem(_gdal.DCAP_CREATE) != "YES":
            raise TypeError(
                "the driver {} does not have write capability".format(assoc_str)
                )
        return driver

#     def set_driver(self, name):
#         """
#         Specify the gdal driver to be used, if it is needed.
#
#         It is not necessary to call this method unless you prefer to override
#         the driver that is automatically identified.
#
#         name is a string that specifies the name of the gdal driver to be used.
#         """
#         driver = _gdal.ogr.GetDriverByName(name)
#         if driver is None:
#             raise TypeError("unrecognized driver name: {!r}".format(name))
#         self._driver = driver


class Description(DataAccess):

    def _process_path(self, iteritems):
        """
        Characterize .path based on the suffixes of its last three levels.

        A 4-tuple of the form (t[0], t[1], t[2], f) is returned, where t is a 3-
        tuple key in *suffix_pattern_to_format and f is its paired value.
        Specifically,
            self.path.endswith(t[0]) --> True
            os.path.dirname(self.path).endswith(t[1]) --> True
            os.path.dirname(os.path.dirname(self.path)).endswith(t[2]) --> True
            f --> string describing the format of the target at .path
        If the suffix pattern is not recognized, None is returned instead of the
        4-tuple.

        iteritems is an iterable of tuples of the form (3-tuple, f) such as
        returned by *suffix_pattern_to_format.iteritems() that specifies the
        scope of permitted path interpretations.
        """
        path = self.path
        path_lower = self.path.lower()
        parent_path = _os.path.dirname(path)
        parent_path_lower = parent_path.lower()
        grandparent_path = _os.path.dirname(parent_path)
        grandparent_path_lower = grandparent_path.lower()
        for ((path_ext, parent_path_ext, grandparent_path_ext),
              format_type) in iteritems:
            if (path_lower.endswith(path_ext) and
                parent_path_lower.endswith(parent_path_ext) and
                grandparent_path_lower.endswith(grandparent_path_ext)):
                break
        else:
            # If all permitted patterns are exhausted, give up.
            return None
        return (path_ext, parent_path_ext, grandparent_path_ext, format_type)

    def _extract_gdb_and_feature_class_paths(self):
        """
        Extract the paths to the file GDB and feature class from .path.

        Returns a 2-tuple of the form (path_to_file_GDB, path_to_feature_class),
        where either or both may be None if the corresponding path is unknown
        (i.e., path_to_feature_class if .path points to a file geodatabase) or
        not applicable (e.g., if .path points to a shapefile). Any child of a
        file geodatabase is assumed to be a feature class.
        """
        gdb_level = self._path_gdb_level
        if gdb_level == -1:
            return (None, None)
        path = self.path
        if gdb_level == 0:
            return (path, None)
        if gdb_level == 1:
            return (_os.path.dirname(path), path)
        return (_os.path.dirname(_os.path.dirname(path)), path)

    def _fetch_format_type_if_supported(self):
        """
        Get name for format type of target at .path, if it is supported.

        If format type of target at .path is not supported, raise an
        AssertionError.
        """
        # If self.path does not match any recognized pattern, raise an
        # AssertionError so that its target will be identified as a
        # regular file or directory.
        format_type = self._fetch_supported_format()
        assert format_type is not None
        return format_type

    # Note: This method should only be called if user promotes gdal
    # after instantiation.
    @staticmethod
    def _get__gdal_path_and_layer_name(self):
        """
        Get 2-tuple (data_source_path, layer_name) as required by gdal.
        """
        if not self._fetch_exists_gdal():
            raise TypeError(
                "gdal cannot see path or it does not exist: {}".format(
                    self.path
                    )
                )
        return self._gdal_path_and_layer_name

    def _fetch_supported_format(self):
        """
        Get format type.

        If data format is not supported, returns None instead.
        """
        format_type = self._process_path(
            _supported_suffix_pattern_to_format.iteritems()
            )
        if format_type is None:
            return None
        return format_type

    def _fetch_gdal_source_and_layer(self, source_only=False, read_only=True):
        """
        Get 2-tuple of gdal (data_source, layer) objects.

        data_source is a gdal.ogr.DataSource and layer is a gdal.ogr.Layer.

        source_only is a boolean that specifies whether data_source should be
        returned alone.

        read_only is a boolean that specifies whether it is permissable for the
        data_source (and hence layer also) to be read-only.
        """
        ## DataSource is now DataSet. Should update names.
        if self.is_in_filesystem:
            path = self.path
            lyr_name = None
        else:
            path, lyr_name = self._gdal_path_and_layer_name
        open_flags = (_gdal.OF_VECTOR |
                      _gdal.OF_READONLY if read_only else _gdal.OF_UPDATE)
        # Note: During testing, the FileGDB driver would raise numerous
        # "Empty Spatial Reference" warnings during opening. Because a
        # file geodatabase has no defined spatial reference, it seems
        # safe to suppress all such warnings.
        if self._path_gdb_level < 0:
            filter_explicit = None
        else:
            filter_explicit = [(1, "Empty Spatial Reference")]
        with _util.GDALWarningFilter(filter_explicit):
            src = _gdal.OpenEx(path, open_flags)
        if src is None:
            raise TypeError(
                "path could not be opened for {}: {}".format(
                    "reading" if read_only else "writing", path
                    )
                )
        if source_only:
            return src
        # Note: gdal's Python bindings require that src be kept alive in
        # order for the layer returned by GetLayer() be kept alive.
        if lyr_name is None:
            lyr = src.GetLayer()
        else:
            lyr = src.GetLayerByName(lyr_name)
        return (src, lyr)

    @staticmethod
    def _get__path_gdb_level(self):
        """
        Return level of .path relative to its file geodatabase.

        -1: .path is not associated with a file geodatabase
         0: .path is a file geodatabase
         1: .path is a child of a file geodatabase
         2: .path is a grandchild of a file geodatabase
        """
        data = self._process_path(
            _gdb_suffix_pattern_to_format.iteritems()
            )
        if data is None:
            return -1
        for level, path_ext in enumerate(data[:3]):
            if path_ext:
                return level

    @staticmethod
    def _get_is_container(self):
        return self.is_vector and not self.is_subpackage and not self.is_package

    @staticmethod
    def _get_is_in_filesystem(self):
        return _os.path.exists(self.path)

    @staticmethod
    def _get_is_vector(self):
        return self.format_type not in ("File", "Directory")


class Information(_util.Instantiable, Description):
    _deleted = False

    def __getattribute__(self, name,
                         names_OK_post_deletion=("_get_parent", "parent")):
        # Prevent any lazy attribute generation (except for .parent)
        # once the target is deleted, since many of the relevant
        # ._get_*()'s assume that the target still exists.
        if (not object.__getattribute__(self, "_deleted") or
            name in names_OK_post_deletion or
            name in object.__getattribute__(self, "__dict__")):
            return object.__getattribute__(self, name)
        raise TypeError("target has been deleted")

    def __repr__(self):
        return "{}({!r}, {})".format(type(self).__name__, self.path,
                                     self.is_subpackage)

    def __init__(self, path, prefer_subpackage=False):
        """
        Get information for a target (geospatial or not). The target must exist.

        For example, if you want to know what sort of geometries are supported
        by the container at a given path, you could use
            Information(path).geom_type
        Or if you wanted to know whether two paths have equivalent spatial
        references, you could use
            test_spatial_reference_equivalence(Information(path1),
                                               Information(path2))

        path is a string that specifies the path to the target. The target's
        path may completely exist in the filesystem (e.g., the path to a .shp
        file) or it may be partially virtual (e.g., the path to a feature class
        within a feature dataset within a file geodatabase .gdb directory).

        prefer_subpackage is a boolean that specifies whether the target should
        be interpreted as a subpackage when there is ambiguity. This ambiguity
        arises because the OpenFileGDB driver in gdal cannot see feature
        datasets. If prefer_subpackage is True and the gdal module is used, any
        specified path within a file geodatabase that is not a feature class is
        assumed to point to a feature dataset.

        Note: The nomenclature used in the current module for the levels of a
        geospatial data hierarchy are:
            container:  The lowest level, which directly contains geometries.
                        Examples: a shapefile, a feature class within a file GDB
            package:    The highest level, which is represented in the
                        filesystem and contains any subpackages.
                        Example: a file geodatabase.
            subpackage: Any level between container and package.
                        Example: a feature dataset within a file geodatabase.
        A given path may have (at most) one of these identities, which are
        assigned in the priority listed above. For example, even though a
        shapefile exists in the filesystem and no higher level exists within its
        data structure, it is a container and not a package.
        """
        self.path = _os.path.abspath(path)
        self.prefer_subpackage = prefer_subpackage
        if not self.exists:
            raise TypeError("path does not exist: {}".format(self.path))

    def add_field(self, name, field):
        """
        Add a field with the specified name.

        name is a string that specifies the name of the field to be added.

        field is a Field or example value that represents the nature of the
        field to be added.

        See .add_fields() for additional documentation and extended
        functionality.
        """
        return self.add_fields({name: field})

    def add_fields(self, fields_dict, ignore_name_conflicts=False,
                   ignore_type_conflicts=False):
        """
        Add fields with the specified names.

        field_dict is a Schema or other mapping of field names (strings) to
        Field's or example values that specifies the fields to be added. For
        example, the .data of a Geometry may be used. For any field_dict value
        that is not a Field, Field(value) is used to interpret the desired
        nature of the field to be added.

        ignore_name_conflicts is a boolean that specifies whether to ignore any
        entry in field_dict whose field name already exists in the target's
        schema.

        ignore_type_conflicts is a boolean that specifies whether to ignore any
        type conflict
        """
        if not self.is_container:
            self._raise_is_not_container()
        new_fields = Schema()
        schema = self.schema
        for field_name, field in fields_dict.iteritems():
            new_fields[field_name] = field
            if field_name in schema:
                if not ignore_name_conflicts:
                    raise TypeError(
                        "a field with this name already exists: {!r}".format(
                            field_name
                            )
                        )
                if (not ignore_type_conflicts and
                    new_fields[field_name] != schema[field_name]):
                    raise TypeError(
                        "a field with this name already exists and is incompatible: {!r} is currently {!r} (not {!r})".format(
                            field_name, schema[field_name],
                            new_fields[field_name]
                            )
                        )
                del new_fields[field_name]
                continue
        del self.schema # Force schema to be re-created on next use.
        return _auto_call()

    def _add_fields_arcpy(self, new_fields, **kwargs):
        for field_name, field in new_fields.iteritems():
            _arcpy.AddField_management(
                self.path, field_name,
                self._find_best_numeric_type_key(_field_type_to_arcpy_type,
                                                 type(field.example)),
                field.width, field.precision, field.width,
                field_is_nullable=field.is_nullable
                )
            if field.default is not None:
                _arcpy.AssignDefaultToField_management(self.path, field_name,
                                                       field.default)
        # Note: At ArcGIS 10.6, attempting to add a reserved field name
        # to a file geodatabase does not raise an error but instead
        # silently appends an underscore. Because the list of reserved
        # field names has changed previously, it seems more "future-
        # proof" to simply test whether the field was indeed added with
        # the specified name.
        ## If an error is raised by future versions of arcpy, could
        ## avoid this test for those versions.
        cur_schema = Information(self.path).schema
        for field_name in new_fields:
            if field_name not in cur_schema:
                raise TypeError(
                    "field could not be added; its name may be reserved (forbidden): {!r}".format(
                        field_name
                        )
                    )

    def _add_fields_gdal(self, new_fields, **kwargs):
        src, lyr = self._fetch_gdal_source_and_layer(read_only=False)
        for field_name, field in new_fields.iteritems():
            gdal_type, gdal_subtype = self._find_best_numeric_type_key(
                _field_type_to_gdal_type_subtype_tuple, type(field.example)
                )
            field_def = _gdal.ogr.FieldDefn(field_name, gdal_type)
            if gdal_subtype:
                field_def.SetSubType(gdal_subtype)
            if field.default is not None:
                field_def.SetDefault(field.default)
            if field.width is not None:
                field_def.SetWidth(field.width)
            if field.precision is not None:
                field_def.SetPrecision(field.precision)
            if not field.is_nullable:
                field_def.SetNullable(False)
            lyr.CreateField(field_def)

    @staticmethod
    def _score_dtype(dtype, kind_to_score={"i": .1, "u": .2, "f": .3}):
        return dtype.itemsize + kind_to_score[dtype.kind]

    @classmethod
    def _find_best_numeric_type_key(cls, d, k, return_value=True):
        """
        Find lowest-bitdepth numpy numeric type that is permitted, compatible.

        d is a dict whose keys specify the permitted numpy numeric types (e.g.,
        numpy.float64). d may also contain other types of keys.

        k is a type that specifies the target. If k is in d, k is treated as the
        best match. Otherwise, k must be a numeric type and the lowest bitdepth
        key in d to which k can be safely cast is treated as the best match,
        with preference given in descending order to signed integers, unsigned
        integers, and floats.

        return_value is a boolean that specifies whether the value in d paired
        to the best match for k should be returned. If return_value is False,
        the best match for k is returned instead.
        """
        if k in d:
            k_best = k
        elif issubclass(k, _numpy.number):
            d_key_dtypes = [d_key().dtype for d_key in d
                            if issubclass(d_key, _numpy.number)]
            d_key_dtypes.sort(key=cls._score_dtype)
            for d_key_dtype in d_key_dtypes:
                if _numpy.can_cast(k, d_key_dtype):
                    k_best = d_key_dtype.type
                    break
            else:
                k_best = None
        else:
            k_best = None
        if k_best is None:
            raise TypeError("not a fully supported type: {}".format(k))
        if return_value:
            return d[k_best]
        return k_best

    def delete(self):
        """
        Delete target.
        """
        if self.is_vector:
            return _auto_call()
        if self.format_type == "File":
            _os.remove(self.path)
        elif self.format_type == "Directory":
            _shutil.rmtree(self.path)
        else:
            self._raise_cannot_delete_format_type()
        try:
            Information(self.path)
        except:
            return
        raise TypeError(
            "target could not be deleted for unknown reason: {!r}".format(
                self.path
                )
            )

    def _delete_arcpy(self):
        _arcpy.Delete_management(self.path)

    def _delete_gdal(self):
        is_datasource = self._path_gdb_level <= 0
        if is_datasource:
            ## Update this code and/or comments.
            driver = self._find_gdal_driver_from_path(raster=False)
            # Note: (Presumably) because the driver was searched for
            # rather than specified, the driver has a generic Delete()
            # (rather than DeleteDataSource(); compare
            # gdal.ogr.GetDriverByName("ESRI Shapefile").
            if driver.Delete(self.path) != _gdal.CE_None:
                raise TypeError("deletion failed: {}".format(self.path))
            return
        if self.is_container:
            src, lyr = self._fetch_gdal_source_and_layer(read_only=False)
            name = lyr.GetName()
            del lyr
            if src.DeleteLayer(name) != _gdal.ogr.OGRERR_NONE:
                raise TypeError("deletion failed: {}".format(self.path))
            return
        self._raise_cannot_delete_format_type()

    @staticmethod
    def _get_exists(self):
        if self.is_in_filesystem:
            return True
        return _auto_call("fetch_exists")

    def _fetch_exists_arcpy(self):
        "Test if self.path exists (inside or outside the filesystem)."
        # If self.path does not exist in the file system, it may still
        # exist if it is a feature class within a file geodatabase.
        return _arcpy.Exists(self.path)

    def _fetch_exists_gdal(self):
        """
        Test if self.path exists *outside* the filesystem.

        If self.path exists (or might exist, if prefer_subpackage initialization
        argument is True), .is_subpackage is also set.
        """
        # If self.path does not exist in the file system, it may still
        # exist if it is a feature class within a file geodatabase.
        gdb_path, fc_path = self._extract_gdb_and_feature_class_paths()
        # If the self.path implies no feature class, then either
        # self.path has nothing to do with file geodatabases (e.g., it
        # is a shapefile or generic file path) or it is itself a path to
        # a non-existent file geodatabase.
        if fc_path is None:
            return False
        # If self.path implies a file geodatabase whose path does not
        # exist, self.path must also not exist.
        if not _os.path.exists(gdb_path):
            return False
        # If self.path implies a feature class that exists, assume that
        # self.path itself exists.
        # Note: This is an assumption because self.path could be
        # something like
        # ...\file_geodatabase\feature_dataset\feature_class
        # and we're only checking that ...\file_geodatabase exists and
        # that some feature class within it is named feature_class.
        # Therefore, feature_class may not be in feature_dataset, but
        # this is difficult (impossible?) to test with the OpenFileGDB
        # driver.
        try:
            src = Information(gdb_path)._fetch_gdal_source_and_layer(True)
        except:
            return False
        fc_name = _os.path.basename(fc_path)
        self._gdal_path_and_layer_name = (gdb_path, fc_name)
        lyr = src.GetLayerByName(fc_name)
        if lyr is None:
            if not self.prefer_subpackage:
                return False
            self.is_subpackage = True
            _warnings.warn(
                "gdal cannot see feature datasets (subpackages) in file geodatabases, but this identity is assumed for the following path, because prefer_subpackage was True at instantiation: {!r}".format(
                    self.path
                    )
                )
        else:
            self.is_subpackage = False
            if self._path_gdb_level > 1:
                _warnings.warn(
                    "gdal cannot see feature datasets (subpackages) in file geodatabases, but because a feature class with the specified name does exist, the feature dataset that is its parent is also assumed to exist: {!r}".format(
                        self.path
                        )
                    )
        return True

    @staticmethod
    def _get_format_type(self):
        """
        Get a descriptive string for .path's format.
        """
        # Note: It is essential that format type be the same whether
        # gdal or arcpy is generally used. For this reason, prefer an
        # internally managed name if possible, which also ensures
        # consistency with Definition objects.
        try:
            return self._fetch_format_type_if_supported()[3]
        except AssertionError:
            # Note: Even a symbolically linked file will be recognized
            # as a file.
            if _os.path.isfile(self.path):
                return "File"
            return "Directory"

    @staticmethod
    def _get_geom_type(self):
        """
        Get the Geometry subclass that best describes .path's Geometry's.

        .path must point to a container.

        Warning: It is not guaranteed that the reported partedness strictly
        applies to all members. For example, the container may report
        MultiPolygon2D whether all members are Polygon2D's, MultiPolygon2D's, or
        a mix of these types.
        """
        if not self.is_container:
            self._raise_is_not_container()
        return _auto_call("fetch_geom_type")

    def _fetch_geom_type_arcpy(self):
        d = _arcpy.Describe(self.path)
        geom_type2D = _geom._arcpy_shape_type_to_geom_type2D[d.shapeType]
        if d.hasZ:
            return _geom._geom_type_to_3D[geom_type2D]
        return geom_type2D

    def _fetch_geom_type_gdal(self):
        src, lyr = self._fetch_gdal_source_and_layer()
        return _geom._wkb_type_to_geom_type[lyr.GetGeomType()]

    @staticmethod
    def _get_is_package(self):
        if not self.is_vector:
            return False
        return _auto_call("fetch_is_package")

    def _fetch_is_package_arcpy(self):
        return _arcpy.Describe(self.path).dataType == "Workspace"

    def _fetch_is_package_gdal(self):
        return self._path_gdb_level == 0

    @staticmethod
    def _get_is_subpackage(self):
        return _auto_call("fetch_is_subpackage")

    def _fetch_is_subpackage_arcpy(self):
        return _arcpy.Describe(self.path).dataType == "FeatureDataset"

    def _fetch_is_subpackage_gdal(self):
        if self._path_gdb_level < 1:
            return False
        # Note: The line below is only used if the user promotes gdal
        # after instantiation (or deletes the .is_subpackage), because
        # otherwise, .is_subpackage would have been set when existence
        # was tested at initialization.
        self._fetch_exists_gdal()
        return self.is_subpackage

    @staticmethod
    def _get_parent(self):
        """
        Get parent of .path (which points to os.path.dirname(self.path).
        """
        return type(self)(_os.path.dirname(self.path), True)

    @staticmethod
    def _get_schema(self):
        """
        Get a Schema representing the target's schema.
        """
        if not self.is_container:
            self._raise_is_not_container()
        return _auto_call("fetch_schema")

    def _fetch_schema_arcpy(self):
        schema = Schema()
        for field in _arcpy.ListFields(self.path):
            field_name = field.name
            try:
                example = _arcpy_type_to_example[field.type]
            except KeyError:
                raise TypeError(
                    "field type of field {!r} is not supported".format(
                        field_name
                        )
                    )
            if not isinstance(example, _OK_field_types):
                # Note: example may be either None, if all fields of a
                # given type are to be ignored, or a function that must
                # be called to disambiguate the best example.
                if example is None:
                    continue
                example = example(self) # *REDEFINITION*
            if isinstance(example, basestring):
                # *REDEFINITION*
                width = field.length
                # *REDEFINITION*
                example = example * width
            elif isinstance(example, _numbers.Number):
                width = field.precision
            else:
                width = None
            schema[field_name] = Field(example, field.defaultValue,
                                       width, field.scale, field.isNullable)
        return schema

    def _fetch_schema_gdal(self):
        src, lyr = self._fetch_gdal_source_and_layer()
        lyr_def = lyr.GetLayerDefn()
        schema = Schema()
        for i in xrange(lyr_def.GetFieldCount()):
            field_def = lyr_def.GetFieldDefn(i)
            field_name = field_def.GetName()
            gdal_field_type = field_def.GetType()
            gdal_field_subtype = field_def.GetSubType()
            try:
                example = _gdal_type_subtype_tuple_to_example[
                    (gdal_field_type, gdal_field_subtype)
                    ]
            except KeyError:
                raise TypeError(
                    "field type of field {!r} is not supported".format(
                        field_name
                        )
                    )
            width = field_def.GetWidth()
            if isinstance(example, basestring):
                # *REDEFINITION*
                example = example * width
            schema[field_name] = Field(example, field_def.GetDefault(),
                                       width, field_def.GetPrecision(),
                                       field_def.IsNullable())
        return schema

    @staticmethod
    def _get_spatial_reference(self):
        """
        Get well-known text representation of a sptial reference.
        """
        if not self.is_vector:
            self._raise_unrecognized_vector_data()
        return _auto_call("fetch_spatial_reference")

    def _fetch_spatial_reference_arcpy(self):
        return _arcpy.Describe(self.path).spatialReference.exporttostring()

    def _fetch_spatial_reference_gdal(self):
        if not self.is_container:
            self._raise_is_not_container()
        src, lyr = self._fetch_gdal_source_and_layer()
        sr = lyr.GetSpatialRef()
        return sr.ExportToWkt()

    def _raise_is_not_container(self):
        raise TypeError(
            "path does not point to a (recognized) container (identified as {}): {!r}".format(
                self.format_type, self.path
                )
            )

    def _raise_cannot_delete_format_type(self):
        raise NotImplementedError(
            "format_type cannot be deleted: {!r}".format(self.format_type)
            )

    def _raise_unrecognized_vector_data(self):
        raise TypeError(
            "path does not point to (recognized) vector data (identified as {}): {!r}".format(
                self.format_type, self.path
                )
            )


class Definition(_util.Instantiable, Description):
    _info = None # This is the default.
    was_created = False # This is the default.

    def __dir__(self, return_set=False, add_names_set=None):
        # Once target is known to exist, include Information's
        # attributes.
        if self._info is not None:
            if add_names_set is None:
                add_names_set = self._info.__dir__(True)
            else:
                add_names_set |= self._info.__dir__(True)
        return Description.__dir__(self, return_set, add_names_set)

    def __getattr__(self, name):
        try:
            return _util.Lazy2.__getattr__(self, name)
        except AttributeError:
            # Once target is known to exist, support access to
            # Information attributes.
            if self._info is None:
                raise
            return getattr(self._info, name)

    def __init__(self, path, geom_type=None, schema=None,
                 spatial_reference=None, prefer_file=False):
        """
        Define expectations for a target that may or may not yet exist.

        The current type is primarily meant for use by functions within the
        current module. Probably its most common direct use by an end user is to
        create a container. For example,
            Definition(...).create()
        will create the entire specified path. In an extreme case, that could
        include nested directories (like os.makedirs()), a file geodatabase, a
        feature dataset, and a feature class. Nonetheless, you will likely find
        that letting WriteCursor create any required target is easier.

        path is a string that specifies the path to the target. The target's
        path may completely exist in the filesystem (e.g., the path to a .shp
        file) or it may be partially virtual (e.g., the path to a feature class
        within a feature dataset within a file geodatabase .gdb directory).

        geom_type is a SingleGeometry subclass that specifies the expected
        geometry type for the target. Specifically, geom_type must be a
        "fully described" subclass, including topological and spatial
        dimensions (e.g., LineString2D). If target is a container, geom_type
        must be specified. Note: In the particular case that path points to a
        path within a package (e.g., a file geodatabase), an unspecified
        geom_type may be interpreted to indicate that the path points to a
        subpackage (which has no geometry type) instead of a container (which
        has a geometry type).

        schema is a Schema that specifies the expected schema for the target.

        spatial_reference is a well-known text representation that specifies the
        expected spatial reference for the target.

        prefer_file is a boolean that specifies whether a non-geospatial path
        should be preferentially interpreted as a file (rather than a folder).

        Note: geom_type, schema, and spatial_reference can be unspecified, but
        this is not recommended if they are relevant for the target. (For
        example, schema is not relevant for a non-container.)
        """
        ## Should geom_type still be a SingleGeometry only?
        self.path = _os.path.abspath(path)
        if geom_type is None:
            if self.is_container:
                raise TypeError("geom_type must be specified for containers")
        ## Which elif block should be retained?
        # elif (issubclass(geom_type, _geom.SingleGeometry) and 
        #       issubclass(geom_type, _util.Instantiable)):
        #     self.geom_type = geom_type
        elif (issubclass(geom_type, _geom.Geometry) and
              issubclass(geom_type, _util.Instantiable)):
            self.geom_type = geom_type
        else:
            raise TypeError(
                "geom_type is not an instantiable subclass of SingleGeometry (with defined topological and spatial dimensions): {}".format(
                    type(geom_type).__name__
                    )
                )
        self.schema = schema
        self.spatial_reference = spatial_reference
        self.prefer_file = prefer_file

    def clear_path(self):
        """
        Delete path, if it exists. Do nothing if it does not exist.
        """
        if not self.path_exists:
            return
        if self._info is None:
            info = Information(self.path)
        else:
            info = self._info
        result = info.delete()
        self._info = None
        del self.path_exists
        return result

    def test_existence(self, test_geom_type=True, test_schema=True,
                       test_spatial_reference=True):
        """
        Test whether target exists with the expected characteristics.

        The existence of the specified path is always tested. If that path
        exists, the fundamental nature of that path is always tested (e.g.,
        .format_type). Finally, more detailed characteristics are tested, as
        constrained by the arguments and nature of the path. (For example, the


        and a boolean is a
        returned to indicate existence. The arguments can limit the other
        characteristics that are tested.
        """
        # Test that .path exists.
        if not self.path_exists:
            return False

        # Test that format types match.
        if self._info is None:
            info = Information(self.path, self.is_subpackage)
        else:
            info = self._info
        if self.format_type != info.format_type:
            return False

        # Test more detailed characteristics, as constrianed by the
        # nature of the target and the argument values. If these
        # characteristics are fully tested, store the corresponding
        # Information object, which effectively adds to self that type's
        # attributes. See .__getattr__().
        if self.is_container or self.is_subpackage:
            if (test_spatial_reference and not
                test_spatial_reference_equivalence(self, info)):
                return False
            if self.is_container:
                if (test_geom_type and
                    self.geom_type != info.geom_type and
                    self.geom_type not in _format_type_to_reported_geom_type_to_geom_types[self.format_type][info.geom_type]):
                    return False
                if test_schema  and self.schema != info.schema:
                    return False
                if test_geom_type and test_schema and test_spatial_reference:
                    self._info = info
            elif test_spatial_reference:
                self._info = info
        else:
            self._info = info
        return True

    @staticmethod
    def _get_format_type(self):
        """
        Get a descriptive string for .path's format.
        """
        if self.is_container or self.is_package or self.is_subpackage:
            return self._fetch_format_type_if_supported()[3]
        elif self.prefer_file:
            return "File"
        return "Directory"

    @staticmethod
    def _get_is_container(self):
        # Set defaults.
        self.is_container = self.is_subpackage = self.is_package = False
        # If self.path is not associated with a file geodatabase, it
        # must be a container, regular file, or regular directory.
        gdb_level = self._path_gdb_level
        if gdb_level == -1:
            if _os.path.splitext(self.path)[1] in _supported_writable_ext_set:
                self.is_container = True
        # If self.path is a file geodatabase, it (obviously) is not a
        # container.
        elif gdb_level == 0:
            self.is_container = False
        # If self.path is a child within a file geodatabase, whether it
        # is a container is ambiguous. Assume that a geom_type would
        # have been specified if it is a container.
        elif gdb_level == 1:
            self.is_container = is_container = hasattr(self, "geom_type")
            self.is_subpackage = not is_container
        # If self.path is a grandchild within the file geodatabase, it
        # must be a
        # container.
        elif gdb_level == 2:
            self.is_container = True
        return self.is_container

    @staticmethod
    def _get_is_package(self):
        self.is_container
        return self.is_package

    @staticmethod
    def _get_is_subpackage(self):
        self.is_container
        return self.is_subpackage

    @staticmethod
    def _get_geom_type(self):
        # Note: If user manually deletes self.geom_type, which is
        # assigned at instantiation, the raised error is deceptive.
        raise TypeError("geom_type is not defined for non-containers")

    @staticmethod
    def _get_parent(self):
        parent = type(self)(_os.path.dirname(self.path))
        if parent.is_subpackage:
            parent.spatial_reference = self.spatial_reference
        return parent

    @staticmethod
    def _get_path_exists(self):
        """
        Test simply that "something" exists at the specified path.
        """
        try:
            Information(self.path)
        except:
            return False
        return True

    def create(self, overwrite=False):
        ## Note: Could eventually permit overwriting.
        """
        Create the defined target.

        overwrite is a boolean that specifies whether .path may be overwritten.
        If overwrise is False and .path exists, a TypeError is raised.
        """
        # If .path already exists, clear it if overwriting is permitted,
        # otherwise error.
        if self.path_exists:
            if not overwrite:
                raise TypeError("path already exists: {!r}".format(self.path))
            self.clear_path()

        # If parent does not exist, create it.
        # Note: In turn, if grandparent doesn't exist, parent.create()
        # will call grandparent.create(), and so on, unto os.makedirs()
        # is called, if necessary.
        if not self.parent.path_exists:
            self.parent.create()

        # Pass vector creation onto module-specific methods, but create
        # intermediate directories or a file immediately.
        if self.is_vector:
            _auto_call()
        elif self.format_type == "Directory":
            _os.makedirs(self.path)
        elif self.format_type == "File":
            open(self.path, 'a').close()
        else:
            raise TypeError(
                "format_type is not creatable: {!r}".format(self.format_type)
                )

        # Record creation.
        # Note: Storing the corresponding Information object effectively
        # adds to self that type's attributes. See .__getattr__().
        self.was_created = True
        self._info = Information(self.path, self.is_subpackage)

        # If a container was just created it, apply the specified
        # schema.
        if self.is_container:
            self._info.add_fields(self.schema)

    def _create_arcpy(self, **kwargs):
        gdb_level = self._path_gdb_level
        dirname = _os.path.dirname(self.path)
        basename = _os.path.basename(self.path)
        if self.is_package:
            if gdb_level:
                raise TypeError("path does not end in .gdb")
            _arcpy.CreateFileGDB_management(dirname, basename)
        elif self.is_subpackage:
            if gdb_level != 1:
                raise TypeError("parent path does not end in .gdb")
            sr = _arcpy.SpatialReference()
            sr.loadFromString(self.spatial_reference)
            _arcpy.CreateFeatureDataset_management(dirname, basename, sr)
        else:
            assert self.is_container
            sr = _arcpy.SpatialReference()
            sr.loadFromString(self.spatial_reference)
            _arcpy.CreateFeatureclass_management(
                dirname, basename,
                _geom._geom_type2D_to_arcpy_shape_type[_geom._geom_type_to_2D[self.geom_type]].upper(),
                None, "DISABLED",
                "ENABLED" if self.geom_type.is_3D else "DISABLED", sr
                )

    def _create_gdal(self, **kwargs):
        # If self.path points to a file geodatabase or is unrelated to a
        # file geodatabase (e.g., a shapefile), that source must be
        # created first.
        is_datasource = self._path_gdb_level <= 0
        if is_datasource:
            driver = self._find_gdal_driver_from_path(False, False)
            # Note: (Presumably) because the driver was searched for
            # rather than specified, the driver has a generic Create()
            # (rather than CreateDataSource()) and a generic call
            # signature (compare
            # gdal.ogr.GetDriverByName("ESRI Shapefile")). The arguments
            # specify the path and four (irrelevant) raster parameters
            # (x-dimension, y-dimension, band count, and bit type).
            ds = driver.Create(self.path, 0, 0, 0, _gdal.GDT_Unknown)
        if self.is_container:
            if not is_datasource:
                ds = self.parent._fetch_gdal_source_and_layer(True, False)
            sr = _gdal.osr.SpatialReference()
            sr.ImportFromWkt(self.spatial_reference)
            kwargs = {}
            if self.parent.is_subpackage:
                kwargs["FEATURE_DATASET"] = _os.path.basename(self.parent.path)
            ds.CreateLayer(
                "out" if is_datasource else _os.path.basename(self.path),
                sr, self.geom_type.wkb_type, **kwargs
                )


class Cursor(DataAccess):
    "Base class to support cursors."
    _arcpy_cursor = None
    _gdal_objects = None
    entered = False

    def __enter__(self):
        return self._on_enter()

    def _on_enter(self):
        # Disallow nested with clauses.
        if self.entered:
            raise RuntimeError("cannot doubly enter cursor")

        # Register that a managed context was entered and return self.
        self.entered = True
        return self

    def __exit__(self, type_, value, traceback):
        return self._on_exit()

    def _on_exit(self):
        # Register that the managed contexted was exited.
        self.entered = False

        # Delete module-specific data.
        if self._arcpy_cursor is not None:
            self._arcpy_cursor.__exit__(None, None, None)
            del self._arcpy_cursor
        if self._gdal_objects is not None:
            # Note: The attribute below is especially important for
            # gdal. In that module, an object (e.g., gdal.ogr.Layer) is
            # often unusable if its parent (e.g., gdal.ogr.DataSource)
            # dies (i.e., becomes fully dereferened), so the attribute
            # that is deleted below is required throughout the "session"
            # represented by the managed context. On the other hand,
            # gdal is only guaranteed to write data when all dependent
            # objects are dead, making the deletion below critical.
            del self._gdal_objects


class ReadCursor(_util.Instantiable, Cursor):

    ## Should also support a sequence of Geometry's for path, in which
    ## case field_names would (presumably) be ignored. This would make
    ## it easier to write functions that can accept either format and
    ## use the same code.
    def __init__(self, path, field_names=None, partedness="COERCE_TO_MULTI",
                 strict=True):
        """
        Read Geometry's from a container, within a managed context.

        path is a string that specifies the path of the target container.

        field_names is an iterable of strings that specifies the names of the
        fields whose values will be available in the .data of each returned
        Geometry. If not specified (None), it will default to all field names.

        partedness is a string that specifies the partedness of the returned
        Geometry's:
            "COERCE_TO_MULTI"   Each Geometry is returned as a MultiGeometry.
            "NATIVE"            Each Geometry is returned as a MultiGeometry
                                unless it has 1 part, in which case it is
                                returned as a SingleGeometry.
            "EXPLODE"           Each Geometry is returned as >=1
                                SingleGeometry's. If multiple SingleGeometry's
                                represent the same record, they will each have
                                the same .ID and equivalent .data.
            "ERROR_IF_MULTI"    Each Geometry is returned as a SingleGeometry,
                                and an error is raised if a Geometry with >1
                                part is encountered.

        strict is a boolean that specifies whether the values stored to .data
        will be coerced to the standard field types used in the current module
        (i.e., by Field). For example, if strict is True, a value might be a
        numpy.int8 instead of a (Python) int. Such "strict" types better
        represent the precision of the read value and can therefore pass this
        precision along. (See "naive" approach in Field documentation.) However,
        operations involving them may be somewhat slower than similar operations
        on built-in Python types.
        """
        self.path = path
        self.field_names = field_names
        self.partedness = _validate_string_option(
            partedness, "partedness", ("COERCE_TO_MULTI", "NATIVE", "EXPLODE",
                                       "ERROR_IF_MULTI")
            )
        self.strict = strict

    def __iter__(self):
        if not self.entered:
            raise RuntimeError("outside managed context ('with' clause)")
        info = Information(self.path)
        if self.field_names is None:
            field_names = list(info.schema)
        else:
            field_names = list(self.field_names)
        multi_geom_type = _geom._geom_type_to_multi[info.geom_type]
        if self.partedness == "COERCE_TO_MULTI":
            wkx_geom_type = multi_geom_type
        else:
            wkx_geom_type = multi_geom_type.member_type
        coerce_partedness = True
        # Note: Suppress (erroneous) "unused variable" code warnings.
        field_names, wkx_geom_type, coerce_partedness
        generator = _auto_call("iter")
        if self.partedness in ("EXPLODE", "ERROR_IF_MULTI"):
            return self._implement_partedness(generator)
        return generator

    def _iter_arcpy(self, info, wkx_geom_type, field_names, coerce_partedness,
                    **kwargs):
        strict = self.strict
        if strict:
            types = [type(info.schema[field_name].example)
                     for field_name in field_names]
        spatial_reference = info.spatial_reference
        ## Note: At 10.4, ArcGIS does not support 3D wkb's, neither ISO
        ## nor PostGIS "extended" forms.
        use_wkt = info.geom_type.is_3D
        if use_wkt:
            from_wkt = wkx_geom_type.from_wkt
            self._arcpy_cursor = _arcpy.da.SearchCursor(
                self.path, ["OID@", "SHAPE@WKT"] + field_names
                )
        else:
            from_wkb = wkx_geom_type.from_wkb
            self._arcpy_cursor = _arcpy.da.SearchCursor(
                self.path, ["OID@", "SHAPE@WKB"] + field_names
                )
        with self._arcpy_cursor as cursor:
            for row in cursor:
                if use_wkt:
                    geom = from_wkt(row[1], coerce_partedness)
                else:
                    geom = from_wkb(str(row[1]), coerce_partedness)
                geom.ID = row[0]
                # Note: .data could be stored as an OrderedStringDict
                # instead, which would enable the use of index keys.
                # However, such a .data would have lower performance,
                # and index keys could be confusing, especially if not
                # all fields in the container's schema are used or when
                # working with Geometry's from different containers. If
                # OrderedStringDict is ever so used, it should support
                # deep copying (to support, for example,
                # LineString.flip()).
                if strict:
                    geom.data = {
                        field_name: type_(value) for field_name, type_, value in
                        zip(field_names, types, row[2:])
                        }
                else:
                    geom.data = {field_name: value for field_name, value in
                                 zip(field_names, row[2:])}
                geom.spatial_reference = spatial_reference
                yield geom

    def _iter_gdal(self, info, wkx_geom_type, field_names, coerce_partedness,
                    **kwargs):
        from_wkb = wkx_geom_type.from_wkb
        get_field_data = info.schema._make_field_data_getter_or_setter_gdal(
            field_names, True, self.strict
            )
        get_ID = _gdal.ogr.Feature.GetFID.im_func
        get_geom = _gdal.ogr.Feature.GetGeometryRef.im_func
        get_wkb = _gdal.ogr.Geometry.ExportToWkb.im_func
        spatial_reference = info.spatial_reference
        self._gdal_objects = (src, lyr) = info._fetch_gdal_source_and_layer()
        for feature in lyr:
            # Note: Specify a little endian wkb.
            geom = from_wkb(get_wkb(get_geom(feature), 1), coerce_partedness)
            geom.ID = get_ID(feature)
            # Note: .data could be stored as an OrderedStringDict
            # instead, which would enable the use of index keys.
            # However, such a .data would have lower performance,
            # and index keys could be confusing, especially if not
            # all fields in the container's schema are used or when
            # working with Geometry's from different containers.
            geom.data = {field_name: value for field_name, value in
                         zip(field_names, get_field_data(feature))}
            geom.spatial_reference = spatial_reference
            # Note: Because of _field_type_to_get_strict_gdal (used in
            # Schema._make_field_data_getter_or_setter_gdal), strict is
            # already enforced, obviating the lines below.
            # if strict:
            #     schema_apply(geom.data)
            yield geom

    def _implement_partedness(self, generator):
        explode = self.partedness == "EXPLODE"
        for geom in generator:
            if isinstance(geom, _geom.SingleGeometry):
                yield geom
            elif explode:
                geom._cascade(True)
                for g in geom:
                    yield g
            else:
                raise TypeError(
                    "A MultiGeometry with >1 part was encountered."
                    )


class WriteCursor(_util.Instantiable, Cursor):
    cache = None
    write_on_exit = True

    def __init__(self, path, mode="a", fields=None, spatial_reference=None,
                 nulled_fields="ERROR", write_on_exit=True):
        """
        Write (or append) Geometry's from a container, within a managed context.

        path is a string that specifies the path of the target container.

        mode is a character that specifies whether each Geometry should be
        appended ("a"), written ("w"), or written after exclusive creation ("x")
        to the target. Similar to Python's built-in open(), "a", "w", and "x"
        will each create the target container if it does not already exist.
        However, if the target container does already exist, "w" will
        *overwrite* it, "x" will error, and "a" will append to it. (Note that
        the concept of mode "x" only appears in Python 3 but is nonetheless used
        here.)

        fields is a Schema, function, or dictionary that implies what fields
        should be created or added to the target container. If fields is a
        function (or other callable object), it will be called just prior to
        writing the first Geometry. If write_on_exit is True, its only argument
        will be the list of all Geometry's to be written. If write_on_exit is
        False, its only argument will be the first Geometry to be written. In
        both cases, it must return a 2-tuple (schema, nulled_field_names_set) as
        returned by Schema.infer_by_union(). schema (a Schema) should be
        populated only with field names to be written to the target container
        from each Geometry, whereas nulled_field_names_set need only contain any
        additional field names (as strings) that should participate in
        nulled_fields functionality. If fields is instead a Schema, the result
        will be the same as if a function had been specified and (fields, set())
        were returned. Finally, if fields is specified by a dictionary (such as
        the .data of a Geometry), the result will be the same as if a function
        had been specified and (Schema.fromdict(fields), set()) were returned.
        If fields is not specified, it defaults to Schema.infer_by_union() if
        write_on_exit is True and Schema.infer_from_geom() if write_on_exit is
        False. (Note that the respective call signatures of these functions and
        the format of their returned objects conform to the same requirements
        imposed on any callable object specified for fields.)

        spatial_reference is a string or function that implies the spatial
        reference of the target container. If spatial_reference is a function
        (or other callable object), it will be called just prior to writing the
        first Geometry. If write_on_exit is True, its only argument will be the
        list of all Geometry's to be written. If write_on_exit is False, its
        only argument will be the first Geometry to be written. In both cases,
        it must return a string that is a well-known text (wkt) representation
        of the spatial reference that the target container should have. If
        spatial_reference is instead a string, the result will be the same as if
        a function had been specified and spatial_reference was returned when it
        was called. If spatial_reference is not specified, it defaults to a
        function that returns a spatial reference wkt compatible with the
        .spatial_reference of the first Geometry (if write_on_exit is False) or
        of all Geometry's to be written (if write_on_exit is True), ignoring any
        .spatial_reference that is unspecified (None). In both cases, if no such
        compatible spatial reference can be identified, an error is raised,
        which prevents (if write_on_exit is True) or terminates (if
        write_on_exit is False) the writing out of Geometry's. If no
        .spatial_reference is specified for the examined Geometry's, that
        function returns None, which is interpreted as an indication to ignore
        the spatial reference of the target container (if it exists) or leave it
        as unspecified as possible (if it must be created). Any callable object
        specified for spatial_reference may also return None to replicate this
        behavior.

        nulled_fields is a string that specifies the action to be taken if any
        field is to be populated only with nulls (None's) and its specification
        cannot be inferred. More precisely, the specified action is taken if any
        field name (effectively) returned by fields()[1] is not present in the
        minimum expected schema. The options for nulled_fields, and the
        resulting action are:
            "DROP":   (silently) do not create or write to the affected fields
            "WARN":   same as for "DROP", but issue a descriptive UserWarning
            "ERROR":  raise a descriptive TypeError (This is the default.)
            None:     same as "ERROR"

        write_on_exit is a boolean that specifies whether Geometry's should be
        cached when .append() is called and only written when the WriteCursor's
        managed context (i.e., the indented block under the "with statement" is
        exited. If write_on_exit is False, each Geometry is instead written
        immediately when .append() is called. Writing on exit has the advantage
        of better-informed schema inference (see fields) and avoids the
        possibility that writing will be interrupted by an incompatible spatial
        reference (see spatial_reference). Writing on exit may also be slightly
        faster but the memory footprint of all Geometry's to be written out
        accumulates prior to exit, which can dramatically slow performance if
        swapping becomes necessary. If write_on_exit is False, .append() returns
        the feature (object) ID, if possible.

        Note: By default (i.e., if write_on_exit is True and fields is not
        specified), if path does not already exist (or mode is "w"),
        Schema.infer_by_union() is used to generate the Schema of the container
        that is created at path. Both Schema.infer_by_union() and the equally
        useful Schema.infer_by_intersection() use a "minimizing" approach to
        inferring the types of fields. Specifically, they examine all values
        that are about to be written to each field and choose the smallest
        suitable Field specification, including .width for string Field's and
        bit-depth for numeric fields. (This is equivalent to applying the second
        option described below to the values that are about to be written to
        each Field.) If you anticipate that the fields may need to fit more
        "extreme" values (i.e., numbers that are more negative, numbers that are
        more positive, or strings that are longer) than those that are initially
        written out, you have a couple options:
            1) Specify a custom function for the fields argument that internally
               calls Schema.infer_*(..., minimize=False). This will force a
               "naive" approach to specifying Field's, but see documentation for
               Field for potential pitfalls with that approach.
            2) Before instantiating WriteCursor, generate a Schema from the most
               extreme values that you anticipate may be stored in each field.
               If the field will hold both positive and negative values, you
               should initially specify the Field by
               example=most_negative_number and then call
               .promote(most_positive_number) on that Field. If the field will
               ultimately contain any floats, be sure that
               most_negative_number and most_positive_number are expressed as
               floats. Finally, specify WriteCursor(..., fields=this_schema).

        Note: Writing out can be resumed, even if interrupted by an
        error.
            with WriteCursor(..., mode="a") as curs:
                for geom in geoms:
                    curs.append(geom)
            # Note: All Geometry's in geoms were written out.
            with curs:
                for geom in more_geoms:
                    curs.append(geom)
            # Note: All Geometry's in more_geoms were written out.
            # Note: Although writing can be resumed, as above, this provides
            # fields (if specified by a function, as is the default) with fewer
            # Geometry's from which to infer the target container's schema,
            # which could cause problems, as discussed further above.
            try:
                with WriteCursor(..., mode=?, write_on_exit=True) as curs2:
                    for geom in geoms:
                        curs.append(geom)
                    raise TypeError()
            except TypeError:
                pass
            # Note: No Geometry's were written out because managed context
            # exited with an error and write_on_exit was True. However, all
            # Geometry's from geoms were cached to curs2.cache.
            with curs2:
                for geom in more_geoms:
                    curs.append(geom)
            # Note: All Geometry's (geoms + more_geoms) were written out,
            # regardless of the mode.
            # Warning: If write_on_exit were instead False for curs2, the entire
            # target container would be deleted, re-created, and populated only
            # with Geometry's from more_geoms when exiting the "with curs2"
            # managed context!
        """
        self._args = locals().copy()
        del self._args["self"]
        self.__dict__.update(self._args)
        # *REDEFINITION*
        mode = _validate_string_option(mode, "mode", ("a", "w", "x"))
        if fields is None:
            # *REDEFINITION*
            fields = (Schema.infer_by_union if write_on_exit
                      else Schema.infer_from_geom)
        if hasattr(fields, "__call__"):
            self._fields = fields
        elif isinstance(fields, Schema):
            self._fields = fields
        elif isinstance(fields, dict):
            self._fields = Schema.fromdict(fields)
        else:
            raise TypeError(
                "fields type {!r} not acceptabled".format(type(fields))
                )
        if spatial_reference is None:
            # *REDEFINITION*
            if write_on_exit:
                spatial_reference = _isolate_spatial_reference
            else:
                spatial_reference = lambda geom: getattr(geom,
                                                         "spatial_reference",
                                                         None)
        if hasattr(spatial_reference, "__call__"):
            self._spatial_reference = spatial_reference
        elif isinstance(spatial_reference, basestring):
            self._spatial_reference = spatial_reference
        else:
            raise TypeError(
                "spatial_reference type {!r} not acceptabled".format(
                    type(spatial_reference)
                    )
                )
        if nulled_fields is None:
            nulled_fields = "ERROR"
        if (isinstance(nulled_fields, basestring) and
            nulled_fields.upper() in ("DROP", "WARN", "ERROR")):
            self.nulled_fields = nulled_fields.upper()
        else:
            raise TypeError("unsupported nulled_fields: {!r}".format(mode))

    def __enter__(self):
        # Execute standard cursor preparation.
        self._on_enter()

        # Prepare to write on exit, if necessary.
        if self.write_on_exit:
            if self.cache is None:
                self.cache = _deque()
            self.append = self.cache.append

        # Return the cursor.
        return self

    def __exit__(self, type_, value, traceback):
        # In case .append() was replaced, revert to the class-level
        # default so that future calls will have the correct behavior.
        try:
            del self.append
        except AttributeError:
            pass

        # Execute standard cursor cleanup.
        self._on_exit()

        # If no Geometry's are cached, or if an error was encountered,
        # return immediately.
        if self.cache is None or not self.cache or value is not None:
            return

        # Re-enter the managed context of self and attempt to write out
        # (and uncache) each geometry in turn.
        try:
            self.write_on_exit = False
            with self:
                cache = self.cache
                for geom in tuple(cache):
                    self.append(geom)
                    del cache[0] # Faster with a deque than a list.
        finally:
            self.write_on_exit = True

    def _process_inferred_schema(self, inferred_schema,
                                 all_None_field_names_set):
        if all_None_field_names_set:
            if self.nulled_fields == "DROP":
                # *REDEFINITION*
                inferred_schema = inferred_schema.difference(
                    all_None_field_names_set
                    )
            else:
                base_message = "the following fields contain only None's{{}}: {}".format(
                    ", ".join(sorted(all_None_field_names_set))
                    )
                if self.nulled_fields == "WARN":
                    _warnings.warn_explicit(
                        base_message.format(" and will not be used"),
                        UserWarning, None, None
                        )
                raise TypeError(base_message.format(""))
        ## Note: This redundant tuple is only intended as a temporary
        ## kludge to avoid bigger changes to code elsewhere.
        return (inferred_schema, inferred_schema)

    def append(self, geom):
        """
        Write out first Geometry, or cache it for writing.

        geom is the Geometry to be written out or cached for writing.
        """
        # Note: The current function is called both when the user
        # specifies write_on_exit=False at instantiation and
        # (internally) to implement writing on exit after a cache is
        # fully populated with Geometry's (in which case .write_on_exit
        # is temporarily set to False). In both cases, the method is
        # only called once (per managed context), to initialize, and
        # then (internally) replaced by a module-specific replacement.
        if not self.entered:
            raise TypeError("cannot append outside of a managed 'with'-context")

        # Because this method should only be called once (per managed
        # context), error if user attempts to call it again, as would
        # only happen in a situation similar to:
        #     with WriteCursor(..., write_on_exit=False) as curs:
        #         append = curs.append
        #         append(geom1)
        #         append(geom2) # Raise error here.
        #     append = WriteCursor.append
        #     with WriteCursor(..., write_on_exit=x) as curs:
        #         append(curs, geom1) # Raise error here if x is True.
        #         append(curs, geom2) # Raise error here if x is False.
        if (self.write_on_exit or
            self.append.im_func is not type(self).append.im_func):
            raise TypeError("do not detach .append from WriteCursor")

        # Determine whether (the first) geom or a stored cache is the
        # relevant "seed" from which to infer the schema and spatial
        # reference, as necessary.
        is_uncached = self.cache is None or not self.cache
        if is_uncached:
            seed = geom
        else:
            seed = self.cache

        # Infer the schema and spatial reference, if they are implicit.
        fields = self._fields
        if isinstance(fields, Schema):
            min_schema = fields
        else:
            # *REDEFINITION*
            fields, min_schema = self._process_inferred_schema(*fields(seed))
        spatial_reference = self._spatial_reference
        if not isinstance(spatial_reference, basestring):
            spatial_reference = spatial_reference(seed) # *REDEFINITION*
            if spatial_reference is None:
                raise TypeError("no spatial reference is specified")

        # If a cache is populated, verify that no spatial reference or
        # geometry conflicts exist.
        # Note: Allow for possibility that geom is a highly derived type
        # (e.g., LineSegment2D instead of LineString2D).
        geom_type = _geom._wkb_type_to_geom_type[geom.wkb_type]
        if not is_uncached:
            # Note: If self._spatial_reference is set to
            # _isolate_spatial_reference, spatial reference consistency
            # was tested when spatial reference was inferred.
            if self._spatial_reference is not _isolate_spatial_reference:
                spat_refs = list(seed)
                spat_refs.append(spatial_reference)
                _isolate_spatial_reference(spat_refs)
            topological_dimension = geom_type.topological_dimension
            for cached_geom in self.cache:
                if cached_geom.topological_dimension != topological_dimension:
                    raise TypeError(
                        "a {} is not compatible with target {} type".format(
                            type(cached_geom).__name__,
                            geom_type.__name__
                            )
                        )
            if geom_type.is_3D:
                for cached_geom in self.cache:
                    if not cached_geom.is_3D:
                        raise TypeError("Geometry is 2D but target is 3D")

        # Define target container and address any conflicts before
        # creating it, if necessary.
        d = Definition(self.path, geom_type, min_schema, spatial_reference)
        if not d.path_exists or self.mode == "w":
            d.create(True)
            ## Delete this assignment when shapefile null kludge is 
            ## retired.
            i = Information(self.path)
        elif self.mode == "x":
            raise TypeError(
                'target exists but mode is "x" (exclusive writing): {}'.format(
                    self.path
                    )
                )
        else:
            i = Information(self.path)
            if d.test_existence(test_schema=False):
                try:
                    # *REDEFINITION*
                    min_schema = i.schema.union(min_schema)
                except:
                    raise TypeError(
                        "target exists, but one or more of its fields cannot receive the data type specified in fields"
                        )
                min_schema.difference_update(i.schema)
                i.add_fields(min_schema)
            elif not i.is_container:
                raise TypeError(
                    "target exists but is not a recognized vector container: {!r}".format(
                        self.path
                        )
                    )
            elif i.geom_type != type(geom):
                raise TypeError(
                    "target exists but has the wrong geom_type: {!r}".format(
                        i.geom_type
                        )
                    )
            elif not test_spatial_reference_equivalence(i, spatial_reference):
                raise TypeError(
                    "target exists but has the wrong spatial reference"
                    )
            else:
                raise TypeError(
                    "target exists but is not compatible for appending"
                    )

        # Direct future calls to self.append() to the relevant module-
        # specific version.
        append, result = _auto_call()
        self.append = append

        # Return the result from appending the first geom.
        return result

    def _append_arcpy(self, geom, is_uncached, fields, spatial_reference,
                      geom_type, **kwargs):
        # Note: This method is called only on the first call of
        # self.append(). The function that it returns is called on all
        # future calls to self.append().
        row_field_names = fields.keys()
        ## Note: At 10.4, ArcGIS does not support 3D wkb's, neither ISO
        ## nor PostGIS "extended" forms.
        use_wkt = geom.is_3D
        if use_wkt:
            row_field_names.append("SHAPE@WKT")
        else:
            row_field_names.append("SHAPE@WKB")
        _arcpy_cursor = _arcpy.da.InsertCursor(self.path, row_field_names)
        def append(geom, is_uncached=is_uncached,
                   spatial_reference=spatial_reference, geom_type=geom_type,
                   field_names=fields.keys(), use_wkt=use_wkt,
                   _insert_row=_arcpy_cursor.insertRow,
                   is_shapefile=kwargs["i"].format_type=="ESRI Shapefile"):
            """
            Write out a Geometry.

            geom is the Geometry to be written out.

            Warning: Allow all other arguments to default!
            """
            if is_uncached:
                if (geom.topological_dimension != geom_type.topological_dimension or
                    geom.is_3D < geom_type.is_3D):
                    raise TypeError(
                        "a {} is not compatible with target {} type".format(
                            type(geom).__name__,
                            geom_type.__name__
                            )
                        )
                _isolate_spatial_reference((geom, spatial_reference))
            row = map(geom.data.get, field_names)
            ## Note: Temporary kludge to ensure that no (abortive) 
            ## attempt is made to write a null to a shapefile. Once this
            ## kludge is retired, also retire is_shapefile argument.
            if is_shapefile and None in row:
                row = [0 if v is None else v for v in row]
            if use_wkt:
                row.append(geom.wkt)
            else:
                row.append(bytearray(geom.wkb))
            return _insert_row(row)
        return append, append(geom)

    def _append_gdal(self, geom, is_uncached, fields, spatial_reference,
                      geom_type, **kwargs):
        # Note: This method is called only on the first call of
        # self.append(). The function that it returns is called on all
        # future calls to self.append().
        i = Information(self.path)
        if self._driver is not None:
            i.set_driver(self._driver)
        src, lyr = i._fetch_gdal_source_and_layer(read_only=False)
        lyr_def = lyr.GetLayerDefn()
        self._gdal_objects = (src, lyr, lyr_def)
        field_names = fields.keys()
        set_field_data = fields._make_field_data_getter_or_setter_gdal(
            field_names, False
            )
        sync_to_disk = lyr.SyncToDisk if is_uncached else None
        def append(geom, is_uncached=is_uncached,
                   spatial_reference=spatial_reference, geom_type=geom_type,
                   field_names=field_names, set_field_data=set_field_data,
                   lyr_def=lyr_def,
                   create_gdal_geom=_gdal.ogr.CreateGeometryFromWkb,
                   create_feature=lyr.CreateFeature, sync_to_disk=sync_to_disk):
            """
            Write out a Geometry.

            geom is the Geometry to be written out.

            Warning: Allow all other arguments to default!
            """
            if is_uncached:
                if (geom.topological_dimension != geom_type.topological_dimension or
                    geom.is_3D < geom_type.is_3D):
                    raise TypeError(
                        "a {} is not compatible with target {} type".format(
                            type(geom).__name__,
                            geom_type.__name__
                            )
                        )
                _isolate_spatial_reference((geom, spatial_reference))
            feature = _gdal.ogr.Feature(lyr_def)
            set_field_data(feature, map(geom.data.get, field_names))
            feature.SetGeometryDirectly(create_gdal_geom(geom.wkb2D))
            create_feature(feature)
            if is_uncached:
                sync_to_disk()
                return feature.GetFID()
            return None
        return append, append(geom)
