"""
Inexpensive, broadly used support utilities.
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
# Note: To avoid circular imports, which are not (well) supported in
# Python, numgeo.geom is not imported here but is instead imported
# (from) in Lazy2.

# Import external.
import collections as _collections
import inspect as _inspect
import itertools as _itertools
import numpy as _numpy
import operator as _operator
import os as _os
import re as _re
import sys as _sys
import time as _time


###############################################################################
# LOCALIZATION                                                                #
###############################################################################

# Derived from built-ins.
_neg_1_1_tuple = (-1, 1)

# Derived from internal.

# Derived from external.
_izip = _itertools.izip
_numpy_arange = _numpy.arange
_numpy_as_strided = _numpy.lib.stride_tricks.as_strided
_numpy_concatenate = _numpy.concatenate
_numpy_divide = _numpy.divide
_numpy_float64 = _numpy.dtype("<f8")
_numpy_int64 = _numpy.dtype("<i8")
_numpy_empty = _numpy.empty
_numpy_not_equal = _numpy.not_equal



###############################################################################
# NON-ARRAY CLASSES                                                           #
###############################################################################

class Object(object):
    pass


class Deconstruction(object):
    "Temporary class to support global assign_deconstructor()."

    get_id = _itertools.count().next
    id_to_weakref_callback_pair = {}

    @classmethod
    def assign_deconstructor(cls, obj, func, post=True, prewarn=True):
        """
        Assign a deconstructor for an object.

        If post and prewarn are both False, the current function is similar to
            type(obj).__del__ = func
        except that func is effectively registered for an instance by assigning
        a one-off sublcass of obj.__class__ to obj.__class__. Even if that type
        is inherited by future instances, only the original instance will
        trigger deconstruction (by func(obj, True)) and post functionality.

        obj is an instance of a new-style class that specifies the object for
        which deconstruction will be supported.

        func is a function that specifices the deconstructor for obj.
        func(instance, other) will be called, if possible, whenever obj or
        another instance of the one-off class is about to be destroyed, where
        instance is the instance about to be destroyed and other is a boolean
        that indicates whether instance is *not* obj. If other is False and
        func returns None, processing of obj will be assumed to have succeeded
        and the action implied by the post argument (if any) will not occur.
        Note that any preexisting .__del__() of the original obj.__class__ must
        be called explicitly by func if necessary.

        post is a boolean, function, or string that specifies the behavior if
        failure is discovered, that is, if it is detected that obj has been
        destroyed but func was not called or returned a value other than None on
        each call. (See also documentation for __del__() in Python's data
        model.) In the event of such a discovered failure, if bool(post) is
        True:
            post's type    Action
            function       post(wr, True), where wr is a dead weakref.ref to obj
            string         warning.warn(post)
            other          warning.warn(some_descriptive_text)

        prewarn is a boolean, function, or string that specifies the behavior if
        the current function appears to have been called within an interactive
        context, which can delay or suppress the calling of func. In the event
        that such a context is suspected, if bool(prewarn) is True:
            prewarn's type    action
            function          prewarn(obj)
            string            warning.warn(prewarn)
            other             warning.warn(some_descriptive_text)
        """
        # Optionally warn if the current context is interactive.
        if prewarn:
            import __main__
            if not hasattr(__main__, "__file__"):
                if hasattr(prewarn, "__call__"):
                    prewarn(obj)
                else:
                    import warnings
                    if isinstance(prewarn, basestring):
                        prewarn_text = prewarn
                    else:
                        prewarn_text = """
The interpreter appears to be interactive, which may delay
deconstruction until the end of a session or even suppress it
altogether.
"""
                    warnings.warn(prewarn_text)

        # Optionally set up weak reference monitoring of obj.
        if post:
            import weakref
            id_ = cls.get_id()
            if not hasattr(post, "__call__"):
                import warnings
                if isinstance(post, basestring):
                    postwarn_text = post
                else:
                    postwarn_text = """
The following instance was not properly deconstructed:
    {!r}
""".format(obj)
                # *REDEFINITION*
                post = lambda wr, warn=warnings.warn, postwarn_text=postwarn_text: warn(postwarn_text)
            callback = cls.wrap(None, id_, post)
            wr = weakref.ref(obj, callback)
            cls.id_to_weakref_callback_pair[id_] = (wr, callback)
        else:
            wr = weakref.ref(obj)
            cls.id_to_weakref_callback_pair[id_] = None

        # Derive one-off subclass of obj.__class__ and assign it to
        # obj.__class__.
        orig_cls = obj.__class__
        orig_cls_name = orig_cls.__name__
        new_cls_name = "{}{}ithDeconstructor".format(
            orig_cls_name,
            "w" if orig_cls_name[-1].upper() == orig_cls_name[-1] else "W"
            )
        obj.__class__ = type(new_cls_name, (orig_cls,),
                             {"__del__": cls.wrap(wr, id_, func)})

        # Return obj.
        return obj

    @classmethod
    def has_deconstructor(cls, obj):
        """
        Test whether obj was assigned a deconstructor by assign_deconstructor().
        """
        try:
            return obj.__class__.__del__.__func__.func_defaults[1] is cls.id_to_weakref_callback_pair
        except:
            return False

    @classmethod
    def wrap(cls, wr_or_None, id_, func):
        def wrapped(obj_or_wr, wr_or_None=wr_or_None,
                    id_to_weakref_callback_pair=cls.id_to_weakref_callback_pair,
                    id_=id_, func=func):
            # If (1) the weak reference stored as a function default
            # matches the calling instance (for __del__()) or is None
            # (for a weak reference callback), (2) the id stored as a
            # function default is still registered in id_to_weakref, and
            # (3) the call to func() indicates success, unregister the
            # id.
            # Note: The first test ensures that the second argument to
            # func is True only if this call relates to the original
            # obj. The second test indicates both that (1)
            # deconstruction has not been cancelled and (2)
            # deconstruction has not yet been successful.
            if wr_or_None is None or obj_or_wr is wr_or_None():
                if (id_ in cls.id_to_weakref_callback_pair and
                    func(obj_or_wr, True) is None):
                    del cls.id_to_weakref_callback_pair[id_]
            else:
                func(obj_or_wr, False)
        return wrapped

_ = Deconstruction()
assign_deconstructor = _.assign_deconstructor
has_deconstructor = _.has_deconstructor
del _


class defaultdict2(dict):
    def __init__(self, default_factory=None):
        """
        dict with key-based default factory.

        Similar to collections.defaultdict, but default_factory() is called with
        the key as the sole argument (rather than default_factory() being called
        with no argument).

        Example:
            d = defaultdict2(int)
            d[3.14] --> 3 # int(3.14) was called.
            e = collections.defaultdict(int)
            e[3.14] --> 0 # int() was called.
        """
        self.default_factory = default_factory
    def __missing__(self, key):
        try:
            val = self[key] = self.default_factory(key)
            return val
        except TypeError:
            if self.default_factory is None:
                raise KeyError(key)
            raise


class _NumpyTypeDict(dict):
    def __init__(self, default_factory=None):
        """
        Support equivalent numpy types as though hash-equivalent.
        
        For example:
            a = type(numpy.uint32(2))
            b = numpy.uint32
            d = {}
            d[a] = None
            b in d  --> can be False
            e = _NumpyTypeDict()
            d[a] = None
            b in e --> always True
        alaways hold
        """
        self.default_factory = default_factory
        
    def __missing__(self, key):
        try:
            if key.__module__ == "numpy":
                key_str = _numpy.dtype(key).name
                for k in self:
                    if _numpy.dtype(k).name == key_str:
                        return self[k]
            raise KeyError(key)
        except:
            if self.default_factory is None:
                raise KeyError(key)
            val = self[key] = self.default_factory()
            return val


class GDALWarningFilter(object):
    """
    Context manager to filter GDAL warnings and other low-level errors.

    Inspired by https://gis.stackexchange.com/a/68042.
    """
    def __init__(self, filter_explicit=(), permit_explicit=()):
        """
        Manage GDAL warnings (etc.) by explicitly filtering/permitting them.

        The current class also manages errors of lower level than warnings, that
        is, errors marked as "debug" or "none".

        filter_explicit is a container of (integer, string) tuples that specify
        each error number, message combination that should be suppressed. In the
        special case that filter_explicit is None, no warnings whatsoever are
        filtered.

        permit_explicit is a container of (integer, string) tuples that specify
        each error code, message combination that should not be suppressed. In
        the special case that permit_explicit is None, all warnings are
        suppressed.

        Note: Only filter_explicit or permit_explicit can be specified (i.e.,
        set to a non-empty container or None), not both.
        """
        # Process arguments.
        if ((filter_explicit or filter_explicit is None) and
            (permit_explicit or permit_explicit is None)):
            raise TypeError(
                "only filter_explicit or permit_explicit may be specified, not both"
                )
        if ((not filter_explicit and filter_explicit is not None) and
            (not permit_explicit and permit_explicit is not None)):
            raise TypeError(
                "either filter_explicit or permit_explicit must be specified"
                )
        self.filter_explicit = filter_explicit
        self.permit_explicit = permit_explicit

        # Initialize error records.
        import gdal
        self._gdal = gdal

    def __enter__(self):
        # In the special case that no filtering should be performed, do
        # not set the error handler.
        if self.filter_explicit is not None:
            self._set_handler()
        return self

    def __exit__(self, type_, value, traceback):
        # In the special case that the error handler was not set, there
        # is no need to unset it.
        if self.filter_explicit is not None:
            self._gdal.PopErrorHandler()

    def _set_handler(self):
        # handler is assigned to avoid a possible crash (see
        # https://trac.osgeo.org/gdal/ticket/5186#comment:4).
        handler = self._handler
        self._gdal.PushErrorHandler(handler)

    def _handler(self, err_level, err_num, err_msg):
        """
        GDAL-compatible error handler. (Used internally.)
        """
        # If error level is above a warning (failure or fatal), ignore
        # it.
        # Note: If gdal.UseExceptions() was called, such an error raises
        # a Python exception.
        if err_level >= self._gdal.CE_Failure or self.permit_explicit is None:
            return

        # Determine whether error should be suppressed based on the
        # arguments specified at initialization.
        if self.permit_explicit:
            for test_err_num, test_err_msg in self.permit_explicit:
                if err_num == test_err_num and err_msg == test_err_msg:
                    break
            else:
                return  # Error not specified, so suppress.
        elif self.filter_explicit:
            for test_err_num, test_err_msg in self.filter_explicit:
                if err_num == test_err_num and err_msg == test_err_msg:
                    return  # Error specified, so suppress.

        # Error should not be suppressed. Briefly turn off error
        # handling, raise the error, and then resume error handling.
        self._gdal.PopErrorHandler()
        self._gdal.Error(self._gdal.CE_Warning, err_num, err_msg)
        self._set_handler()


class LogPrintTiming(object):
    _format_attr_name = "{}_{}_time".format
    _format_attr_label = "{}={!r}".format
    _format_field_label = "{!r}={!r}".format
    _format_keyword_text = "{} {}{}".format
    _format_kwarg_text_for_nonstr_val = "{} = {!r}".format
    _format_kwarg_text_for_str_val = '{} = r"{}"'.format
    _format_line = "{}{}: {}{}{}\n".format
    _format_tagged_kwarg_text_for_nonstr_val = (
        _format_kwarg_text_for_nonstr_val.__self__ + "{}"
        ).format
    _format_tagged_kwarg_text_for_str_val = (
        _format_kwarg_text_for_str_val.__self__ + "{}"
        ).format
    _format_tagged_kwarg_text = "{} = {!r}".format
    _format_untimed_line = "{}{}{}\n".format

    def __init__(self, log, stdout=True, desc=".ID"):
        """
        Facilitates formatting, logging, and printing of steps and their timing.

        log is a file object that specifies the file to which lines should be
        written. If log is instead None, logging is disabled.

        stdout is a boolean that specifies whether lines should be writted to
        stdout (e.g., printed to the screen).

        desc is a string or function that specifies how the descriptive text for
        a Geometry will be generated by default. More precisely, desc specifies
        the default value for the desc argument of .generate_geom_text().
        """
        import time
        self._time = time
        self.log = log
        self.stdout = stdout
        self.default_desc = desc

    def generate_geom_text(self, geom, desc=None):
        """
        Generate and return a useful text description of a Geometry.

        geom is a Geometry for which descriptive text should be generated.

        desc is a string or function that specifies how the descriptive text is
        generated. It must have one of the following forms:
            ".attr_name" --> base text on geom.attr_name
            ".data['field_name']" --> base text on geom.data['field_name']
            func() --> func(geom) is returned, without modification
        """
        if desc is None:
            desc = self.default_desc
        if not isinstance(desc, basestring):
            return desc(geom)
        # Note: Requiring a leading "." makes the function somewhat
        # more extensible, as special strings (e.g., one whose
        # .format() could be used to generate the descriptive text)
        # could potentially be accommodated in the future with
        # backward compatibility.
        if not desc.startswith("."):
            raise TypeError('desc must start with "." if it is a string')
        if desc.startswith(".data["):
            field_name = desc.partition("[")[2][1:-2]
            value = geom.data[field_name]
            label = self._format_field_label(field_name, value)
        else:
            attr_name = desc[1:]
            value = _operator.attrgetter(attr_name)(geom)
            label = self._format_attr_label(attr_name, value)
        return "{} ({})".format(type(geom).__name__, label)

    def write_timed(self, text, geom_text="", indent="*", time=None):
        """
        Write out a time-stamped line.

        text is a string that specifies the text to write out.

        geom_text is a string or Geometry that determines what text in the line
        (if any) will be used to describe the corresponding Geometry. If
        geom_text is a string, it is not modified. If geom_text is instead a
        Geometry, geom_text is reset to .generate_geom_text(geom_text).

        indent is a string that specifies the text at the beginning of the line,
        that is, the text that comes immediately before the timestamp.

        time is a numeric value that specifies the seconds since the epoch to
        which the timestamp will correspond. If time is None, time.time() is
        called to populate it.
        """
        self._process_line(text, geom_text, indent, time)

    def write_untimed(self, text, geom_text=""):
        """
        Write out a line (without a timestamp).

        See .write_timed() for argument descriptions.
        """
        self._process_line(text, geom_text, untimed=True)

    def write_empty_lines(self, line_count=1):
        """
        Write out a number of empty lines.

        line_count is a numeric value that specifies the number of empty lines
        to be written.
        """
        self._process_line("\n" * (line_count - 1), untimed=True)

    def write_kwargs(self, kwargs, func=None, include_defaults=True,
                     comment_defaults=True, header=None, include_unmatched=True,
                     unmatched_suffix="  # [no keyword match]"):
        """
        Write out keyword argument name-value pairs.

        Keyword argument name-value pairs are written out, optionally following
        a header, in either call-signature (if func is specified) or
        alphabetical (if func is None) order. If func is specified (not None), a
        list of the names of any unmatched arguments (whether or not they were
        written out) is returned. (For consistency, an empty list is returned if
        func is None.)

        kwargs is a dictionary that specifies keyword arguments paired to their
        corresponding values.

        func is a function or (new-style) class that specifies/implies the
        function to which kwargs corresponds. If func is a class, arguments are
        matched to func.__init__()'s call signature.

        include_defaults is a boolean that specifies whether defaulted argument
        name-value pairs not present in kwargs should also be written out.
        
        comment_defaults is a boolean that specifies whether all arguments that
        have the same values as the corresponding defaults in func's call 
        signature should be commented out.

        header is a string that specifies a line that will be written out before
        writing out the argument name-value pairs. If header is None and func is
        specified, a header is automatically generated. Otherwise, if header
        evaluates False, no header is written out.

        include_unmatched is a boolean that specifies whether name-value pairs
        for argument names not explicitly belonging to func's call signature
        (e.g., those captured by **kwargs) should be written out.

        unmatched_suffix is a string that specifies text that will be written
        out immediately after (and on the same line as) each name-value pair for
        unmatched arguments. This argument is ignored if include_unmatched is
        False.

        Note: If func is None, the following arguments are irrelevant and
        therefore ignored: include_defaults, include_unmatched, and
        unmatched_suffix.
        """
        if func is None:
            names = kwargs.keys()
            names.sort()
        else:
            if isinstance(func, type):
                src = func
                func = func.__init__  # *REASSIGNMENT*
            else:
                src = func
            all_arg_names, _, _, func_defaults = _inspect.getargspec(func)
            if include_defaults:
                if func_defaults is not None:
                    default_kwargs = dict(
                        zip(all_arg_names[-len(func_defaults):], func_defaults)
                        )
                    default_kwargs.update(kwargs)
                    kwargs = default_kwargs  # *REASSIGNMENT*
            names = [arg_name for arg_name in all_arg_names
                     if arg_name in kwargs]
            if header is None:
                header = "# {}.{} arguments{}:".format(
                    src.__module__, src.__name__, "" if names else " (if any)"
                    )
        process_line = self._process_line
        if header:
            process_line(header, untimed=True)
        format_kwarg_text_for_nonstr_val = (
            self._format_kwarg_text_for_nonstr_val
            )
        format_kwarg_text_for_str_val = self._format_kwarg_text_for_str_val
        for name in names:
            val = kwargs[name]
            if isinstance(val, basestring):
                # Note: Raw string literals are more easily read by
                # humans, perhaps especially for Windows paths.
                format_this_line = format_kwarg_text_for_str_val
            else:
                format_this_line = format_kwarg_text_for_nonstr_val
            process_line(format_this_line(name, val), untimed=True)
        if func is None:
            unmatched_names = []
        else:
            unmatched_names = [name for name in kwargs if name not in names]
            unmatched_names.sort()
            if include_unmatched:
                format_tagged_kwarg_text_for_nonstr_val = (
                    self._format_tagged_kwarg_text_for_nonstr_val
                    )
                format_tagged_kwarg_text_for_str_val = (
                    self._format_tagged_kwarg_text_for_str_val
                    )
                for name in unmatched_names:
                    val = kwargs[name]
                    if isinstance(val, basestring):
                        # Note: Raw string literals are more easily read
                        # by humans, perhaps especially for Windows
                        # paths.
                        format_this_line = format_tagged_kwarg_text_for_str_val
                    else:
                        format_this_line = (
                            format_tagged_kwarg_text_for_nonstr_val
                            )
                    process_line(format_this_line(name, val, unmatched_suffix),
                                 untimed=True)
        process_line("", untimed=True)
        return unmatched_names

    def start(self, keyword, geom_text="", addendum="", indent="", time=None):
        """
        Register and write out the start of a step.

        keyword is a string that specifies a name for the step. It must be a
        viable attribute name (e.g., must not contain spaces, begin with a
        number, etc.) and be unique among all step names. Any underscores will
        be replaced by spaces when writing out.

        addendum is a string that specifies the text (if any) that should be
        inserted immediately after the keyword in the generated line.

        See .write_timed() for the remaining argument descriptions.
        """
        text, suffix = self._process(keyword, addendum, time)
        self._process_line(text, geom_text, indent, time, suffix)

    def end(self, keyword, geom_text="", addendum="", indent=" ", time=None):
        """
        Register and write out the end of a step.

        See .start() for all argument descriptions.
        """
        text, suffix = self._process(keyword, addendum, time, False)
        self._process_line(text, geom_text, indent, time, suffix)

    def _process(self, keyword, addendum, epoch_time, start=True):
        """
        Facilitate .start() and .end() functionality.

        A tuple of the form (text, suffix) is returned.
        """
        if epoch_time is None:
            epoch_time = _time.time()
        start_attr_name = self._format_attr_name(keyword, "start")
        if start:
            setattr(self, start_attr_name, epoch_time)
            suffix = ""
        else:
            end_attr_name = self._format_attr_name(keyword, "end")
            setattr(self, end_attr_name, epoch_time)
            suffix = " (took {:.1f} minutes)".format(
                (epoch_time - getattr(self, start_attr_name)) / 60.
                )
        return (self._format_keyword_text("started" if start else "ended",
                                          keyword.replace("_", " "), addendum),
                suffix)

    def _process_line(self, text, geom_text="", indent="", epoch_time=None,
                      suffix="", untimed=False):
        """
        Write out a line after generating any missing bits.
        """
        if not isinstance(geom_text, basestring):
            # *REASSIGNMENT*
            geom_text = ", for " + self.generate_geom_text(geom_text)
        if untimed:
            line = self._format_untimed_line(text, geom_text, suffix)
        else:
            if epoch_time is None:
                epoch_time = _time.time()
            line = self._format_line(
                indent, _time.asctime(_time.localtime(epoch_time)),
                text, geom_text, suffix
                )
        if self.log is not None:
            self.log.write(line)
            self.log.flush()
        if self.stdout:
            _sys.stdout.write(line)
            _sys.stdout.flush()


class MemoryMonitor(object):
    _stopped = False
    max_vms = None
    recent_max_vms = None
    start_vms = None

    def __init__(self, res=0.01, pid=None):
        import multiprocessing
        import psutil
        if pid is None:
            pid = _os.getpid()  # *REDEFINITION*
        self.res = res
        self.pid = pid
        self._parent_conn, child_conn = multiprocessing.Pipe()
        if self.start_vms is None:
            self.start_vms = psutil.Process(pid).memory_info().vms
        p = multiprocessing.Process(target=_monitor_memory, args=(res, pid,
                                                                  child_conn))
        p.start()

    def check(self):
        result = self.stop()
        import gc
        gc.collect()
        self.__init__(self.res, self.pid)
        return result

    def stop(self):
        self._stopped = True
        self._parent_conn.send(None)
        self.recent_max_vms = self._parent_conn.recv() - self.start_vms
        del self._parent_conn
        if self.max_vms is None:
            self.max_vms = self.recent_max_vms
        else:
            self.max_vms = max(self.recent_max_vms, self.max_vms)
        return (self.start_vms, self.recent_max_vms, self.max_vms)



###############################################################################
# NON-ARRAY UTILITY FUNCTIONS                                                 #
###############################################################################

def _dict_to_set(d, keys):
    """
    Return a set of values paired to specified keys.

    The current function is equivalent to:
        lambda d, keys: {d[key] for key in keys}
    is defined to avoid a language-level error in Python 2.7 where
        def some_function():
            ...
            s = {d[key] for key in keys}
            del d
    unnecessarily raises a (rather unhelpful) SyntaxError. (See
    https://bugs.python.org/issue4617.) Such lines can be easily replaced by
        def some_function():
            ...
            s = _dict_to_set(d, keys)
            del d
    which does not raise the aforementioned SyntaxError.
    """
    return {d[key] for key in keys}


def read_settings(path):
    """
    Read settings from an external file.

    The external file must be simply formatted like:
        # These are settings.
        a = 0
        b = "one"
        c = {"two": 2, "three": True, "four": None}
        \"""
        A long comment.
        \"""
        d = 5
    A dictionary equivalent to the global namespace of the external file is
    returned.

    path is a string that specifies the path to the external settings file.
    """
    import ast
    literal_eval = ast.literal_eval
    settings_dict = {}
    with open(path) as f:
        open_tri_quote = False  # Initialize.
        for n, line in enumerate(f):
            if open_tri_quote:
                if '"""' in line:
                    open_tri_quote = False
                    continue
                continue
            bare_line = line.strip()
            if bare_line.startswith("#") or not bare_line:
                continue
            if bare_line.startswith('"""'):
                open_tri_quote = True
                continue
            var_name, _, val = bare_line.partition("=")
            try:
                settings_dict[var_name.rstrip()] = literal_eval(val.lstrip())
            except:
                raise TypeError(
                    "settings file is not correctly formatted on line {}: {}".format(
                        n, path
                        )
                    )
    return settings_dict


def _monitor_memory(res, pid, child_conn):
    """
    Function to monitor memory. (Supports MemoryMonitor.)

    Note: This function must be picklable and therefore must be defined at
    the top level of the current module.
    """
    import psutil
    p = psutil.Process(pid)
    max_vms = 0  # Initialize.
    while not child_conn.poll(res):
        max_vms = max(max_vms, p.memory_info().vms)
    child_conn.send(max_vms)


def _reverse_dict(d, safe=True):
    """
    Reverse a dict, mapping values to keys.

    safe is a boolean that specifies whether the uniqueness of d's values
    should be tested. (If d's values are not unique, some key-value pairs
    will not be represented in the returned dict.)
    """
    d_rev = {v: k for k, v in d.iteritems()}
    if safe and len(d_rev) < len(d):
        _test_uniqueness(d, True, "value(s)")
    return d_rev


def slide_pairwise(iterable, slice_instead=False):
    """
    Return iterable as iterable of sliding pair tuples (or slices).

    Ex: list(slide_pairwise(range(4))) --> [(0, 1), (1, 2), (2, 3)]
    Ex: list(slide_pairwise(range(4), True)) --> [[0, 1], [1, 2], [2, 3]]

    iterable is any iterable.

    slice_instead is a boolean that specifies whether pairwise slices should be
    returned instead.

    Credit: Adapted from itertools documentation.
    """
    if slice_instead:
        return [iterable[n:n+2] for n in xrange(len(iterable) - 1)]
    a, b = _itertools.tee(iterable)
    next(b)
    return _itertools.izip(a, b)


def _test_uniqueness(seq, skip_test=False, error_word=None):
    """
    Test whether all items in a sequence are unique.

    seq is the sequence whose items will be tested for uniqueness.

    skip_test is a boolean that specifies whether seq's items should be
    tested for uniqueness (rather than assumed to be non-unique). It only
    makes sense to specify skip_test=False if error_word is not None.

    error_word is a string that specifies the wording of the error that is
    raised if seq's items are not unique. Specifically, it should be the plural
    (or possibly plural) name of seq's item types. For example, a generic
    error_word would be "value(s)". If error_word is not specified, False is
    returned rather than an error raised.
    """
    if not skip_test and len(seq) == len(set(seq)):
        return True
    if error_word is None:
        return False
    # Note: iter() is used to avoid Counter using seq's values, if seq
    # is a mapping.
    counts = _collections.Counter(iter(seq))
    non_unique_values = [x for x in seq if counts[x] > 1]
    raise TypeError(
        "{} {} are non-unique, including: {!r}".format(
            len(non_unique_values), error_word, non_unique_values[0]
            )
        )


def validate_string_option(value, arg_name, valid_options,
                           number_placeholder=False):
    """
    Validate string option against a container of valid options.

    If value is not in valid_options, a descriptive TypeError is raised.
    Otherwise, the string in valid_options that matches value will be
    returned. (This match is not case sensitive.)

    value is a string that specifies the user option to be validated.

    arg_name is a string that specifies the label that will be used to refer
    to value if any error is raised.

    valid_options is a container of all valid options.

    number_placeholder is a boolean that specifies whether the symbol "#" should
    be interpreted as a placeholder for any numeric value. For example, if value
    is "SCALE_33.3%" and a member of valid_options is "SCALE_#%", value would be
    treated as valid and "SCALE_#%" returned.
    """
    try:
        if value in valid_options:
            return value
        value_upper = value.upper()
        for valid_option in valid_options:
            if value_upper == valid_option.upper():
                return valid_option
        if number_placeholder:
            for valid_option in valid_options:
                if "#" in valid_option and _re.match(
                    valid_option.replace("#", "[0-9.-]+") + "$", value
                    ) is not None:
                    return valid_option
    except:
        pass
    raise TypeError(
        "{} is not one of {}: {!r}".format(
            arg_name, ", ".join(valid_options), value
            )
        )



###############################################################################
# ARRAY CLASSES                                                               #
###############################################################################

class _SortIter2D(object):

    def __init__(self, a, reverse=False, arg=False, minimum=None, maximum=None):
        """
        Create sorted iterator over 2D float array, with small memory footprint.

        The following two blocks print the same values, in the same order:
            a = numpy.random.rand(3, 3)
            for x in numpy.sort(a, None):  # Option 1. (Creates new array.)
                print x
            for x in _SortIter2D(a):       # Option 2. (Modifies a.)
                print x
        However, the first option uses more memory (scales with a.size rather
        than len(a)), whereas the second option is slower and modifies a.

        a is a 2D float array that specifies the array whose sorted values will
        be iterated over. Iteration is over the flattened array, and a will be
        modified by the iteration. In addition, any nan's will be ignored, but
        len(self) will return the length assuming no nan's are present.

        reverse is a boolean that specifies whether values should be iterated
        over in reverse (descending) order.

        arg is a boolean that specifies whether the indices of the sorted
        values (in the flattened array), rather than the values themselves,
        should be returned.

        minimum is an integer or float that specifies the minimum value which
        should be iterated over. As a convenience, in the special case that
        minimum is specified as negative infinity, negatively infinite values
        are excluded. To include such values, minimum must be None (the
        default).

        maximum is an integer or float that specifies the maximum value which
        should be iterated over. As a convenience, in the special case that
        minimum is specified as positive infinity, positively infinite values
        are excluded. To include such values, maximum must be None (the
        default).

        Note: One application for the current type is to iterate over a
        numpy.memmap that is too large to fit into memory.
        """
        self.a = a
        self.reverse = reverse
        self.arg = arg
        if minimum is not None or maximum is not None:
            raise NotImplementedError("minimum and maximum must both be None")

    def __iter__(self):
        # Common preparation.
        import bottleneck
        a = self.a
        if self.reverse:
            nan_extremum = bottleneck.nanmax
            nan_arg_extremum = bottleneck.nanargmax
        else:
            nan_extremum = bottleneck.nanmin
            nan_arg_extremum = bottleneck.nanargmin
        arg = self.arg

        # Yield full-array extremum (or its argument).
        try:
            a_flat = a.ravel()
            a_flat_extremum_idx = nan_arg_extremum(a_flat)
            if arg:
                yield a_flat_extremum_idx
            else:
                yield a_flat[a_flat_extremum_idx]
        except ValueError:
            pass
        else:

            # Prepare for deeper iteration (than full-array extremum).
            nan = _numpy.nan
            a_flat[a_flat_extremum_idx] = nan
            extreme_by_row = nan_extremum(a, 1)
            row_count, col_count = a.shape
            try:
                while True:
                    row_idx = nan_arg_extremum(extreme_by_row)
                    row = a[row_idx]
                    col_idx = nan_arg_extremum(row)
                    if arg:
                        yield row_idx*col_count + col_idx
                    else:
                        yield row[col_idx]
                    row[col_idx] = nan
                    extreme_by_row[row_idx] = nan_extremum(row)
            except ValueError:
                pass

    def __len__(self):
        return self.a.size


class _MemmapNumpy(object):

    @classmethod
    def __getattr__(cls, name):
        value = getattr(_numpy, name)
        if hasattr(value, "__call__"):
            setattr(cls, name, staticmethod(value))
        else:
            setattr(cls, name, value)
        return value

    def __init__(self, filename_func=None, memmap_version=None, temp=True):
        if filename_func is None:
            self.filename_func = _itertools.repeat(None).next
        else:
            self.filename_func = filename_func
        self.memmap_version = memmap_version
        self.temp = temp

    def _copy_to_memmap(self, a):
        return copy_to_memmap(a, self.filename_func(), self.memmap_version,
                              self.temp)

    def arange(self, *args, **kwargs):
        # Note: Though code could be more memory inefficient, the
        # extreme inefficiency of iteratively populating an array via
        # (x)range is deemed too high.
        return self._copy_to_memmap(_numpy.arange(*args, **kwargs))

    def array(self, object, dtype=None, copy=True, order="K", subok=False,
              ndmin=0):
        # Note: Though code could be more memory inefficient, the
        # generality of array() would be difficult to replicate.
        return self._copy_to_memmap(
            _numpy.array(object, dtype, copy, order, subok, ndmin)
            )

    def empty(self, shape, dtype=_numpy.float, order="C"):
        return open_memmap(shape=shape, dtype=dtype,
                           fortran_order=(order.upper() == "F"),
                           version=self.memmap_version, temp=self.temp)

    def zeros(self, shape, dtype=_numpy.float, order="C"):
        a = self.empty(shape, dtype, order)
        a.fill(0)
        return a

    def ones(self, shape, dtype=_numpy.float, order="C"):
        a = self.empty(shape, dtype, order)
        a.fill(1)
        return a



###############################################################################
# ARRAY UTILITY FUNCTIONS                                                     #
###############################################################################

def _force_divide(num, denom, out=None):
    old_seterr_dict = _numpy.seterr(divide="ignore", invalid="ignore")
    try:
        return _numpy_divide(num, denom, out)
    finally:
        _numpy.seterr(**old_seterr_dict)


def copy_to_memmap(a, filename=None, version=None, temp=False):
    """
    Create and return memory-mapped copy of a.

    a is an array that specifies the array to be memory-mapped.

    Note: The undocumented arguments have the same meaning as the arguments
    of the same name in open_memmap.
    """
    a2 = open_memmap(filename, "w+", a.dtype, a.shape, version=version,
                     temp=temp)
    a2[:] = a
    return a2


def _deconstruct_memory_map(memory_map, hint=True):
    """
    Delete a memory-mapped file after cancelling any pending writes to disk.

    The current function is designed for cases where memory mapping is used
    merely to prevent exhausting memory (similar to paging) so that the data
    need not persist. A boolean is returned to indicate whether the termination
    was both executed and appeared to be successful.

    memory_map is a numpy.memmap or mmap.mmap that specifies the memory-
    mapping to be terminated. If memory_map is a numpy.memmap and does not
    own its corresponding mmap.mmap (i.e., memory_map.base is not that
    mmap.mmap), the current function effectively does nothing except return
    False.

    hint is a string or boolean. If hint is a string, it specifies the path to
    the file mapped by memory_map, which must be a mmap.mmap. If hint is instead
    True, memory_map must be a numpy.memmap, in which case path is set to
    memory_map.filename. Finally, if memory_map evaluates False, the current
    function effectively does nothing except return False (which makes it
    compatible with assign_deconstructor()).

    Warning: If memory_map is accessed after the current function is called,
    the interpreter will freeze. It is therefore preferable to call the
    current function immediately before memory_map is destroyed, perhaps
    using assign_deconstructor().
    """
    # Isolate the mmap and determine its ownership, if applicable.
    if not hint:
        return False
    if isinstance(hint, basestring):
        mmap_instance = memory_map
        path = hint
    else:
        import mmap
        if isinstance(memory_map.base, mmap.mmap):
            mmap_instance = memory_map.base
            path = memory_map.filename
        else:
            return False

    # Resize memory map to effectively cancel pending writes to
    # disk. Then close memory map and delete its .npy file.
    # Note: Resizing file to 0 would freeze the interpreter (presumably,
    # only on Windows, as suggested by the mmap.mmap documentation).
    mmap_instance.resize(1)
    mmap_instance.close()
    _os.remove(path)
    return None


def hstack_flat(arrays, commingled=False):
    """
    Stack flat arrays in sequence as columns.

    arrays is a sequece of arrays.

    commingled is a boolean that specifies whether flat and non-flat arrays may
    be commingled. If commingled is True, each non-flat arrays must have the
    same row count as the item count of each flat array.
    """
    if commingled:
        return _numpy_concatenate([a.reshape(_neg_1_1_tuple)
                                   if len(a.shape) == 1 else a
                                   for a in arrays],
                                  1)
    return _numpy_concatenate([a.reshape(_neg_1_1_tuple) for a in arrays], 1)


def _is_subset(a, b, return_unique_a=False, copy_a=True, copy_b=True,
               a_is_unique=False, b_is_unique=False, sort_a=True, sort_b=True):
    """
    Test that a is a subset of b, where a and b are sortable flat arrays.

    The current function is equivalent to (but generally fast than):
        set(a.tolist()).issubset(b.tolist())
    It is especially fast in special cases, which include but are not limited
    to:
        1) a and b are identical (i.e., the same object) --> True
        2) a has more unique values than b's length --> False
        3) a and b are disjoint --> False

    a is a flat array.

    b is a flat array.

    return_unique_a is a boolean that specifies whether a tuple of the form
    (boolean, unique_a_array) should instead be returned, where boolean is
    the same value that would be returned if return_unique_a were False and
    unique_a is an array that contains the unique values from a. If
    a_is_unique is True, unique_a_array is a. Otherwise, unique_a_array is
    guaranteed to be sorted.

    copy_a is a boolean that specifies whether a should be copied internally
    prior to any modification.

    copy_b is a boolean that specifies whether b should be copied internally
    prior to any modification.

    a_is_unique is a boolean that specifies whether the values in a are
    unique.

    b_is_unique is a boolean that specifies whether the values in b are
    unique.

    sort_a is a boolean that specifies whether the values in a must be
    sorted. If sort_a is False, the values in a must already be sorted.

    sort_b is a boolean that specifies whether the values in b must be
    sorted. If sort_b is False, the values in b must already be sorted.

    Note: After return_unique_a, all remaining arguments are merely provided
    for optimization. They may be specified appropriately for a possible
    performance benefit, or allowed to default without any change to the
    final result. However, specifying any of them inappropriately may cause
    an erroneous test or other undesirable results!
    """
    # As a trivial test, a is an (improper) subset of b if a is b.
    a_is_b = a is b
    # Note: If a is b but return_unique_a is True, a unique version of a
    # must first be found before returning.
    if not return_unique_a and a_is_b:
        return True

    # Ensure that a is unique.
    if not a_is_unique:
        a = _uniquify_flat_array(a, copy_a, sort_a) # *REDEFINITION*
    if a_is_b:
        return (True, a)

    # If (possibly not unique) b is smaller than (now unique) a, a
    # cannot possibly fit into b.
    if len(b) < len(a):
        if return_unique_a:
            return (False, a)
        return False

    # Ensure that b is unique and sorted.
    if not b_is_unique:
        # *REDEFINITION*
        b = _uniquify_flat_array(b, copy_b, sort_b)
    elif sort_b:
        if copy_b:
            b = b.copy() # *REDEFINITION*
        b.sort()

    # If the extrema from a are not present in b, a cannot be a subset of
    # b. Otherwise, resort to a full test.
    if sort_a:
        a_max = a.max()
    else:
        a_max = a[-1]
    idxN = b.searchsorted(a_max)
    # Note: idxN == len(b) if a.max() > b.max().
    if idxN == len(b) or b[idxN] != a_max:
        if return_unique_a:
            return (False, a)
        return False
    if sort_a:
        a_min = a.min()
    else:
        a_min = a[0]
    idx0 = b.searchsorted(a_min)
    if b[idx0] != a_min:
        if return_unique_a:
            return (False, a)
        return False

    # Test that the subset of b's values that are within the range of
    # a's values contains at least as many values as a (or else a cannot
    # possibly fit into b).
    if idxN - idx0 + 1 < len(a):
        if return_unique_a:
            return (False, a)
        return False

    # Resort to a full test.
    # Note: Approximately optimized (based on testing).
    if len(b) // len(a) >= 5:
        result = _uniquify_flat_array(
            _numpy_concatenate((a, b)), False, True, True
            ) == len(b)
    else:
        result = not len(_numpy.setdiff1d(a, b, True))
    if return_unique_a:
        return (result, a)
    return result


def open_memmap(filename=None, mode=None, dtype=None, shape=None,
                fortran_order=False, version=None, temp=False):
    """
    Open a memory-mapped array from an .npy file (optionally created).

    filename is a string that specifies the name of the file on disk. It
    cannot be a file-like object. If filename is not specified (None), a name
    is automatically generated using tempfile.NamedTemporaryFile.

    mode is a string that specifies the mode in which the file specified by
    filename is opened. If filename is None, mode defaults to "w+". Otherwise,
    mode defaults to "r+". Modes "r" and "c" are also permitted. (See
    numpy.lib.format.open_memmap().)

    temp is a boolean that specifies whether the returned copy's memory-
    mapped file (path) should be deleted upon deletion of the instance.

    Note: The undocumented arguments have the same meaning as the arguments
    of the same name in numpy.lib.format.open_memmap().
    """
    # If filename is not specified (None), generate a filename in the
    # user's temporary directory.
    if filename is None:
        # Note: Mode must be "w+" to permit creation of the file.
        if mode is None:
            mode = "w+"  # *REDEFINITION*
        elif mode != "w+":
            raise TypeError("mode must be 'w+'")
        # Note: Although the code below does not guarantee that the
        # generated filename will still be available when passed to
        # numpy.lib.format.open_memmap() further below, it is considered
        # reasonably safe.
        import tempfile
        # *REDEFINITION*
        filename = tempfile.NamedTemporaryFile(suffix=".npy", delete=temp).name

    # Default mode to reading and writing.
    elif mode is None:
        mode = "r+"

    # Create numpy.memmap instance and optionally return it.
    m = _numpy.lib.format.open_memmap(filename, mode, dtype, shape,
                                      fortran_order, version)

    # Optionally register memmap for self-destruct upon finalization.
    if temp:
        import tempfile
        dirs = {tempfile.gettempdir(), _os.path.dirname(filename)}
        assign_deconstructor(
            m, _deconstruct_memory_map,
            lambda wr, instance, mmap=m._mmap, path=filename: _deconstruct_memory_map(mmap, path),
            """
The interpreter appears to be interactive. Deletion of temporary memory-
mapped .npy files (essentially used as named swap files) may be delayed
or fail, and therefore may accumulate across the current and possibly
future sessions. After interpreter termination (especially if
termination is not normal), the following director{} (and possibly
others) may contain .npy files that should be manually deleted:
    {}
""".format("ies" if len(dirs) > 1 else "y", "\n    ".join(dirs) + ".\n")
            )

    # Return memmap.
    return m


def _slide_flat(a):
    """
    Return (1, 2) sliding window view of a flat array .

    The sliding windows are returned within an array, so that the shape of
    the output arary is (n - 1, 2) for an input array of shape (n,). The
    output array is not contiguous, but the individual windows are
    contiguous if the input array is contiguous.
    """
    # Note: Even for values of a known bit depth (e.g., 64-bit floats),
    # only contiguous flat arrays will always have the same "column"
    # step sizes.
    col_step, = a.strides
    return _numpy_as_strided(a, (len(a) - 1, 2), (col_step, col_step))


def _take2(a, indices, in_place=False, memmap=False, axis=0, out=None,
           mode="clip"):
    """
    Take elements from an array along an axis.

    The current function is a wrapper for a.take() that defaults to axis 0
    and the faster "clip" mode (instead of that method's "raise" default).
    It also adds the arguments in_place and memmap.

    in_place is a boolean that specifies whether the object specified for
    indices should also be assigned to out. If in_place is True, indices
    must be an array (not merely array-like) and out should not be
    specified.

    memmap is a boolean that specifies that the returned array should be a
    "true" (mmap.mmap-backed) numpy.memmap if a is a numpy.memmap that owns
    its mmap.mmap (i.e., whose .base is a subtype of mmap.mmap). More
    specifically, the return array will be backed by its own mmap.mmap with the
    same persistence (temporary or not) as that of a, whose file is located in
    the same directory as the file that backs a. If both in_place and memmap are
    True, an error is raised.

    Note: All remaining arguments are documented in numpy.take(), but note
    the different defaults for axis and mode.

    Warning: The returned array will invariably have the same type as a if a is
    a subtype of numpy.ndarray, including numpy.memmap. However, if a is a
    subtype of numpy.memmap but memmap is False or a does not own its mmap.mmap,
    the returned array will (deceptively) not be backed by a file.
    """
    # Note: In testing, modes "clip" and "wrap" were equally fast, but
    # if disallowed (i.e., negative or overly large) indices are used,
    # the behavior that results from "clip" would probably be easier to
    # recognize and diagnose, hence its use as the default. Conversely,
    # mode "raise" (numpy.take()'s default) is slower.
    if in_place:
        if memmap:
            raise TypeError("in_place and memmap cannot both evaluate True")
        if out is not None and out is not indices:
            raise TypeError("in_place and out arguments are incompatible")
        out = indices
    elif memmap and isinstance(a, _numpy.memmap):
        import mmap
        if isinstance(a.base, mmap.mmap):
            import tempfile
            path = tempfile.NamedTemporaryFile(
                suffix=".npy", dir=_os.path.dirname(a.filename)
                ).name
            shape_list = list(a.shape)
            shape_list[axis] = len(indices)
            # Note: The argument fortran_order is allowed to default,
            # and version is a guess.
            out = open_memmap(path, "w+", a.dtype, tuple(shape_list),
                              version=(2, 0), temp=has_deconstructor(a))
    return a.take(indices, axis, out, mode)


def _triu_indices_flat(n):
    """
    Return the flat indices for the upper triangle of an array of shape (n, n).

    n is an integer that specifies the length (or, equivalently, width) of the
    square array for which flat-equivalent indices should be returned.
    """
    if n < 2:
        raise TypeError("n must be >= 2")
    arrays = []
    arrays_append = arrays.append
    # Note: This approach is somewhat faster than one that populates an
    # array initialized by arange(1, (n*n - n)//2 + 1) and then
    # increments it over slices.
    for idx0, idxN_plus_1 in _izip(xrange(1, n*n, n + 1),
                                   xrange(n, n*n + 1, n)):
        arrays_append(_numpy_arange(idx0, idxN_plus_1))
    return _numpy_concatenate(arrays)


def _union_flat_arrays(a, b):
    """
    Union two flat arrays.

    A unique, sorted flat array of all values represented in a or b
    (inclusive) is returned.

    a is a flat array (shape (-1,)) whose values are to be unioned with b.

    b is a flat array (shape (-1,)) whose values are to be unioned with a.
    """
    return _uniquify_flat_array(_numpy_concatenate((a, b)), False)


def _uniquify_flat_array(a, copy=True, sort=True, count_only=False):
    """
    Uniquify a flat array.

    A sorted flat array of unique values from a is returned.

    a is a flat array (shape (-1,)) whose values are to be uniquified.

    copy is a boolean that specifies whether a copy of a should be made
    internally prior to sorting. If copy is False, a will be sorted in place
    (unless sort is False, in which case the current argument is ignored).

    sort is a boolean that specifies whether a or its copy (if copy is True)
    should be sorted. If sort is False, a must already be sorted.

    count_only is a boolean that specifies whether the number of unique values
    should be retunred (rather than a unique array).
    """
    if sort:
        if copy:
            a = a.copy() # *REDEFINITION*
        a.sort()
    count = len(a)
    if not count:
        return a
    bools = _numpy_empty((len(a),), _numpy.bool8)
    bools[0] = True
    _numpy_not_equal(a[1:], a[:-1], bools[1:])
    if count_only:
        return bools.sum()
    return a[bools]


def unravel_index2D(index, column_count=2):
    """
    Convert a flat index to a corresponding 2-tuple of 2D indices.

    Similar to numpy.unravel_index() but faster for the supported case.

    index is an integer that specifies the flat index to be converted.

    column_count is an integer that specifies the column count of the 2D
    array to which the flat index refers.
    """
    return (index // column_count, index % column_count)


def unravel_indices2D(indices_array, column_count=2):
    """
    Convert array of flat indices to corresponding array of 2D indices.

    Similar to numpy.unravel_index() but faster for the supported case and
    returns a single array of shape (len(indices_array), 2).

    indices_array is an array of integers that specifies the flat indices to
    be converted.

    column_count is an integer that specifies the column count of the 2D
    array to which the flat indices refer.
    """
    e = _numpy_empty((len(indices_array), 2), _numpy_int64)
    _numpy.floor_divide(indices_array, column_count, e[:,0])
    _numpy.fmod(indices_array, column_count, e[:,1])
    return e



###############################################################################
# LAZY BASE CLASSES                                                           #
###############################################################################

class Instantiable(object):
    """
    Generic base class to undo instantiation-blocking by Lazy2.
    """
    __new__ = object.__new__


class InstantiableList(Instantiable):
    """
    List-friendly base class to undo instantiation-blocking by Lazy2.
    """
    __new__ = list.__new__


class Lazy(object):
    """
    Base class to support lazy attribute generation.

    The functionality supported by the current base class is similar to what
    can be achieved with properties:
        @property
        def attribute_name(self):
            if not hasattr(self, "_attribute_name"):
                self._attribute_name = some_function()
            return self._attribute_name
    However, it avoids the need to implement such a code template for each such
    attribute name and avoids function calls on all uses of the attribute after
    its assignment. The implementation in the current base class is also a bit
    more transparent, in that
        instance.__dict__["attribute_name"] == instance.attribute_name --> True
    .attribute_name also does not unexpectedly assume a new (and possibly
    erroneous value) if the user naively assigns to ._attribute_name. Finally,
    releasing the calculated value (e.g., to reduce the memory footprint or
    force recalculation) is straightforwardly achieved (e.g., by the end user)
    by
        del instance.attribute_name

    See also documentation for .__getattr__().

    Warning: If you wish to specify a hidden "get" method that does not
    accidentally implement lazy attribution, you might consider naming your
    method ._get__*() to avoid the name mangling that .__get*() would cause, or
    better still, simply use a different word, such as ._fetch_*(), which avoids
    .__dir__() falsely reporting ._* as available.
    """

    def __dir__(self, return_set=False, lazy_only=False, deletable_only=False):
        """
        Return all current and lazy attributes names.

        return_set is a boolean that specifies whether the attribute names
        should be returned as a set (rather than a sorted list), which is
        slightly faster.

        lazy_only is a boolean that specifies whether only lazy-supported
        attribute names should be returned.

        deletable_only is a boolean that specifies whether only lazy-supported
        attribute names whose associated ._get_*()'s do not have a .deletable
        attribute set to False.

        Note: If lazy_only or deletable_only is True, the current method may be
        called as a function (rather than an (un)bound method) if an appropriate
        class is provided as the first argument:
            self.__dir__.__func__(type(self), lazy_only=True)
        This is primarily useful to determine the lazy attribute names at the
        level of a specific class when a strict instance of that class is not
        available (even if an instance of a subtype may be available):
            issubclass(SubClass, ParentClass) --> True
            subclass_instance = SubClass()
            parent_class_lazy_names1 = subclass_instance.__dir__.__func__(
                ParentClass, lazy_only=True
                )
            parent_class_lazy_names2 = ParentClass.__dir__.__func__(
                ParentClass, lazy_only=True
                )
            parent_class_lazy_names1 == parent_class_lazy_names2 --> True
        """
        if lazy_only or deletable_only:
            if isinstance(self, type):
                # self is actually a class.
                class_dir = dir(self)
            else:
                class_dir = dir(self.__class__)
            # Note: Unlike further below, a set is not necessary here
            # (because all items are guaranteed to be unique).
            # Nonetheless, return_set will typically be true when
            # lazy_only or deletable_only is True.
            dir_set = {name[5:] for name in class_dir if name[:5] == "_get_"
                       and (not deletable_only or
                       getattr(getattr(self, name), "deletable", True))}
            if return_set:
                return dir_set
            return sorted(dir_set)
        dir_set = set(self.__dict__)
        class_dir = dir(self.__class__)
        dir_set.update(class_dir)
        dir_set.update([name[5:] for name in class_dir if name[:5] == "_get_"])
        if return_set:
            return dir_set
        return sorted(dir_set)

    def __getattr__(self, name):
        """
        Attempt lazy attribute generation.

        The current method is only called if the specified attribute name cannot
        be found. (See Python data model.) In that case, a class-level method
        _get_name() is searched for, where name is replaced with the attribute
        name. If such a method is found, it is called (with the instance (self)
        as the only argument), the result assigned to .name (where name is
        replaced with the attribute name), and the result returned. Otherwise,
        if no such method is found, a standardly-formatted AttributeError is
        raised.
        """
        try:
            # Note: To make Lazy attribute generation robust to instance
            # attribute collision, this "self" could be changed to
            # "type(self)" in the line below.
            func = object.__getattribute__(self, "_get_" + name)
        except AttributeError:
            # name is not lazy, so raise standard error.
            object.__getattribute__(self, name)
        result = func(self)
        setattr(self, name, result)
        return result

    def __clear_lazy(self, allow=None, disallow=None):
        """
        Clear "lazy" attributes, which are generated only when needed.

        Not only are lazy attributes "lazily" generated at the moment when they
        are first needed, but their generated values are also stored for reuse.
        This benefits performance at the cost of memory footprint. Therefore,
        clearing these lazy attributes may release some memory, though this
        depends on some details of implementation. It is also not guaranteed
        that all lazy attributes can be re-generated after such a clearing. For
        this reason, a lazy attribute can be excluded from clearing by the
        current function if its associated ._get_*() has a .deletable attribute
        set to False.

        allow is an iterable of strings that specifies the attribute names that
        are allowed to be cleared by the current method call. If any lazy
        attribute's name does not occur in this iterable, it will not be
        cleared. Any string (or other object) in allow that does not correspond
        to a lazy attribute's name is ignored. If allow is not specified (None),
        all lazy attributes are treated as deletable by default (subject to
        the aforementioned .deletable behavior and the disallow argument).

        disallow is an iterable of strings that specifies the attribute names
        that are not allowed to be cleared by the current method call. If any
        lazy attribute's name occurs in this iterable (even if it also occurs in
        allow), it will not be cleared. Any string (or other object) in disallow
        that does not correspond to a lazy attribute's name is ignored.

        Warning: Any attribute assigned by the user that has the same name as a
        lazy attribute may be deleted by the current method.
        """
        attr_names = self.__dir__(True, True, True)
        if allow is not None:
            attr_names.intersection_update(allow)
        if disallow is not None:
            attr_names.difference_update(disallow)
        self_dict_pop = self.__dict__.pop
        for name in attr_names:
            self_dict_pop(name, None)


class Lazy2Type(type):
    """
    Type that primes Lazy2 subclasses for enhanced lazy attribute use.
    """
    _undeletable_initialization_base_attr_names_set = {"coords_array",
                                                       "boundary"}

    def __init__(cls, name, bases, attrs_dict):
        # Prepare.
        attrgetter = _operator.attrgetter
        cls_is_spatial_dimension_aware = hasattr(cls, "is_3D")
        if cls_is_spatial_dimension_aware:
            cls_is_3D = cls.is_3D
            if cls_is_3D:
                suffix = "3D"
            else:
                suffix = "2D"
            make_assign_spatial_dimension_specific_aliases_for_initialization_attributes = cls._make_assign_spatial_dimension_specific_aliases_for_initialization_attributes

            # Iterate over class-level attribute names.
            # Note: set(attrs_dict) is not equivalent to set(dir(cls)),
            # as the only the latter contains inherited methods.
            cls_attr_names_set = set(dir(cls))
            for attr_name in cls_attr_names_set:

                # Copy spatial-dimension-specific non-lazy attributes to
                # their spatial-dimension-generic aliases (e.g.,
                # .wkb_type2D --> .wkb_type).
                if attr_name[:5] != "_get_":
                    if attr_name[-2:] != suffix:
                        continue
                    base_attr_name = attr_name[:-2]
                    # Note: Do not overwrite any already specified
                    # spatial-dimension-generic alias.
                    if base_attr_name in cls_attr_names_set:
                        continue
                    setattr(cls, base_attr_name, getattr(cls, attr_name))
                    continue

                # Create spatial-dimension-specific ._get_*2D/3D() to
                # get the corresponding generic attribute, after
                # creating a counterpart Geometry2D if necessary.
                # Note: Only assign a spatial-dimension-specific
                # ._get_*() the name would not be redundant or
                # confusing (e.g., ._get__2D3D).
                if attr_name[-1] == "D":
                    continue
                setattr(cls, attr_name + suffix,
                        staticmethod(attrgetter(attr_name[5:])))
                if cls_is_3D:
                    setattr(cls, attr_name + "2D",
                            staticmethod(attrgetter("_2D." + attr_name[5:])))

        # Create example, if possible.
        try:
            example = cls.make_example()
        except:
            return

        # Prefer assigning new _get_*()'s to cls's base (ignoring
        # Instantiable and its subclasses), if .__init__() is inherited.
        # (For example, this ensures that LineSegment2D inherits these
        # _get_*()'s from LineString via LineSegment, because
        # LineSegment(2D) does not inherit from LineString2D.)
        cls_init_func = cls.__init__.im_func
        for parent_of_cls in bases:
            if issubclass(parent_of_cls, Instantiable):
                continue
            break
        uninherited_attr_names_set = set(attrs_dict)
        if cls_init_func == parent_of_cls.__init__.im_func:
            target_cls = parent_of_cls
            # Note: Treat attributes defined by the base class from
            # which .__init__() is inherited as though they were
            # uninherited (for purposes of avoiding conflicts).
            uninherited_attr_names_set.update(parent_of_cls.__dict__)
        else:
            target_cls = cls

        # Identify those attributes assigned by .__init__(), and assign
        # a corresponding ._get_*() for each attribute's name.
        initialization_base_attr_names = list(example.__dict__)
        make_initialize_and_get = cls._make_initialize_and_get
        undeletable_initialization_base_attr_names_set = cls._undeletable_initialization_base_attr_names_set
        # Note: See note on circular imports near the top of this
        # module.
        from .geom import _geom_type_to_3D
        for base_attr_name in initialization_base_attr_names:
            # Assign base attribute name to _get_*() (e.g.,
            # _get_length()).
            base_method_name = "_get_" + base_attr_name
            # Note: Do not overwrite any preexisting class-level method
            # (e.g., if a ._get_length() exists that may obviate
            # initialization).
            if base_method_name not in uninherited_attr_names_set:
                smethod = make_initialize_and_get(cls_init_func, base_attr_name)
                if base_attr_name in undeletable_initialization_base_attr_names_set:
                    smethod.__func__.deletable = False
                setattr(target_cls, base_method_name, smethod)
            # Note: Only assign a spatial-dimension-specific ._get_*()
            # if class is spatial-dimension-aware and the name would not
            # be redundant (e.g., ._get_length2D2D()).
            if (not cls_is_spatial_dimension_aware or
                base_attr_name[-1] == "D"):
                continue
            # Assign ._get_*2D/3D() to spatial-dimension-aware class,
            # not its base, to avoid erroneous inheritance.
            alias_method_name = base_method_name + suffix
            # Note: Do not overwrite any preexisting class-level method
            # (e.g., if a ._get_length2D() exists that may obviate
            # initialization).
            if alias_method_name not in uninherited_attr_names_set:
                setattr(
                    cls, alias_method_name,
                    make_assign_spatial_dimension_specific_aliases_for_initialization_attributes(
                        base_attr_name + suffix, initialization_base_attr_names
                        )
                    )
            # If class is 2D, a 3D counterpart exists, and ._get_*2D()
            # does not already exist for that counterpart, assign it.
            # Note: No such conflict is anticipated in the current code.
            if cls_is_3D or cls not in _geom_type_to_3D:
                continue
            alias_method_name2D = base_method_name + "2D"
            cls3D = _geom_type_to_3D[cls]
            if alias_method_name2D in cls3D.__dict__:
                continue
            setattr(cls3D, alias_method_name2D,
                    staticmethod(attrgetter("_2D." + base_attr_name)))

    @staticmethod
    def _make_initialize_and_get(init, attr_name):
        """
        Helper function called by Lazy2Type.__init__().

        The staticmethod returned by the current function is only called when a
        Geometry has not been initialized and a (generic) attribute assigned at
        initialization is used (e.g., .length, but not .length2D). The returned
        staticmethod performs initialization and returns the requested value.
        """
        def initialize_and_get(self, init=init, attr_name=attr_name):
            init(self)
            return getattr(self, attr_name)
        return staticmethod(initialize_and_get)

    @staticmethod
    def _make_assign_spatial_dimension_specific_aliases_for_initialization_attributes(
        spatial_dimension_specific_alias, initialization_base_attr_names
        ):
        """
        Helper function called by Lazy2Type.__init__().

        The staticmethod returned by the current function is only called when a
        spatial-dimension-specific alias of an attribute assigned at
        initialization is required (e.g., .length2D for LineString2D). It
        triggers initialization, if necessary, and copies all initialization
        attributes to their spatial-dimension-specific aliases (e.g., .length
        --> .length2D). (The rationale for this batch copying is that if one
        spatial-dimension-specific alias is used, it is likely that others will
        be used as well.)
        """
        def assign_spatial_dimension_specific_aliases_for_initialization_attributes(
            self, spatial_dimension_specific_alias=spatial_dimension_specific_alias,
            initialization_base_attr_names=initialization_base_attr_names
            ):
            if self.is_3D:
                suffix = "3D"
            else:
                suffix = "2D"
            for base_attr_name in initialization_base_attr_names:
                # Note: If, and only if, initialization has not yet been
                # performed, it will be triggered by getattr().
                setattr(self, base_attr_name + suffix,
                        getattr(self, base_attr_name))
            return getattr(self, spatial_dimension_specific_alias)
        return staticmethod(
            assign_spatial_dimension_specific_aliases_for_initialization_attributes
            )


class Lazy2(Lazy):
    """
    Base class to support enhanced lazy attribute generation.

    The enhancements relative to Lazy are:
        1) If subclass has a .make_example(), the attribute names assigned
           during initialization are used to generate corresponding
           ._get_name()'s that are registered as class methods on either the
           subclass or, if its .__init__() is inherited, its immediate base
           class.
           Ex: A LineString2D created by .make_fast() will be lazily
               initialized by LineString2D.__init__() when an attribute
               assigned by that method is required.
           Ex: A LineSegment2D will call LineString.__init__() if an
               attribute assigned by that method is required.
        2) If subclass is spatial-dimension-aware, spatial dimension aliases
           are supported.
           Ex: If subclass is 2D and has a .wkb_type2D, it is copied to
               .wkb_type.
           Ex: If subclass is 2D and has a ._get_length(), ._get_length2D()
               is effectively set to gettattr("length").
           Ex: If subclass is 3D and its 2D counterpart has a
               ._get_length(), using .length2D on an instance of subclass
               will cause a hidden 2D copy to be created (if one does not
               already exist) and its .length set as .length2D and returned.
        3) Prevents instantiation of subclass unless inheritance prioritizes
           Instantiable.
    """
    __metaclass__ = Lazy2Type  # Here lies most of the "magic."

    def __new__(cls, *args, **kwargs):
        raise TypeError(
            "try instantiating one of the following types (or their subtypes) instead: " +
            ", ".join([subcls.__name__ for subcls in cls.__subclasses__()])
            )
