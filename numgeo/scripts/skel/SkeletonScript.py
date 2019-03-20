"""
Script-like demo code for deriving skeletons from input polygons.
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
__all__ = ["process"]


##############################################################################
# IMPORT                                                                     #
##############################################################################

# Record time.
import time
if __name__ == "__main__":
    script_time0 = time.time()
else:
    script_time0 = None

# Import.
import inspect
import itertools
import math
import numgeo.geom
import numgeo.skel
import numgeo.util
import numgeo.vec_io
import numpy
import os
try:
    import psutil
except ImportError:
    psutil = None
import sys



###############################################################################
# FUNCTIONS                                                                   #
###############################################################################

# Define class to facilitating printing to log and/or stdout.

def process(
    # Path to input polygon data.
    in_path,

    # EasySkeleton arguments without defaults.
    interval,  # analogous to resolution
    min_normalized_length,  # higher --> simpler skeleton

    # Output paths.
    out_path_prefix="",  # default prefix for all outputs
    out_path_suffix="",  # default suffix for all outputs
    out_dir_path=None,  # default output directory
    log_out_path=None,  # text file
    skel_out_path=None,  # skeleton
    copy_out_path=None,  # polygon copies + intervals
    cut_out_path=None,  # nodes deleted by cutting
    skelp_out_path=None,  # skeleton nodes
    vor_out_path=None,  # Voronoi segments
    vorp_out_path=None,  # Voronoi cell vertices
    samp_out_path=None,  # boundary sample points
    unpar_out_path=None,  # unused graph edges
    test_out_path=None,  # original polygon or its proxy
    rot_out_path=None,  # 0-vertex rotated polygon

    # Non-output options.
    monitor_memory=True,  # Monitor memory usage?
    # Note: The option below will delete data soon after it is
    # outputted rather than store it for return, thereby reducing the
    # peak memory footprint.
    reduce_peak_memory=True,
    rotate_0_vertex=False,  # Rotate "0" vertex if isolation fails?
    interpolate_3D=False,  # Interpolate z-coordinates if possible?

    # Output options.
    output_log=True,  # Output sidecar text file?
    print_log=True,  # Print log to screen?
    output_any_vectors=True,  # Output any vector data? (Overrides.)
    # Note: The option below specifies whether input polygons should
    # merely be copied, with information regarding intervals appended.
    # (Overrides.)
    output_copy_only=False,
    output_skeleton=True,  # Output derived skeleton?
    # Warning: The memory footprint for the options below can be very
    # large (especially outputting the Voronoi cell vertices) and is
    # not accounted for when optimizing for available memory (i.e.,
    # specifying 0 for interval and/or template_interval). If you enable
    # them, you should consider setting reduce_peak_memory (further
    # above) to True.
    output_cut_nodes=True,  # Output nodes deleted by cutting?
    output_skeleton_nodes=False,  # Output nodes of derived skeleton?
    output_vor_segs=False,  # Output Voronoi segments?
    output_vor_verts=False,  # Output Voronoi cell vertices?
    output_samps=False,  # Output boundary sample points?
    output_unpar=True,  # Output unused graph edges?
    output_test_poly=True,  # Output testing (orig. or proxy) poly.?
    output_rot_poly=True,  # Output rotated polygon?

    # numgeo.skel.EasySkeleton() arguments with defaults.
    **skel_kwargs
    ):
    """
    Derive skeletons (e.g., centerlines) from input polygons.

    The current function provides high-level support for deriving skeletons
    from polygons in an input file. Progressively lower-level support is
    provided by numgeo.skel.EasySkeleton and numgeo.skel.Skeleton (and other 
    classes in numgeo.skel). The current function also facilitates 
    outputting the results.
    
    in_path is a string that specifies the path to the input polygons (e.g., 
    a shapefile).
    
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
    
    out_path_prefix is a string that specifies the first characters of each 
    default output name (e.g., shapefile name or file geodatabase feature 
    class name).
    
    out_path_suffix is a string that specifies the last characters of each 
    default output name (e.g., shapefile name (before the .shp extension) or 
    file geodatabase feature class name).
    
    out_dir_path is a string that specifies the path to the output directory 
    for each default path. If out_dir_path is not specified (None), it 
    defaults to the directory that contains in_path.
    
    log_out_path is a string that specifies the path to which the log should 
    be written. (See note further below.)
    
    skel_out_path is a string that specifies the path to which each skeleton 
    should be written. (See note further below.)
    
    copy_out_path is a string that specifies the path to which a copy of 
    each input polygon should be written. Each copy contains information on 
    suggested interval values based on available memory. (See note further 
    below.)
    
    cut_out_path is a string that specifies the path to which cut nodes, if
    any, should be written. (See note further below.)
    
    skelp_out_path is a string that specifies the path to which skeleton 
    nodes should be written. (See note further below.)
    
    vor_out_path is a string that specifies the path to which Voronoi 
    segments should be written. (See note further below.)
    
    vorp_out_path is a string that specifies the path to which Voronoi nodes
    should be written. (See note further below.)
    
    samp_out_path is a string that specifies the path to which boundary 
    samples should be written. (See note further below.)
    
    unpar_out_path is a string that specifies the path to which Voronoi 
    segments remaining in the graph skeleton, if any, should be written. 
    (See note further below.)
    
    test_out_path is a string that specifies the path to which each test 
    polygon should be written. The test polygon is a polygon that is at 
    least based on the input polygon and is used intenally for containment 
    tests. It is typically a proxy but may have identical geometry to the 
    input polygon. (See note further below.)
    
    rot_out_path is a string that specifies the path to which a possibly 
    rotated copy of each input polygon should be written. See 
    rotate_0_vertex. (See also note further below.)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    monitor_memory is a boolean that specifies whether a process should be 
    spawned to monitor memory use during execution.
    
    reduce_peak_memory is a boolean that specifies whether the peak memory
    footprint should be minimized.
    
    rotate_0_vertex is a boolean that specifies whether the "0-vertex", at 
    which an input polygon closes, should be "rotated", if necessary, to 
    subsequent vertices. The input polygon is not rotated, but its 0-vertex 
    hops around the boundary the input polygon. Such rotation may be helpful 
    in the special case that one wishes to use an especially coarse interval 
    that is just slightly too coarse for processing to otherwise succeed 
    (i.e., for the graph skeleton to be isolated).
    
    interpolate_3D is a boolean that specifies whether each skeleton, which 
    is derived in the x-y plane, should be (somewhat crudely) draped on the 
    surface defined by a 3D input polygon. interpolate_3D is ignored if the 
    input polygon is not 3D.
    
    output_log is a boolean that specifies whether a log should be written.    
    
    print_log is a boolean that specifies whether each line of the log 
    should also print to the screen.
    
    output_any_vectors is a boolean that specifies whether any vector data 
    should be outputted. output_any_vectors overrides all other vector data 
    output options.
    
    output_copy_only is a boolean that specifies whether a copy of each 
    polygon should be the only vector output. output_copy_only overrides all 
    vector data output options below.
    
    output_skeleton is a boolean that specifies whether skeletons should be 
    outputted.
    
    output_cut_nodes is a boolean that specifies whether cut nodes, if any,
    should be outputted.
    
    output_skeleton_nodes is a boolean that specifies whether skeleton nodes 
    should be outputted.
    
    output_vor_segs is a boolean that specifies whether Voronoi segments 
    should be outputted.
    
    output_vor_verts is a boolean that specifies whether Voronoi vertices 
    should be outputted.
    
    output_samps is a boolean that specifies whether boundary samples should 
    be outputted.
    
    output_unpar is a boolean that specifies whether Voronoi segments 
    remaining in the graph skeleton, if any, should be outputted.
    
    output_test_poly is a boolean that specifies whether test polygons 
    should be outputted.
    
    output_rot_poly is a boolean that specifies whether a possibly rotated 
    copy of each input polygon should be outputted.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    Any additional keyword arguments are passed to EasySkeleton.__init__()
    when it is called internally.
    
    Note: Argument descriptions for interval and min_normalized_length
    arguments are duplicated from EasySkeleton.__init__() and may refer to
    other arguments to that function and/or to EasySkeleton methods. More
    generally, documentation of EasySkeleton methods provides more detailed
    descriptions of how skeletonization is executed and what controls can be
    placed on that execution.
    
    Note: Each *_out_path argument (e.g., skel_out_path) will default to a 
    path in out_dir_path if the paired output_* argument (e.g., 
    output_skeleton) is set to True. That path will include the *_out_path 
    argument prefix (e.g., "skel"). The single exception is log_out_path, if 
    out_dir_path is a file geodatabase or feature dataset within a file 
    geodatabase, in which case log_out_path defaults to the directory 
    containing the file geodatabase. For vector data, the path may point to 
    a feature class in a file geodatabase (e.g., /path/to/data.gdb/
    feature_class) or a shapefile. In either case, the path must not exist 
    before the function is called. Internally, gdal is preferred to read and 
    write vector data, but if gdal cannot complete an operation and ESRI's 
    arcpy module is available, the operation will be attempted a second time 
    using that module. In the outputted vector_data, the following field 
    names have the indicating meanings:
        ComputeGB   memory footprint used by computation (not writing), in GB
        CumMaxGB    cumulative peak memory footprint, in GB
        Cut         whether cutting was required (True/False)
        FromVorIdx  Voronoi index of line's start
        Interval    interval used to derive the skeleton
        Kind        kind of line (e.g., branch) or node (e.g., hub)
        Length2D    length of the line in the x-y plane
        LwrBndInvl  lower bound on optimized interval
        MinFineIvl  minimum fine interval (for templating)
        NormLength  normalized length
        NoTail2D    length of the line, excluding any tail, in the x-y plane
        NoTailNL    normalized length, excluding any tail
        OptIntrvl   optimized interval (for targeted memory footprint)
        OrigID      feature ID of input polygon.
        Rotated     whether 0-vertex was rotated (True/False)
        SampIdx     boundary sample index
        SampIdx1    boundary sample index associated with Voronoi segment
        SampIdx2    boundary sample index associated with Voronoi segment
        TargMemGB   targeted memory footprint, in GB
        TmplIntrvl  template interval used to derive the skeleton, if any
        TmplNLXOff  template branches of lesser norm. length were discarded
        ToVorIdx    Voronoi index of line's end
        UprBndInvl  upper bound on optimized interval
        VorIdx      Voronoi vertex index

    Warning: It is strongly suggested that the current function only be
    called with keyword (rather than positional) arguments, except for
    in_path, as future versions may not preserve the same order.
    
    See also:
        process.external():
            Similar function that accepts an external settings file.
    """
    # Store calling arguments.
    orig_kwargs = locals().copy()
    final_kwargs = orig_kwargs.copy()
    
    # Standardize path to output directory.
    if out_dir_path is None:
        out_dir_path = os.path.dirname(in_path)  # *REASSIGNMENT*
    final_kwargs["out_dir_path"] = os.path.abspath(out_dir_path)

    # Derive unspecified output paths.
    base, ext = os.path.splitext(os.path.basename(in_path))
    format_out_path = "{}_{{}}{}{{}}".format(
        os.path.join(out_dir_path, "{}{}".format(out_path_prefix, base)),
        out_path_suffix
        ).format
    for k, v in final_kwargs.items():
        if k.endswith("_out_path") and v is None:
            if k != "log_out_path":
                final_kwargs[k] = format_out_path(k.partition("_")[0], ext)
                continue

            # In the special case of the log out-path, use the .txt
            # extension and extricate the file's placement from within a
            # file geodatabase, if necessary.
            final_kwargs[k] = format_out_path(k.partition("_")[0], ".txt")
            # If ".gdb" occurs in the out-directory-path, progressively
            # truncate that path until the level with that string is
            # dropped or the maximum reasonable number of levels are
            # dropped.
            if ".gdb" in out_dir_path.lower():
                log_out_dir_path = out_dir_path  # Initialize.
                for n in xrange(1, 3):
                    prev_log_out_dir_path = log_out_dir_path
                    log_out_dir_path = os.path.dirname(log_out_dir_path)
                    if ".gdb" in os.path.basename(prev_log_out_dir_path):
                        break
                else:
                    # Although the out-directory-path includes ".gdb",
                    # there are too many levels below the level with
                    # that string to be accommodated within a file
                    # geodatabase (even with a feature dataset).
                    # Therefore, assume that the occurrence of ".gdb" is
                    # due to an unfortunately-named directory and retain
                    # the original (naive) log out-path.
                    continue
                # Update log out-path to be in the same directory as
                # that which contains the file geodatabase.
                final_kwargs[k] = os.path.join(
                    log_out_dir_path, os.path.basename(final_kwargs[k])
                    )

    # Optionally apply vector output override.
    if not output_any_vectors:
        for k, v in final_kwargs.items():
            if k.startswith("output_") and k != "output_log":
                final_kwargs[k] = False

    # Validate monitor_memory option.
    if monitor_memory is None:
        monitor_memory = psutil is not None
    elif monitor_memory and psutil is None:
        raise TypeError("memory monitoring requires the psutil module")

    # Call main function.
    return _process(orig_kwargs, **final_kwargs)

def _process_external(settings_path, **kwargs):
    """
    Derive skeletons (e.g., centerlines) from input polygons.
    
    settings_path is a string that specifies the path to a file from which 
    default settings will be read. For example, if settings_path is the log 
    outputted by a previous call and no (overriding) arguments are specified,
    the exact same processing as created the log will be repreated.
    
    All remaining arguments are as documented in process().
    """
    final_kwargs = numgeo.util.read_settings(settings_path)
    final_kwargs.update(kwargs)
    return process(**final_kwargs)
process.external = _process_external

def _process(orig_kwargs, in_path, out_path_prefix, out_path_suffix, 
             out_dir_path, log_out_path, skel_out_path, copy_out_path, 
             cut_out_path, skelp_out_path, vor_out_path, vorp_out_path, 
             samp_out_path, unpar_out_path, test_out_path, rot_out_path,
             monitor_memory, reduce_peak_memory, rotate_0_vertex, 
             interpolate_3D, output_log, print_log, output_any_vectors, 
             output_copy_only, output_skeleton, output_cut_nodes, 
             output_skeleton_nodes, output_vor_segs, output_vor_verts, 
             output_samps, output_unpar, output_test_poly, output_rot_poly, 
             skel_kwargs, **more_skel_kwargs):

    # Store calling argument values.
    skel_kwargs.update(more_skel_kwargs)
    del more_skel_kwargs
    process_kwargs = locals().copy()
    del process_kwargs["orig_kwargs"]
    del process_kwargs["skel_kwargs"]

    # Remember start time.
    time0 = time.time()

    # Begin monitoring memory.
    if monitor_memory:
        overall_mem_mon = numgeo.util.MemoryMonitor()

    # Write out settings to log file and/or stdout (or neither).
    if not output_log:
        log_out_path = os.devnull
    with open(log_out_path, "w") as log:
        logger = numgeo.util.LogPrintTiming(log, print_log)
        unmatched_arg_names = logger.write_kwargs(
            skel_kwargs, numgeo.skel.EasySkeleton, include_unmatched=False
            )
        logger.write_kwargs(
            {arg_name: value for arg_name, value in skel_kwargs.iteritems()
             if arg_name in unmatched_arg_names},
            numgeo.skel.Voronoi2D, False
            )
        logger.write_kwargs(process_kwargs, process)
        logger.write_untimed('\n"""\nLog...')
        logger.start("overall_processing",
                     addendum=" (computations + writing out for all polygons)",
                     time=time0)

        # Optionally prepare for optimizing only.
        if output_copy_only:
            estimate_arg_names = inspect.getargspec(
                numgeo.skel.EasySkeleton.estimate_safe_interval
                )[0]
            estimate_kwargs = {k: skel_kwargs[k] for k in skel_kwargs
                               if k in estimate_arg_names}
            inf_substitute = -1.

        # Read in each polygon.
        default_desc = logger.default_desc
        with numgeo.vec_io.ReadCursor(in_path,
                                      partedness="ERROR_IF_MULTI") as in_data:
            for poly in in_data:
                logger.start("polygon_processing", poly,
                             " (computations + writing out)")
                logger.default_desc = default_desc  # Possibly restore.
                spatial_reference = poly.spatial_reference

                # Optionally compute interval optimization data.
                if output_copy_only:
                    # Note: Estimate the minimum fine interval first to
                    # avoid any residual memory footprint for the next
                    # set of estimates.
                    logger.start("optimizing_intervals", poly)
                    min_fine_interval = numgeo.skel.EasySkeleton.estimate_safe_interval(
                        poly, include_partitioning=False, **estimate_kwargs
                        )[0]
                    (opt_interval,
                     lower_bound_interval,
                     upper_bound_interval,
                     cut_bool,
                     targ_mem) = numgeo.skel.EasySkeleton.estimate_safe_interval(
                        poly, include_partitioning=True, **estimate_kwargs
                        )
                    logger.end("optimizing_intervals", poly)
                    logger.start("polygon_copy_writeout", poly)
                    with numgeo.vec_io.WriteCursor(copy_out_path) as out_copy_data:
                        # Do not attempt to write to fields that are not
                        # writable for file geodatabaes.
                        for field_name in poly.data.keys():
                            if field_name.upper() in ("SHAPE_LENGTH",
                                                      "SHAPE_AREA"):
                                del poly.data[field_name]
                        # Note: Any existing fields with the same names
                        # will be overwritten.
                        poly.data["OrigID"] = poly.ID
                        poly.data["OptIntrvl"] = opt_interval
                        poly.data["LwrBndInvl"] = lower_bound_interval
                        if math.isinf(upper_bound_interval):
                            poly.data["UprBndInvl"] = inf_substitute
                        else:
                            poly.data["UprBndInvl"] = upper_bound_interval
                        poly.data["MinFineIvl"] = min_fine_interval
                        poly.data["Cut"] = repr(cut_bool)
                        poly.data["TargMemGB"] = targ_mem
                        out_copy_data.append(poly)
                    logger.end("polygon_copy_writeout", poly)
                    logger.end("polygon_processing", poly,
                               " (computations + writing out")
                    if reduce_peak_memory:
                        del out_copy_data, poly
                        poly = None  # Avoid debugger flags.
                    continue

                # Optionally rotate the "0" vertex of each ring in the
                # input polygon to successive vertices until skeleton
                # isolation succeeds or all rotations for the ring with
                # the highest vertex count are exhausted.
                # Note: This is a very special-use functionality. Under
                # normal circumstance, simply choosing a finer sampling
                # interval or (less ideally) using the "CUT"
                # isolation_mode option are better options. However, 0-
                # vertex rotation could potentially be useful in niche
                # cases where a skeleton must be derived at a coarse
                # sampling interval and cutting must be avoided. The
                # reason that 0-vertex rotation works at all is that
                # sampling starts at the 0-vertex, so (in general)
                # different sample points are interpolated along the
                # input polygon's boundary with each rotation.
                logger.start("computations", poly)
                if monitor_memory:
                    computations_mem_mon = numgeo.util.MemoryMonitor()
                if rotate_0_vertex:
                    max_vertex_count = max([len(ring.coords_array)
                                            for ring in poly.boundary])
                    if poly.is_3D:
                        LineStringXD = numgeo.geom.LineString3D
                        PolygonXD = numgeo.geom.Polygon3D
                    else:
                        LineStringXD = numgeo.geom.LineString2D
                        PolygonXD = numgeo.geom.Polygon2D
                    time0_rotation = time.clock()
                    for rotation_number in xrange(max_vertex_count):
                        try:
                            skel = numgeo.skel.EasySkeleton(poly, **skel_kwargs)
                        except TypeError as e:
                            # If any error other than a skeleton
                            # isolation error was raised, re-raise it
                            # exactly.
                            type_, value, traceback = sys.exc_info()
                            if "skeleton could not be isolated" not in e.args[0]:
                                raise type_, value, traceback
                            new_boundary = []
                            for ring in poly.boundary:
                                old_coords_array = ring.coords_array
                                new_coords_array = numpy.empty(
                                    old_coords_array.shape
                                    )
                                new_coords_array[1:] = old_coords_array[:-1]
                                new_coords_array[0] = new_coords_array[-1]
                                new_boundary.append(LineStringXD(new_coords_array))
                            new_poly = PolygonXD(new_boundary)
                            new_poly.ID = poly.ID
                            new_poly.spatial_reference = spatial_reference
                            new_poly.data = poly.data
                            poly = new_poly  # *REASSIGNMENT*
                            if rotation_number == 0:
                                approx_max_rot_minutes = int(
                                    (time.clock() - time0_rotation)
                                    * max_vertex_count
                                    ) // 60
                                raw_input(
                                    "0-vertex rotation is necessary but could take up to ~{} minutes if all rotations are attempted. Press Enter to continue.".format(
                                        approx_max_rot_minutes
                                        )
                                    )
                                logger.start("vertex_rotation", poly)
                        except MemoryError:
                            logger.end("computations", poly, " (MemoryError!)")
                            raise
                        else:
                            if rotation_number:
                                logger.end("vertex_rotation", poly)
                            break
                    else:
                        logger.end("vertex_rotation", poly,
                                   ", which failed to isolate skeleton")
                        value.args = value.args + (
                            "(0-vertex rotation was attempted {} times.)".format(
                                rotation_number
                                ),
                            )
                        raise type_, value, traceback

                # Derive skeleton for polygon.
                else:
                    try:
                        skel = numgeo.skel.EasySkeleton(poly, **skel_kwargs)
                    except MemoryError:
                        logger.end("computations", poly, " (MemoryError!)")
                        raise
                logger.end("computations", poly)
                if monitor_memory:
                    computations_mem_mon.stop()

                # Prepare for 3D interpolation, if necessary.
                # *REASSIGNMENT*
                interpolate_3D = interpolate_3D and poly.is_3D
                if interpolate_3D:
                    MultiPointXD = numgeo.geom.MultiPoint3D
                else:
                    MultiPointXD = numgeo.geom.MultiPoint2D

                # Note: Data are written out in an order that reduces
                # the peak memory footprint if reduce_peak_memory is set
                # to True.

                # Write out (partitioned) skeleton parts.
                if output_skeleton:
                    logger.start("skeleton_writeout", poly)
                    with numgeo.vec_io.WriteCursor(skel_out_path) as out_skel_data:
                        parts = skel.get_lines()
                        # Optionally interpolate the z-coordinate.
                        if interpolate_3D:
                            logger.start("interpolation3D", poly)
                            # *REDEFINITION*
                            parts = skel.interpolate_3D(parts)
                            logger.end("interpolation3D", poly)
                        # Note: Memory use is fetched only once per
                        # input polygon for sake of performance.
                        # Note: Memory use is stored in GB because byte
                        # integers can become too large to convert to
                        # (32-bit) C long, raising an OverFlowError on
                        # write.
                        if monitor_memory:
                            computations_GB = (
                                computations_mem_mon.max_vms / 2.**30.
                                )
                            cum_max_GB = overall_mem_mon.check()[2] / 2.**30.
                        for part in parts:
                            if interpolate_3D:
                                part2D = part.original
                            else:
                                part2D = part
                            part.data["Kind"] = skel.describe_line(part2D,
                                                                   False)
                            from_vor_idx, to_vor_idx = part._aligned_key.tuple
                            part.data["FromVorIdx"] = from_vor_idx
                            part.data["ToVorIdx"] = to_vor_idx
                            part.data["Length2D"] = part.length2D
                            part.data["NoTail2D"] = part.untailed_length2D
                            if hasattr(part, "stem_width"):
                                part.data["NormLength"] = part2D.normalized_length
                                part.data["NoTailNL"] = 2. * part2D.untailed_length2D / part2D.stem_width
                            # Note: These intervals have explicit
                            # values, even if they were specified
                            # implicitly at initialization (by 0's).
                            part.data["Interval"] = skel.interval
                            # Note: template_interval and
                            # template_normalized_length fields will
                            # only be output if >=1 polygons use a
                            # template.
                            if skel.template_interval is not None:
                                part.data["TmplIntrvl"] = skel.template_interval
                                part.data["TmplNLXOff"] = skel.template_normalized_length_cutoff
                            if monitor_memory:
                                part.data["ComputeGB"] = computations_GB
                                part.data["CumMaxGB"] = cum_max_GB
                            part.data["OrigID"] = poly.ID
                            out_skel_data.append(part)
                        if reduce_peak_memory:
                            del parts, part
                    if reduce_peak_memory:
                        del out_skel_data
                    logger.end("skeleton_writeout", poly)

                # Write out nodes deleted during cutting (if any).
                if output_cut_nodes:
                    logger.start("cut_node_writeout", poly)
                    cut_node_data = skel.get_nodes(
                        "cut", False, interpolate_3D=interpolate_3D
                        )
                    if cut_node_data is None:
                        logger.end(
                            "cut_node_writeout", poly,
                            " [cancelled because there are no such nodes]"
                            )
                    else:
                        # Note: If the memory footprint should be
                        # reduced, avoid accumulating Point's in
                        # out_cut_data.cache. Instead, create and
                        # immediately write out each one.
                        with numgeo.vec_io.WriteCursor(
                            cut_out_path, write_on_exit=not reduce_peak_memory
                            ) as out_cut_data:
                            (cut_node_idxs_array,
                            cut_node_coords_array, _) = cut_node_data
                            cut_node_idxs = cut_node_idxs_array.tolist()
                            cut_nodes = MultiPointXD(cut_node_coords_array)
                            if reduce_peak_memory:
                                # *REASSIGNMENT*
                                cut_nodes = cut_nodes.iterate_only()
                                del cut_node_data, cut_node_idxs_array
                            for idx, node in itertools.izip(cut_node_idxs,
                                                            cut_nodes):
                                node.spatial_reference = spatial_reference
                                node.data["VorIdx"] = idx
                                node.data["Kind"] = "cut"
                                node.data["OrigID"] = poly.ID
                                out_cut_data.append(node)
                            if reduce_peak_memory:
                                del (cut_node_coords_array, cut_node_idxs,
                                     cut_nodes, node)
                        if reduce_peak_memory:
                            del out_cut_data
                        logger.end("cut_node_writeout", poly)

                # Write out (partitioned) skeleton nodes.
                if output_skeleton_nodes:
                    logger.start("node_writeout", poly)
                    # Note: If the memory footprint should be reduced,
                    # avoid accumulating Point's in
                    # out_skelp_data.cache. Instead, create and
                    # immediately write out each one.
                    with numgeo.vec_io.WriteCursor(
                        skelp_out_path, write_on_exit=not reduce_peak_memory
                        ) as out_skelp_data:
                        if not reduce_peak_memory:
                            node_datas = []
                        for kind in ("stub", "hub"):
                            node_data = skel.get_nodes(
                                kind, False, True, interpolate_3D=interpolate_3D
                                )
                            if node_data is None:
                                continue
                            (node_idxs_array,
                             node_coords_array, _) = node_data
                            node_idxs = node_idxs_array.tolist()
                            nodes = MultiPointXD(node_coords_array)
                            if reduce_peak_memory:
                                # *REASSIGNMENT*
                                nodes = nodes.iterate_only()
                                del node_data, node_idxs_array
                            else:
                                node_datas.append(node_data)
                            for idx, node in itertools.izip(node_idxs, nodes):
                                node.spatial_reference = spatial_reference
                                node.data["VorIdx"] = idx
                                node.data["Kind"] = kind
                                node.data["OrigID"] = poly.ID
                                out_skelp_data.append(node)
                            if reduce_peak_memory:
                                del node_coords_array, node_idxs, nodes, node
                    if reduce_peak_memory:
                        del out_skelp_data
                    logger.end("node_writeout", poly)

                # Write out unused graph edges.
                if output_unpar:
                    logger.start("unused_edge_writeout", poly)
                    graph_edges = skel.get_lines("graph")
                    if graph_edges:
                        with numgeo.vec_io.WriteCursor(unpar_out_path) as out_unpar_data:
                            for edge in skel.get_lines("graph"):
                                edge.data["Kind"] = skel.describe_line(edge)
                                from_vor_idx, to_vor_idx = edge._aligned_key.tuple
                                edge.data["FromVorIdx"] = from_vor_idx
                                edge.data["ToVorIdx"] = to_vor_idx
                                edge.data["Length2D"] = edge.length2D
                                edge.data["OrigID"] = poly.ID
                                out_unpar_data.append(edge)
                        logger.end("unused_edge_writeout", poly)
                        if reduce_peak_memory:
                            del graph_edges, edge
                    else:
                        logger.end(
                            "unused_edge_writeout", poly,
                            " [cancelled because there are no such edges]"
                            )

                # Optionally free memory by replacing skel object with a
                # substitute to which is attached only the attributes
                # necessary to support the write outs further below.
                if reduce_peak_memory:
                    substitute_skel = numgeo.util.Object()
                    substitute_skel.sampled_coords_array = skel.sampled_coords_array
                    substitute_skel.test_poly = skel.test_poly
                    substitute_skel.voronoi = getattr(skel, "voronoi", None)
                    del skel
                    skel = substitute_skel  # *REDEFINITION*
                    del substitute_skel

                # Write out testing polygon (i.e., the original polygon
                # or its proxy).
                if output_test_poly:
                    logger.start("testing_poly_writeout", poly)
                    with numgeo.vec_io.WriteCursor(test_out_path) as out_test_data:
                        test_poly = skel.test_poly
                        # For consistency, do not inherit field data.
                        if test_poly is poly:
                            # *REDEFINITION*
                            test_poly = numgeo.geom.Polygon2D(test_poly.boundary)
                            test_poly.spatial_reference = spatial_reference
                        test_poly.data["OrigID"] = poly.ID
                        out_test_data.append(test_poly)
                    if reduce_peak_memory:
                        del out_test_data, test_poly, skel.test_poly
                    logger.end("testing_poly_writeout", poly)

                # Write out (possibly rotated) input polygon if 0-vertex
                # rotation was allowed.
                if rotate_0_vertex and output_rot_poly:
                    logger.start("rotated_version_writeout", poly)
                    with numgeo.vec_io.WriteCursor(rot_out_path) as out_rot_data:
                        # Do not attempt to write to fields that are not
                        # writable for file geodatabaes.
                        for field_name in poly.data.keys():
                            if field_name.upper() in ("SHAPE_LENGTH",
                                                      "SHAPE_AREA"):
                                del poly.data[field_name]
                        poly.data["OrigID"] = poly.ID
                        poly.data["Rotated"] = repr(bool(rotation_number))
                        out_rot_data.append(poly)
                    if reduce_peak_memory:
                        del out_rot_data
                        # Discard poly after permanently storing its
                        # descriptive text.
                        logger.default_desc = lambda poly, desc=logger.generate_geom_text(poly): desc
                        poly = None
                    logger.end("rotated_version_writeout", poly)

                # Write out sample points from the input polygon's
                # boundary.
                if output_samps:
                    logger.start("sample_writeout", poly)
                    samps = MultiPointXD(skel.sampled_coords_array)
                    if reduce_peak_memory:
                        samps = samps.iterate_only()  # *REASSIGNMENT*
                    # Note: If the memory footprint should be reduced,
                    # avoid accumulating Point's in out_samp_data.cache.
                    # Instead, create and immediately write out each
                    # one.
                    with numgeo.vec_io.WriteCursor(
                        samp_out_path, write_on_exit=not reduce_peak_memory
                        ) as out_samp_data:
                        for idx, point in enumerate(samps):
                            point.spatial_reference = spatial_reference
                            point.data["SampIdx"] = idx
                            point.data["OrigID"] = poly.ID
                            out_samp_data.append(point)
                    if reduce_peak_memory:
                        del samps, out_samp_data, skel.sampled_coords_array, point
                    logger.end("sample_writeout", poly)

                # Write out Voronoi cell vertices.
                if output_vor_verts:
                    logger.start("voronoi_vertex_writeout", poly)
                    if skel.voronoi is None:
                        logger.end(
                            "voronoi_vertex_writeout", poly,
                            " [cancelled because no Voronoi data were recorded, presumably due to memory_option choice"
                            )
                    else:
                        # Note: If the memory footprint should be
                        # reduced, avoid accumulating Point's in
                        # out_vorp_data.cache. Instead, create and
                        # immediately write out each one.
                        with numgeo.vec_io.WriteCursor(
                            vorp_out_path, write_on_exit=not reduce_peak_memory
                            ) as out_vorp_data:
                            for point in skel.voronoi.iter_points():
                                point.data["VorIdx"] = point.out_idx
                                point.data["OrigID"] = poly.ID
                                out_vorp_data.append(point)
                        if reduce_peak_memory:
                            del out_vorp_data, point
                        logger.end("voronoi_vertex_writeout", poly)

                # Write out Voronoi segments.
                if output_vor_segs:
                    logger.start("segment_writeout", poly)
                    if skel.voronoi is None:
                        logger.end(
                            "segment_writeout", poly,
                            " [cancelled because no Voronoi data were recorded, presumably due to memory_option choice"
                            )
                    else:
                        vor_lines = skel.voronoi.iter_lines()
                        if not reduce_peak_memory:
                            # *REASSIGNMENT*
                            vor_lines = list(vor_lines)
                        # Note: If the memory footprint should be
                        # reduced, avoid accumulating LineString's in
                        # out_vor_data.cache. Instead, create and
                        # immediately write out each one.
                        with numgeo.vec_io.WriteCursor(
                            vor_out_path, write_on_exit=not reduce_peak_memory
                            ) as out_vor_data:
                            for line in vor_lines:
                                (line.data["FromVorIdx"],
                                 line.data["ToVorIdx"]) = line.out_idxs
                                (line.data["SampIdx1"],
                                 line.data["SampIdx2"]) = line.in_idxs_array.tolist()
                                line.data["Length2D"] = line.length2D
                                line.data["OrigID"] = poly.ID
                                out_vor_data.append(line)
                        if reduce_peak_memory:
                            del vor_lines, out_vor_data, line
                        logger.end("segment_writeout", poly)

                # End processing of current polygon.
                if reduce_peak_memory:
                    del skel
                logger.end("polygon_processing", poly,
                           " (computations + writing out")

        # End.
        logger.end("overall_processing",
                   addendum=" (computations + writing out for all polygons)")
        if monitor_memory:
            logger.write_timed(
                "peak memory footprint: {} GB".format(
                    overall_mem_mon.stop()[2] / 2.**30.
                    ),
                )
        logger.write_untimed('"""')



###############################################################################
# SCRIPT SUPPORT                                                              #
###############################################################################

# Optionally execute module as script.
if __name__ == "__main__":
    process.external(*sys.argv[1:])
