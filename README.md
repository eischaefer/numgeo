# Numgeo, a geospatial package

*Note: This file is best viewed with support for the Markdown markup language.*

Numgeo is a package of Python modules that, at present, primarily supports **skeletonization**. 

In simple terms, the skeleton of a polygon collapses an elongate polygon to a linear form. For example, the  outline of a river is a polygon (with a nonzero area). However, we often think of a river in terms of its skeleton, such as when a river is drawn as one or more squiggly lines on a map. Even though those lines do not represent the river's width, they still capture a useful expression of the river's geometry.



## The Algorithm, in brief

The algorithm builds on a long history of work by others. It also has a lot of moving parts, but these are the highlights:

1. Sample points along the boundary of an input polygon, including around any holes.
2. Compute a Voronoi diagram. (https://en.wikipedia.org/wiki/Voronoi_diagram)
3. Isolate the "graph skeleton" from the other extraneous bits  in the Voronoi diagram (which together make up the graph skeleton's "complement"). For example, each hole in polygon has its own skeleton that should be discarded.
4. "Partition out" paths from the graph skeleton to incrementally construct the "partitioned skeleton".
   - This is a bit like moving Lego blocks (segments) from one toy box (skeleton) to another, stacking some of them together during the transfer (to form continuous paths of many segments).
5. Optionally add "tails" to the paths so that they extend to polygon's boundary rather than stopping short.
6. Optionally prune away undesired paths to simplify the skeleton and remove noise.
   - This is accomplished using the normalized length metric.



## Quick start

If you're keen to get started, consider the following code (*but don't run it just yet!*):

```python
from numgeo.scripts.skel import process
process(r"path/to/polygon_shapefile.shp", interval=0.1, min_normalized_length=0.)
```

`interval`, in effect, specifies the desired resolution of the output skeleton, in map units (e.g., meters). If the narrowest constriction in your input polygons is *x* map units, you should specify `interval` *<0.5x*.

`min_normalized_length`, in effect, specifies how simple you want the output skeleton to be, with higher values corresponding to greater simplicity.  Any value less than 1. results in no simplification, so the "raw" skeleton itself is output, which may have a lot of unwanted bits.

Therefore, if you were to run the example code, you'd derive a "raw" skeleton at a resolution of 0.1 m (if your map unit is one meter). Because you did not specify an output path, the skeleton would default to *path/to/polygon_shapefile_skel.shp*.

#### Quick tips:

1. It is *highly* recommended that you read the documentation on Installation before installing numgeo.
2. Processing time and memory use both increase exponentially with progressively smaller `interval` values, so be careful when choosing a value.
3. You might consider calling `process(...)` a few times, each time with different `interval` and `min_normalized_length` values, to build an intuition for how these arguments affect the output skeleton.
   - In that case, you can specify `out_path_prefix` to avoid overwriting previous outputs. For example, `out_path_prefix="A_"` would output to *path/to/A_polygon_shapefile_skel.shp*.

