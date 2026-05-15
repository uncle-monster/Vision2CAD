# Writes stl to obj file.
from OCC.Extend.DataExchange import write_stl_file
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
import numpy as np
import trimesh
import os


def stl_to_obj(stl_file, obj_file):
    trimesh.load_mesh(stl_file).export(obj_file)


# Converts CAD solid to obj file through intermediate stl.
def cad_to_obj(shape, obj_file):
    stl_file = obj_file[:-3] + 'stl'
    write_stl_file(shape, stl_file)
    stl_to_obj(stl_file, obj_file)
    os.system("rm " + stl_file)


def cad_to_step(shape, step_file):
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    writer.Write(step_file)