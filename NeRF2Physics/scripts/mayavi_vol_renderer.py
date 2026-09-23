"""
Low-level mayavi volumetric-rendering helpers, adapted from
wisp/trainers/tracker/mayavi_vol_renderer.py in the same author's pixi-wisp
repo (https://github.com/barikata1984/pixi-wisp), trimmed to the static
single-shot rendering this project needs (no per-epoch video/tracking).

Kept as-is from that source: the headless EGL->OSMesa fallback (this
container's /dev/dri/renderD* exists but isn't readable/writable by the
non-root user, which crashes mayavi's default EGL offscreen path outright
rather than raising a catchable error) and the ctf/otf colormap construction.
"""
import glob
import os

import tvtk.pyface.tvtk_scene as _tvtk_scene
from mayavi import mlab
from mayavi.core.lut_manager import LUTManager
from tvtk.util import ctf  # ColorTransferFunction


def _egl_render_node_usable() -> bool:
    """True if at least one DRI render node is readable/writable (needed by EGL)."""
    return any(os.access(n, os.R_OK | os.W_OK) for n in glob.glob("/dev/dri/renderD*"))


class _NoEGLTVTK:
    """Proxy over the ``tvtk`` namespace that hides ``EGLRenderWindow``."""

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name):
        if name == "EGLRenderWindow":
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_real"), name)


if (
    not getattr(_tvtk_scene, "_wisp_force_osmesa", False)
    and hasattr(_tvtk_scene.tvtk, "OSOpenGLRenderWindow")
    and not _egl_render_node_usable()
):
    _orig_create_control = _tvtk_scene.TVTKScene._create_control

    def _create_control(self, parent):
        real_tvtk = _tvtk_scene.tvtk
        if self.off_screen_rendering:
            _tvtk_scene.tvtk = _NoEGLTVTK(real_tvtk)
        try:
            return _orig_create_control(self, parent)
        finally:
            _tvtk_scene.tvtk = real_tvtk

    _tvtk_scene.TVTKScene._create_control = _create_control
    _tvtk_scene._wisp_force_osmesa = True


def get_custom_colormap(name, num_colors=256, opacity=1.0, transparent_input=-1.0):
    """Build a (color transfer function, opacity transfer function) pair from a
    named mayavi/VTK colormap. Scalar values <= transparent_input render fully
    transparent -- used here to hide unoccupied voxels (see
    visualize_density_volume_mayavi.py)."""
    lm = LUTManager(number_of_colors=num_colors, lut_mode=name, show_scalar_bar=True)
    rgbs = lm.lut.table.to_array()[:, :3]
    rgbs = rgbs.astype(float) / (num_colors - 1)
    _ctf = ctf.ColorTransferFunction()
    for i, rgb in enumerate(rgbs):
        _ctf.add_rgb_point(i / (num_colors - 1), *rgb)

    _otf = ctf.PiecewiseFunction()
    _otf.add_point(transparent_input, 0.0)  # transparent
    _otf.add_point(0.0, opacity)
    _otf.add_point(1.0, opacity)
    _ctf.range = [transparent_input, 1.0]

    return _ctf, _otf


def init_mayavi_vol_renderer(s, x=None, y=None, z=None, size=720, _ctf=None, _otf=None, shade=True):
    """Create a mayavi figure + volume actor for a scalar field ``s`` (optionally
    on an explicit ``x``/``y``/``z`` grid)."""
    black = (0, 0, 0)
    white = (1, 1, 1)
    size = (size, size)
    figure = mlab.figure(bgcolor=white, fgcolor=black, size=size)

    inputs = [s]
    if x is not None and y is not None and z is not None:
        inputs = [x, y, z] + inputs

    scalar_field = mlab.pipeline.scalar_field(*inputs)
    volume = mlab.pipeline.volume(scalar_field, figure=figure)

    if _ctf is not None:
        volume._ctf = _ctf
        volume._volume_property.set_color(_ctf)
        volume.update_ctf = True

    if _otf is not None:
        volume._otf = _otf
        volume._volume_property.set_scalar_opacity(_otf)

    volume._volume_property.shade = shade

    return figure, volume
