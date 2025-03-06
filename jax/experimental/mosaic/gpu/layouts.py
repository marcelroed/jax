# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layout utilities."""

import re

from jax._src.lib.mlir import ir
from . import fragmented_array as fa


_splat_fragmented_layout_attr_pattern = re.compile(
    r"^#mosaic_gpu.WGSplatFragLayout<\[(?P<shape>.*)\]>$"
)


def to_splat_fragmented_layout_attr(layout: fa.WGSplatFragLayout) -> ir.Attribute:
  """Constructs a #mosaic_gpu.WGSplatFragLayout attribute from a WGSplatFragLayout."""
  return ir.Attribute.parse(
      f"#mosaic_gpu.WGSplatFragLayout<{list(layout.shape)}>"
  )


def from_splat_fragmented_layout_attr(attr: ir.Attribute) -> fa.WGSplatFragLayout:
  """Constructs a WGSplatFragLayout from a #mosaic_gpu.WGSplatFragLayout attribute.

  Raises:
    ValueError: If the attribute is not a #mosaic_gpu.WGSplatFragLayout
      attribute.
  """
  match = _splat_fragmented_layout_attr_pattern.fullmatch(str(attr))
  if not match:
    raise ValueError(
        f"Expected a #mosaic_gpu.WGSplatFragLayout attribute, got {attr}"
    )

  return fa.WGSplatFragLayout(
      shape=tuple(int(s) for s in match.group("shape").split(","))
  )


def is_splat_fragmented_layout(attr: ir.Attribute) -> bool:
  return bool(_splat_fragmented_layout_attr_pattern.search(str(attr)))


_strided_fragmented_layout_attr_pattern = re.compile(
    r"^#mosaic_gpu.WGStridedFragLayout<\[(?P<shape>.*)\],"
    r" (?P<vector_size>\d+)>$"
)

def to_strided_fragmented_layout_attr(
    layout: fa.WGStridedFragLayout,
) -> ir.Attribute:
  """Constructs a #mosaic_gpu.WGStridedFragLayout attribute from a WGStridedFragLayout."""
  return ir.Attribute.parse(
      f"#mosaic_gpu.WGStridedFragLayout<{list(layout.shape)},"
      f" {layout.vec_size}>"
  )


def from_strided_fragmented_layout_attr(
    attr: ir.Attribute,
) -> fa.WGStridedFragLayout:
  """Constructs a WGStridedFragLayout from a #mosaic_gpu.WGStridedFragLayout attribute.

  Raises:
    ValueError: If the attribute is not a #mosaic_gpu.WGStridedFragLayout
      attribute.
  """
  match = _strided_fragmented_layout_attr_pattern.fullmatch(str(attr))
  if not match:
    raise ValueError(
        f"Expected a #mosaic_gpu.WGStridedFragLayout attribute, got {attr}"
    )

  return fa.WGStridedFragLayout(
      shape=tuple(int(s) for s in match.group("shape").split(",")),
      vec_size=int(match.group("vector_size")),
  )


def is_strided_fragmented_layout(attr: ir.Attribute) -> bool:
  return bool(_strided_fragmented_layout_attr_pattern.search(str(attr)))


def to_layout_attr(
    layout: (
        fa.WGSplatFragLayout
        | fa.WGStridedFragLayout
        | fa.TiledLayout
        | fa.WGMMARowFragLayout
    ),
) -> ir.Attribute:
  """Constructs an MLIR attribute that corresponds to the given layout."""
  match layout:
    case fa.WGSplatFragLayout():
      return to_splat_fragmented_layout_attr(layout)
    case fa.WGStridedFragLayout():
      return to_strided_fragmented_layout_attr(layout)
    case fa.WGMMA_LAYOUT:
      return ir.Attribute.parse("#mosaic_gpu.WGMMAFragLayout")
    case fa.WGMMARowFragLayout():
      return ir.Attribute.parse("#mosaic_gpu.WGMMARowFragLayout")
    case _:
      raise NotImplementedError(
          f"Unsupported layout for conversion to MLIR attribute: {layout}"
      )


_wgmma_fragmented_layout_attr_pattern = re.compile(
    r"^#mosaic_gpu.WGMMAFragLayout$"
)


def is_wgmma_fragmented_layout(attr: ir.Attribute) -> bool:
  return bool(_wgmma_fragmented_layout_attr_pattern.search(str(attr)))


_wgmma_row_fragmented_layout_attr_pattern = re.compile(
    r"^#mosaic_gpu.WGMMARowFragLayout$"
)


def is_wgmma_row_fragmented_layout(attr: ir.Attribute) -> bool:
  return bool(_wgmma_row_fragmented_layout_attr_pattern.search(str(attr)))


def from_layout_attr(
    attr: ir.Attribute,
) -> (
    fa.WGSplatFragLayout
    | fa.WGStridedFragLayout
    | fa.TiledLayout
    | fa.WGMMARowFragLayout
):
  """Constructs a layout from an MLIR attribute."""
  if is_splat_fragmented_layout(attr):
    return from_splat_fragmented_layout_attr(attr)
  elif is_strided_fragmented_layout(attr):
    return from_strided_fragmented_layout_attr(attr)
  elif is_wgmma_fragmented_layout(attr):
    return fa.WGMMA_LAYOUT
  elif is_wgmma_row_fragmented_layout(attr):
    return fa.WGMMARowFragLayout()
  else:
    raise NotImplementedError(
        f"Unsupported layout for conversion from MLIR attribute: {attr}"
    )
