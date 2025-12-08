"""
EASE-Grid 2.0 Southern Hemisphere 50km Grid Generator.

This module provides a production-ready implementation for creating
EASE-Grid Version 2 Polar Southern Hemisphere 50km coordinate grids.

Reference:
    NSIDC EASE-Grid 2.0 Projection: https://nsidc.org/ease/ease-grid-projection-gt

Author: Data Engineering Team
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from pyproj import CRS, Transformer


@dataclass(frozen=True)
class EaseGrid2S50KmConfig:
    """Immutable configuration for EASE-Grid 2.0 Southern Hemisphere 50km.
    
    Attributes:
        grid_cell_size_m: Grid cell size in meters (50,000m = 50km).
        num_cols: Number of columns in the grid.
        num_rows: Number of rows in the grid.
        col_origin_offset: Column origin offset (r0).
        row_origin_offset: Row origin offset (s0).
        center_latitude: Projection center latitude (-90 for South Pole).
        center_longitude: Projection center longitude.
        false_easting: False easting in meters.
        false_northing: False northing in meters.
        ellipsoid: Reference ellipsoid name.
        datum: Geodetic datum name.
    """
    grid_cell_size_m: float = 50000.0
    num_cols: int = 216
    num_rows: int = 216
    col_origin_offset: float = 108.0
    row_origin_offset: float = 108.0
    center_latitude: float = -90.0
    center_longitude: float = 0.0
    false_easting: float = 0.0
    false_northing: float = 0.0
    ellipsoid: str = "WGS84"
    datum: str = "WGS84"


class EaseGrid2SouthernHemisphere50Km:
    """EASE-Grid Version 2 Polar Southern Hemisphere 50km Grid.
    
    This class implements the EASE-Grid 2.0 projection for the Southern
    Hemisphere at 50km resolution, providing methods for coordinate
    transformation between geographic (lat/lon) and grid (row/col) coordinates.
    
    The EASE-Grid 2.0 uses Lambert Azimuthal Equal-Area (LAEA) projection
    centered at the South Pole (-90°, 0°) with WGS84 ellipsoid.
    
    Example:
        >>> grid = EaseGrid2SouthernHemisphere50Km()
        >>> lon, lat = grid.get_grid_coordinates()
        >>> row, col = grid.geographic_to_grid(-70.0, 45.0)
        >>> lon, lat = grid.grid_to_geographic(100, 100)
    
    Attributes:
        config: Grid configuration parameters.
        crs: Coordinate Reference System object.
    """
    
    # EPSG code for EASE-Grid 2.0 South
    EPSG_CODE: int = 6932
    
    # Proj4 string for explicit projection definition
    PROJ4_STRING: str = (
        "+proj=laea +lat_0=-90 +lon_0=0 +x_0=0 +y_0=0 "
        "+ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    )

    def __init__(self, config: Optional[EaseGrid2S50KmConfig] = None) -> None:
        """Initialize the EASE-Grid 2.0 Southern Hemisphere 50km grid.
        
        Args:
            config: Optional custom configuration. Uses default 50km config if None.
        """
        self.config = config or EaseGrid2S50KmConfig()
        self._crs = CRS.from_proj4(self.PROJ4_STRING)
        self._transformer_to_geo = Transformer.from_crs(
            self._crs, CRS.from_epsg(4326), always_xy=True
        )
        self._transformer_from_geo = Transformer.from_crs(
            CRS.from_epsg(4326), self._crs, always_xy=True
        )
        self._longitude: Optional[NDArray[np.float64]] = None
        self._latitude: Optional[NDArray[np.float64]] = None

    @property
    def crs(self) -> CRS:
        """Return the Coordinate Reference System."""
        return self._crs

    @property
    def shape(self) -> Tuple[int, int]:
        """Return grid shape as (rows, cols)."""
        return (self.config.num_rows, self.config.num_cols)

    @property
    def geotransform(self) -> Tuple[float, float, float, float, float, float]:
        """Return GDAL-style affine geotransform coefficients.
        
        Returns:
            Tuple of (x_origin, pixel_width, x_rotation, y_origin, y_rotation, pixel_height).
        """
        x_origin = -self.config.col_origin_offset * self.config.grid_cell_size_m
        y_origin = self.config.row_origin_offset * self.config.grid_cell_size_m
        return (
            x_origin,
            self.config.grid_cell_size_m,
            0.0,
            y_origin,
            0.0,
            -self.config.grid_cell_size_m,
        )

    def get_grid_coordinates(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute longitude and latitude for all grid cell centers.
        
        Returns:
            Tuple of (longitude, latitude) arrays with shape (num_rows, num_cols).
            Longitude values are in degrees East (-180 to 180).
            Latitude values are in degrees North (-90 to 90).
        
        Note:
            Results are cached after first computation for performance.
        """
        if self._longitude is None or self._latitude is None:
            self._compute_grid_coordinates()
        return self._longitude, self._latitude

    def _compute_grid_coordinates(self) -> None:
        """Compute and cache grid coordinates."""
        # Create 1-based row and column indices
        cols = np.arange(1, self.config.num_cols + 1, dtype=np.float64)
        rows = np.arange(1, self.config.num_rows + 1, dtype=np.float64)
        col_grid, row_grid = np.meshgrid(cols, rows)
        
        # Convert grid indices to projected coordinates (meters)
        x_proj = (col_grid - self.config.col_origin_offset - 1) * self.config.grid_cell_size_m
        y_proj = (row_grid - self.config.row_origin_offset - 1) * -self.config.grid_cell_size_m
        
        # Transform to geographic coordinates
        self._longitude, self._latitude = self._transformer_to_geo.transform(x_proj, y_proj)

    def geographic_to_grid(
        self, 
        latitude: float | NDArray[np.float64], 
        longitude: float | NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Convert geographic coordinates to grid row/column indices.
        
        Args:
            latitude: Latitude in degrees North (-90 to 90).
            longitude: Longitude in degrees East (-180 to 180).
        
        Returns:
            Tuple of (row, col) indices (1-based).
        
        Example:
            >>> grid = EaseGrid2SouthernHemisphere50Km()
            >>> row, col = grid.geographic_to_grid(-70.0, 45.0)
        """
        x_proj, y_proj = self._transformer_from_geo.transform(longitude, latitude)
        
        col = self.config.col_origin_offset + x_proj / self.config.grid_cell_size_m + 1
        row = self.config.row_origin_offset - y_proj / self.config.grid_cell_size_m + 1
        
        return np.asarray(row), np.asarray(col)

    def grid_to_geographic(
        self, 
        row: float | NDArray[np.float64], 
        col: float | NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Convert grid row/column indices to geographic coordinates.
        
        Args:
            row: Row index (1-based).
            col: Column index (1-based).
        
        Returns:
            Tuple of (longitude, latitude) in degrees.
        
        Example:
            >>> grid = EaseGrid2SouthernHemisphere50Km()
            >>> lon, lat = grid.grid_to_geographic(100, 100)
        """
        x_proj = (col - self.config.col_origin_offset - 1) * self.config.grid_cell_size_m
        y_proj = (row - self.config.row_origin_offset - 1) * -self.config.grid_cell_size_m
        
        longitude, latitude = self._transformer_to_geo.transform(x_proj, y_proj)
        return np.asarray(longitude), np.asarray(latitude)

    def is_valid_grid_index(
        self, 
        row: float | NDArray[np.float64], 
        col: float | NDArray[np.float64]
    ) -> NDArray[np.bool_]:
        """Check if grid indices are within valid bounds.
        
        Args:
            row: Row index (1-based).
            col: Column index (1-based).
        
        Returns:
            Boolean array indicating valid indices.
        """
        row_arr = np.asarray(row)
        col_arr = np.asarray(col)
        
        valid_row = (row_arr >= 1) & (row_arr <= self.config.num_rows)
        valid_col = (col_arr >= 1) & (col_arr <= self.config.num_cols)
        
        return valid_row & valid_col


def create_ease_grid_2_south_50km() -> EaseGrid2SouthernHemisphere50Km:
    """Factory function to create EASE-Grid 2.0 Southern Hemisphere 50km grid.
    
    This is the recommended entry point for creating grid instances,
    following the factory pattern for cleaner API design.
    
    Returns:
        Configured EaseGrid2SouthernHemisphere50Km instance.
    
    Example:
        >>> grid = create_ease_grid_2_south_50km()
        >>> lon, lat = grid.get_grid_coordinates()
        >>> print(f"Grid shape: {grid.shape}")
        Grid shape: (216, 216)
    """
    return EaseGrid2SouthernHemisphere50Km()


# Module-level convenience function
def get_s2_50km_coordinates() -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get longitude and latitude arrays for EASE-Grid 2.0 South 50km.
    
    Convenience function for quick access to grid coordinates without
    instantiating the full grid object.
    
    Returns:
        Tuple of (longitude, latitude) arrays with shape (216, 216).
    
    Example:
        >>> lon, lat = get_s2_50km_coordinates()
        >>> print(lon.shape, lat.shape)
        (216, 216) (216, 216)
    """
    grid = create_ease_grid_2_south_50km()
    return grid.get_grid_coordinates()


if __name__ == "__main__":
    # Example usage and validation
    grid = create_ease_grid_2_south_50km()
    
    print("=" * 60)
    print("EASE-Grid 2.0 Southern Hemisphere 50km Grid")
    print("=" * 60)
    print(f"Grid Shape: {grid.shape}")
    print(f"Cell Size: {grid.config.grid_cell_size_m / 1000:.0f} km")
    print(f"CRS: {grid.crs.name}")
    print(f"Geotransform: {grid.geotransform}")
    print()
    
    # Compute coordinates
    lon, lat = grid.get_grid_coordinates()
    print(f"Longitude range: [{lon.min():.2f}°, {lon.max():.2f}°]")
    print(f"Latitude range: [{lat.min():.2f}°, {lat.max():.2f}°]")
    print()
    
    # Test forward/inverse transformation
    test_lat, test_lon = -70.0, 45.0
    row, col = grid.geographic_to_grid(test_lat, test_lon)
    lon_back, lat_back = grid.grid_to_geographic(row, col)
    
    print(f"Forward transform: ({test_lat}°, {test_lon}°) -> (row={row:.2f}, col={col:.2f})")
    print(f"Inverse transform: (row={row:.2f}, col={col:.2f}) -> ({lat_back:.6f}°, {lon_back:.6f}°)")
    print(f"Round-trip error: {abs(test_lat - lat_back):.2e}°, {abs(test_lon - lon_back):.2e}°")
