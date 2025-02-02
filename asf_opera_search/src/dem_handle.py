import boto3
from botocore import UNSIGNED
from botocore.config import Config
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import box
import math

def get_dem_tiles(bounds, buffer_degrees=0.1):
    """
    Get the list of required GLO-30 DEM tiles given bounds.
    
    Parameters:
    -----------
    bounds : tuple
        (min_lon, min_lat, max_lon, max_lat) in geographic coordinates
    buffer_degrees : float
        Buffer to add around the bounds to ensure coverage
        
    Returns:
    --------
    list
        List of tuples containing (lat, lon) for required tiles
    """
    # Add buffer and round to nearest degree
    min_lon, min_lat, max_lon, max_lat = bounds
    min_lon = math.floor(min_lon - buffer_degrees)
    min_lat = math.floor(min_lat - buffer_degrees)
    max_lon = math.ceil(max_lon + buffer_degrees)
    max_lat = math.ceil(max_lat + buffer_degrees)
    
    tiles = []
    for lat in range(min_lat, max_lat):
        for lon in range(min_lon, max_lon):
            tiles.append((lat, lon))
    
    return tiles

def get_dem_url(lat, lon, download_path=None):
    """
    Generate S3 URL for a GLO-30 DEM tile and optionally download it.
    
    Parameters:
    -----------
    lat : int
        Latitude of tile corner
    lon : int
        Longitude of tile corner
    download_path : str, optional
        Directory to save the DEM tile if downloading is requested
        
    Returns:
    --------
    str
        S3 URL for the DEM tile or path to downloaded file
    """
    # Format coordinates for filename
    lat_str = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
    lon_str = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
    
    # Construct directory and filename
    dir_name = f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"
    
    url = f"s3://copernicus-dem-30m/{dir_name}/{dir_name}.tif"
    
    # If download is requested
    if download_path:
        try:
            # Create download directory if it doesn't exist
            os.makedirs(download_path, exist_ok=True)
            
            # Construct local file path
            local_path = os.path.join(download_path, f"{dir_name}.tif")
            
            # Skip if file already exists
            if os.path.exists(local_path):
                print(f"DEM tile already exists: {local_path}")
                return local_path
            
            print(f"Downloading {dir_name}.tif...")
            
            # Initialize S3 client with unsigned config
            s3_client = boto3.client('s3',
                                   region_name='us-west-2',
                                   config=Config(signature_version=UNSIGNED))
            
            # Download file
            s3_client.download_file('copernicus-dem-30m', 
                                  f"{dir_name}/{dir_name}.tif", 
                                  local_path)
            
            print(f"Successfully downloaded to {local_path}")
            return local_path
            
        except Exception as e:
            print(f"Error downloading {dir_name}.tif: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            return url
    
    return url

def merge_dem_streams(tile_coordinates, output_path=None, download_path=None):
    """
    Merge multiple DEM tiles streamed from S3.
    
    Parameters:
    -----------
    tile_coordinates : list
        List of (lat, lon) tuples for required tiles
    output_path : str, optional
        Path to save merged DEM. If None, only returns the data
        
    Returns:
    --------
    tuple
        (merged_data, transform, crs)
    """
    # Configure environment for S3 access
    environ = {
        'AWS_NO_SIGN_REQUEST': 'YES',
        'AWS_ACCESS_KEY_ID': '',
        'AWS_SECRET_ACCESS_KEY': '',
        'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
        'AWS_REGION': 'us-west-2',
        'AWS_S3_ENDPOINT': 's3.amazonaws.com'
    }
    
    # Set GDAL configurations
    import os
    os.environ.update(environ)
    
    # Open all raster files (either as streams or from downloaded files)
    src_files = []
    for lat, lon in tile_coordinates:
        url = get_dem_url(lat, lon, download_path)
        try:
            if download_path and os.path.exists(url):  # url will be local path if downloaded
                src = rasterio.open(url)
            else:
                src = rasterio.open(url, environ=environ)
            src_files.append(src)
        except Exception as e:
            print(f"Error opening {url}: {e}")
    
    if not src_files:
        raise ValueError("No valid DEM tiles to merge")
    
    # Merge rasters
    merged_data, transform = merge(src_files)
    
    # Get CRS from first file
    crs = src_files[0].crs
    
    # Save merged DEM if output path provided
    if output_path:
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=merged_data.shape[1],
            width=merged_data.shape[2],
            count=1,
            dtype=merged_data.dtype,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(merged_data)
    
    # Close all files
    for src in src_files:
        src.close()
    
    return merged_data[0], transform, crs

def download_glo30_tile(lat, lon, output_dir):
    """
    Download a single GLO-30 DEM tile from AWS S3.
    
    Parameters:
    -----------
    lat : int
        Latitude of tile corner
    lon : int
        Longitude of tile corner
    output_dir : str
        Directory to save downloaded tiles
        
    Returns:
    --------
    str
        Path to downloaded file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Format coordinates for filename
    # Note: GLO-30 uses 00 padding for both lat and lon
    lat_str = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
    lon_str = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
    
    # Construct filename according to GLO-30 convention
    filename = f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM.tif"
    local_path = os.path.join(output_dir, filename)
    
    # Check if file already exists
    if os.path.exists(local_path):
        print(f"DEM tile already exists: {filename}")
        return local_path
    
    try:
        print(f"Downloading {filename}...")
        
        # Initialize S3 client with unsigned config (for public access)
        s3_client = boto3.client('s3', 
                                region_name='us-west-2',
                                config=Config(signature_version=UNSIGNED))
        
        # Construct S3 path
        # The file is inside a directory with the same name (without .tif)
        dir_name = f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"
        bucket_name = 'copernicus-dem-30m'
        s3_key = f"{dir_name}/{dir_name}.tif"
        
        # Download file from S3
        s3_client.download_file(bucket_name, s3_key, local_path)
        
        print(f"Successfully downloaded {filename}")
        return local_path
    
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return None

def merge_dem_tiles(tile_paths):
    """
    Merge multiple DEM tiles into a single raster.
    
    Parameters:
    -----------
    tile_paths : list
        List of paths to DEM tiles
        
    Returns:
    --------
    tuple
        (merged_data, transform, crs)
    """
    # Open all raster files
    src_files = [rasterio.open(path) for path in tile_paths if path is not None]
    
    if not src_files:
        raise ValueError("No valid DEM tiles to merge")
    
    # Merge rasters
    merged_data, transform = merge(src_files)
    
    # Get CRS from first file (they should all be the same)
    crs = src_files[0].crs
    
    # Close all files
    for src in src_files:
        src.close()
    
    return merged_data, transform, crs

def reproject_dem_to_cslc(dem_data, dem_transform, dem_crs, params, output_path=None):
    """
    Reproject merged DEM to match CSLC coordinates.
    
    Parameters:
    -----------
    dem_data : numpy.ndarray
        Merged DEM data
    dem_transform : affine.Affine
        Transform of merged DEM
    dem_crs : CRS
        CRS of merged DEM
    params : dict
        CSLC parameters containing coordinate information
    output_path : str, optional
        Path to save reprojected DEM
        
    Returns:
    --------
    numpy.ndarray
        Reprojected DEM data
    """
    # Get target parameters from CSLC
    dst_crs = f"EPSG:{params['epsg']}"
    x_coords = params['x_coordinates']
    y_coords = params['y_coordinates']
    
    # Calculate target shape and transform
    dst_shape = (len(y_coords), len(x_coords))
    dst_bounds = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    dst_transform = rasterio.transform.from_bounds(*dst_bounds, dst_shape[1], dst_shape[0])
    
    # Initialize output array
    dst_dem = np.zeros(dst_shape, dtype=dem_data.dtype)
    
    # Perform reprojection
    reproject(
        source=dem_data,
        destination=dst_dem,
        src_transform=dem_transform,
        src_crs=dem_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear
    )
    
    # Save reprojected DEM if output path provided
    if output_path:
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=dst_shape[0],
            width=dst_shape[1],
            count=1,
            dtype=dst_dem.dtype,
            crs=dst_crs,
            transform=dst_transform
        ) as dst:
            dst.write(dst_dem, 1)
    
    return dst_dem

def plot_dem(dem_data, title="Digital Elevation Model"):
    """
    Visualize DEM data.
    
    Parameters:
    -----------
    dem_data : numpy.ndarray
        DEM elevation data
    title : str
        Plot title
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot DEM
    im = ax.imshow(dem_data, cmap='terrain')
    
    # Add colorbar with matching height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='Elevation (m)')
    
    # Set title and turn off axis
    ax.set_title(title)
    ax.axis('off')
    
    plt.show()

def process_dem_for_cslc(params, merged_output=None, final_output=None, download_path=None):
    """
    Main function to process DEM data for CSLC coverage.
    
    Parameters:
    -----------
    params : dict
        CSLC parameters containing coordinate information
    merged_output : str, optional
        Path to save merged DEM
    final_output : str, optional
        Path to save final reprojected DEM
        
    Returns:
    --------
    numpy.ndarray
        Processed DEM data matching CSLC coordinates
    """
    # Convert CSLC bounds to geographic coordinates
    from pyproj import Transformer
    
    transformer = Transformer.from_crs(f"EPSG:{params['epsg']}", "EPSG:4326", always_xy=True)
    
    # Get bounds in geographic coordinates
    x_coords = params['x_coordinates']
    y_coords = params['y_coordinates']
    
    corners = [
        transformer.transform(min(x_coords), min(y_coords)),
        transformer.transform(max(x_coords), max(y_coords))
    ]
    
    bounds = (
        min(c[0] for c in corners),
        min(c[1] for c in corners),
        max(c[0] for c in corners),
        max(c[1] for c in corners)
    )
    
    # Get required DEM tiles
    tiles = get_dem_tiles(bounds)
    
    # Merge tiles (either streaming or downloaded)
    merged_data, transform, crs = merge_dem_streams(tiles, merged_output, download_path)
    
    # Reproject to match CSLC
    dem_reprojected = reproject_dem_to_cslc(merged_data, transform, crs, params, final_output)
    
    return dem_reprojected