from typing import Dict, Tuple, Any
import numpy as np
import h5py

def read_opera_cslc(hdf_path: str, polarization: str = 'VV', deramping_flag: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read and process OPERA CSLC data from HDF file.
    
    Args:
        hdf_path (str): Path to the HDF file
        polarization (str): Polarization type to read
        deramping_flag (bool): Flag to apply deramping correction
        
    Returns:
        Tuple containing:
            - np.ndarray: Corrected CSLC data
            - Dict[str, Any]: Dictionary containing all parameters directly
    """
    # Define constant paths
    PATHS = {
        'grid': 'data',
        'metadata': 'metadata/processing_information/input_burst_metadata',
        'id': 'identification'
    }
    
    with h5py.File(hdf_path, 'r') as h5:
        # Read all data
        cslc = h5[f'{PATHS["grid"]}/{polarization}'][:]
        azimuth_phase = h5[f'{PATHS["grid"]}/azimuth_carrier_phase'][:]
        flatten_phase = h5[f'{PATHS["grid"]}/flattening_phase'][:]
        
        # Create flat dictionary of all parameters
        parameters = {
            'azimuth_phase': azimuth_phase,
            'flatten_phase': flatten_phase,
            'x_coordinates': h5[f'{PATHS["grid"]}/x_coordinates'][:],
            'y_coordinates': h5[f'{PATHS["grid"]}/y_coordinates'][:],
            'x_spacing': int(h5[f'{PATHS["grid"]}/x_spacing'][()]),
            'y_spacing': int(h5[f'{PATHS["grid"]}/y_spacing'][()]),
            'epsg': int(h5[f'{PATHS["grid"]}/projection'][()]),
            'sensing_start': h5[f'{PATHS["metadata"]}/sensing_start'][()].decode(),
            'sensing_stop': h5[f'{PATHS["metadata"]}/sensing_stop'][()].decode(),
            'dimensions': h5[f'{PATHS["metadata"]}/shape'][:],
            'bounding_polygon': h5[f'{PATHS["id"]}/bounding_polygon'][()].decode(),
            'orbit_direction': h5[f'{PATHS["id"]}/orbit_pass_direction'][()].decode(),
            'burst_id': h5[f'{PATHS["id"]}/burst_id'][()].decode(),
            'center_lon': h5[f'{PATHS["metadata"]}/center'][0],
            'center_lat': h5[f'{PATHS["metadata"]}/center'][1],
            'wavelength': h5[f'{PATHS["metadata"]}/wavelength'][()]
        }
        
        # Process CSLC correction
        if deramping_flag:
            phase_correction = np.exp(1j * (azimuth_phase + flatten_phase))
            cslc = cslc * np.conj(phase_correction)

    return cslc, parameters