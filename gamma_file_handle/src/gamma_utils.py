from enum import Enum
import numpy as np
from typing import Dict, List, Union, Tuple, Optional

class DataType(Enum):
    """Enumeration of supported data types"""
    SCOMPLEX = 'scomplex'  # 4-byte complex integers
    FCOMPLEX = 'fcomplex'  # 8-byte complex floats
    FLOAT = 'float'        # 4-byte floats
    BYTE = 'byte'         # 1-byte unsigned integers

    @classmethod
    def from_string(cls, value: str) -> 'DataType':
        """Convert string to DataType, case-insensitive"""
        try:
            return cls(value.lower())
        except ValueError:
            # Try matching uppercase string to enum value
            try:
                return next(t for t in cls if t.value.upper() == value.upper())
            except StopIteration:
                raise ValueError(f"Invalid data type: {value}. Valid types are: {[t.value for t in cls]}")

def read_binary_file(filename: str, width: int, length: int, 
                    data_type: Union[DataType, str], big_endian: bool = True) -> np.ndarray:
    """
    Read a binary file containing complex, float, or byte data
    
    Parameters:
    -----------
    filename : str
        Path to the binary file
    width : int
        Width of the image/data
    length : int
        Length of the image/data
    data_type : Union[DataType, str]
        Type of data to read: 'scomplex', 'fcomplex', 'float', or 'byte'
        Can be provided as string or DataType enum
    big_endian : bool, optional
        If True, read as big-endian (default)
        If False, read as little-endian
        Note: endianness doesn't affect byte data
        
    Returns:
    --------
    numpy.ndarray
        Array of shape (length, width)
    """
    # Convert string to DataType if necessary
    if isinstance(data_type, str):
        data_type = DataType.from_string(data_type)
    
    # Set endianness prefix
    endian = '>' if big_endian else '<'
    
    # Define data types for different formats
    if data_type == DataType.SCOMPLEX:
        dt = np.dtype([('real', f'{endian}i2'), ('imag', f'{endian}i2')])
        convert_to_complex = True
    elif data_type == DataType.FCOMPLEX:
        dt = np.dtype([('real', f'{endian}f4'), ('imag', f'{endian}f4')])
        convert_to_complex = True
    elif data_type == DataType.FLOAT:
        dt = np.dtype(f'{endian}f4')
        convert_to_complex = False
    elif data_type == DataType.BYTE:
        dt = np.dtype('u1')  # unsigned 1-byte integer (endianness doesn't matter)
        convert_to_complex = False
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    # Read the binary file
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=dt)
    
    if convert_to_complex:
        data = data['real'] + 1j * data['imag']
    
    # Reshape the array according to the specified dimensions
    data = data.reshape(length, width)
    
    return data

def write_binary_file(filename: str, data: np.ndarray, 
                     data_type: Union[DataType, str], big_endian: bool = True) -> None:
    """
    Write a numpy array to a binary file
    
    Parameters:
    -----------
    filename : str
        Path to the output binary file
    data : numpy.ndarray
        Data to write (2D array)
    data_type : Union[DataType, str]
        Type of data to write: 'scomplex', 'fcomplex', 'float', or 'byte'
        Can be provided as string or DataType enum
    big_endian : bool, optional
        If True, write as big-endian (default)
        If False, write as little-endian
        Note: endianness doesn't affect byte data
    """
    # Convert string to DataType if necessary
    if isinstance(data_type, str):
        data_type = DataType.from_string(data_type)
    
    # Set endianness prefix
    endian = '>' if big_endian else '<'
    
    # Prepare data based on type
    if data_type == DataType.SCOMPLEX:
        output_data = np.empty(data.shape, dtype=[('real', f'{endian}i2'), ('imag', f'{endian}i2')])
        output_data['real'] = np.real(data).astype(np.int16)
        output_data['imag'] = np.imag(data).astype(np.int16)
    elif data_type == DataType.FCOMPLEX:
        output_data = np.empty(data.shape, dtype=[('real', f'{endian}f4'), ('imag', f'{endian}f4')])
        output_data['real'] = np.real(data).astype(np.float32)
        output_data['imag'] = np.imag(data).astype(np.float32)
    elif data_type == DataType.FLOAT:
        output_data = data.astype(f'{endian}f4')
    elif data_type == DataType.BYTE:
        # Ensure data is in valid byte range [0, 255]
        if data.min() < 0 or data.max() > 255:
            raise ValueError("Byte data must be in range [0, 255]")
        output_data = data.astype('u1')  # unsigned 1-byte integer
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    # Write to file
    with open(filename, 'wb') as f:
        output_data.tofile(f)

class SARParameters:
    def __init__(self):
        self.metadata: Dict[str, Union[str, float, int]] = {}
        self.state_vectors: Dict[str, np.ndarray] = {
            'position': None,  # Will be nx3 array for n state vectors
            'velocity': None   # Will be nx3 array for n state vectors
        }
        self.state_vector_times: np.ndarray = None  # Will be n-length array

def parse_sar_parameter_file(file_path: str) -> SARParameters:
    """
    Parse a SAR parameter file and extract key parameters.
    
    Args:
        file_path (str): Path to the parameter file
        
    Returns:
        SARParameters: Object containing parsed metadata and state vectors
    """
    params = SARParameters()
    
    # Temporary storage for state vectors
    positions = []
    velocities = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # First pass: get number of state vectors
    for line in lines:
        if 'number_of_state_vectors' in line:
            num_vectors = int(line.split(':')[1].strip())
            break
    
    # Initialize arrays
    positions = np.zeros((num_vectors, 3))
    velocities = np.zeros((num_vectors, 3))
    times = np.zeros(num_vectors)
    
    # Parse state vector information
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Get timing information
        if 'time_of_first_state_vector' in line:
            first_time = float(line.split(':')[1].strip().split()[0])
        elif 'state_vector_interval' in line:
            time_interval = float(line.split(':')[1].strip().split()[0])
            
        # Parse position vectors
        if 'state_vector_position' in line:
            parts = line.split(':')
            vector_num = int(parts[0].split('_')[-1]) - 1  # Convert to 0-based index
            values = [float(x) for x in parts[1].strip().split()[:3]]  # Get x, y, z
            positions[vector_num] = values
            
        # Parse velocity vectors
        elif 'state_vector_velocity' in line:
            parts = line.split(':')
            vector_num = int(parts[0].split('_')[-1]) - 1  # Convert to 0-based index
            values = [float(x) for x in parts[1].strip().split()[:3]]  # Get x, y, z
            velocities[vector_num] = values
            
        # Parse other metadata
        elif ':' in line:
            parts = line.split(':')
            key = parts[0].strip()
            value_part = parts[1].strip()
            
            # Extract the first value and unit if present
            value_parts = value_part.split()
            if not value_parts:
                continue
                
            try:
                # Try to convert to float if possible
                value = float(value_parts[0])
            except ValueError:
                # Keep as string if not a number
                value = value_parts[0]
                
            params.metadata[key] = value
    
    # Calculate times for each state vector
    times = np.array([first_time + i * time_interval for i in range(num_vectors)])
    
    # Store in params object
    params.state_vectors['position'] = positions
    params.state_vectors['velocity'] = velocities
    params.state_vector_times = times
    
    return params

def get_state_vectors(params: SARParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get state vectors and their corresponding times.
    
    Args:
        params (SARParameters): Parsed parameters
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - Times (nx1 array)
            - Positions (nx3 array)
            - Velocities (nx3 array)
    """
    return params.state_vector_times, params.state_vectors['position'], params.state_vectors['velocity']

def print_state_vectors(params: SARParameters):
    """
    Print state vectors in a readable format.
    
    Args:
        params (SARParameters): Parsed parameters
    """
    times, positions, velocities = get_state_vectors(params)
    
    print("\nState Vectors:")
    print("-------------")
    for i in range(len(times)):
        print(f"\nVector {i+1}:")
        print(f"Time: {times[i]:.3f} seconds")
        print(f"Position (m): {positions[i][0]:.4f}, {positions[i][1]:.4f}, {positions[i][2]:.4f}")
        print(f"Velocity (m/s): {velocities[i][0]:.4f}, {velocities[i][1]:.4f}, {velocities[i][2]:.4f}")

def format_sar_parameters(params: SARParameters) -> str:
    """
    Format the parsed SAR parameters into a readable string.
    
    Args:
        params (SARParameters): Parsed parameters
        
    Returns:
        str: Formatted string representation
    """
    output = []
    
    # Format metadata
    output.append("=== SAR Parameters ===")
    for key, value in params.metadata.items():
        output.append(f"{key}: {value}")
    
    # Format state vectors
    output.append("\n=== State Vectors ===")
    for i, sv in enumerate(params.state_vectors, 1):
        output.append(f"\nState Vector {i}:")
        if 'position' in sv:
            output.append(f"  Position (m): {sv['position']}")
        if 'velocity' in sv:
            output.append(f"  Velocity (m/s): {sv['velocity']}")
    
    return "\n".join(output)
