# python_dewesoft
A module for reading DEWESoft files and an example program

## Motivation

DEWESoft, https://dewesoft.com/, develops and manufactures versatile and easy-to-use data acquisition systems - the ultimate tools for test and measurement engineers. They provide a Python module for reading their binary files DXD/D7D and an example program, available at https://download.dewesoft.com/.

However, the programs are rather rudimentary and, as a part of data processing, a higher-level interface to the DWS data files was needed. This is the result.

## The module

The main function, `read_dws(filename, fields=None, rename=None, mixed_sample_rates=False, dll=None)` in the `DWDataReader` module, reads a DWS file and returns either a list of fields and other information about the file or data from the file in the form of a Pandas DataFrame. This is the help for the function:

```
    Reads fields from a DEWESoft file and returns a pandas DataFrame
    
    Args:
        - filename: input filename
        
        - fields: A list of fields to extract. If not specified, information
            about the file will be returned. Empty list means read all fields.
            Note that selection is performed before renaming, so names refer
            to original field names from the DEWESoft file.
            
        - rename: A dict used to rename fields in the resulting DataFrame. If
            the key is an integer, it is assumed that this is the zero-based
            index of the field to rename. Otherwise it is assumed to be the
            original field name.
        
        - mixed_sample_rate: If false, only fields with identical sample rates
            will be allowed. The sampling rate of the returned data will be the
            sample rate of the selected fields. E.g., if the file contains
            fields sampled at 100Hz, 20Hz and 10Hz, and you select fields with
            20Hz, this will be the sampling rate of the returned data.
            
            If true, mixed sample rates will be allowed and the missing values
            will be filled with NaN-s. The sample rate in this case will be the
            maximum of all fields (e.g., 100Hz in the previous example).
            
            Note that only integer sample rate ratios are handled.
            
        - dll: optional handle to dll, obtained with open_dll(). Use when
            calling this function many times, to prevent opening/closing the
            DLL.
            
    Returns:
        If there is no 'fields' input, returns a dict with file info:
            - sampling_rate (integer)
            - start_store_time (datetime)
            - duration (seconds)
            - number_of_channels (int)
            - channel_info: list of tuples, for each channel:
                - channel index (int), zero-based
                - channel name (str)
                - renamed channel name (str), possibly the same as channel name
                - unit (str)
                - number of samples (int)
                - sampling rate ratio divisor (int), 1 for full sampling rate

        If there is 'fields' input:        
            - data: pandas DataFrame
            
    Notes:
        - This function cannot handle array_size > 1.
        - When handling large ammounts of data, Python's garbage coolection
            does not always function. If you get memory errors, `import gc` and
            then periodically call `cg.collect()`.
```

## Usage example

An example of the use of the module is in the supplied script `dws2hdf5.py`, that reads DWS files and converts them to datasets in on or more HDF5 files. This is the help for the script:

```
usage: dws2hdf5.py [-h] [--dst DST] [--new] [--list] [--noprogress]
                   [--fields {0,1,2}] [--rename FILENAME:FROM:TO[:FROM:TO ...]
                   [FILENAME:FROM:TO[:FROM:TO ...] ...]] [--mixedsamplerates]
                   src [src ...]

Reads DXD/D7D (DWS) files and saves results to HDF5 file

positional arguments:
  src                   Source file(s). Can include wildcards

optional arguments:
  -h, --help            show this help message and exit
  --dst DST             Destination file. If unspecified, one HDF5 file per
                        one DWS file will be produced (default: None)
  --new                 Only processes new files, checks existence of
                        destination file and dataset within the file (default:
                        False)
  --list                List file info (default: False)
  --noprogress          Do not output progress (default: False)
  --fields {0,1,2}      Select fields in DWS based on names in DWS files. 0:
                        all fields; 1: 'ACC and not Freq' + 'T_'; 2: 'AI'
                        (default: None)
  --rename FILENAME:FROM:TO[:FROM:TO ...] [FILENAME:FROM:TO[:FROM:TO ...] ...]
                        Rename fields when reading files. FILENAME is a
                        regular expression. If FROM is an integer, it is
                        assumed to be zero-based column number, else it is
                        assumed to be field name (default: None)
  --mixedsamplerates    Allow mixed sample rates (default: False)
```  
