#----------------------------------------------------------------------------------------------------------------
# DWDataReader "header" for Python and module for reading
#----------------------------------------------------------------------------------------------------------------
# Author: Dewesoft,
#         Jan Kalin <jan.kalin@zag.si>
#         Uros Bohinc <uros.bohinc@zag.si>
# Notes:
#   - requires DWDataReaderLib.dll 4.0.0.0 or later
#   - tested with Python 3.4 and Python 2.7.15
#   - additionally implemented reading of events data
#----------------------------------------------------------------------------------------------------------------

import collections
import _ctypes
from ctypes import Structure, c_char, c_int, c_uint, c_ulong, cdll, c_char_p, c_double, create_string_buffer, cast, byref, POINTER, c_int64
import datetime
from dateutil import tz
from enum import Enum
import platform
import re
import os
import subprocess
import sys

import numpy as np
import pandas as pd

###########################################################################
# Can only run on Windows
###########################################################################

###########################################################################
# Constants
###########################################################################

INT_SIZE = 4 # size of integer
DOUBLE_SIZE = 8 # size of double

###########################################################################
# Enums and Structures for access to DEWESoft files
###########################################################################


class DWStatus(Enum):
    DWSTAT_OK = 0
    DWSTAT_ERROR = 1
    DWSTAT_ERROR_FILE_CANNOT_OPEN = 2
    DWSTAT_ERROR_FILE_ALREADY_IN_USE = 3
    DWSTAT_ERROR_FILE_CORRUPT = 4
    DWSTAT_ERROR_NO_MEMORY_ALLOC = 5
    DWSTAT_ERROR_CREATE_DEST_FILE = 6
    DWSTAT_ERROR_EXTRACTING_FILE = 7
    DWSTAT_ERROR_CANNOT_OPEN_EXTRACTED_FILE = 8

class DWChannelProps(Enum):
	DW_DATA_TYPE = 0
	DW_DATA_TYPE_LEN_BYTES = 1
	DW_CH_INDEX = 2
	DW_CH_INDEX_LEN = 3
	DW_CH_TYPE = 4
	DW_CH_SCALE = 5
	DW_CH_OFFSET = 6
	DW_CH_XML = 7
	DW_CH_XML_LEN = 8
	DW_CH_XMLPROPS = 9
	DW_CH_XMLPROPS_LEN = 10

class DWChannelType(Enum):
	DW_CH_TYPE_SYNC = 0 # sync
	DW_CH_TYPE_ASYNC = 1 # async
	DW_CH_TYPE_SV = 2 # single value
	
class DWFileInfo(Structure):
    _fields_ =\
    [
        ("sample_rate", c_double),
        ("start_store_time", c_double),
        ("duration", c_double)
    ]
	
class DWChannel(Structure):
    _fields_ =\
    [
        ("index", c_int),
        ("name", c_char * 100),
        ("unit", c_char * 20),
        ("description", c_char * 200),
        ("color", c_uint),
        ("array_size", c_int),
        ("data_type", c_int)
    ]

class DWEvent(Structure):
    _pack_ = 1
    _fields_ = [("event_type", c_int),
                ("time_stamp", c_double),
                ("_event_text", c_char * 200)]

    @property
    def event_text(self):
        """Readable description of the event"""
        return self._event_text.decode(encoding='ISO-8859-1')

    def __str__(self):
        return "{0.time_stamp} {0.event_text}".format(self)


class DWReducedValue(Structure):
	_fields_ =\
	[
		("time_stamp", c_double),
        ("ave", c_double),
        ("min", c_double),
        ("max", c_double),
        ("rms", c_double)
	]
	
class DWArrayInfo(Structure):
	_fields_ =\
	[
        ("index", c_int),
        ("name", c_char * 100),
        ("unit", c_char * 20),
        ("size", c_int)
	]
	
class DWCANPortData(Structure):
	_fields_ =\
	[
        ("arb_id", c_ulong),
        ("data", c_char * 8)
	]
	
class DWComplex(Structure):
	_fields_ =\
	[
        ("re", c_double),
        ("im", c_double)
	]
	
class DWEventType(Enum):
	etStart = 1
	etStop = 2
	etTrigger = 3
	etVStart = 11
	etVStop = 12
	etKeyboard = 20
	etNotice = 21
	etVoice = 22
	etModule = 24	
	
class DWStoreType(Enum):
	ST_ALWAYS_FAST = 0
	ST_ALWAYS_SLOW = 1
	ST_FAST_ON_TRIGGER = 2
	ST_FAST_ON_TRIGGER_SLOW_OTH = 3
	
class DWDataType(Enum):
	dtByte = 0
	dtShortInt = 1
	dtSmallInt = 2
	dtWord = 3
	dtInteger = 4
	dtSingle = 5
	dtInt64 = 6
	dtDouble = 7
	dtLongword = 8
	dtComplexSingle = 9
	dtComplexDouble = 10
	dtText = 11
	dtBinary = 12
	dtCANPortData = 13

def DWRaiseError(err_str):
    print(err_str)
    sys.exit(-1)

###########################################################################
# Reader stuff
###########################################################################

def glibc_version():
    lines = subprocess.check_output(["ldd", "--version"]).split("\n")
    for line in lines:
        match = re.search("(\d+\.\d+)$", line)
        if match:
            return float(match.group(1))
    raise RuntimeError("Cannot determine GLIBC version with 'ldd --version'")
    

def open_dll(libname = None):
    """Opens dewesoft dll.
    
    Result can be used in read_dws(). Call close_dll() when done, best with try: finally: blocks
    """

    if not libname:
        libname = os.path.dirname(os.path.abspath(__file__)) + '/'
        if platform.architecture()[0] == '32bit':
            libname += 'DWDataReaderLib'
        else:
            libname += 'DWDataReaderLib64'
        if platform.system() == 'Windows':
            libname += ".dll"
        else:
            if glibc_version() < 2.14:
                libname += "-4.2.0.4"
            libname += ".so"

    dll = cdll.LoadLibrary(libname)
    result = dll.DWInit()
    if result != DWStatus.DWSTAT_OK.value:
        raise RuntimeError("DWInit() failed: {}".format(result))
    return dll 


def close_dll(dll):
    """Closes dewesoft dll"""

    result = dll.DWDeInit()
    if result != DWStatus.DWSTAT_OK.value:
        raise RuntimeError("DWDeInit() failed: {}".format(result))
    if platform.system() == 'Windows':
        _ctypes.FreeLibrary(dll._handle)
    else:
        _ctypes.dlclose(dll._handle)


def read_dws(filename, fields=None, rename=None, scale=None, mixed_sample_rates=False, downsample=None, dll=None):
    """Reads fields from a DEWESoft file and returns a pandas DataFrame
    
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
            
        - scale: A dict used to scale data.  If the key is the renamed field
            name. The value is a tuple and the data is scaled using formula
            value -> value*tuple[0] + tuple[1]
        
        - mixed_sample_rate: If false, only fields with identical sample rates
            will be allowed. The sampling rate of the returned data will be the
            sample rate of the selected fields. E.g., if the file contains
            fields sampled at 100Hz, 20Hz and 10Hz, and you select fields with
            20Hz, this will be the sampling rate of the returned data.
            
            If true, mixed sample rates will be allowed and the missing values
            will be filled with NaN-s. The sample rate in this case will be the
            maximum of all fields (e.g., 100Hz in the previous example).
            
            Note that only integer sample rate ratios are allowed.
            
        - downsample: If defined, the data will be downsampled with this ratio
            with the use of np.mean()
            
        - dll: Optional handle to dll, obtained with open_dll(). Use when
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
        
    """

    if dll:
        mydll = dll
    else:
        mydll = open_dll()

    fileopened = False
    try:
        # add additional data reader
        if mydll.DWAddReader() != DWStatus.DWSTAT_OK.value:
            raise RuntimeError("DWAddReader() failed")
        
        # get number of open data readers
        num = c_int()
        if mydll.DWGetNumReaders(byref(num)) != DWStatus.DWSTAT_OK.value:
           raise RuntimeError("DWGetNumReaders() failed")
        
        # open data file
        file_name = c_char_p(filename.encode())
        file_info = DWFileInfo(0, 0, 0)
        if mydll.DWOpenDataFile(file_name, byref(file_info)) != DWStatus.DWSTAT_OK.value:
            raise RuntimeError("DWOpenDataFile() failed")
        fileopened = True

        # get events
        time_stamp = []
        event_type = []
        event_text = []
        nEvents = mydll.DWGetEventListCount()
        if nEvents:
            events_ = (DWEvent * nEvents)()
            stat = mydll.DWGetEventList(events_)
            if stat:
                raise DWError(stat)
            for e in events_:
                time_stamp.append(e.time_stamp)
                event_type.append(e.event_type)
                event_text.append(e.event_text)		
                
        events=pd.DataFrame(
            data = {'type': event_type, 'text': event_text},
            index = time_stamp)						

        # Check right now for downsampling parameter errors        
        if downsample and downsample != int(downsample):
            raise ValueError("Invalid downsample ratio {}")
                
        # Get start store time and round it to milliseconds
        sst = (datetime.datetime(1899, 12, 30) + datetime.timedelta(days=file_info.start_store_time)).replace(tzinfo=tz.tzutc()).astimezone(tz.tzlocal())
					        
        # get num channels
        num = mydll.DWGetChannelListCount()
        if num == -1:
            raise RuntimeError("DWGetChannelListCount() failed")
        
        # get channel list
        ch_list = (DWChannel * num)()
        if mydll.DWGetChannelList(byref(ch_list)) != DWStatus.DWSTAT_OK.value:
            raise RuntimeError("DWGetChannelList() failed")				
            
        # Get channel info
        sample_cnts = []
        for idx in range(num):
            dw_ch_index = c_int(ch_list[idx].index)
            sample_cnt = c_int()
            try:
                sample_cnt = mydll.DWGetScaledSamplesCount(dw_ch_index)
            except:
                raise RuntimeError("DWGetScaledSamplesCount() failed with an Access Violation")
            if sample_cnt < 0:
                raise RuntimeError("DWGetScaledSamplesCount() failed")
            else:
                sample_cnts.append(sample_cnt)
            if not idx:
                sample_cnt = 1
                dewesoftdata = create_string_buffer(DOUBLE_SIZE * sample_cnt * ch_list[idx].array_size)
                time_stamp = create_string_buffer(DOUBLE_SIZE * sample_cnt)
                p_data = cast(dewesoftdata, POINTER(c_double))
                p_time_stamp = cast(time_stamp, POINTER(c_double))
                if mydll.DWGetScaledSamples(dw_ch_index, c_int64(0), sample_cnt, p_data, p_time_stamp) != DWStatus.DWSTAT_OK.value:
                    raise RuntimeError("DWGetScaledSamples() failed")
                sat = sst - datetime.timedelta(seconds=p_time_stamp[0])

        
        # Calculate sample ratio
        max_sample_cnt = max(sample_cnts)
        srdiv = [max_sample_cnt/x if x and not max_sample_cnt % x else np.nan for x in sample_cnts]

        # Perhaps rename
        def field_name(idx):
            try:
                return rename[str(idx)].decode('UTF-8')
            except:
                try:
                    return rename[ch_list[idx].name.decode('UTF-8')]
                except:
                    return ch_list[idx].name.decode('UTF-8')
            
        # Perhaps just return 
        if fields == None:
            return {'sample_rate': file_info.sample_rate,
                    'start_store_time': sst,
                    'start_acquisition_time': sat,
                    'duration': file_info.duration,
                    'number_of_channels': num,
					'events': events,
                    'channels': [(x, ch_list[x].name.decode('UTF-8'), field_name(x), ch_list[x].unit.decode('UTF-8'), sample_cnts[x], srdiv[x], ch_list[x].array_size) for x in range(num)]}

        # Get columns and check for duplicate names
        columns = [x for x in range(num) if (not len(fields) or ch_list[x].name.decode('UTF-8') in fields) and not np.isnan(srdiv[x])]
        multiple = [item for item, count in list(collections.Counter([field_name(x) for x in columns]).items()) if count > 1]
        if len(multiple):
            raise RuntimeError("Multiple occurrences of field name(s): {}".format(", ".join(multiple)))
            
        # Allocate data for the largest block of data
        max_idx = np.argmax(sample_cnts)
        max_sample_cnt = sample_cnts[max_idx]
        dewesoftdata = create_string_buffer(DOUBLE_SIZE * max_sample_cnt * ch_list[max_idx].array_size)
        time_stamp = create_string_buffer(DOUBLE_SIZE * max_sample_cnt)
        p_data = cast(dewesoftdata, POINTER(c_double))
        p_time_stamp = cast(time_stamp, POINTER(c_double))
        
        # If we're allowing mixed sample rates, we need to find the 'fastest' timestamps and we need to have at least one field with full sample rate
        if mixed_sample_rates:
            dw_ch_index = c_int(ch_list[max_idx].index)
            if mydll.DWGetScaledSamples(dw_ch_index, c_int64(0), max_sample_cnt, p_data, p_time_stamp) != DWStatus.DWSTAT_OK.value:
                raise RuntimeError("DWGetScaledSamples() failed")
            tss = np.full((max_sample_cnt,), np.datetime64(sst.strftime("%Y-%m-%d %H:%M:%S.%f")), dtype='datetime64[us]')
            dts = (np.array(p_time_stamp[:max_sample_cnt])*1e6).astype('timedelta64[us]')
            tss += dts
            data = pd.DataFrame(columns=[field_name(x) for x in columns], index=tss, dtype=float)
        else:
            data = None

        # Perhaps scale
        def scale_data(field_name, data):
            try:
                scale_factor = scale[field_name]
                return np.array(data)*scale_factor[0] + scale_factor[1]
            except:
                return data

        # channel loop
        fieldcount = 0
        for i in columns:
            # Cannot process this
            if ch_list[i].array_size > 1:
                raise RuntimeError("Cannot read data with array_size > 1")
                
            # number of samples
            dw_ch_index = c_int(ch_list[i].index)
            sample_cnt = c_int()
            sample_cnt = mydll.DWGetScaledSamplesCount(dw_ch_index)
            if sample_cnt < 0:
                raise RuntimeError("DWGetScaledSamplesCount() failed")
        
            # get actual data
            if mydll.DWGetScaledSamples(dw_ch_index, c_int64(0), sample_cnt, p_data, p_time_stamp) != DWStatus.DWSTAT_OK.value:
                raise RuntimeError("DWGetScaledSamples() failed")

            # Copy data to array
            try:
                data.empty
            except:
                tss = np.full((sample_cnt,), np.datetime64(sat.strftime("%Y-%m-%d %H:%M:%S.%f")), dtype='datetime64[us]')
                dts = (np.array(p_time_stamp[:sample_cnt])*1e6).astype('timedelta64[us]')
                tss += dts
                data = pd.DataFrame(columns=[field_name(x) for x in columns], index=tss, dtype=float)
            if not mixed_sample_rates and len(data) != sample_cnt:
                raise RuntimeError("Mismached number of samples between two selected fields")
            data.iloc[::srdiv[i] if mixed_sample_rates else 1, data.columns.get_loc(field_name(i))] = scale_data(field_name(i), [p_data[x] for x in range(sample_cnt)])
            fieldcount += 1
        
        # Length and srdiv check
        if len(fields) and fieldcount != len(fields):
            raise RuntimeError("Not all fields read")
        
        # Remove buffers
        del dewesoftdata
        del time_stamp
        
        # And done. Perhaps downsample
        if downsample:
            if len(data) < downsample:
                raise RuntimeError("Cannot downsample less than {} samples".format(downsample))
            else:
                rule = "{}us".format(downsample*(dts[1]-dts[0])/np.timedelta64(1,'us'))
                return data.resample(rule).mean()
        return data

    finally:        

        # close data file
        if fileopened and mydll.DWCloseDataFile() != DWStatus.DWSTAT_OK.value:
            raise RuntimeError("DWCloseDataFile() failed")
        
        # Close dll if we have opened it
        if not dll:
            close_dll(mydll)
