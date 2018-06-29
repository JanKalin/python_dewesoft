"""restore_net_use.py: Reads DEWESoft files as Pandas DataFrame and writes
them to HDF5
"""

__author__ = "Jan Kalin"
__credits__ = ["Jan Kalin"]
__license__ = "MIT"
__maintainer__ = "Jan Kalin"
__email__ = "jan.kalin@zag.si"
__status__ = "Development"

###########################################################################

import argparse
import datetime
import gc
import glob as globglob
import h5py
import math
import os
import re
import warnings
import sys

from dewesoft import DWDataReader

#%%

parser = argparse.ArgumentParser(description="Reads DXD/D7D (DWS) files and saves results to HDF5 file", fromfile_prefix_chars='@', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("src", help="Source file(s). Can include wildcards", nargs='+')
parser.add_argument("--dst", help="Destination file. If unspecified, one HDF5 file per one DWS file will be produced")
parser.add_argument("--list", help="List file info", action='store_true')
parser.add_argument("--noprogress", help="Do not output progress", action='store_true')
parser.add_argument("--fields", help="Select fields in DWS based on names in DWS files. 0: all fields; 1: 'ACC and not Freq' + 'T_'; 2: 'AI'", type=int, choices=[0, 1, 2])
parser.add_argument("--rename", help="Rename fields when reading files. FILENAME is a regular expression. If FROM is an integer, it is assumed to be zero-based column number, else it is assumed to be field name", nargs='+', metavar="FILENAME:FROM:TO[:FROM:TO ...]")
parser.add_argument("--mixedsamplerates", help="Allow mixed sample rates", action='store_true')

args = parser.parse_args()
sys.stdout.flush()

#%%
filenames = sorted({item for sublist in [globglob.glob(x) for x in args.src] for item in sublist})
if not filenames:
    raise ValueError("No files selected")
loglen = int(math.ceil(math.log(len(filenames), 10)))
formatstr = "{{}} {{:0{0}}}/{{:0{0}}} {{}},".format(loglen)

for filename in filenames:
    if os.path.splitext(filename)[1].lower() not in ['.dxd', '.d7d']:
        warnings.warn("Possible non-DWS file specified. Perhaps missing argument --dst?")

if args.dst:
    try:
        with h5py.File(args.dst, 'r') as hf:
            datasets = hf.keys()
    except:
        datasets = []

#%%
rename = []
if args.rename:
    for item in [x.split(':') for x in args.rename]:
        rename.append((item[0], {x:y for (x,y) in zip(item[1::2], item[2::2])})) 

#%%
dll = DWDataReader.open_dll()
try:
    for idx, src in enumerate(filenames, 1):
        if not args.noprogress:
            print formatstr.format(datetime.datetime.now(), idx, len(filenames), src), 
        try:
            localrename = {}
            for key, value in rename:
                if re.search(key, os.path.basename(src)):
                    localrename.update(value)
            file_info = DWDataReader.read_dws(src, rename=localrename, dll=dll)
            if args.list:
                print file_info
                continue
            
            allfields = [x[1] for x in file_info['channels']]
            if args.fields == 0:
                qty_myfields = [('', [])]
            elif args.fields == 1:
                qty_myfields = zip(['a', 'T'], [[x for x in allfields if x[:3] == 'ACC' and x[-4:] != 'Freq'], [x for x in allfields if x[:2] == "T_"]])
            elif args.fields == 2:
                qty_myfields = zip(['a'], [[x for x in allfields if x[:2] == 'AI']])
            else:
                raise ValueError("Invalid or missing argument for option --fields")
                
            datasetroot = file_info['ts0'].strftime("%Y_%m_%d_%H_%M_%S_%f")
            if args.dst:
                filename = args.dst
            else:
                filename = "{}.hdf5".format(datasetroot)
                try:
                    with h5py.File(filename, 'r') as hf:
                        datasets = hf.keys()
                except:
                    datasets = []

            for qty, myfields in qty_myfields:
                dataset = "{}_{}".format(qty, datasetroot)
                if dataset in datasets:
                    if not args.noprogress:
                        print "{} exists,".format(dataset),
                    continue
                data = DWDataReader.read_dws(src, myfields, rename=localrename, mixed_sample_rates=args.mixedsamplerates, dll=dll)
                if not len(data.index):
                    continue
                data.to_hdf(filename, dataset, complevel=5, complib='zlib')
                if not args.noprogress:
                    print "{},".format(dataset),
                del data
            gc.collect(), 
            if not args.noprogress:
                print "done"
        except RuntimeError as e:
            print "{} problems: {}".format(src, e.message)
finally:
    if dll:
        DWDataReader.close_dll(dll)

