# Some common imports
import pandas, datetime, calendar, os, psutil

# Function to digest SDs and VDATA from HDF files and return xarray datasets.
# Attempts to be smart about missing data and attributes and such
# but HDF is an abomination, and seems to be difficult to get
# information about coordinate variables and such, especially for
# things like Cloudsat data, where some data is stored in VDATA and
# other data stored in SDs. This probably needs some work...
from pyhdf.SD import SD, SDC
from pyhdf.HDF import HDF
import pyhdf.VS
def open_dataset_hdf(filename, variables=None, drop_variables=[]):
    da_dict = {}
    
    # First read SD (scientific datasets)
    sd = SD(filename)
    if variables is None: 
        data_vars = sd.datasets().keys()
    else:
        data_vars = variables
    for dname in data_vars:
        
        if dname in drop_variables: continue
        if dname not in sd.datasets().keys(): continue
        
        sds = sd.select(dname)

        # get (masked) data
        d = np.where(sds[:] != sds.getfillvalue(), sds[:], np.nan)

        # check for more masks
        if 'missing' in sds.attributes():
            d[d == sds.missing] = np.nan

        # unpack data
        if 'offset' in sds.attributes() and 'factor' in sds.attributes():
            d = d / sds.factor + sds.offset

        # coordinate variables...how to do this?! Look for VDATA?
        # just save as DataArray for now, without coordinate variables...
        dims = [sds.dim(i).info()[0] for i in range(len(sds.dimensions()))]

        da_dict[dname] = xr.DataArray(d, dims=dims, attrs=sds.attributes(), name=dname)

        # Close this dataset
        sds.endaccess()

    # Close file
    sd.end()
    
    # ...now read VDATA...
    hdf = HDF(filename)
    vs = hdf.vstart()
    if variables is None: 
        data_vars, *__ = zip(*vs.vdatainfo())
    else:
        data_vars = variables
    for vname in data_vars:
        
        if vname in drop_variables: continue
        if vname not in [v[0] for v in vs.vdatainfo()]: continue
            
        # attach vdata
        vd = vs.attach(vname)

        # get vdata info
        nrec, mode, fields, *__ = vd.inquire()
        if nrec == 0:
            vd.detach()
            continue

        # read data
        d = np.array(vd[:]).squeeze()

        # make sure not to overwrite coordinate variables
        if all([vname not in da.dims for v, da in da_dict.items()]):
            da_dict[vname] = xr.DataArray(d)

        vd.detach()

    # clean up
    vs.end()

    # HDF files do not always close cleanly, so close manually
    hdf.close()
    
    return xr.Dataset(da_dict)
    
        
def get_geoprof_time(ds, return_datetime=False):
    """
    Read time stamp and return seconds since Epoch
    """

    # read data
    profile_sec = ds['Profile_time']
    start_tai = ds['TAI_start']

    # TAI time for each profile
    time_tai = (profile_sec + start_tai)  # seconds since 00:00:00 Jan 1

    # get epoch for TAI start of 00:00:00 Jan 1
    tai_base = calendar.timegm((1993, 1, 1, 0, 0, 0))

    # get epoch time as a datetime
    epoch = time_tai + tai_base
    
    # get array of datetime objects
    dt = np.array([datetime.datetime.utcfromtimestamp(t) for t in epoch.values])
    
    # return xarray DataArray
    if return_datetime:
        return dt
    else:
        return xr.DataArray(dt, dims=profile_sec.dims, coords=profile_sec.coords, name='time')
