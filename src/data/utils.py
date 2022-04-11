import numpy as np
import operator

def itemgettergeneral(keys,listdict, cast_to_np_array=False):
    """itemgettergeneral accept a single key or a iterable? and extract 
    from list of dict listdict.

    Due to numpy ufunc limitatation (Cannot construct a ufunc with more 
    than 32 operands) we switch from np,vectorize to list(map(x)).

    Parameters
    ----------

    keys     : single key or a iterable? (list only?)
    listdict : list of dict 
    cast_to_np_array : bool cast the output to np.array

    Returns
    -------

    numpy array : each dimension correspond to the key in input keys.

    """
    import numpy as np
    import operator


    if isinstance(keys,list):
        outdata = list(map(operator.itemgetter(*keys),listdict))
    else:
        outdata = list(map(operator.itemgetter(keys),listdict))

    if cast_to_np_array:
        return np.array(outdata)
    else:
        return outdata

class Wavelenght:
    """Class to handle wavelenght structure, with useful self.nearest_index method
    """
    def __init__(self,name, start_wav,end_wav,resolution):
        self.name = name
        self.wavelenghts = np.arange(start_wav,end_wav,resolution)
        self.start_wav= start_wav
        self.end_wav=end_wav
        self.resolution = resolution
    def __str__(self):
        return f'Wavelenght(name="{self.name}",start={self.start_wav},end={self.end_wav},res={self.resolution})'
    def nearest_index(self,value,verbose=False):
        if (value < self.start_wav) or (value > self.end_wav):
            raise ValueError(f'{value} is outside Wavelenght range [{self.start_wav},{self.end_wav}]')

        closest=(np.abs(self.wavelenghts - value)).argmin()
        
        if verbose:
            print(f'closest point to Wavelenght(start={self.start_wav},end={self.end_wav},res={self.resolution})')
        return closest

    
# dictextractor = lambda keys,dic : dict(zip(keys,itemgetter(*keys)(dic)))
#%timeit 29.7 µs ± 8.93 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

dictextractor = lambda keys,dic : {k:dic[k] for k in keys}
#%timeit 29.2 µs ± 6.36 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

# python defintion of DATA_QUALITY_INDEX field
DATA_QUALITY_INDEX = [
    ('Dark Scan',{
    '0' : 'shutter not engaged',
    '1:' : 'shutter engaged'}),
    ('Temperature 1',{
    '0' : 'Temperature does not exceed 15 deg C threshold.',
    '1' : 'Temperature exceeds 15 deg C threshold but less than 25 deg C threshold.',
    '2' : 'Temperature exceeds 25 deg C threshold but less than 40 deg C threshold.',
    '3' : 'Temperature exceeds 40 deg C threshold.'}),
    ('Temperature 2',{
    '0' : 'Temperature does not exceed 15 deg C threshold.',
    '1' : 'Temperature exceeds 15 deg C threshold but less than 25 deg C threshold.',
    '2' : 'Temperature exceeds 25 deg C threshold but less than 40 deg C threshold.',
    '3' : 'Temperature exceeds 40 deg C threshold.'}),
    ('Grating Temperature',{
    '0' : 'Temperature does not exceed 15 deg C threshold.',
    '1' : 'Temperature exceeds 15 deg C threshold but less than 25 deg C threshold.',
    '2' : 'Temperature exceeds 25 deg C threshold but less than 40 deg C threshold.',
    '3' : 'Temperature exceeds 40 deg C threshold.'}),
    ('Anomalous Pixels',{
    '0' : 0,
    '1' : 1,
    '2' : 2,
    '3' : 3,
    '4' : 4,
    '5' : 5,
    '6' : 6,
    '7' : 7,
    '8' : 8,
    '9' : 9}),
    ('Partial Data',{
    '0' : 'No partial data.',
    '1' : 'Partial data exists.'}),
    ('Saturation',{
    '0' : 'No pixels saturated.',
    '1' : 'Saturated pixels exist.'}),
    ('Low Signal Level',{
    '0' : 'Signal level not below -32768 threshold.',
    '1' : 'Signal level below -32768 threshold.'}),
    ('Low VIS Wavelength Uncertainty',{
    '0' : 'Uncertainty not above TBD threshold at low wavelengths.',
    '1' : 'Uncertainty above TBD threshold at low wavelengths.'}),
    ('High VIS Wavelength Uncertainty',{
        '0' : 'Uncertainty not above TBD threshold at high wavelengths.',
        '1' : 'Uncertainty above TBD threshold at high wavelengths.'}),
    ('UVVS Operating',{
        '0' : 'UVVS is not scanning during readout.',
        '1' : 'UVVS is scanning during readout.'}),
    ('UVVS Noise Spike',{
        '0' : 'No noise spike detected.',
        '1' : 'Noise spike detected.'}),
    ('SPICE Version Epoch',{
        '0' : 'No SPICE',
        '1' : 'Predict',
        '2' : 'Actual'}),
    ('Dark Saturation',{
        '0' : 'All pixels in data record contain at least four unsaturated dark frames.',
        '1' : 'One or more pixels in data record contain three or fewer unsaturated dark frames.'}),
    ('SpareO',{ '0' : 'None'}),
    ('SpareP',{ '0' : 'None'})
]


def extract_data_quality_index(dqi):
    extract = lambda d: [i for i in d if i != '-']
    return dict(zip(
            [k.replace(" ", "_") for k,v in DATA_QUALITY_INDEX],
             [int(k) for k in extract(dqi['DATA_QUALITY_INDEX'])]
             )
        )

def np_to_sql_arr(x):
    """return a tuple from iterable input x, with el >= 1.0e32 set as None"""
    return tuple(el.astype(float) if not el >= 1.0e32 else None for el in x)

# add to the table_data
def extract_regridded_sp(sp,wav_grid):
    import scipy.interpolate
    
    wav = sp['CHANNEL_WAVELENGTHS']
    iof_sp = sp['IOF_SPECTRUM_DATA']
    photom_iof_sp = sp['PHOTOM_IOF_SPECTRUM_DATA']
    
    interpolator = scipy.interpolate.interp1d(wav, photom_iof_sp, kind='linear', fill_value='extrapolate')
    photom_iof_sp_2nm = interpolator(wav_grid)
    interpolator = scipy.interpolate.interp1d(wav, iof_sp, kind='linear', fill_value='extrapolate')
    iof_sp_2nm = interpolator(wav_grid)
    return {'photom_iof_sp_2nm': np_to_sql_arr(photom_iof_sp_2nm),
            'iof_sp_2nm' : np_to_sql_arr(iof_sp_2nm)}

def fov_coord_extractor(sp,
                        shapelyze=True,
                        geoalchemy2ize=True,
                        srid=4326,
                        coordinates_names = None):
    if not coordinates_names:
        coordinates_names = ['TARGET_LONGITUDE_SET','TARGET_LATITUDE_SET']

    coord = np.array([l for l in  operator.itemgetter(*coordinates_names)(sp)],dtype=np.float32)
    
    # If there is a NaN in the coord, return none. 
    # It is not possible to use it!!
    if ( coord >= 1.0e+32).any() or  np.isnan(coord).any() : 
        return None
    else:
        coord_tuples = np.array([(i,j) for i,j in zip(*coord)])
        coord_dict = {'center': coord_tuples[0]}
        coord_dict['fov'] = coord_tuples[[1,3,2,4,1],:]
        if shapelyze:
            import shapely.geometry
            coord_dict['center'] = 'SRID={};{}'.format(srid,shapely.geometry.Point(*coord_dict['center']).wkt)
            coord_dict['fov'] = 'SRID={};{}'.format(srid,shapely.geometry.Polygon(coord_dict['fov']).wkt)
        if geoalchemy2ize:
            from geoalchemy2 import shape
            coord_dict['center'] = shape.from_shape(coord_dict['center'],srid=srid)
            coord_dict['fov'] = shape.from_shape(coord_dict['fov'],srid=srid)            
        return coord_dict