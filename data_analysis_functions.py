import toolbox_scs as tb
import extra_data as ed
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import BaselineRemoval as br
from multiprocessing import Pool
import pkg_resources


################################################################################
######################## Camera settings, Viking calibration ###################
################################################################################

def get_camera_gain(run):
    ''' Get gain of the camera in the Viking spectrometer for a specified run.
        Inputs:
        ------
        run: extra_data DataCollection
            information on the run
            
        Outputs:
        ------
        int
            gain
    '''
    sel = run.select_trains([0])
    gain = sel.get_array('SCS_EXP_NEWTON/CAM/CAMERA', 'preampGain.value').item()
    gain_dict = {0: 1, 1: 2, 2: 4}
    return gain_dict[gain]


def photoelectronsPerCount(gain, mode='HS'):
    ''' Conversion factor from camera gain to photoelectrons
        per count. The values can be found
        in the camera datasheet but they have been slightly corrected
        for High Sensitivity mode after analysis of runs 1204, 1207 and 1208.

        Inputs:
        ------
        gain: int
            camera gain; allowed values are 1, 2, or 4
        mode: string
            High sensitivity 'HS' or high capacity 'HC'

        Outputs:
        ------
        float:
            photoelectrons per count
    '''
    if mode=='HS':
        pe_dict = {1: 4., 2: 2.05, 4: 0.97}
    else:
        pe_dict = {1: 17.9, 2: 9., 4: 4.5}
    return pe_dict[gain]


def get_photoelectronsPerCount(run, gain):
    sel = run.select_trains([0])
    hc = sel.get_array('SCS_EXP_NEWTON/CAM/CAMERA',
                       'HighCapacity.value').item()
    mode = 'HS' if hc == 0 else 'HC'
    return photoelectronsPerCount(gain, mode)
    

def calibrate_viking(x, runNB):
    """
    Calibration of Viking spectrometer using 2nd order polynomial,
    based on analysis of:
    - WK 41: runs 118-124 (proposal 2593), valid for runs 1 - 851
    - WK 41: run 1716 (proposal 2937), valid for runs 1299 - 1718
    - WK39: runs 894 - 902, valid for runs ***
    """
    if runNB <= 851: # Proposal 2593
        coeffs = [1.20078832e-05, 8.06816040e-02, 8.63353911e+02]
    elif runNB >= 1289: # Proposal 2937
        coeffs = [1.21597814e-05, 8.03133850e-02, 8.63770237e+02]
    elif runNB >=874 and runNB <= 1288: # Proposal 2937
        coeffs = [1.17677273e-05, 7.90306688e-02, 8.52688452e+02]
    return np.poly1d(coeffs)(x)


################################################################################
############################## Filters transmission ############################
################################################################################

def filterTransmission(filterStr, energy=None):
    ''' Returns transmission of Al filters of specified thickness
        for optional photon energy range.
        Inputs:
        ------
        filterStr: string
            description of the filter with element name and thickness
            in microns; accepted values are 'Al3.5', 'Al5', 'Al10', 'Al15'
        energy: xarray or 1d ndarray
            energy axis for energy-dependent transmission
        
        Outputs
        -------
        transmission: float or xarray or 1d ndarray
            If energy is provided, 1d array with same length is returned.
    '''
    if energy is None:
        dict_Tr = {'Al3.5': 0.281, # measured, run 757, 764
                   'Al5': 0.14,
                   'Al10': 0.025, # measured, runs 167, 168
                   'Al13.5': 5.0e-3,
                   'Al15': 2.8e-3,
                  }
        if filterStr not in dict_Tr:
            return 1.
        return dict_Tr[filterStr]
    
    resource_package = __name__
    resource_path = '/Al_att_length.txt'
    fileName = pkg_resources.resource_stream(resource_package,
                                             resource_path)
    e_eV, attLength = np.loadtxt(fileName, skiprows=3, unpack=True)
    dict_length = {'NoFilter': 0.,
                   'Al3.5': 3.43, # measured, runs 757, 764
                   'Al5': 5.,
                   'Al10': 10.05, # measured, runs 167, 168
                   'Al13.5': 13.5,
                   'Al15': 15.,
                  }
    if filterStr in dict_length:
        length = dict_length[filterStr]
        tr = np.interp(energy, e_eV, np.exp(-length/attLength))
    else:
        print('Unknown filter - setting transmission to 1.')
        tr = np.ones(energy.shape)
    if isinstance(energy, xr.DataArray):
        tr = xr.DataArray(tr, dims=energy.dims, coords=energy.coords,
                          name='filterTr')

    return tr
        
        
################################################################################
############################## Baseline subtraction ############################
################################################################################

def removePolyBaseline(x, spectra, deg=8, signalRange=[910, 970]):
    """
    Removes a polynomial baseline to a signal, assuming a fixed
    position for the signal.
    Parameters
    ----------
    x: array-like, shape(M,)
        x-axis
    spectra: array-like, shape(M,) or (N, M,)
        the signals to subtract a baseline from. If 2d, the signals
        are assumed to be stacked on the first axis.
    deg: int
        the polynomial degree for fitting a baseline
    signalRange: list of type(x), length 2
        the x-interval where to expect the signal. The baseline is fitted to
        all regions except the one defined by the interval.
    Output
    ------
    spectra_nobl: array-like, shape(M,) or (N, M,)
        the baseline subtracted spectra
    
    """
    mask = (x<signalRange[0]) | (x>signalRange[1])
    if isinstance(x, xr.DataArray):
        x_bl = x.where(mask, drop=True)
        bl = spectra.sel(x=x_bl)
    else:
        x_bl = x[mask]
        if len(spectra.shape) == 1:
            bl = spectra[mask]
        else:
            bl = spectra[:, mask]
    fit = np.polyfit(x_bl, bl.T, deg)
    if len(spectra.shape) == 1:
        return spectra - np.poly1d(fit)(x)
    final_bl = np.empty(spectra.shape)
    for t in range(spectra.shape[0]):
        final_bl[t] = np.poly1d(fit[:, t])(x)
    return spectra - final_bl


def removeBaseline(spectra, degree=5, repitition=100, gradient=0.001):
    ''' Baseline removal using the BaselineRemoval package.
        Inputs:
        ------
        spectra: 1d or 2d ndarray or xarray
        If 2d, it is assumed that spectra are stacked along axis=0
        
        degree, repitition, gradient: int, int, float
            parameters for IModPoly baseline removal algorithm
            
        Output:
        -------
        spectra_corr: the baseline-subtracted spectra
    '''
    spectra_corr = np.empty((spectra.shape))
    if len(spectra.shape) == 2:
        for i in range(spectra.shape[0]):
            obj = br.BaselineRemoval(spectra[i])
            spectra_corr[i] = obj.IModPoly(degree, repitition, gradient)
    else:
        obj = br.BaselineRemoval(spectra)
        spectra_corr = obj.IModPoly(degree, repitition, gradient)
    if isinstance(spectra, xr.DataArray):
        spectra_corr = xr.DataArray(spectra_corr, dims=spectra.dims,
                                    coords=spectra.coords)
    return spectra_corr


def remove_sample_baseline(mdata, degree=5, repitition=100, gradient=0.001):
    ''' Removes baseline from spectra for concatenated sample runs,
        and calculates the standard deviation and mean error of the
        spectra with valid trainIds. Modifies the datasets in mdata.
        Inputs:
        ------
        mdata: dict 
            data restructured with respect to transmission and run type
            
        Outputs:
        ------
        
    '''
    for r in mdata:
        for k in ['ref', 'sample']:
            ds = mdata[r][k]
            if k=='ref':
                tid = ds.trainId
            else:
                tid = ds.valid_tid
            mdata[r][k]['spectrum_nobl_avg'] = removeBaseline(
                ds.spectrum.sel(trainId=tid).mean(dim='trainId'),
                degree, repitition, gradient)
            ds['spectrum_std'] = ds.spectrum.sel(trainId=tid).std(dim='trainId')
            ds['spectrum_meanError'] = ds.spectrum_std / np.sqrt(tid.size)
            ds.attrs['Tr_from_data'] = ds.transmission.sel(
                trainId=tid).mean(dim='trainId').values * 1e-2
    return

################################################################################
######################### Load run and list of runs ############################
################################################################################

def get_run(proposal, runNB, fields, darkNB=None, roi=[0,2048, 0,512], 
            tid_shift=-1, use_dark=True, errors=False,
            signalRange=[920, 960]):
    ''' Creates dataset for a run, subtracts dark background from the Viking
        spectrometer image and calculates the spectrum.
        Inputs:
        ------
        proposal: int
            proposal number
            
        runNB: int
            run number
            
        fields: list 
            list of mnemonics to load into the memory
            
        darkNB: int
            dark run number
            
        use_dark: bool
            if True, subtracts the dark background from the Viking spectrometer image
            
        tid_shift: int
            overall shift for the train IDs
            
        roi: list or 1d ndarray
            region of interest for the Viking spectrometer image;
            [xMin, xMax, yMin, yMax]
            
        errors: bool
            If true, also calculates mean spectrum with baseline removed, standard 
            deviation of the spectra and standard error of the mean
            
        Outputs:
        ----
        run: extra_data DataCollection
            information on the run
        
        ds: xarray Dataset
            data recorded in the run
    '''
    rois = {'newton': {'newton': {'roi': ed.by_index[roi[2]:roi[3], roi[0]:roi[1]],
                                    'dim': ['y', 'x']},
                       }
           }
    fields += ['newton', 'XTD10_SA3', 'transmission', 'scannerX']
    # remove duplicates
    fields = list(set(fields))
    other_fields = [f for f in fields if f != 'newton']
    ds = xr.Dataset()
    run, ds = tb.load(proposal, runNB, other_fields)
    run, newton = tb.load(proposal, runNB, 'newton', rois=rois)
    newton = newton.assign_coords({'trainId': newton.trainId + tid_shift})
    ds = ds.merge(newton.newton, join='inner')
    if darkNB is not None and use_dark is True:
        dark_run, dark_ds = tb.load(proposal, darkNB, 'newton', rois=rois)
        ds['dark'] = dark_ds.newton.mean(dim='trainId')
        newton_nobg = ds['newton'] - ds['dark']
        ds['spectrum'] = newton_nobg.sum(dim='y')
    else:
        ds['spectrum'] = ds['newton'].sum(dim='y')
            
    ds = ds.sortby(ds.trainId)
    ds['trainId'] = ds.trainId.astype(int)
    ds.attrs['darkNB'] = darkNB
    ds.attrs['runNB'] = runNB
    ds.attrs['gain'] = get_camera_gain(run)
    ds.attrs['countsToPhotoEl'] = get_photoelectronsPerCount(run, ds.attrs['gain'])
    
    energy = calibrate_viking(ds.x, runNB)
    ds = ds.assign_coords({'x': energy})
    
    ds['spectrum_nobl'] = removePolyBaseline(ds.x, ds.spectrum, 
                                             signalRange=signalRange)
    if errors:
        ds['spectrum_std'] = ds.spectrum_nobl.std(dim='trainId')
        ds['spectrum_meanError'] = ds.spectrum_nobl / np.sqrt(ds.trainId.size)
    
    return run, ds


def concatenateRuns(data, runList):
    ''' Concatenates data for specified runs into a single dataset.
        Inputs:
        ------
        data: dict
            dictionary with xarray Datasets for the selected runs
            
        runList: dict
            dictionary containing information on the runs
            
        Outputs:
        ----
        ds: xarray Dataset
            contatenated dataset
    '''
    runNB = [data[r].attrs['runNB'] for r in runList]
    ds = xr.concat([data[r] for r in runList], dim='trainId', data_vars='minimal')
    ds.attrs['runNB'] = runNB
    return ds


def update_runDict(runDict, runBounds, allRuns):
    if len(runBounds) == 1:
        runDict.update({runBounds[0]: allRuns[runBounds[0]]})
    else:
        partialDict = {i: allRuns[i] for i in range(runBounds[0], 
                                                    runBounds[1]+1) if i in allRuns.keys()}
        runDict.update(partialDict)
    return


def generate_partial_runList(runBoundsList, allRuns):
    ''' Generates a dictionary with data for selected runs.
        Inputs:
        ------
        runBoundsList: list
            array containing run numbers to be included in the output.
            If it contains only integers: if it has one element, only 
            this run number is added to the output; otherwise, the first
            and second element represent the boundaries for the range of
            run numbers to be included in the output (boundaries included)
            If it contains lists: the output contains the union of the
            specified run numbers, each element is treated as a list of 
            integers (see above).
            
        allRuns: dict
            dictionary containing the information on runs from which the
            subset is to be generated
            
        Outputs:
        ------
        outDict: dict
            data for selected runs
    '''
    runDict = {}
    if any(isinstance(el, list) for el in runBoundsList):
        for runBounds in runBoundsList:
            update_runDict(runDict, runBounds, allRuns)
    else:
        update_runDict(runDict, runBoundsList, allRuns)
            
    # remove duplicates
    outDict = {}
    for key, value in runDict.items():
        if key not in outDict.keys():
            outDict[key] = value
            
    return outDict


def get_data_for_runList_mp(proposal, runList, fields, roi,
                            inputData={}, append=False, errors=False,
                            nprocesses=None):
    '''
    Generates a dictionary containing datasets for the selected runs.
    For reference runs also removes baseline from spectra, and calculates
    the standard deviation and mean error of the spectra.
    
    Parameters
    ----------
    proposal: int
        proposal number

    runList: dict
        dictionary containing information on the runs

    fields: list 
        list of mnemonics to load into the memory

    roi: list or 1d ndarray
        region of interest for the Viking spectrometer image; [xMin, xMax,
        yMin, yMax]

    inputData: dict
        dictionary with xarray Datasets, optional; the runs contained in
        this dictionary are omitted

    append: bool
        If true, the data is appended to inputData, else a copy is created to
        which the missing runs are added

    errors: bool
        If true, also calculates mean spectrum with baseline removed, standard
        deviation of the spectra and standard error of the mean
    nprocesses: int, optional
        The number of processes to perform the loading of data. If None, equals
        to min(16, number of runs).
        
    Outputs
    -------
    data: dictionary
        dictionary with xarray Datasets for the selected runs
    '''
    if append:
        data = inputData
    else:
        data = inputData.copy()
    runNBs = [r for r in runList if r not in data]
    if len(runNBs) == 0:
        return data
    if nprocesses is None:
        nprocesses = min(16, len(runNBs))
    args = [(proposal, runNB, fields, runList[runNB][0], roi) for runNB in runNBs]
    print(f'loading {len(runNBs)} runs {runNBs} with {nprocesses} processes.')
    with Pool(nprocesses) as pool:
        result = pool.starmap(get_run, args)
    
    for i, runNB in enumerate(runNBs):
        params = runList[runNB]
        ds = result[i][1]
        darkNB = params[0]
        if params[2] == 'ref':
            #ds['spectrum_nobl_avg'] = removePolyBaseline(ds.x, ds.spectrum.mean(dim='trainId'))
            ds['spectrum_std'] = ds.spectrum_nobl.std(dim='trainId')
            ds['spectrum_meanError'] = ds.spectrum_std / np.sqrt(ds.trainId.size)
            if 'transmission' in ds:
                ds.attrs['Tr_from_data'] = ds.transmission.mean(dim='trainId').values * 1e-2
        ds.attrs['Tr'] = params[1]
        ds.attrs['sample'] = params[2]
        ds.attrs['filter'] = params[3]
        ds['filterTr'] = filterTransmission(params[3], ds.x)
        if len(params) > 4:
            ds.attrs['pumpEnergy'] = params[4]
        if len(params) > 5:
            ds.attrs['sampleThickness'] = params[5]
        if len(params) > 6:
            ds.attrs['delay'] = params[6]
        ds['scalingFactor'] = ds.attrs['countsToPhotoEl'] / ds['filterTr']
        data[runNB] = ds
    
    if append:
        return
    else:
        return data


################################################################################
########################## Restructure list of runs ############################
################################################################################

def get_data_for_runList(proposal, runList, fields, roi, 
                         inputData={}, append=False, errors=False):
    ''' Generates a dictionary containing datasets for the selected
        runs. For reference runs also removes baseline from spectra,
        and calculates the standard deviation and mean error of the spectra.
        Inputs:
        ------
        proposal: int
            proposal number
            
        runList: dict
            dictionary containing information on the runs
            
        fields: list 
            list of mnemonics to load into the memory
            
        roi: list or 1d ndarray
            region of interest for the Viking spectrometer image; 
            [xMin, xMax, yMin, yMax]
            
        inputData: dict
            dictionary with xarray Datasets, optional; the runs contained
            in this dictionary are omitted
        
        append: bool
            If true, the data is appended to inputData, else a copy is
            created to which the missing runs are added
            
        errors: bool
            If true, also calculates mean spectrum with baseline removed,
            standard deviation of the spectra and standard error of the mean
            
        Outputs:
        ------
        data: dictionary
            dictionary with xarray Datasets for the selected runs
    '''
    if append:
        data = inputData
    else:
        data = inputData.copy()
        
    for runNB,params in runList.items():
        if runNB in data:
            continue
        print(runNB)
        darkNB = params[0]
        run, ds = get_run(proposal, runNB, fields, darkNB=darkNB, roi=roi, errors=errors)
        if params[2] == 'ref':
            #ds['spectrum_nobl_avg'] = removePolyBaseline(ds.x, ds.spectrum.mean(dim='trainId'))
            ds['spectrum_std'] = ds.spectrum_nobl.std(dim='trainId')
            ds['spectrum_meanError'] = ds.spectrum_std / np.sqrt(ds.trainId.size)
            if 'transmission' in ds:
                ds.attrs['Tr_from_data'] = ds.transmission.mean(dim='trainId').values * 1e-2
        ds.attrs['Tr'] = params[1]
        ds.attrs['sample'] = params[2]
        ds.attrs['filterTr'] = filterTransmission(params[3], ds.x)
        if len(params) > 4:
            ds.attrs['pumpEnergy'] = params[4]
        if len(params) > 5:
            ds.attrs['sampleThickness'] = params[5]
        if len(params) > 6:
            ds.attrs['delay'] = params[6]
        ds.attrs['scalingFactor'] = ds.attrs['countsToPhotoEl'] / ds.attrs['filterTr']
        data[runNB] = ds
    
    if append:
        return
    else:
        return data


def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def find_same_runs(runList):
    ''' Finds sample runs with the same experimental parameters
        (excluding number of dark run).
        Inputs:
        ------
        runList: dict
            dictionary containing information on the runs
            
        Outputs:
        ------
        same_runs: list
            list of lists which contain the numbers of runs with
            same experimental parameters
    '''
    same_runs = []
    for r in runList:
        params = runList[r][1:]
        s = [k for k in runList if runList[k][1:] == params and params[1] != 'ref']
        if len(s) > 0:
            same_runs.append(s)
    same_runs = unique(same_runs)
    return same_runs


def partial_name_for_restructure(dataset, specifier):
    if specifier == 'Tr':
        name = f'Tr{dataset.attrs["Tr"]}'
    elif specifier == 'pumpEnergy':
        name = f'pump{dataset.attrs["pumpEnergy"]}'
    elif specifier == 'both':
        name = f'Tr{dataset.attrs["Tr"]}_pump{dataset.attrs["pumpEnergy"]}'
    elif specifier == 'thickness':
        name = f'Cu{dataset.attrs["sampleThickness"]}'
    elif specifier == 'delay':
        name = f'{dataset.attrs["delay"]}fs'
    return name


def names_for_restructure(dataset, specifier):
    if isinstance(specifier, str):
        specifier = [specifier]
    name = ''
    for i in specifier:
        name += partial_name_for_restructure(dataset, i) + '_'
    return name[:-1]


def restructure_data(runList, data, specifier='Tr'):
    ''' Divides the data for specified run numbers according to
        GATT transmission, and for each transmission according to
        run type (reference or sample). Concatenates sample runs with
        same experimental parameters into a single dataset.
        Inputs:
        ------
        runList: dict
            dictionary containing information on the runs
            
        data: dict 
            dictionary with xarray Datasets for the selected runs
            
        specifier: string or list, optional, default='Tr'.
            string specifying by which parameter to restructure the data,
            currently implemented 'Tr', 'pumpEnergy', 'thickness' and 'both'
            (this one combines 'Tr' and 'pumpEnergy', it is obsolete with 
            added list capabitity, but retained for backward compatibility)
            If it is a list, the names are strings created by mergins the 
            names for inidivual specifiers in the list; 

        Outputs:
        ------
        mdata: dict
            dictionary of which the keys are strings specifying the transmision
            and values are dictionaries. Each (sub-)dictionary contains xarray
            Datasets for the reference and sample runs
    '''
    mdata = {}
    for r in runList:
        if 'ref' in runList[r]:
            name = names_for_restructure(data[r], specifier)
            if name not in mdata:
                mdata[name] = {}
            mdata[name]['ref'] = data[r]

    same_runs = find_same_runs(runList)
    for runs in same_runs:
        ds = concatenateRuns(data, runs)
        name = names_for_restructure(data[runs[0]], specifier)
        if name not in mdata:
            mdata[name] = {}
        mdata[name]['sample'] = ds
    return mdata


################################################################################
######################### Filter relevant train Ids ############################
################################################################################

def plot_counts_by_trainIds(ds, xLim, plotYRange = None, figsize=None):
    ''' Plot total number of couns on newton camera (integrated image)
        as function of trainId for given run.
        Inputs:
        ------
        ds: xarray Dataset
            data recorded in the run
            
        xLim: list or 1d ndarray
            boundaries for integration over photon energy
            
        plotYrange: None or list or 1d ndarray
            y range for the plot, optional
            
        figsize: None or tuple
            plot size, optional
        
        Outputs:
        ------
        
    '''
    spectral_sum = ds.sel(x=slice(xLim[0], xLim[1])).newton.sum(dim=['x', 'y'])
    if figsize is None:
        plt.figure(figsize=(5,6))
    else:
        plt.figure(figsize=figsize)
    plt.plot(np.arange(ds.trainId.size), spectral_sum)
    plt.xlabel('trainId')
    plt.ylabel('counts')
    if plotYRange is not None:
        plt.ylim(plotYRange[0], plotYRange[1])
    plt.show()
    return


def filter_trainIds_with_count_thresholds(mdata, thresholdList, xLim):
    ''' Drop trainIds for which the total number of counts on the newton camera
        is outside the specified range. Modifies the datasets in mdata.
        Inputs:
        ------
        mdata: dict
            data restructured with respect to transmission and run type
            
        thresholdList: dict
            dictionary of which keys are strings specifying transmission
            and values are lists with boundaries for the counts
            
        xLim: list or 1d ndarray
            boundaries for integration over photon energy
            
        Outputs:
        ------
        
    '''
    for r, thresholds in thresholdList.items():
        spectral_sum = mdata[r]['sample'].sel(x=slice(xLim[0], xLim[1])).newton.sum(dim=['x', 'y'])
        filtered_sum = spectral_sum.where(spectral_sum > thresholds[0])
        filtered_sum = filtered_sum.where(filtered_sum < thresholds[1], drop=True)
        valid_tid = filtered_sum.trainId.dropna(dim='trainId')
        mdata[r]['sample']['valid_tid'] = mdata[r]['sample'].trainId.isin(valid_tid)
    return


def filter_trainIds_with_index(mdata, indexList):
    ''' Drop trainIds for which the total number of counts on the newton camera
        is outside the specified range. Modifies the datasets in mdata.
        Inputs:
        ------
        mdata: dict
            data restructured with respect to specifier and run type
            
        indexList: dict
            dictionary of which keys are strings specifiers and values are
            lists with boundaries for the trainId
            indices
            
        Outputs:
        ------
        
    '''
    for r, indices in indexList.items():
        if indices==None:
            continue
        all_indices = []
        for i in indices:
            valid_tid = mdata[r]['sample'].trainId.sel(
                trainId=slice(mdata[r]['sample'].trainId[i[0]],
                              mdata[r]['sample'].trainId[i[1]]))
            all_indices += list(valid_tid.values)
        all_indices = np.unique(all_indices)
        mdata[r]['sample']['valid_tid'] = mdata[r]['sample'].trainId.isin(all_indices)
        
    return


def filter_sample_rastering(ds, xStart=None, xStop=None, tolerance=0.1, use_scannerX=True,
                            frac=0.75, min_threshold=None, max_threshold=None,
                            spectralRange=[920, 960], tid_indices=None, xgm_norm=True,
                            plot=True):
    """
    Filters the rastered data by scanner X position, threshold on spectrum sum,
    and train Id indices.
    First filtering is where scanner X is lower than xStart - tolerance and
    higher than xStop + tolerance. It also scans the derivative of the stage and
    filters out trainIds where the scannerX is stopped for more than 5 consecutive
    trains.
    Second filtering is where spectral sum accros spectralRange is within
    [min_threshold, maxt_threshold].
    Third filtering is to allow specific ranges of indices.
    
    Parameters
    ----------
    ds: xarray Dataset
        The dataset containing rastered sample spectra
    xStart: float
        value in mm of start position of scannerX. If None, equals scannerX[0]
    xStop: float
        value in mm of stop position of scannerX. If None, equals scannerX[-1]
    tolerance: float
        trainIds where xStart - tolerance < scannerX < xStop + tolerance
    spectralRange: list of float
        the spectral range for which to compute the sum for threshold filtering
    frac: float, optional
        Only used if min_threshold is None. Sets min_threshold to frac * median.
    min_threshold: float, optional
        the minimum spectral sum value. If None, equals frac * median.
    max_threshold: float, optional
        the maximum spectral sum value. If None, no upper limit for filtering.
    tid_indices: list of list of int
    xgm_norm: bool
        If True, normalizes the spectral sum by the XTD10 XGM
    plot: bool
        plot the results of the filtering operation if True.
    
    Output
    ------
    valid_tid: xarray DataArray of boolean with dim='trainId'
        the flitered train Ids. Adds this variable to the original dataset.
    """
    scannerX_mask = ds.trainId.astype(bool)
    if use_scannerX:
        #if xStart is None:
        #    xStart = ds.scannerX[0]
        #if xStop is None:
        #    xStop = ds.scannerX[-1]
        #scannerX_mask = (ds.scannerX < xStart - tolerance) & (ds.scannerX > xStop + tolerance)
        scannerX_diff = ds.scannerX.diff(dim='trainId')
        mask = np.ones(ds.scannerX.shape, dtype=bool)
        n = 5
        for i, v in enumerate(scannerX_diff[:-n].values):
            if v==0 and scannerX_diff[i+n]==0:
                mask[i:i+n] = False
        #scannerX_mask = scannerX_mask & mask
        scannerX_mask = xr.DataArray(mask, dims=['trainId'], 
                                     coords={'trainId': ds.scannerX.trainId})
    specsum = ds.spectrum.sel(x=slice(spectralRange[0],
                                      spectralRange[1])).sum(dim='x')
    if xgm_norm:
        specsum = specsum / ds.XTD10_SA3.mean(dim='sa3_pId') * ds.XTD10_SA3.mean()
    
    if min_threshold is None:
        min_threshold = frac * specsum.where(
            specsum >= specsum.mean()).median()
    threshold_mask = specsum > min_threshold
    if max_threshold is not None:
        threshold_mask = threshold_mask & (specsum < max_threshold)
    
    indices_mask = ds.trainId.astype(bool)
    if tid_indices is not None:
        indices_mask = ~indices_mask
        for iMin, iMax in tid_indices:
            indices_mask[iMin: iMax] = True
    valid_mask = scannerX_mask & threshold_mask & indices_mask
    ds['valid_tid'] = valid_mask
    if plot:
        # transform train Id values into indices
        ind = np.argwhere(ds.trainId.isin(ds.trainId).values)
        scanX_ind = np.argwhere(scannerX_mask.values)
        valid_ind = np.argwhere(valid_mask.values)
        plt.figure(figsize=(8,4))
        plt.plot(ind, ds.scannerX, 'o-', ms=3, label='all scannerX')
        plt.plot(scanX_ind,
                 ds.scannerX.where(scannerX_mask, drop=True),
                 '*-', label='filtered scannerX')
        plt.legend(loc='lower left')
        plt.ylabel('scannerX position [mm]')
        plt.xlabel('Train Id index')

        plt.twinx()
        plt.plot(ind, specsum, color='C4', alpha=0.5, label='all')
        plt.plot(valid_ind, specsum.where(valid_mask, drop=True),
                 color='k', lw=2, alpha=0.5, label='filtered')
        plt.axhline(min_threshold, ls='--', color='grey')
        plt.grid()
        if max_threshold is not None:
            plt.axhline(max_threshold, ls='--', color='grey')
        plt.ylabel(f'sum of spectrum over {spectralRange}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        
    return valid_mask


################################################################################
########################## Compute and  plot XAS ###############################
################################################################################

def compute_XAS(mdata, thickness, sortby=None, beamlineTr=0.32):
    if sortby=='Tr':
        keys = [mdata[k]['ref'].attrs['Tr'] for k in mdata.keys()]
        idx = np.argsort(keys) 
        ordered = np.array([k for k in mdata.keys()])[idx]
    else:
        ordered = [k for k in mdata.keys()]
    result = {}
    for r in ordered:
        ds = xr.Dataset()
        ref = mdata[r]['ref'].spectrum_nobl_avg
        attrs_ref = mdata[r]['ref'].attrs
        avg_penergy_ref = mdata[r]['ref'].XTD10_SA3.mean().values
        Tr_ref = beamlineTr*avg_penergy_ref*attrs_ref['Tr_from_data']
        ref_err = mdata[r]['ref'].spectrum_meanError
        ref_scaling = mdata[r]['ref']['scalingFactor']
        
        sample = mdata[r]['sample'].spectrum_nobl_avg
        attrs_sample = mdata[r]['sample'].attrs
        avg_penergy_sample = mdata[r]['sample'].XTD10_SA3.mean().values
        Tr_sample = beamlineTr*avg_penergy_sample*attrs_sample['Tr_from_data']
        sample_err = mdata[r]['sample'].spectrum_meanError
        sample_scaling = mdata[r]['sample']['scalingFactor']
        
        ds['ref'] = ref * ref_scaling
        ds['sample'] = sample * sample_scaling
        ds['ref_err'] = ref_err * ref_scaling
        ds['sample_err'] = sample_err * sample_scaling
        ds['absorption'] = ds.ref / ds.sample
        # assume zero covariance between ref and sample spectra... check!
        ds['absorption_error'] = np.abs(ds.absorption) * np.sqrt(
            ds.ref_err**2 / ds.ref**2 + ds.sample_err**2 / ds.sample**2)
        ds['absorptionCoef'] = np.log(ds.absorption)/thickness
        ds['absorptionCoef_error'] = 1 / thickness * np.sqrt(
            ds.ref_err**2 / ds.ref**2 + ds.sample_err**2 / ds.sample**2)
        for at in mdata[r]['ref'].attrs:
            ds.attrs[at] = mdata[r]['ref'].attrs[at]
        ds.attrs['Tr_from_data_sample'] = mdata[r]['sample'].attrs['Tr_from_data']
        ds.attrs['n_ref'] = mdata[r]['ref'].trainId.size
        ds.attrs['n_sample'] = mdata[r]['sample'].trainId.size
        result[r] = ds
        
    return result


def plot_XAS(mdata, thickness, title='', plotXRange=None, plotAbsRange=None,
             plotAbsCoefRange=None, plotErrors=False, save=False,
             saveTitle='', legend='', sortby=None, ncol=3):
    ''' Plot reference spectra, sample spectra and effective absorption coefficient
        for a range of transmissions.
        Inputs:
        ------
        mdata: dict 
            data restructured with respect to transmission and run type
            
        thickness: float
            target thickness in microns
            
        title: string
            title of the plot, optional
            
        plotXRange: None or list or 1d ndarray
            x range for the plot (photon energies), optional
            
        plotAbsRange: None or list or 1d ndarray
            y range for the absorption plot, optional
            
        plotAbsCoefRange: None or list or 1d ndarray
            y range for the absorption coefficient plot, optional
            
        plotErrors: bool
            If True, plots the 95% confidence interval of the spectrum mean, optional
            
        save: bool
            If true, saves figure, optional
            
        saveTitle: string
            partial name for of the file to which the plot is saved

        Outputs:
        ------
        
    '''
    beamlineTransmission = 0.36
    Lalpha = 929.7
    L3edge = 932.7
    L2edge = 952.3
    nPlots = 4
    if sortby=='Tr':
        keys = [mdata[k]['ref'].attrs['Tr'] for k in mdata.keys()]
        idx = np.argsort(keys) 
        ordered = np.array([k for k in mdata.keys()])[idx]
    else:
        ordered = [k for k in mdata.keys()]
    colors = iter(cm.rainbow(np.linspace(0, 1, len(ordered))))
    fig, ax = plt.subplots(nPlots, 1, sharex=True, figsize=(8,15))
    for r in ordered:
        c = next(colors)
        ref = mdata[r]['ref'].spectrum_nobl_avg
        attrs_ref = mdata[r]['ref'].attrs
        avg_penergy_ref = mdata[r]['ref'].XTD10_SA3.mean().values
        Tr_ref = beamlineTransmission*avg_penergy_ref*attrs_ref['Tr_from_data']
        ref_err = mdata[r]['ref'].spectrum_meanError
        ref_scaling = mdata[r]['ref']['scalingFactor']
        
        sample = mdata[r]['sample'].spectrum_nobl_avg
        attrs_sample = mdata[r]['sample'].attrs
        avg_penergy_sample = mdata[r]['sample'].XTD10_SA3.mean().values
        Tr_sample = beamlineTransmission*avg_penergy_sample*attrs_sample['Tr_from_data']
        sample_err = mdata[r]['sample'].spectrum_meanError
        sample_scaling = mdata[r]['sample']['scalingFactor']
        
        absorption = (ref * ref_scaling) / (sample * sample_scaling)
        absorptionCoef = np.log(absorption)/thickness
        
        ax[0].plot(mdata[r]['ref'].x, ref * ref_scaling, color=c,
                   label=f'{Tr_sample:.2f} $\mu$J, {attrs_sample["Tr_from_data"]*100:.1g}%, ')#
                   #f'{np.floor(attrs_sample["delay"]):.0f} fs')
        ax[1].plot(mdata[r]['sample'].x, sample * sample_scaling, color=c,
                   label=f'{Tr_sample:.2f} $\mu$J, {attrs_sample["Tr_from_data"]*100:.1g}%, ')#
                   #f'{np.floor(attrs_sample["delay"]):.0f} fs')
        ax[2].plot(mdata[r]['sample'].x, absorption, color=c,
                   label=f'{Tr_sample:.2f} $\mu$J, {attrs_sample["Tr_from_data"]*100:.1g}%, ')#
                   #f'{np.floor(attrs_sample["delay"]):.0f} fs')
        ax[3].plot(mdata[r]['sample'].x, absorptionCoef, color=c,
                   label=f'{Tr_sample:.2f} $\mu$J, {attrs_sample["Tr_from_data"]*100:.1g}%, ')#
                   #f'{np.floor(attrs_sample["delay"]):.0f} fs')
        
        if plotErrors:
            ax[0].fill_between(mdata[r]['ref'].x,
                               (ref - 1.96 * ref_err) * ref_scaling,
                               (ref + 1.96 * ref_err) * ref_scaling, color=c,
                               label='_nolegend_', alpha=0.4)
            ax[1].fill_between(mdata[r]['sample'].x,
                               (sample - 1.96 * sample_err) * sample_scaling,
                               (sample + 1.96 * sample_err) * sample_scaling, color=c,
                               label='_nolegend_', alpha=0.4)
            
            # assume zero covariance between reference and sample spectra... check!
            absorption_error = np.abs(absorption) * np.sqrt(ref_err**2 / ref**2 + sample_err**2 / sample**2)
            ax[2].fill_between(mdata[r]['sample'].x,
                               absorption - 1.96 * absorption_error,
                               absorption + 1.96 * absorption_error,  color=c,
                               label='_nolegend_', alpha=0.4)
            
            absorptionCoef_error = 1 / thickness * np.sqrt(ref_err**2 / ref**2 + sample_err**2 / sample**2)
            ax[3].fill_between(mdata[r]['sample'].x,
                               absorptionCoef - 1.96 * absorptionCoef_error,
                               absorptionCoef + 1.96 * absorptionCoef_error, color=c,
                               label='_nolegend_', alpha=0.4)
        
    for i in range(nPlots):
        ax[i].axvline(L3edge, ls='--', color='grey')
        ax[i].axvline(L2edge, ls='--', color='grey')
        ax[i].axvline(Lalpha, ls='--', color='darkgrey')
        ax[i].legend(ncol=4, fontsize=8) #, bbox_to_anchor=(1.04,1.0), loc='upper left')
        ax[i].grid()
        
    ax[0].set_ylabel('reference')
    ax[1].set_ylabel('sample')
    ax[2].set_ylabel('reference / sample')
    ax[3].set_xlabel('photon energy [eV]')
    ax[3].set_ylabel('XAS -log(sample/ref)/thickness')

    if plotAbsRange is not None:
        ax[2].set_ylim(plotAbsRange[0], plotAbsRange[1])
    if plotAbsCoefRange is not None:
        ax[3].set_ylim(plotAbsCoefRange[0], plotAbsCoefRange[1])
    if plotXRange is not None:
        plt.xlim(plotXRange[0], plotXRange[1])
    ax[0].set_title(title)
    fig.tight_layout()
    
    if save:
        fig.patch.set_alpha(1)
        plt.savefig('XAS_'+saveTitle+'.png', dpi=300, bbox_inches='tight')
    return fig, ax
