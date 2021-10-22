import toolbox_scs as tb
import extra_data as ed
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import BaselineRemoval as br
import json

def load_runs_from_file(fname):
    ''' Loads run data from a text file to a dictionary.
        Inputs:
        ------
        fname: string
            name of the file
            
        Outputs:
        ------
        dict: dictionary with run numbers as keys and an array of run parameters as values
    '''
    with open(fname) as f:
        data = f.read()
    dataDict =  json.loads(data)
    return {int(key): value for key, value in dataDict.items()}

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
    gain = sel.get_array('SCS_EXP_NEWTON/CAM/CAMERA', 'preampGain.value')
    if gain == 0:
        return 1
    if gain == 1:
        return 2
    if gain == 2:
        return 4

def photoelectronsPerCount_HS(gain):
    ''' Conversion factor from camera gain to photoelectrons per count for High Sensitivity mode.
        Inputs:
        ------
        gain: int
            camera gain; allowed values are 1, 2, or 4
            
        Outputs:
        ------
        int:
            photoelectrons per count
    '''
    if gain==1:
        return 4
    if gain==2:
        return 2
    if gain==4:
        return 0.9
    
def photoelectronsPerCount_HC(gain):
    ''' Conversion factor from camera gain to photoelectrons per count for High Capacity mode.
        Inputs:
        ------
        gain: int
            camera gain; allowed values are 1, 2, or 4
            
        Outputs:
        ------
        int:
            photoelectrons per count
    '''
    if gain==1:
        return 17.9
    if gain==2:
        return 9
    if gain==4:
        return 4.5
    
def get_photoelectronsPerCount(run, gain):
    sel = run.select_trains([0])
    hc = sel.get_array('SCS_EXP_NEWTON/CAM/CAMERA', 'HighCapacity.value').item()
    if hc == 0:
        return photoelectronsPerCount_HS(gain)
    else:
        return photoelectronsPerCount_HC(gain)
    

def filterTransmission(filterStr):
    ''' Returns transmission of Al filters of specified thickness.
        Inputs:
        ------
        filterStr: string
            description of the filter with element name and thickness in microns; accepted values are 'Al3.5',  'Al5' or 'Al10'
            
        Outputs:
        ------
        float
            transmission
    '''
    if filterStr == 'Al3.5':
        return 0.25
    if filterStr == 'Al5':
        return 0.14
    if filterStr == 'Al10':
        return 0.02
    if filterStr == 'Al15':
        return 2.8e-3
    if filterStr == 'Al13.5':
        return 5e-3
    return 1

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

def get_run(proposal, runNB, fields, darkNB=None, use_dark=True, 
            tid_shift=-1, roi=[0,2048, 0,512], errors=False,
            signalRange=[920, 960]):
    ''' Creates dataset for a run, subtracts dark background from the Viking spectrometer image and calculates the spectrum.
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
            if True it subtracts the dark background from the Viking spectrometer image
            
        tid_shift: int
            overall shift for the train IDs
            
        roi: list or 1d ndarray
            region of interest for the Viking spectrometer image; [xMin, xMax, yMin, yMax]
            
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
        dark = dark_ds.newton.mean(dim='trainId')
        newton_nobg = ds['newton'] - dark
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
    
    ds['spectrum_nobl'] = removePolyBaseline(ds.x, ds.spectrum, signalRange=signalRange)
    if errors:
        ds['spectrum_std'] = ds.spectrum_nobl.std(dim='trainId')
        ds['spectrum_meanError'] = ds.spectrum_nobl / np.sqrt(ds.trainId.size)
    
    return run, ds

def calibrate_viking(x, runNB):
    """
    Calibration of Viking spectrometer using 2nd order polynomial,
    based on analysis of runs 118-124 (proposal 2593) and run 1716
    (proposal 2937).
    """
    if runNB <= 851: # Proposal 2593
        coeffs = [1.20078832e-05, 8.06816040e-02, 8.63353911e+02]
    elif runNB >= 1289: # Proposal 2937
        coeffs = [1.21597814e-05, 8.03133850e-02, 8.63770237e+02]
    return np.poly1d(coeffs)(x)



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
        partialDict = {i: allRuns[i] for i in range(runBounds[0], runBounds[1]+1) if i in allRuns.keys()}
        runDict.update(partialDict)
    return

def generate_partial_runList(runBoundsList, allRuns):
    ''' Generates a dictionary with data for selected runs.
        Inputs:
        ------
        runBoundsList: list
            array containing run numbers to be included in the output.
            If it contains only integers: if it has one element, only this run number is added to the output; otherwise, the first and second element represent the boundaries for the range of run numbers to be included in the output (boundaries included)
            If it contains lists: the output contains the union of the specified run numbers, each element is treated as a list of integers (see above)
            
        allRuns: dict
            dictionary containing the information on runs from which the subset is to be generated
            
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

def get_data_for_runList(proposal, runList, fields, roi, inputData={}, append=False, errors=False):
    ''' Generates a dictionary containing datasets for the selected runs. For reference runs also removes baseline from spectra, and calculates the standard deviation and mean error of the spectra.
        Inputs:
        ------
        proposal: int
            proposal number
            
        runList: dict
            dictionary containing information on the runs
            
        fields: list 
            list of mnemonics to load into the memory
            
        roi: list or 1d ndarray
            region of interest for the Viking spectrometer image; [xMin, xMax, yMin, yMax]
            
        inputData: dict
            dictionary with xarray Datasets, optional; the runs contained in this dictionary are omitted
        
        append: bool
            If true, the data is appended to inputData, else a copy is created to which the missing runs are added
            
        errors: bool
            If true, also calculates mean spectrum with baseline removed, standard deviation of the spectra and standard error of the mean
            
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
            ds.attrs['Tr_from_data'] = ds.transmission.mean(dim='trainId').values * 1e-2
        ds.attrs['Tr'] = params[1]
        ds.attrs['sample'] = params[2]
        ds.attrs['filterTr'] = filterTransmission(params[3])
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
    ''' Finds sample runs with the same experimental parameters (excluding number of dark run).
        Inputs:
        ------
        runList: dict
            dictionary containing information on the runs
            
        Outputs:
        ------
        same_runs: list
            list of lists which contain the numbers of runs with same experimental parameters
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
    ''' Divides the data for specified run numbers according to GATT transmission, and for each transmission according to run type (reference or sample). Concatenates sample runs with same experimental parameters into a single dataset.
        Inputs:
        ------
        runList: dict
            dictionary containing information on the runs
            
        data: dict 
            dictionary with xarray Datasets for the selected runs
            
        specifier: string or list
            string specifying by which parameter to restructure the data, currently implemented 'Tr', 'pumpEnergy', 'thickness' and 'both' (this one combines 'Tr' and 'pumpEnergy', it is obsolete with added list capabitity, but retained for backward compatibility)
            If it is a list, the names are strings created by mergins the names for inidivual specifiers in the list; 
            optional, default is 'Tr'
        
        Outputs:
        ------
        mdata: dict
            dictionary of which the keys are strings specifying the transmision and values are dictionaries. Each (sub-)dictionary contains xarray Datasets for the reference and sample runs
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

def plot_counts_by_trainIds(ds, xLim, plotYRange = None, figsize=None):
    ''' Plot total number of couns on newton camera (integrated image) as function of trainId for given run.
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
            dictionary of which keys are strings specifying transmission and values are lists with boundaries for the counts
            
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
            dictionary of which keys are strings specifiers and values are lists with boundaries for the trainId
            indices
            
        Outputs:
        ------
        
    '''
    for r, indices in indexList.items():
        if indices==None:
            continue
        all_indices = []
        for i in indices:
            valid_tid = mdata[r]['sample'].trainId.sel(trainId=slice(mdata[r]['sample'].trainId[i[0]], 
                                                             mdata[r]['sample'].trainId[i[1]]))
                                              
            all_indices += list(valid_tid.values)
        all_indices = np.unique(all_indices)
        mdata[r]['sample']['valid_tid'] = mdata[r]['sample'].trainId.isin(all_indices)
        
    return

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
            if k=='ref':
                tid = mdata[r][k].trainId
            else:
                tid = mdata[r][k].valid_tid
            mdata[r][k]['spectrum_nobl_avg'] = removeBaseline(mdata[r][k].spectrum.sel(trainId=tid).mean(dim='trainId'),
                                                              degree, repitition, gradient)
            mdata[r][k]['spectrum_std'] = mdata[r][k].spectrum.sel(trainId=tid).std(dim='trainId')
            mdata[r][k]['spectrum_meanError'] = mdata[r][k].spectrum_std / np.sqrt(tid.size)
            mdata[r][k].attrs['Tr_from_data'] = mdata[r][k].transmission.sel(trainId=tid).mean(dim='trainId').values * 1e-2
    return

def plot_XAS(mdata, thickness, title='', plotXRange=None, plotAbsRange=None,
             plotAbsCoefRange=None, plotErrors=False, save=False,
             saveTitle='', legend=''):
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
    beamlineTransmission = 0.32
    L3edge = 932.7
    L2edge = 952.3
    nPlots = 4
    
    fig, ax = plt.subplots(nPlots, 1, sharex=True, figsize=(6,15))
    for r in mdata:
        ref = mdata[r]['ref'].spectrum_nobl_avg
        attrs_ref = mdata[r]['ref'].attrs
        avg_penergy_ref = mdata[r]['ref'].XTD10_SA3.mean().values
        Tr_ref = beamlineTransmission*avg_penergy_ref*attrs_ref['Tr_from_data']
        ref_err = mdata[r]['ref'].spectrum_meanError
        ref_scaling = attrs_ref['scalingFactor']
        
        sample = mdata[r]['sample'].spectrum_nobl_avg
        attrs_sample = mdata[r]['sample'].attrs
        avg_penergy_sample = mdata[r]['sample'].XTD10_SA3.mean().values
        Tr_sample = beamlineTransmission*avg_penergy_sample*attrs_sample['Tr_from_data']
        sample_err = mdata[r]['sample'].spectrum_meanError
        sample_scaling = attrs_sample['scalingFactor']
        
        absorption = (ref * ref_scaling) / (sample * sample_scaling)
        absorptionCoef = np.log(absorption)/thickness
        
        ax[0].plot(mdata[r]['ref'].x, ref * ref_scaling, 
                   label=f'{Tr_sample:.2f} $\mu$J, {attrs_sample["Tr_from_data"]*100:.2f}%, ')#
                   #f'{np.floor(attrs_sample["delay"]):.0f} fs')
        ax[1].plot(mdata[r]['sample'].x, sample * sample_scaling,
                   label=f'{Tr_sample:.2f} $\mu$J, {attrs_sample["Tr_from_data"]*100:.2f}%, ')#
                   #f'{np.floor(attrs_sample["delay"]):.0f} fs')
        ax[2].plot(mdata[r]['sample'].x, absorption,
                   label=f'{Tr_sample:.2f} $\mu$J, {attrs_sample["Tr_from_data"]*100:.2f}%, ')#
                   #f'{np.floor(attrs_sample["delay"]):.0f} fs')
        ax[3].plot(mdata[r]['sample'].x, absorptionCoef,
                   label=f'{Tr_sample:.2f} $\mu$J, {attrs_sample["Tr_from_data"]*100:.2f}%, ')#
                   #f'{np.floor(attrs_sample["delay"]):.0f} fs')
        
        if plotErrors:
            ax[0].fill_between(mdata[r]['ref'].x,
                               (ref - 1.96 * ref_err) * ref_scaling,
                               (ref + 1.96 * ref_err) * ref_scaling,
                               label='_nolegend_', alpha=0.4)
            ax[1].fill_between(mdata[r]['sample'].x,
                               (sample - 1.96 * sample_err) * sample_scaling,
                               (sample + 1.96 * sample_err) * sample_scaling,
                               label='_nolegend_', alpha=0.4)
            
            # assume zero covariance between reference and sample spectra... check!
            absorption_error = np.abs(absorption) * np.sqrt(ref_err**2 / ref**2 + sample_err**2 / sample**2)
            ax[2].fill_between(mdata[r]['sample'].x, 
                               absorption - 1.96 * absorption_error,
                               absorption + 1.96 * absorption_error, 
                               label='_nolegend_', alpha=0.4)
            
            absorptionCoef_error = 1 / thickness * np.sqrt(ref_err**2 / ref**2 + sample_err**2 / sample**2)
            ax[3].fill_between(mdata[r]['sample'].x, 
                               absorptionCoef - 1.96 * absorptionCoef_error,
                               absorptionCoef + 1.96 * absorptionCoef_error,
                               label='_nolegend_', alpha=0.4)
        
    for i in range(nPlots):
        ax[i].axvline(L3edge, ls='--', color='grey')
        ax[i].axvline(L2edge, ls='--', color='grey')
        ax[i].legend(ncol=1, fontsize=8) #, bbox_to_anchor=(1.04,1.0), loc='upper left')
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
    # fig.tight_layout()
    
    if save:
        fig.patch.set_alpha(1)
        plt.savefig('XAS_'+saveTitle+'.png', dpi=300, bbox_inches='tight')