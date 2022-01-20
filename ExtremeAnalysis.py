import pandas as pd
import numpy as np
from datetime import date, time
import os
import xarray
import cftime as cft
from scipy.stats import genextreme as gev
from datetime import  datetime
from shutil import copyfile
from glob import glob
import itertools
from scipy.optimize import curve_fit


class ExtremePrecIndexFunctions():
    def __init__(self):
        self.functionList = [self.CDD, self.CWD, self.rx1day, self.rx3day, self.rx5day, self.r20mm, self.Prec95p, self.Prec99p]

    def CDD(self,S):
    #  print('Shape S: ', S.shape)
        
        ind_CDD=[]
        S_no_nan = S[~np.isnan(S)]
        N = len(S)
        N2 = len(S_no_nan)
        if ((N2/N) < 0.3): 
            ind_CDD = np.empty(1)
            ind_CDD = np.nan
        else:
            temp = 0
            ind_CDD = 0 
            j =0
            while (j < N2):
                while (j < N2 ) and (S_no_nan[j] < 1.0 ):
                    j += 1
                    temp +=1
                if ind_CDD < temp:
                    ind_CDD = temp
                temp = 0
                j += 1 
        return ind_CDD
    
    def CWD(self,S):
        ind_CWD=[]
        S_no_nan = S[~np.isnan(S)]
        N = len(S)
        N2 = len(S_no_nan)
        if ((N2/N) < 0.3): 
            ind_CWD = np.empty(1)
            ind_CWD = np.nan
        else:
            temp = 0
            ind_CWD = 0 
            j =0
            while (j < N2):
                while (j < N2 ) and (S_no_nan[j] > 1.0 ):
                    j += 1
                    temp +=1
                if ind_CWD < temp:
                    ind_CWD = temp
                temp = 0
                j += 1 
        return ind_CWD
    
    def rx1day(self,S):
        S_no_nan = S[~np.isnan(S)]
        N = len(S)
        N2 = len(S_no_nan)
        if ((N2/N) < 0.3):
            return np.nan 
        return S.max()
    
    def rx3day(self, S):
        ind_R3d=[]
        S_no_nan = S[~np.isnan(S)]
        N = len(S)
        N2 = len(S_no_nan)
        if ((N2/N) < 0.3): 
            ind_R3d = np.empty(1)
            ind_R3d = np.nan
        else:
            temp = 0
            ind_R3d = 0 
            for i in range(0,N-2):
                if (~np.isnan(S[i])) and  (~np.isnan(S[i+1]))  and  (~np.isnan(S[i+2])):
                    temp = S[i] + S[i+1] + S[i+2]
                if ind_R3d < temp:
                    ind_R3d = temp
        return ind_R3d
    
    def rx5day(self, S):
        ind_R3d=[]
        S_no_nan = S[~np.isnan(S)]
        N = len(S)
        N2 = len(S_no_nan)
        if ((N2/N) < 0.3): 
            ind_R5d = np.empty(1)
            ind_R5d = np.nan
        else:
            temp = 0
            ind_R5d = 0 
            for i in range(0,N-4):
                if (~np.isnan(S[i])) and  (~np.isnan(S[i+1]))  and  (~np.isnan(S[i+2]))  and  (~np.isnan(S[i+3])) and  (~np.isnan(S[i+4])):
                    temp = S[i] + S[i+1] + S[i+2] + S[i+3] + S[i+4]
                if ind_R5d < temp:
                    ind_R5d = temp
        return ind_R5d
    
    def r20mm(self, S):
        S_no_nan = S[~np.isnan(S)]
        N = len(S)
        N2 = len(S_no_nan)
        if ((N2/N) < 0.3):
            return np.nan 
        return np.count_nonzero(S >= 20)
    
    def Prec95p(self, S):
        S_no_nan = S[~np.isnan(S)]
        N = len(S)
        N2 = len(S_no_nan)
        if ((N2/N) < 0.3):
            return np.nan
        S = pd.Series(S)
        return S.quantile(q=0.95)
    
    def Prec99p(self, S):
        S_no_nan = S[~np.isnan(S)]
        N = len(S)
        N2 = len(S_no_nan)
        if ((N2/N) < 0.3):
            return np.nan 
        S = pd.Series(S)
        return S.quantile(q=0.99)
    

    def mapoverlatlon(self, ds, funct):
 
        return xarray.apply_ufunc(funct, 
                    ds,
                    input_core_dims=[["time"]],
                    exclude_dims=set(("time",)),
                    vectorize=True,                 
                    )
    
class SpatialOperations():
    def __init__(self):
        self.cities ={
            'Banjarmasin': [-3.26,  -3.37, 114.54 , 114.65],
            'Pangkalpinang': [-2.07,  -2.16, 106.06, 106.18],
            'Ternate1' : [0.921, 0.747, 127.288, 127.395],
            'Ternate2' : [0.482, 0.431, 127.38, 127.441],
            'Ternate3' : [1.354, 1.279, 126.356, 126.417],
            'Ternate4' : [0.99, 0.955, 126.126, 126.163],
            # 'Ternate': [1.36, 0.43, 126.12, 127.44],
            'Bandar Lampung': [-5.33, -5.53, 105.18, 105.35],
            'Mataram':[-8.55,  -8.62 ,116.06, 116.16],
            'Samarinda': [0.71, 0.3, 117.04, 117.31],            
            'Pekanbaru': [0.61, 0.41, 101.36, 101.52],
            'Gorontalo': [0.6, 0.5, 123.0, 123.08],
            'Cirebon': [-6.68, -6.8, 108.51, 108.59],
            'Kupang': [-10.12, -10.22, 123.54, 123.68]
        }
        self.era5data = '/content/drive/MyDrive/Bahan Pelatihan/DataIklimEkstrim/ERA5'
        self.GEVparamDir = '/content/drive/MyDrive/Bahan Pelatihan/DataIklimEkstrim/GEVparams'
        self.DATADIR = '/content/drive/MyDrive/Bahan Pelatihan/DataIklimEkstrim/chirps/'
        self.MOVE2GDRIVE = True
        self.GDRIVELOC = '/content/drive/MyDrive/Bahan Pelatihan/DataIklimEkstrim/CityECI2'
        self.CityETCCDIEra5 = '/content/drive/MyDrive/Bahan Pelatihan/DataIklimEkstrim/ETCCDIEra5'
        self.GCMDIRS = '/content/drive/MyDrive/Bahan Pelatihan/DataIklimEkstrim/GCM'
        self.ReferenceFile = '/content/drive/MyDrive/Bahan Pelatihan/DataIklimEkstrim/wc2.1_30s_prec_01.tif'
        self.DATAFROMDRIVE = False
        self.GCMdir = '/content/drive/MyDrive/Bahan Pelatihan/DataIklimEkstrim/GCM'
        self.GCMs = ['MIROC-ESM', 'IPSL-CM5A-LR', 'HadGEM2-ES', 'bcc-csm1-1', 'MIROC5', 'GFDL-ESM2M', 'CSIRO-Mk3-6-0', 'NorESM1-M', 'CCSM4'] 
        self.BaselineGCMdir = '/content/drive/MyDrive/Bahan Pelatihan/DataIklimEkstrim/baselineGCM2'
        self.polynomParamsDir = '/content/drive/MyDrive/Bahan Pelatihan/DataIklimEkstrim/PolynomParams2'
        self.correctedDataDir = '/content/drive/MyDrive/Bahan Pelatihan/DataIklimEkstrim/correctedData2'
        self.QCDir = '/content/drive/MyDrive/Bahan Pelatihan/DataIklimEkstrim/QCData'
        self.SHPDir = '/content/drive/MyDrive/Bahan Pelatihan/DataIklimEkstrim/SHPs'

        self.paramToGCMParamDict = {
            'CDD' : 'cddETCCDI',
            'CWD' : 'cwdETCCDI',
            'r20mm': 'r20mmETCCDI',
            'rx1day': 'rx1dayETCCDI',
            'rx5day': 'rx5dayETCCDI',
            'Prec95p': 'r95pETCCDI', 
            'Prec99p': 'r99pETCCDI'
        }
    
    def mapoverlatlon(self, ds, funct):
        return xarray.apply_ufunc(funct, 
                    ds,
                    input_core_dims=[["time"]],
                    exclude_dims=set(("time",)),
                    vectorize=True,                 
                    )
    
    def getGridReference(self, city, ref='/content/drive/MyDrive/Bahan Pelatihan/DataIklimEkstrim/wc2.1_30s_prec_01.tif'):
        if self.DATAFROMDRIVE:
            ref = self.ReferenceFile

        ref = xarray.open_rasterio(ref)
        ref = ref[0,:,:]

        max_lat, min_lat, min_lon, max_lon = self.cities[city]
        max_lat = max_lat + 0.2
        max_lon = max_lon + 0.2
        min_lat = min_lat - 0.2
        min_lon = min_lon - 0.2

        mask_lon = (ref.x >= min_lon) & (ref.x <= max_lon)
        mask_lat = (ref.y >= min_lat) & (ref.y <= max_lat)
        cropped_ref = ref.where(mask_lon & mask_lat, drop=True)
        return cropped_ref
    
    def getMergedCropRegridedData(self, city):
        # if self.DATAFROMDRIVE:
        #     with xarray.open_dataset(os.path.join(self.GDRIVELOC, 'Merged_'+city+'.nc')) as dataset:

        ref = self.getGridReference(city)
        print('pocessing for ', city)
        max_lat, min_lat, min_lon, max_lon = self.cities[city]
        max_lat = max_lat + 0.2
        max_lon = max_lon + 0.2
        min_lat = min_lat - 0.2
        min_lon = min_lon - 0.2
        dss = []
        for i in range(1991,2021,1):
            print(i)
            with xarray.open_dataset(os.path.join(self.DATADIR, 'chirps-v2.0.'+str(i)+'.days_p05.nc')) as dataset:
                prec = dataset['precip']
            mask_lon = (prec.longitude >= min_lon) & (prec.longitude <= max_lon)
            mask_lat = (prec.latitude >= min_lat) & (prec.latitude <= max_lat)
            cropped_ds = prec.where(mask_lon & mask_lat, drop=True)
            dss.append(cropped_ds)
        merged_noregrid = xarray.merge(dss)
        merged = merged_noregrid.interp(latitude=ref.y, longitude=ref.x, method="linear")
        merged = merged.drop(['latitude', 'longitude'])
        merged.to_netcdf('Merged_'+city+'.nc')
        if self.MOVE2GDRIVE:
            copyfile('Merged_'+city+'.nc', os.path.join(self.GDRIVELOC, 'Merged_'+city+'.nc'))
        return merged
    
    def run_all_obs(self, ECIO):
        for city in self.cities.keys():
            ref = self.getGridReference(city)
            merged = self.getMergedCropRegridedData(city)

            for index_f in ECIO.functionList:
                eci = merged.groupby('time.year').map(self.mapoverlatlon, funct=index_f)
                nc_filename = city+'_'+index_f.__name__ + '.nc'
                eci.to_netcdf(nc_filename)
                copyfile(nc_filename, os.path.join(self.GDRIVELOC, nc_filename))

    def calculateECIndexEra5(self, ECIO):
        for city in self.cities:
            ref = self.getGridReference(city)
            max_lat, min_lat, min_lon, max_lon = self.cities[city]
            max_lat = max_lat + 0.2
            max_lon = max_lon + 0.2
            min_lat = min_lat - 0.2
            min_lon = min_lon - 0.2
            with xarray.open_dataset(os.path.join(self.DATADIR, 'chirps-v2.0.'+str(1991)+'.days_p05.nc')) as dataset:
                prec = dataset['precip']
            mask_lon = (prec.longitude >= min_lon) & (prec.longitude <= max_lon)
            mask_lat = (prec.latitude >= min_lat) & (prec.latitude <= max_lat)
            ref2 = prec.where(mask_lon & mask_lat, drop=True)

            #### grep era5 2016 data
            era5 = xarray.open_dataset(os.path.join(self.era5data, 'era5_2016.nc'))
            era5_5km_2016 = era5.interp(y=ref2.latitude, x=ref.longitude, method='linear')
            
            ### grep era5 2020 data
            era5 = xarray.open_dataset(os.path.join(self.era5data, 'era5_2020.nc'))
            era5_5km_2020 = era5.interp(y=ref2.latitude, x=ref.longitude, method='linear')

            # fill up the 2020 data
            merged_2020 = xarray.merge([era5_5km_2020['total_precipitation'][:,:,:], era5_5km_2016['total_precipitation'][191,:,:] ])
            merged_2020['time'] = np.arange('2020-01', '2021-01', dtype='datetime64[D]')

            ### combine all era5 data
            dss = []
            for year in range(1991, 2020):
                with xarray.open_dataset(os.path.join(self.era5data, 'era5_{}.nc'.format(year))) as era5:
                    era5_prec = era5['total_precipitation']  
                era5_5km = era5_prec.interp(y=ref2.latitude, x=ref.longitude, method='linear')
                dss.append(era5_5km)
            dss.append(merged_2020)
            merged_era5 = xarray.merge(dss)

            dss = []
            for i in range(1991,2021,1):
                print(i)
                with xarray.open_dataset(os.path.join(self.DATADIR, 'chirps-v2.0.'+str(i)+'.days_p05.nc')) as dataset:
                    prec = dataset['precip']
                mask_lon = (prec.longitude >= min_lon) & (prec.longitude <= max_lon)
                mask_lat = (prec.latitude >= min_lat) & (prec.latitude <= max_lat)
                cropped_ds = prec.where(mask_lon & mask_lat, drop=True)
                dss.append(cropped_ds)
            merged_noregrid = xarray.merge(dss)

            filled_chirps = xarray.where(merged_noregrid.isnull(), merged_era5, merged_noregrid )
            for index_f in ECIO.functionList:
                eci = filled_chirps.groupby('time.year').map(self.mapoverlatlon, funct=index_f)
                nc_filename = city+'_'+index_f.__name__ + '.nc'
                eci.to_netcdf(nc_filename)
                copyfile(nc_filename, os.path.join(self.CityETCCDIEra5, nc_filename))
    # def run_selectedf_obs(self, FunList):
    #     for city in self.cities.keys():
    #         ref = self.getGridReference(city)
    #         merged = self.getMergedCropRegridedData(city)

    #         for index_f in FunList:
    #             eci = merged.groupby('time.year').map(self.mapoverlatlon, funct=Fun)
    #             nc_filename = city+'_'+index_f.__name__ + '.nc'
    #             eci.to_netcdf(nc_filename)
    #             copyfile(nc_filename, os.path.join(self.GDRIVELOC, nc_filename))

    def calculateGEVParam(S):
        params = gev.fit(S)
        # print('S and params shape', S.shape, params)
        return np.array(params)

    def downloadGCM(self, model, base_urls):
        '''
        links = [
            'https://crd-data-donnees-rdc.ec.gc.ca/CCCMA/products/CLIMDEX/CMIP5/historical/MIROC5/r3i1p1/v20120710/base_1961-1990/',
            'https://crd-data-donnees-rdc.ec.gc.ca/CCCMA/products/CLIMDEX/CMIP5/rcp45/MIROC5/r3i1p1/v20120710/historical_MIROC5_r3i1p1_v20120710_historical-base_1961-1990/',
            'https://crd-data-donnees-rdc.ec.gc.ca/CCCMA/products/CLIMDEX/CMIP5/rcp85/MIROC5/r3i1p1/v20120710/historical_MIROC5_r3i1p1_v20120710_historical-base_1961-1990/'
        ]
        downloadGCM('MIROC5', links)
        !mv /content/MIROC5 /content/drive/MyDrive/GCM/MIROC5
        '''
        import requests
        import os
        # import wget
        from parallel_sync import wget
        from bs4 import BeautifulSoup
        vars = ['cddETCCDI_yr', 'cwdETCCDI_yr', 'r20mmETCCDI_yr', 'r95pETCCDI_yr', 'r99pETCCDI_yr', 'rx1dayETCCDI_yr', 'rx5dayETCCDI_yr']
        # tup_vars = tuple(vars)
        outdir = '/content/' + model
        os.mkdir(outdir)

        for url in base_urls:
            r = requests.get(url)
            soup = BeautifulSoup(r.text)
            files = []
            for item in soup.find_all("a")[5:]:
                if item.text.startswith(tuple(vars)):
                    # print(item.text)
                    files.append(item.text)
            urls = [url+f for f in files]
            # [i.text if i.text.endswith('.nc') for i in ]
            # type(tuple(vars))
            wget.download(outdir, urls)

    def Combine_baseline_model(self, city='Mataram', rcp='rcp45',gcm = 'CCSM4',  
                               param='CDD' ):
        modelparam = self.paramToGCMParamDict[param]
        hist = xarray.open_dataset( glob( os.path.join(self.GCMdir, gcm) + '/'+
                                         modelparam+'_yr_'+gcm+'_historical' +'*.nc')[0]) 
        future = xarray.open_dataset(glob( os.path.join(self.GCMdir, gcm) + '/'+
                                         modelparam+'_yr_'+gcm+'_'+rcp +'*.nc')[0])
        
        last_index = len(hist['time'])
        hist_crop = hist.isel(time=slice( last_index - 15, last_index ))

        last_index = len(future['time'])
        future_crop = future.isel(time=slice(1,16))

        baseline_model = xarray.concat([hist_crop, future_crop], dim='time')

        newtime = []
        # baseline_model['time'] 
        for t in baseline_model.time.values:
            try:
                year = t.year
            except:
                year = int(str(t).split('-')[0])
            newtime.append(datetime(year, 1, 1))
        baseline_model['time'] = newtime

        index_data = baseline_model[modelparam]
 
        ref = self.getGridReference(city)
        max_lat, min_lat, min_lon, max_lon = self.cities[city]
        max_lat = max_lat + 0.2
        max_lon = max_lon + 0.2
        min_lat = min_lat - 0.2
        min_lon = min_lon - 0.2

        # mask_lon = (ref.x >= min_lon) & (ref.x <= max_lon)
        # mask_lat = (ref.y >= min_lat) & (ref.y <= max_lat)
        # cropped_ref = ref.where(mask_lon & mask_lat, drop=True)
        # cropped_ref
        if param in ['CDD', 'CWD', 'r20mm']:
            index_data = index_data.astype('timedelta64[D]') / np.timedelta64(1, 'D')
            index_data = index_data.astype('int')
        cropped_ds = index_data.interp(lat=ref.y, lon=ref.x, method="linear")
        cropped_ds = cropped_ds.drop(['lat', 'lon'])
        # filename_baseline= city+'_baseline_model'+rcp+'.nc'
        # cropped_ds.to_netcdf(filename_baseline)

        # if self.MOVE2GDRIVE:
        #     copyfile(filename, os.path.join(self.GDRIVELOC, filename))
        
        return cropped_ds

    def run_all_baseline_models(self):
        ECIO = ExtremePrecIndexFunctions()
        
        
        for city in self.cities.keys():
            for gcm in self.GCMs:
                for rcp in ['rcp45', 'rcp85']:
                    print(city, gcm, rcp)
                    for index_f in [ECIO.CDD, ECIO.CWD, ECIO.Prec95p, ECIO.Prec99p, ECIO.rx1day, ECIO.rx5day, ECIO.r20mm]:
                        print(index_f.__name__)
                        combined = self.Combine_baseline_model(city=city, rcp=rcp, gcm=gcm, param=index_f.__name__)
                        nc_filename = city+'_'+index_f.__name__ + '_baseline_'+ rcp+'_'+gcm+'.nc'
                        combined.to_netcdf(nc_filename)
                        copyfile(nc_filename, os.path.join(self.BaselineGCMdir, nc_filename))
                        
    def getObsData(self, city, param):
        nc_filename = city+'_'+param + '.nc'
        # copyfile(nc_filename, os.path.join(self.GDRIVELOC, nc_filename))
        
        with xarray.open_dataset(os.path.join(self.GDRIVELOC, nc_filename)) as dataset:
            obs_data = dataset['precip']
        return obs_data

    def getModelBaselineData(self, city, param, rcp, gcm):
        nc_filename = city+'_'+ param + '_baseline_'+ rcp+'_'+gcm+'.nc'
        with xarray.open_dataset(os.path.join(self.BaselineGCMdir, nc_filename)) as dataset:
            mod_data = dataset[self.paramToGCMParamDict[param]]
        return mod_data

    def calculateGEVObs(self, city, param):
        
        def calculateGEVParam(S):
            S_no_nan = S[~np.isnan(S)]
            N = len(S)
            N2 = len(S_no_nan)
            if ((N2/N) < 0.3):
                return np.array([np.nan, np.nan, np.nan])
                
            params = gev.fit(S)
            # print('S and params shape', S.shape, params)
            return np.array(params)
        # calculateGEVParam(S)
        
        obs_data = self.getObsData(city, param)
        gevParams = xarray.apply_ufunc(
            calculateGEVParam,
            obs_data,#.isel(x=20,y=20),
            input_core_dims=[["year"]],
            output_core_dims=[['param']],
            exclude_dims=set(("year",)),
            vectorize=True,
        )
        return gevParams

    def run_all_calculateGEVObsParam(self):
        for city in self.cities.keys():
            print(city)
            for index_f in [ECIO.CDD, ECIO.CWD, ECIO.Prec95p, ECIO.Prec99p, ECIO.rx1day, ECIO.rx5day, ECIO.r20mm]:
                print(index_f.__name__)
                gevParams = self.calculateGEVObs(city, index_f.__name__)
                nc_filename = city + '_' + index_f.__name__ + '_Obs_GEVparams.nc'
                gevParams.to_netcdf(nc_filename)
                copyfile(nc_filename, os.path.join(self.GEVparamDir, nc_filename))

    def calculateGEVModelParam(self, city, gcm, rcp, param):
        
        def calculateGEVParam(S):
            S_no_nan = S[~np.isnan(S)]
            N = len(S)
            N2 = len(S_no_nan)
            if ((N2/N) < 0.3):
                return np.array([np.nan, np.nan, np.nan])
            
            S = S[~np.isnan(S)]
            params = gev.fit(S)
            # print('S and params shape', S.shape, params)
            return np.array(params)

        baseline_mod = self.getModelBaselineData(city, param, rcp, gcm)
        if (param in ['CDD', 'CWD', 'r20mm']) and (gcm == 'HadGEM2-ES'):
            baseline_mod[14,:,:] = np.nan

        gevParams = xarray.apply_ufunc(
            calculateGEVParam,
            baseline_mod,#.isel(x=20,y=20),
            input_core_dims=[["time"]],
            output_core_dims=[['param']],
            exclude_dims=set(("time",)),
            vectorize=True,
        )
        return gevParams



    def run_all_calculateGEVModelParam(self):
        for city in self.cities.keys():
            for gcm in self.GCMs:
                for rcp in ['rcp45', 'rcp85']:
                    print(city, gcm, rcp)
                    for index_f in [ECIO.CDD, ECIO.CWD, ECIO.Prec95p, ECIO.Prec99p, ECIO.rx1day, ECIO.rx5day, ECIO.r20mm]:
                        print(index_f.__name__)
                        gevModParams = self.calculateGEVModelParam(city=city, gcm=gcm, rcp=rcp, param=index_f.__name__)
                        nc_filename = city + '_' + index_f.__name__ + '_' + rcp+gcm + '_ModGEVparams.nc'
                        gevModParams.to_netcdf(nc_filename)
                        copyfile(nc_filename, os.path.join(self.GEVparamDir, nc_filename))
    
    def getGEVParams(self, city, gcm, rcp, param):
        if gcm in ['Obs', 'obs', 'observation']:
            nc_filename = city + '_' + param + '_Obs_GEVparams.nc'
            varname = 'precip'
        else:
            nc_filename = city + '_' + param + '_' + rcp+gcm + '_ModGEVparams.nc'
            varname = self.paramToGCMParamDict[param]
        with xarray.open_dataset(os.path.join(self.GEVparamDir, nc_filename)) as dataset:
            data_params = dataset[varname]

        return data_params

    def getInverseCDF(self, city, gcm, param, rcp = ''):
        def calculateInverseCDF(params):
            # print('params', params.shape)
            # print(S.shape)
            # max_val = S.max()
            q= np.linspace(0,1, 1000)
            params_no_nan = params[~np.isnan(params)]
            N = len(params)
            N2 = len(params_no_nan)
            if ((N2/N) < 0.3):
                out = np.empty(len(q))
                out[:] = np.nan
                return out
            # S = S[~np.isnan(S)]
            inverse_cdf = gev.ppf(q, params[0], params[1], params[2])
            return inverse_cdf

        GEVparams = self.getGEVParams(city, gcm, rcp, param)
        # if gcm in ['obs', 'Obs', 'observation']:
        #     dataa = self.getObsData(city, param)
        #     yot = 'year'
            
        # else:
        #     dataa = self.getModelBaselineData( city, param, rcp, gcm)
        #     yot = 'time'
        inverse_cdf_obs = xarray.apply_ufunc(
            calculateInverseCDF,
            GEVparams,
            input_core_dims= [['param']],
            output_core_dims=[['inverse_cdf']],
            exclude_dims=set(('param',)),
            vectorize=True,
        )
        return inverse_cdf_obs

    def getCDF(self, city, gcm,param, rcp='' ):
                
        def calculateGEVParam(S,params):
            # print('params', params.shape)
            # print(S.shape)
            max_val = S.max()
            x= np.linspace(0,max_val, 1000)
            S_no_nan = S[~np.isnan(S)]
            N = len(S)
            N2 = len(S_no_nan)
            if ((N2/N) < 0.3):
                out = np.empty(len(x))
                out[:] = np.nan
                return out
            S = S[~np.isnan(S)]
            cdf = gev.cdf(x, params[0], params[1], params[2])
            return cdf
        
        GEVparams = self.getGEVParams(city, gcm, rcp, param)
        if gcm in ['obs', 'Obs', 'observation']:
            dataa = self.getObsData(city, param)
            yot = 'year'
            
        else:
            dataa = self.getModelBaselineData( city, param, rcp, gcm)
            yot = 'time'

        cdf_obs = xarray.apply_ufunc(
            calculateGEVParam,
            dataa,
            GEVparams,
            input_core_dims= [[yot], ['param']],
            output_core_dims=[['cdf']],
            exclude_dims=set((yot,)),
            vectorize=True,
        )
        # cdcdf_obsf['param'] = np.linspace(0,max_val, 1000)
        return cdf_obs

    def generatePolyfitParamsR2(self, city, gcm, param, rcp):

        def polyfitfunct(X, Y, threshold):
            deg = 3
            X_no_nan = X[~np.isnan(X)]
            N = len(X)
            N2 = len(X_no_nan)
            if ((N2/N) < 0.3):
                # print('X nan', N2/N)
                out = np.empty(deg + 2)
                out[:] = np.nan
                return out
            
            Y_no_nan = Y[~np.isnan(Y)]
            N = len(Y)
            N2 = len(Y_no_nan)
            if ((N2/N) < 0.3):
                # print('Y nan', N2/N)
                out = np.empty(deg + 2)
                out[:] = np.nan
                return out

            lc = np.where(np.logical_or((np.logical_or( np.isneginf(X) , X<=0)), np.logical_or( Y<=0, np.isneginf(Y) )))
            if lc[0].size == 0:
                li = 0
            else:
                li = lc[0].max() + 1
            rc = np.where(np.logical_or( np.isposinf(X), np.isposinf(Y) ))
            if param in ['CDD', 'CWD','r20mm']:
                rc = np.where( np.logical_or(X>365, Y>365) )
            else:
                rc = np.where(np.logical_or( X>2*threshold, Y>2*threshold )) 
            if rc[0].size == 0:
                ri = len(X)
            else:
                ri = rc[0].min() 
            Xnew = X[li:ri]
            Ynew = Y[li:ri]
            # print('X ',Xnew.min(), Xnew.max())
            # print('Y ', Ynew.min(), Ynew.max())
            # for i in range(len(Xnew)):
                # print(Xnew[i], Ynew[i])
            
            def fit_func(x, a, b, c):
                # Curve fitting function
                return a * x**3 + b * x**2 + c * x  # d=0 is implied

            params = curve_fit(fit_func, Xnew, Ynew)
            [a, b, c] = params[0]

            # params = np.polyfit(Xnew, Ynew, deg=deg)
            params = np.array([a,b,c,0])
            z = np.poly1d(params)
            # z = np.poly1d(params)
            r_squared = 1 - (  sum((Ynew-z(Xnew))**2) / ( sum( (Ynew-Ynew.mean())**2) )  )
            # print(r_squared)
            # polyparams = np.array( [a, b, c])
            return np.append(params ,r_squared)

        obs_inv_cdf = self.getInverseCDF(city, 'obs', param, rcp)
        mod_inv_cdf = self.getInverseCDF(city, gcm, param, rcp)
        obs_data = self.getObsData(city, param)
        mod_data = self.getModelBaselineData( city, param, rcp, gcm)
        tho = obs_data.max().values
        thm = mod_data.max().values
        if tho > thm:
            threshold = tho
        else:
            threshold = thm

        # print('threshold :', threshold)
        polyParams =xarray.apply_ufunc(
            polyfitfunct,
            mod_inv_cdf,
            obs_inv_cdf,
            threshold,
            input_core_dims= [['inverse_cdf'], ['inverse_cdf'], []],
            output_core_dims=[['polynomParamsR2']],
            exclude_dims=set(('inverse_cdf',)),
            vectorize=True,
        )

        return polyParams
    
    def run_all_generatePolyfitParamsR2(self):
        self.cities ={
            'Banjarmasin': [-3.26,  -3.37, 114.54 , 114.65],
            'Pangkalpinang': [-2.07,  -2.16, 106.06, 106.18],
            # 'Ternate1' : [0.921, 0.747, 127.288, 127.395],
            # 'Ternate2' : [0.482, 0.431, 127.38, 127.441],
            # 'Ternate3' : [1.354, 1.279, 126.356, 126.417],
            # 'Ternate4' : [0.99, 0.955, 126.126, 126.163],
            # # 'Ternate': [1.36, 0.43, 126.12, 127.44],
            # 'Bandar Lampung': [-5.33, -5.53, 105.18, 105.35],
            # 'Mataram':[-8.55,  -8.62 ,116.06, 116.16],
            # 'Samarinda': [0.71, 0.3, 117.04, 117.31],            
            # 'Pekanbaru': [0.61, 0.41, 101.36, 101.52],
            # 'Gorontalo': [0.6, 0.5, 123.0, 123.08],
            # 'Cirebon': [-6.68, -6.8, 108.51, 108.59],
            # 'Kupang': [-10.12, -10.22, 123.54, 123.68]
        }

        for city in self.cities.keys():
            for gcm in self.GCMs:
                for rcp in ['rcp45', 'rcp85']:
                    print(city, gcm, rcp)
                    for index_f in [ECIO.CDD, ECIO.CWD, ECIO.Prec95p, ECIO.Prec99p, ECIO.rx1day, ECIO.rx5day, ECIO.r20mm]:
                        print(index_f.__name__)
                        gevModParams = self.generatePolyfitParamsR2(city=city, gcm=gcm, rcp=rcp, param=index_f.__name__)
                        nc_filename = city + '_' + index_f.__name__ + '_' + rcp+gcm + '_PolynomParamsR2.nc'
                        gevModParams.to_netcdf(nc_filename)
                        copyfile(nc_filename, os.path.join(self.polynomParamsDir, nc_filename))
                
    def getPolynomParamsR2(self, city, gcm, rcp, param):

        nc_filename = city + '_' + param + '_' + rcp+gcm + '_PolynomParamsR2.nc'
        varname = '__xarray_dataarray_variable__'
        with xarray.open_dataset(os.path.join(self.polynomParamsDir, nc_filename)) as dataset:
            data_params = dataset[varname]

        return data_params

    
    def ExtremeDownscalingRun(self, city, gcm, rcp, param, data2correct):
        # ECIO = ExtremePrecIndexFunctions()
        # So = SpatialOperations()
        # city = 'Banjarmasin'
        # param = 'Prec99p'
        # gcm = So.GCMs[2]
        # rcp = 'rcp45'
        # data2correct = 'rcp' #### baseline rcp45 or rcp


        polyparams = self.getPolynomParamsR2( city, gcm, rcp, param)

                
        if data2correct == 'baseline':
            cropped_ds = self.getModelBaselineData(city, param, rcp, gcm)
        else:
            modelparam = self.paramToGCMParamDict[param]
            with xarray.open_mfdataset(glob( os.path.join(self.GCMdir, gcm) + '/'+ modelparam+'_yr_'+gcm+'_'+rcp +'*.nc')[0]) as f:
                future_crop = f.isel(time=slice(16,16+50))
            index_data = future_crop[modelparam]

            ref = self.getGridReference(city)
            max_lat, min_lat, min_lon, max_lon = self.cities[city]
            max_lat = max_lat + 0.2
            max_lon = max_lon + 0.2
            min_lat = min_lat - 0.2
            min_lon = min_lon - 0.2

            # mask_lon = (ref.x >= min_lon) & (ref.x <= max_lon)
            # mask_lat = (ref.y >= min_lat) & (ref.y <= max_lat)
            # cropped_ref = ref.where(mask_lon & mask_lat, drop=True)
            # cropped_ref
            if param in ['CDD', 'CWD', 'r20mm']:
                index_data = index_data.astype('timedelta64[D]') / np.timedelta64(1, 'D')
                index_data = index_data.astype('int')
            cropped_ds = index_data.interp(lat=ref.y, lon=ref.x, method="linear")
            cropped_ds = cropped_ds.drop(['lat', 'lon'])

        def correctingData(Sdata, Spolyparams):
            # print(Sdata.shape, Spolyparams.shape)
            deg = 3
            z = np.poly1d(Spolyparams[:-1])
            res = z(Sdata)
            return res
        # dataa = dataa.rename_dims({
        #     'lat':'y',

        # })
        correctedData = xarray.apply_ufunc(
            correctingData,
            cropped_ds.chunk(
                {"x": 2, "y": 2}
            ),
            polyparams,
            input_core_dims= [['time'], ['polynomParamsR2']],
            output_core_dims=[['time']],
            exclude_dims=set(('time',)),
            vectorize=True,
            dask="parallelized",
            output_dtypes=[cropped_ds.dtype],
            output_sizes = {'time':len(cropped_ds.time)}
        )
        return correctedData


    def run_all_ExtremeDownscalingRun(self):
        self.cities ={
            'Banjarmasin': [-3.26,  -3.37, 114.54 , 114.65],
            'Pangkalpinang': [-2.07,  -2.16, 106.06, 106.18],
            # 'Ternate1' : [0.921, 0.747, 127.288, 127.395],
            # 'Ternate2' : [0.482, 0.431, 127.38, 127.441],
            # 'Ternate3' : [1.354, 1.279, 126.356, 126.417],
            # 'Ternate4' : [0.99, 0.955, 126.126, 126.163],
            # # 'Ternate': [1.36, 0.43, 126.12, 127.44],
            # 'Bandar Lampung': [-5.33, -5.53, 105.18, 105.35],
            # 'Mataram':[-8.55,  -8.62 ,116.06, 116.16],
            # 'Samarinda': [0.71, 0.3, 117.04, 117.31],            
            # 'Pekanbaru': [0.61, 0.41, 101.36, 101.52],
            # 'Gorontalo': [0.6, 0.5, 123.0, 123.08],
            # 'Cirebon': [-6.68, -6.8, 108.51, 108.59],
            # 'Kupang': [-10.12, -10.22, 123.54, 123.68]
        }

        for city in self.cities.keys():
            for gcm in self.GCMs:
                for rcp in ['rcp45', 'rcp85']:
                    print(city, gcm, rcp)
                    for index_f in [ECIO.CDD, ECIO.CWD, ECIO.Prec95p, ECIO.Prec99p, ECIO.rx1day, ECIO.rx5day, ECIO.r20mm]:
                        for data2correct in ['baseline', 'rcp']:
                                
                            print(data2correct, index_f.__name__)

                            correctedData = self.ExtremeDownscalingRun(city=city, gcm=gcm, rcp=rcp, param=index_f.__name__, data2correct=data2correct)
                            if data2correct == 'baseline':

                                nc_filename = city + '_' + index_f.__name__ + '_' + rcp+gcm + '_baseline_corrected.nc'
                            else:
                                nc_filename = city + '_' + index_f.__name__ + '_' + rcp+gcm + '_'+ rcp+'_corrected.nc'
                            correctedData.to_netcdf(nc_filename)
                            copyfile(nc_filename, os.path.join(self.correctedDataDir, nc_filename))
    
    def getCorrectedData(self,  city, gcm, rcp, param, data2correct):
        if data2correct == 'baseline':
            nc_filename = city + '_' + param + '_' + rcp+gcm + '_baseline_corrected.nc'
        else:
            nc_filename = city + '_' + param + '_' + rcp+gcm + '_'+ rcp+'_corrected.nc'
        
        varname = '__xarray_dataarray_variable__'
        with xarray.open_dataset(os.path.join(self.correctedDataDir, nc_filename)) as dataset:
            data_params = dataset[varname]

        return data_params
        ##### plot the result (?)
    
    def generateQualityControl(self, city, gcm, rcp, param, data2correct):
        correctedData = self.getCorrectedData(city, gcm, rcp, param, data2correct)
        def calculate_std(S):
            S_no_nan = S[~np.isnan(S)]
            return S_no_nan.std()

        def neighbours_of(i, j,  len_i, len_j):
            """Positions of neighbours (includes out of bounds but excludes cell itself)."""
            ui = i +1 if i + 1 < len_i else  i
            li = i - 1 if i - 1 >= 0 else  i
            uj = j + 1 if j + 1 < len_j else j
            lj = j - 1 if j - 1 >= 0 else j
            neighbours = list(itertools.product(range(li, ui+1), range(lj, uj + 1)))
            neighbours.remove((i, j))
            return neighbours

        stds = xarray.apply_ufunc(
            calculate_std,
            correctedData, 
            input_core_dims= [['time']],
                # output_core_dims=[['time']],
            exclude_dims=set(('time',)),
            vectorize=True,
        )#.plot()#values.flatten()
        prob_grid = xarray.where(stds>4*stds.mean(), 1, 0 )
        # prob_grid

        xs, ys =  np.meshgrid(range(len(correctedData.x.values)),range(len(correctedData.y.values) ),  )

        xs = xarray.DataArray(xs, dims=['y', 'x'])
        ys = xarray.DataArray(ys, dims=['y', 'x'])
        xs['y'] = correctedData.y.values 
        xs['x'] = correctedData.x.values
        ys['y'] = correctedData.y.values 
        ys['x'] = correctedData.x.values

        time_index = xarray.DataArray(range(len(correctedData.time.values)), dims=['time'])

        def correctingData(Sdata,t_i, x, y):
            # Sdata = QC.isel(y=44, x=26)

            if np.any(Sdata<=0) or (prob_grid.isel(x=x,y=y).values == 1):


                # Sdatanan = Sdata
                # if param in ['CWD', 'CDD', 'r20mm']:
                #     Sdatanan = np.where(Sdata > 365, np.nan, Sdata)
                # Sdatanan = np.where(Sdatanan <= 0, np.nan, Sdatanan)
                # S_no_nan = Sdatanan[~np.isnan(Sdatanan)]
                # # S_no_nan.std()
                # std = S_no_nan.std()
                # mean = S_no_nan.mean()
                # mean_neig_arr = []
                Sdata = np.array(Sdata)
                for t in t_i:
                    neigs = []
                    for i, j in neighbours_of(x, y, len(correctedData.x), len(correctedData.y)):
                        add = True
                        if correctedData.isel(time=t, x=i, y=j).values <=0:
                            add = False
                        if prob_grid.isel(x=i,y=j).values == 1:
                            add = False
                        if add:
                            neigs.append(correctedData.isel(time=t, x=i, y=j).values)

                    arr = np.array(neigs)

                    mean_neig = arr.mean()

                    if param in ['CWD', 'CDD', 'r20mm']:
                        if Sdata[t] >= 365:
                            Sdata[t] = mean_neig
                    if Sdata[t] <= 0 :
                        Sdata[t] = mean_neig
                    
                    if (prob_grid.isel(x=x,y=y).values == 1):
                        cond = False
                        try:
                             cond= ( Sdata[t] > arr.max() ) or (Sdata[t] < arr.min())
                        except ValueError:
                            print('x y t',x, y, t)
                            print(arr)
                            if t - 1 > 0:
                                Sdata[t] = Sdata[t-1] 
                        
                        if cond:
                            Sdata[t] = mean_neig 
                    # mean_neig_arr.append(mean_neig)
                
                # mean_neigh_arr = np.array(mean_neigh_arr)
                # print(std)
                # help(Sdatanan.std)
                
                # Sdata = np.where(Sdata <= 0, mean_neig, Sdata)
                # Sdata = np.where(Sdata > mean + 3*std, mean_neig, Sdata)
                # Sdata = np.where(Sdata < mean - 3*std, mean_neig, Sdata)
            return Sdata

        QC =xarray.apply_ufunc(
            correctingData,
            correctedData,
            time_index,
            xs,
            ys,
            input_core_dims= [['time'],['time'], [], []],
            output_core_dims=[['time']],
            exclude_dims=set(('time',)),
            vectorize=True,
        )
        return QC
    
    def run_all_QualityControl(self):
        self.cities ={
            'Banjarmasin': [-3.26,  -3.37, 114.54 , 114.65],
            'Pangkalpinang': [-2.07,  -2.16, 106.06, 106.18],
            # 'Ternate1' : [0.921, 0.747, 127.288, 127.395],
            # 'Ternate2' : [0.482, 0.431, 127.38, 127.441],
            # 'Ternate3' : [1.354, 1.279, 126.356, 126.417],
            # 'Ternate4' : [0.99, 0.955, 126.126, 126.163],
            # # 'Ternate': [1.36, 0.43, 126.12, 127.44],
            # 'Bandar Lampung': [-5.33, -5.53, 105.18, 105.35],
            # 'Mataram':[-8.55,  -8.62 ,116.06, 116.16],
            # 'Samarinda': [0.71, 0.3, 117.04, 117.31],            
            # 'Pekanbaru': [0.61, 0.41, 101.36, 101.52],
            # 'Gorontalo': [0.6, 0.5, 123.0, 123.08],
            # 'Cirebon': [-6.68, -6.8, 108.51, 108.59],
            # 'Kupang': [-10.12, -10.22, 123.54, 123.68]
        }

        for city in self.cities.keys():
            for gcm in self.GCMs:
                for rcp in ['rcp45', 'rcp85']:
                    print(city, gcm, rcp)
                    for index_f in [ECIO.CDD, ECIO.CWD, ECIO.Prec95p, ECIO.Prec99p, ECIO.rx1day, ECIO.rx5day, ECIO.r20mm]:
                        for data2correct in ['baseline', 'rcp']:
                                
                            print(data2correct, index_f.__name__)

                            QC = self.generateQualityControl(city=city, gcm=gcm, rcp=rcp, param=index_f.__name__, data2correct=data2correct)
                            if data2correct == 'baseline':

                                nc_filename = city + '_' + index_f.__name__ + '_' + rcp+gcm + '_baseline_QC.nc'
                            else:
                                nc_filename = city + '_' + index_f.__name__ + '_' + rcp+gcm + '_'+ rcp+'_QC.nc'
                            QC.to_netcdf(nc_filename)
                            copyfile(nc_filename, os.path.join(self.QCDir, nc_filename))
    
    def getQCData(self, city, gcm, rcp, param, data2correct):
        if data2correct == 'baseline':
            nc_filename = city + '_' + param + '_' + rcp+gcm + '_baseline_QC.nc'
        else:
            nc_filename = city + '_' + param + '_' + rcp+gcm + '_'+ rcp+'_QC.nc'

        varname = '__xarray_dataarray_variable__'
        with xarray.open_dataset(os.path.join(self.QCDir, nc_filename)) as dataset:
            data_params = dataset[varname]

        return data_params

    
    def calculate_trends(self, data_x, var='time'):

        def inner_calculate_trend(S):
        # print(S.shape)
            S_no_nan = S[~np.isnan(S)]
            N = len(S)
            N2 = len(S_no_nan)
            if ((N2/N) < 0.3):
                return np.array([np.nan])
            
            x = np.array(range(len(S)))
            a, b = np.polyfit(x, S, 1)
            return a

        trends = xarray.apply_ufunc(
            inner_calculate_trend,
            data_x,#.isel(x=20,y=20),
            input_core_dims=[[var]],
            # output_core_dims=[['param']],
            exclude_dims=set((var,)),
            vectorize=True,
        )
        return trends


