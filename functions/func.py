###########################################################################################################
######################################## IMPORT PACKAGES ##################################################
###########################################################################################################

# General
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean 
from datetime import datetime
from calendar import monthrange
from tqdm import tqdm 
import requests
from bs4 import BeautifulSoup
import re
from string import digits
import time
import math
import itertools
import platform
import multiprocessing as mp
#import xgboost as xgb

# models 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


#from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.x13 import x13_arima_analysis as x13

###########################################################################################################
######################################## SMALL HELPER FUNCTIONS ###########################################
###########################################################################################################

# Chunkifier for lists:

def chunks(l, n):
    """
    Chunkifies a list into a list of lists (chunks) with length `n`. Last chunk will have length between 1 <= n
       
    Parameters:
    ===========
    l: A list
    n: int that must be larger than or equal to 1
 
    Example:
    ========
        chunkified_list = list(chunks(l=list_example, n = 6))
    """
    for i in range(0, len(l), n):
        yield l[i:i+n]

        
# Reindex a single series/list relative to itself

def reindex(x):
    """
    Reindex a pd.series/list relative to itself
    
    Parameters:
    ===========
    x: pd.series, np.array, list or similar
 
    Example:
    ========
        reindex(x = df['col_name'])
        df.apply(lambda col: reindex(col) if col.name in terms else col) #Applying to a subset of columns in a df defined by terms
    """
    if x.max()>0:
        return ((x / x.max())*100).round(2)
    else:
        return x


# Global id keys for merges

def global_id(geo='all'):
    """
    Common merge function. Returns a pandas dataframe with ID keys for all types of data - including the unique, common ID.
     
    Example:
    ========
        x=global_id()
    """
    if geo.lower()=='all':
        data = {'ID': ['North Denmark', 'Central Denmark', 'Southern Denmark', 'Capital', 'Zealand', # DK
                       'Blekinge', 'Dalarna', 'Gotland', 'Gävleborg','Halland', 'Jämtland', 'Jönköping','Kalmar','Kronoberg',  'Norrbotten', 'Skåne', 'Stockholm', 'Södermanland', 'Uppsala', 'Värmland', 'Västerbotten', 
                       'Västernorrland','Västmanland', 'Västra Götalands','Örebro',  'Östergötland', # SE
                      'Akershus', 'Aust-Agder', 'Buskerud', 'Finnmark', 'Hedmark', 'Hordaland', 'Møre og Romsdal', 'Nord-Trøndelag', 'Nordland', 'Oppland', 'Oslo', 'Østfold', 'Rogaland', 'Sogn og Fjordane',
                       'Sør-Trøndelag', 'Telemark', 'Troms', 'Trøndelag', 'Vest-Agder', 'Vestfold'], # NO

               'trends': ['DK-81', 'DK-82', 'DK-83', 'DK-84', 'DK-85', # DK
                         'SE-K', 'SE-W', 'SE-I', 'SE-X', 'SE-N', 'SE-Z', 'SE-F', 'SE-H', 'SE-G', 'SE-BD', 'SE-M', 'SE-AB', 'SE-D', 'SE-C', 'SE-S', 'SE-AC', 'SE-Y', 'SE-U', 'SE-O', 'SE-T', 'SE-E', # SE
                         'NO-02', 'NO-09', 'NO-06', 'NO-20', 'NO-04', 'NO-12', 'NO-15', 'NO-17', 'NO-18', 'NO-05', 'NO-03', 'NO-01', 'NO-11', 'NO-14', 'NO-16', 'NO-08', 'NO-19', 'NO-50', 'NO-10', 'NO-07'], # NO

               'target#': ['081', '082', '083', '084', '085', # DK
                           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, # SE
                           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], # NO

               'target': ['Region Nordjylland', 'Region Midtjylland', 'Region Syddanmark', 'Region Hovedstaden', 'Region Sjælland', # DK
                          'Blekinge län','Dalarnas län','Gotlands län', 'Gävleborgs län','Hallands län', 'Jämtlands län', 'Jönköpings län', 'Kalmar län', 'Kronobergs län', 'Norrbottens län', 'Skåne län',
                          'Stockholms län', 'Södermanlands län', 'Uppsala län', 'Värmlands län', 'Västerbottens län', 'Västernorrlands län', 'Västmanlands län', 'Västra Götalands län', 'Örebro län', 'Östergötlands län', # SE
                          '02 Akershus', '09 Aust-Agder', '06 Buskerud', '20 Finnmark - Finnmárku', '04 Hedmark',  '12 Hordaland','15 Møre og Romsdal','17 Nord-Trøndelag (-2017)', '18 Nordland','05 Oppland', '03 Oslo', 
                          '01 Østfold', '11 Rogaland', '14 Sogn og Fjordane', '16 Sør-Trøndelag (-2017)', '08 Telemark', '19 Troms - Romsa', '50 Trøndelag', '10 Vest-Agder', '07 Vestfold'], # NO


               'jobindex': ['region-nordjylland', 'region-midtjylland', 'region-syddanmark', 'region-hovedstaden', 'region-sjaelland', # DK
                            'blekinge-laen', 'dalarnas-laen', 'gotlands-laen', 'gaevleborgs-laen', 'hallands-laen', 'jaemtlands-laen',
                            'joenkoepings-laen', 'kalmar-laen', 'kronobergs-laen', 'norrbottens-laen', 'skaane-laen', 'stockholms-laen',
                            'soedermanlands-laen', 'uppsala-laen', 'vaermlands-laen', 'vaesterbottens-laen', 'vaesternorrland-laen',
                            'vaestmanlands-laen', 'vaestra-goetalands-laen', 'oerebro-laen', 'oestergoetlands-laen', # SE
                            'akershus', 'aust-agder', 'buskerud', 'finnmark', 'hedmark', 'hordaland', 'moere-og-romsdal', 'nord-troendelag','nordland', 'oppland', 
                             'oslo', 'oestfold', 'rogaland', 'sogn-og-fjordane', 'soer-troendelag', 'telemark', 'troms', np.nan, 'vest-agder', 'vestfold'] # NO 
               }
        x = pd.DataFrame(data = data)
        return x
    #DK
    elif geo.lower()=='dk':
        data = {'ID': ['North Denmark', 'Central Denmark', 'Southern Denmark', 'Capital', 'Zealand'],

               'trends': ['DK-81', 'DK-82', 'DK-83', 'DK-84', 'DK-85'],

               'target#': ['081', '082', '083', '084', '085'],# NO

               'target#long' : ['1081', '1082', '1083', '1084', '1085'], 

               'target': ['Region Nordjylland', 'Region Midtjylland', 'Region Syddanmark', 'Region Hovedstaden', 'Region Sjælland'],


               'jobindex': ['region-nordjylland', 'region-midtjylland', 'region-syddanmark', 'region-hovedstaden', 'region-sjaelland'] 
               }
        x = pd.DataFrame(data = data)
        return x
    
    #NO
    elif geo.lower()=='no':
        data = {'ID': ['Akershus', 'Aust-Agder', 'Buskerud', 'Finnmark', 'Hedmark', 'Hordaland', 'Møre og Romsdal', 'Nord-Trøndelag', 'Nordland', 'Oppland', 'Oslo', 'Østfold', 'Rogaland', 'Sogn og Fjordane',
                       'Sør-Trøndelag', 'Telemark', 'Troms', 'Trøndelag', 'Vest-Agder', 'Vestfold'], # NO

               'trends': ['NO-02', 'NO-09', 'NO-06', 'NO-20', 'NO-04', 'NO-12', 'NO-15', 'NO-17', 'NO-18', 'NO-05', 'NO-03', 'NO-01', 'NO-11', 'NO-14', 'NO-16', 'NO-08', 'NO-19', np.nan, 'NO-10', 'NO-07'], # NO

               'target#': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], # NO

               'target': ['02 Akershus', '09 Aust-Agder', '06 Buskerud', '20 Finnmark - Finnmárku', '04 Hedmark',  '12 Hordaland','15 Møre og Romsdal','17 Nord-Trøndelag (-2017)', '18 Nordland','05 Oppland', '03 Oslo', 
                          '01 Østfold', '11 Rogaland', '14 Sogn og Fjordane', '16 Sør-Trøndelag (-2017)', '08 Telemark', '19 Troms - Romsa', '50 Trøndelag', '10 Vest-Agder', '07 Vestfold'], # NO


               'jobindex': ['akershus', 'aust-agder', 'buskerud', 'finnmark', 'hedmark', 'hordaland', 'moere-og-romsdal', 'nord-troendelag','nordland', 'oppland', 
                             'oslo', 'oestfold', 'rogaland', 'sogn-og-fjordane', 'soer-troendelag', 'telemark', 'troms', np.nan, 'vest-agder', 'vestfold'] # NO
               }
        x = pd.DataFrame(data = data)
        return x
    #SE
    elif geo.lower()=='se':
        data = {'ID': ['Blekinge', 'Dalarna', 'Gotland', 'Gävleborg','Halland', 'Jämtland', 'Jönköping','Kalmar','Kronoberg',  'Norrbotten', 'Skåne', 'Stockholm', 'Södermanland', 'Uppsala', 'Värmland', 'Västerbotten', 
                       'Västernorrland','Västmanland', 'Västra Götalands','Örebro',  'Östergötland'],

               'trends': ['SE-K', 'SE-W', 'SE-I', 'SE-X', 'SE-N', 'SE-Z', 'SE-F', 'SE-H', 'SE-G', 'SE-BD', 'SE-M', 'SE-AB', 'SE-D', 'SE-C', 'SE-S', 'SE-AC', 'SE-Y', 'SE-U', 'SE-O', 'SE-T', 'SE-E'],

               'target#': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],

               'target': ['Blekinge län','Dalarnas län','Gotlands län', 'Gävleborgs län','Hallands län', 'Jämtlands län', 'Jönköpings län', 'Kalmar län', 'Kronobergs län', 'Norrbottens län', 'Skåne län',
                          'Stockholms län', 'Södermanlands län', 'Uppsala län', 'Värmlands län', 'Västerbottens län', 'Västernorrlands län', 'Västmanlands län', 'Västra Götalands län', 'Örebro län', 'Östergötlands län'],


               'jobindex': ['blekinge-laen', 'dalarnas-laen', 'gotlands-laen', 'gaevleborgs-laen', 'hallands-laen', 'jaemtlands-laen',
                            'joenkoepings-laen', 'kalmar-laen', 'kronobergs-laen', 'norrbottens-laen', 'skaane-laen', 'stockholms-laen',
                            'soedermanlands-laen', 'uppsala-laen', 'vaermlands-laen', 'vaesterbottens-laen', 'vaesternorrland-laen',
                            'vaestmanlands-laen', 'vaestra-goetalands-laen', 'oerebro-laen', 'oestergoetlands-laen'] 
               }
        x = pd.DataFrame(data = data)
        return x
    
    
# Search term list generator

def term_list(geo = 'DK'):
    """
    Returns a list of search terms for the specified country geo-code.
    
    Parameters:
    ===========
    geo: geo-code string for the specific country
    
    Example:
    ========
        term_list(geo = 'NO')
    """
    
    #DK
    if geo == 'DK':
        kontant = ['kontanthjælp + "kontanthjælp sats" + "kontanthjælp satser"', 'cv + "cv eksempel" + "cv skabelon"']

        job = ['job + jobopslag + "job opslag"', 'jobcenter + jobcentre + "job center" + "job centre"',
        '"ledige job" + "ledige jobs" + "ledig stilling" + "ledige stillinger"',
           'ledig + ledighed', 'arbejdsløs + arbejdsløshed']

        jobbank = ['jobindex + "job index"', 'ofir + ofir.dk + "ofir jobportal"',
               'jobnet + "jobnet.dk" + "jobnet cv"']

        akasse = ['akasse + akasser + "a-kasse" + "a-kasser" + "a kasse"', 'dagpenge + "dagpenge regler" + dagpengeregler',
             'dagpengesats + "dagpenge sats" + dagpengesatser + "dagpenge satser"']

        akasse_names = ['akademikernes + "akademikernes a-kasse" + "akademikernes akasse" + "ingeniørernes akasse" + iak', 'ase + "ase a-kasse" + "ase akasse"',
                    'bupl + "Børne- og Ungdomspædagogernes Landsforbund" + "Børne- og Ungdomspædagogernes a-kasse"', '"lærernes a-kasse" + "lærernes a kasse"',
                    'faglig fælles a-kasse + "3f" + "3f a-kasse" + "3f a kasse"', 'foa + "fag og arbejde" + "fag og arbejde a-kasse" + "fag og arbejde a kasse"',
                    'hk + hk a-kasse + "hk a kasse" + "hk danmark"', 'krifa + "krifa a kasse" + "krifa a-kasse"'] 
        
        nye = ['fyret']

        terms = kontant + job + jobbank + akasse + akasse_names + nye
        return terms
    #SE
    elif geo == 'SE':
        
        # helper = ['fottbol']

        job = ['jobb + platsannonser + platsannons', 'arbetsförmedlingen',
        '"lediga platser" + "lediga jobb" + vakans',
          'arbetslös', '"cv exempel" + cv + "cv-mall" + "cv mall"']
        
        kontant = ['försörjningsstöd + socialbidrag + "socialbidrag krav" + "ekonomiskt bistånd"']

        jobbank = ['jobbsafari', 'platsbanken + "platsbanken arbetsförmedlingen"']

        akasse = ['a-kassa + akassa + "a kassa"', 'arbetslöshetsersättningen + ersättning + ersättningsperiode']

        akasse_names = ['unionen + htf + sif + tjänstemannaförbundet + "Svenska industritjänstemannaförbundet"', 'lo + landsorganisationen + "landsorganisationen sverige"',
                       'tco', 'saco'] 
        

        terms = job + kontant + jobbank + akasse + akasse_names
        return terms
    #NO
    elif geo == 'NO':

        # helper = ['fotball']
        
        job = ['jobb + stillingsannose + "stilling annonse"',
        '"ledig jobb" + "ledige jobber" + "ledig stilling" + "ledige stillinger" ',
           'arbeidsledighet', 'cv +"cv eksempel" + "cv mal"']

        kontant = ['sosialhjelp', 'meldekort + "elektronisk meldekort"']

        jobbank = ['jobbsafari', 'arbeidsplassen + "arbeidsplassen nav.no" + "arbeidsplassen nav" + "nav ledige stillinger"', 'finn + "finn no"']

        akasse = ['dagpenger + "arbeids og velferdsetaten" + "dagpenger krav"']

        akasse_names = ['nav + "nav no" + "nav jobb-kontor" + "nav jobb kontor" + "nav jobbkontor"', 'lo + landsorganisasjonen + "landsorganisasjonen norge"',
                       'unio +  sykepleierforbund + utdanningsforbundet + "politiets fellesforbund"', 'ys', 'akademikerne'] 

        terms =  job + kontant + jobbank + akasse + akasse_names
        return terms
    #ALL
#     elif geo = 'ALL'


def GT_dict():
    """
    Returns a dict where to get overview of uinderlying search terms for GT variables across countries.

    Example:
    ========
        GT_dict=GT_dict()
    """  
    result = {'GT_DK_0': 'kontanthjælp + "kontanthjælp sats" + "kontanthjælp satser"',
    'GT_DK_1': 'cv + "cv eksempel" + "cv skabelon"',
    'GT_DK_2': 'job + jobopslag + "job opslag"',
    'GT_DK_3': 'jobcenter + jobcentre + "job center" + "job centre"',
    'GT_DK_4': '"ledige job" + "ledige jobs" + "ledig stilling" + "ledige stillinger"',
    'GT_DK_5': 'ledig + ledighed',
    'GT_DK_6': 'arbejdsløs + arbejdsløshed',
    'GT_DK_7': 'jobindex + "job index"',
    'GT_DK_8': 'ofir + ofir.dk + "ofir jobportal"',
    'GT_DK_9': 'jobnet + "jobnet.dk" + "jobnet cv"',
    'GT_DK_10': 'akasse + akasser + "a-kasse" + "a-kasser" + "a kasse"',
    'GT_DK_11': 'dagpenge + "dagpenge regler" + dagpengeregler',
    'GT_DK_12': 'dagpengesats + "dagpenge sats" + dagpengesatser + "dagpenge satser"',
    'GT_DK_13': 'akademikernes + "akademikernes a-kasse" + "akademikernes akasse" + "ingeniørernes akasse" + iak',
    'GT_DK_14': 'ase + "ase a-kasse" + "ase akasse"',
    'GT_DK_15': 'bupl + "Børne- og Ungdomspædagogernes Landsforbund" + "Børne- og Ungdomspædagogernes a-kasse"',
    'GT_DK_16': '"lærernes a-kasse" + "lærernes a kasse"',
    'GT_DK_17': 'faglig fælles a-kasse + "3f" + "3f a-kasse" + "3f a kasse"',
    'GT_DK_18': 'foa + "fag og arbejde" + "fag og arbejde a-kasse" + "fag og arbejde a kasse"',
    'GT_DK_19': 'hk + hk a-kasse + "hk a kasse" + "hk danmark"',
    'GT_DK_20': 'krifa + "krifa a kasse" + "krifa a-kasse"',
    'GT_DK_21': 'fyret',
    'GT_SE_0': 'jobb + platsannonser + platsannons',
    'GT_SE_1': 'arbetsförmedlingen',
    'GT_SE_2': '"lediga platser" + "lediga jobb" + vakans',
    'GT_SE_3': 'arbetslös',
    'GT_SE_4': '"cv exempel" + cv + "cv-mall" + "cv mall"',
    'GT_SE_5': 'försörjningsstöd + socialbidrag + "socialbidrag krav" + "ekonomiskt bistånd"',
    'GT_SE_6': 'jobbsafari',
    'GT_SE_7': 'platsbanken + "platsbanken arbetsförmedlingen"',
    'GT_SE_8': 'a-kassa + akassa + "a kassa"',
    'GT_SE_9': 'arbetslöshetsersättningen + ersättning + ersättningsperiode',
    'GT_SE_10': 'unionen + htf + sif + tjänstemannaförbundet + "Svenska industritjänstemannaförbundet"',
    'GT_SE_11': 'lo + landsorganisationen + "landsorganisationen sverige"',
    'GT_SE_12': 'tco',
    'GT_SE_13': 'saco',
    'GT_NO_0': 'jobb + stillingsannose + "stilling annonse"',
    'GT_NO_1': '"ledig jobb" + "ledige jobber" + "ledig stilling" + "ledige stillinger" ',
    'GT_NO_2': 'arbeidsledighet',
    'GT_NO_3': 'cv +"cv eksempel" + "cv mal"',
    'GT_NO_4': 'sosialhjelp',
    'GT_NO_5': 'meldekort + "elektronisk meldekort"',
    'GT_NO_6': 'jobbsafari',
    'GT_NO_7': 'arbeidsplassen + "arbeidsplassen nav.no" + "arbeidsplassen nav" + "nav ledige stillinger"',
    'GT_NO_8': 'finn + "finn no"',
    'GT_NO_9': 'dagpenger + "arbeids og velferdsetaten" + "dagpenger krav"',
    'GT_NO_10': 'nav + "nav no" + "nav jobb-kontor" + "nav jobb kontor" + "nav jobbkontor"',
    'GT_NO_11': 'lo + landsorganisasjonen + "landsorganisasjonen norge"',
    'GT_NO_12': 'unio +  sykepleierforbund + utdanningsforbundet + "politiets fellesforbund"',
    'GT_NO_13': 'ys',
    'GT_NO_14': 'akademikerne'} 
    return result 

def create_interaction(df,var1,var2):
    name = var1 + "*" + var2
    df[name] = pd.Series(df[var1] * df[var2], name=name)



        
# Change data format from quarterly to monthly to merge on the existing data

def quarter_to_month(df, variable_names = ['region','date', 'population', 'pop_danish_share'], orig_format = '%YK%m'):
    """
    Change data format from quarterly to monthly to merge on the existing data
    
    Parameters:
    ===========
    df: pandas dataframe
    variable_names: a list of variable names in the dataframe
    orig_format: original date format
    
    Example:
    ========
        quarter_to_month(df = df, variable_names = ['region','date', 'population', 'pop_danish_share'])
    """
    
    #Change data to monthly data with the quarterly figures 
    df= pd.DataFrame(np.repeat(df.values,3, axis=0))
    variable_range = range(len(variable_names))
    df = df.rename(index=str, columns={i:variable_names[i] for i in variable_range})
    df['year'] = pd.to_datetime(df['date'], format=orig_format).dt.to_period('y')
    df['month'] = df.groupby(['ID','year']).cumcount()+1
    df['date'] = pd.to_datetime(df.year.astype(str) + '-' + df.month.astype(str))
    df= df.drop(['year', 'month'], axis = 1)
    return(df)


# Find highest average correlation

def find_highest_corr(df):
    mean_corr = []
    list_corr = []
    variable_lags = ['jobs', 'jobs_lag', 'jobs_lag_2', 'jobs_lag_3', 'jobs_lag_4', 'jobs_lag_5', 'jobs_lag_6', 'jobs_lag_7', 'jobs_lag_8', 'jobs_lag_9', 'jobs_lag_10']
    for variable in variable_lags:
        for geo in df['ID'].unique():
            list_corr.append(df[['target_actual', variable]][df['ID']==geo].corr()['target_actual'][1])
        mean_corr.append(mean(list_corr))

    df_mean_corr = pd.DataFrame(list(zip(variable_lags, mean_corr)), columns = ['lag', 'mean_corr'])
    max_corr = df_mean_corr[df_mean_corr['mean_corr'] == min(df_mean_corr['mean_corr'])].iloc[0].lag
    return(max_corr, df_mean_corr)

# From month to quarter format
def to_quarter(x):
    if x.month ==1: 
        x = x.replace(month = 1)
    elif x.month ==2: 
        x =  x.replace(month = 4)
    elif x.month ==3: 
        x =  x.replace(month = 7)
    else: 
        x =  x.replace(month = 10)
    return(x)


# Create linespace around a number with a given factor
def grid_bestpar(best_par_random, num=200, factor=10):
    """
    Creates a linespace around a number with multiplicative factor. Also appends the number itself to the linespace
    
    Parameters:
    ===========
    best_par_random: float. The parameter which the range is to be constructed around
    num: Number of floats in the linespace
    factor: Multiplier that determines range of the linespace
    
    Example:
    ========
       alphas = grid_bestpar(0.32896, num=25, factor = 2)
    """
    
    #Creates linspace around best parameter
    alphas = np.linspace(1/factor*best_par_random, factor*best_par_random, num=num)
    
    #Append best parameter to linspace
    alphas = np.append(alphas, best_par_random)
    
    return alphas


# List of col names for 
def seasadj_col_list():
    col_list = ['target_actual', 'jobs', 'sector_information_technology',
       'sector_engineering_technology', 'sector_management_staff',
       'sector_trade_service', 'sector_industry_craft',
       'sector_sales_communication', 'sector_teaching',
       'sector_office_finance', 'sector_social_health', 'sector_other', 'GT_0', 'GT_1',
       'GT_2', 'GT_3', 'GT_4', 'GT_5', 'GT_6', 'GT_7', 'GT_8', 'GT_9', 'GT_10',
       'GT_11', 'GT_12', 'GT_13', 'GT_14', 'GT_15', 'GT_16', 'GT_17', 'GT_18',
       'GT_19']
    return col_list


# change certain columns to abs change 
def abs_percentage_change(df, 
                          var_abs_change =  ['target_actual', 'GT_0', 'GT_1', 'GT_2', 'GT_3', 'GT_4', 'GT_5', 'GT_6', 'GT_7', 'GT_8', 'GT_9', 'GT_10', 'GT_11', 'GT_12', 'GT_13', 'GT_14', 'GT_15', 'GT_16', 'GT_17', 'GT_18', 'GT_19',
                                              'target_lag', 'jobs', 'sector_information_technology', 'sector_engineering_technology', 'sector_management_staff', 'sector_trade_service', 'sector_industry_craft', 'sector_sales_communication',
                                              'sector_teaching', 'sector_office_finance', 'sector_social_health', 'sector_other'],
                          var_pct_change =  []
                          ):
    """
    Redefines variable values to abs change or pct changee
    
    Parameters:
    ===========
    df: input dataframe
    var_abs_change: column names of columns to be replaced by absolute changes
    var_pct_change: column names of columns to be replaced by percentage changes
    
    Example:
    ========
       abs_percentage_change(df = df_analysis, 
                           var_abs_change =  ['target_actual', 'GT_0', 'GT_1', 'GT_2', 'GT_3', 'GT_4', 'GT_5', 'GT_6', 'GT_7', 'GT_8', 'GT_9', 'GT_10', 'GT_11', 'GT_12', 'GT_13', 'GT_14', 'GT_15', 'GT_16', 'GT_17', 'GT_18', 'GT_19',
                           'target_lag', 'jobs', 'sector_information_technology', 'sector_engineering_technology', 'sector_management_staff', 'sector_trade_service', 'sector_industry_craft', 'sector_sales_communication', 'sector_teaching',
                           'sector_office_finance', 'sector_social_health', 'sector_other'],
                           var_pct_change =  [],
                          )
    """
    result = pd.DataFrame()
        
    for id_ in list(df.loc[:, df.columns.str.contains('ID_')]):
            
        temp1 = df[df[id_] == 1]
        temp1.sort_values(by = 'date')
            
        #subset data for percentage_change
        df_pct =temp1[var_pct_change]
        df_pct =df_pct.pct_change()
                
        #subset data which should be abs change
        df_abs =temp1[var_abs_change]
        df_abs =df_abs.diff()
                
        #subset data which should not be changed
        var_not_change = list(set(list(df))-set(var_pct_change)-set(var_abs_change))
        df_not_change =temp1[var_not_change]
        
        full = df_not_change.merge(right =df_pct,  how = 'left', left_index=True, right_index=True)
        full = full.merge(right =df_abs,  how = 'left', left_index=True, right_index=True)
        result = pd.concat([result,full])
        
    return result
    

# Add squared terms to df
def add_poly_terms(df,
                   poly_columns = ['target_actual', 'GT_0', 'GT_1', 'GT_2', 'GT_3', 'GT_4', 'GT_5', 'GT_6', 'GT_7', 'GT_8', 'GT_9', 'GT_10', 'GT_11', 'GT_12', 'GT_13', 'GT_14', 'GT_15', 'GT_16', 'GT_17', 'GT_18', 'GT_19', 'target_lag',
                                   'jobs', 'sector_information_technology', 'sector_engineering_technology', 'sector_management_staff', 'sector_trade_service', 'sector_industry_craft', 'sector_sales_communication', 'sector_teaching',
                                   'sector_office_finance', 'sector_social_health', 'sector_other']
                  ):
    
    """
    append squared terms to dataframe
    
    Parameters:
    ===========
    df: input dataframe
    poly_columns: column names of columns which should be included as squared terms as well.
    
    Example:
    ========
    add_poly_terms(df = df_analysis, 
                   poly_columns = ['target_actual', 'GT_0', 'GT_1', 'GT_2', 'GT_3', 'GT_4', 'GT_5', 'GT_6', 'GT_7', 'GT_8', 'GT_9', 'GT_10', 'GT_11', 'GT_12', 'GT_13', 'GT_14', 'GT_15', 'GT_16', 'GT_17', 'GT_18', 'GT_19', 'target_lag',
                   'jobs', 'sector_information_technology', 'sector_engineering_technology', 'sector_management_staff', 'sector_trade_service', 'sector_industry_craft', 'sector_sales_communication', 'sector_teaching',
                   'sector_office_finance', 'sector_social_health', 'sector_other']
                  )
    """
    df_poly =df[poly_columns]
    df_poly = np.square(df_poly)
    df_poly.columns = [str(col) + '^2' for col in df_poly.columns]
    df = df.merge(df_poly, right_index=True, left_index=True)
    
    return df


###########################################################################################################
######################################## GRAPHS AND TABLES ################################################
###########################################################################################################

# List of plots for comparsion of two time series for a single geo
def time_corr_plot(df, var_list, target, geo, ylabel1, ylabel2, GT=True):
    if GT==True:
        GT = GT_dict()
        for term in var_list:
            #fig plot
            fig, ax1 = plt.subplots(figsize = (10,4))

            #Target plot
            color = 'darkred'
            ax1.set_ylabel(ylabel1, color = color)
            ax1.plot(df[[target, 'date']][df['ID']==geo].set_index('date'), color = color)

            #Printing correlation
            ax1.text(x = pd.Timestamp("2006-08-01"), y = df[[target, 'date']][df['ID']==geo][target].max()*1.05 ,
                    s = 'Correlation: ' + str(round(df[[target, term]][df['ID']==geo].corr()[target][1], 2)))    

            #Term plot
            ax2 = ax1.twinx()

            color = 'darkblue'
            ax2.set_ylabel(str(GT[term]) + ylabel2, color = color, fontsize = 8)
            ax2.plot(df[[term, 'date']][df['ID']==geo].set_index('date'), color = color)


            #Adding geo
            plt.title('Geo: ' + str(geo))
    else:
        for term in var_list:
            #fig plot
            fig, ax1 = plt.subplots(figsize = (10,4))

            #Target plot
            color = 'darkred'
            ax1.set_ylabel(ylabel1, color = color)
            ax1.plot(df[[target, 'date']][df['ID']==geo].set_index('date'), color = color)

            #Printing correlation
            ax1.text(x = pd.Timestamp("2006-08-01"), y = df[[target, 'date']][df['ID']==geo][target].max()*1.05 ,
                    s = 'Correlation: ' + str(round(df[[target, term]][df['ID']==geo].corr()[target][1], 2)))    

            #Term plot
            ax2 = ax1.twinx()

            color = 'darkblue'
            ax2.set_ylabel(str(term) + ylabel2, color = color, fontsize = 8)
            ax2.plot(df[[term, 'date']][df['ID']==geo].set_index('date'), color = color)


            #Adding geo
            plt.title('Geo: ' + str(geo))
        
# time variable plot        
def time_variable_plot(data, x, y, hue, x_label, y_label, title, legend_title): 
    plt.figure(figsize=(15,6))
    ax = sns.lineplot(x=x, y=y, hue=hue, data=data)

    # axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(title)

    # legend 
    ax.legend(title = legend_title)


    plt.show()        


###########################################################################################################
######################################## MODELS ###########################################################
###########################################################################################################

# Calculate RMSE

def rmse(y_pred, y_true):
    return np.sqrt(mean_squared_error(y_pred, y_true))


# AR(1) model

# See D'Amuri & Marcucci 2017
# https://stackoverflow.com/questions/54152368/statsmodels-ols-with-rolling-window-problem

def ar_1(df, win, step, dependent):

    #initial variables 
    regions = list(set(list(df.ID)))
    end = math.floor((df.shape[0]-step)/len(regions))
    rng = np.arange(start = win+1, stop = end, step = step) # win +1 as no validation
    result = {}
    j = 0

    for i in rng:
        list_X = []
        list_y_test = []
        list_const = []
        list_beta = []
        list_y_pred = []
        list_y_actual = []
        list_rmse = []
        list_y_hat_low = []
        list_y_hat_high = []
        list_rmse_low = []
        list_rmse_high = []
        list_abs_error = []
        j +=1
        
        #print(i)

        for region in regions:

            df_region = df[df.ID == region].sort_values('date')
            df_region = df_region[['date', dependent]].set_index('date')

            # create window X and test data set
            X = df_region.iloc[:i+1].tail(win+1)
            y_test = df_region.iloc[i+1:i+2]

            # save data frame to result
            list_X.append(X)
            list_y_test.append(y_test)

            # Run AR model from statsmodels
            model = AR(np.array(X))
            model_fit = model.fit(maxlag = 1)
    
            y_hat_conf = model_fit.conf_int(alpha=0.05)
            
            #print(model_fit.params)
            
            #allow for unitroot
            
            if len(list(model_fit.params)) ==1:
                const == np.nan
                beta == np.nan
                
                y_hat_conf = np.nan
                list_y_actual.append(np.nan)
                list_const.append(np.nan)
                list_beta.append(np.nan)
                list_y_pred.append(np.nan)
                list_y_hat_low.append(np.nan)
                list_y_hat_high.append(np.nan)
                list_rmse.append(np.nan) 
                list_rmse_low.append(np.nan) 
                list_rmse_high.append(np.nan)
                list_abs_error.append(np.nan)
                
            else:
                beta = list(model_fit.params)[1]
                const = list(model_fit.params)[0]

                y_hat_conf = model_fit.conf_int(alpha=0.05)
                list_y_actual.append(round(y_test[dependent].iloc[0],2))
                list_const.append(const)
                list_beta.append(beta)
                list_y_pred.append((round(const + X[dependent][-1]* beta,2)))
                list_y_hat_low.append((round( y_hat_conf[0][0] + X[dependent][-1]* y_hat_conf[1][0],2)))
                list_y_hat_high.append((round( y_hat_conf[0][1] + X[dependent][-1]* y_hat_conf[1][1],2)))
                list_rmse.append(rmse([round(const + X[dependent][-1]*beta,2)], [y_test[dependent].iloc[0]])) 
                list_rmse_low.append(rmse([round( y_hat_conf[0][0] + X[dependent][-1]* y_hat_conf[1][0],2)], [y_test[dependent].iloc[0]])) 
                list_rmse_high.append(rmse([round( y_hat_conf[0][1] + X[dependent][-1]* y_hat_conf[1][1],2)], [y_test[dependent].iloc[0]]))
                list_abs_error.append(abs(y_test[dependent].iloc[0]-(round(const + X[dependent][-1]* beta,2))))
                
        result[j] = {}    
        result[j]['X'] = list_X
        result[j]['y_test'] = list_y_test    
        result[j]['y_actual'] = list_y_actual
        result[j]['const'] = list_const
        result[j]['beta'] = list_beta
        result[j]['y_pred'] = list_y_pred
        result[j]['y_pred_low']= list_y_hat_low
        result[j]['y_pred_high']= list_y_hat_high
        result[j]['rmse'] = list_rmse
        result[j]['rmse_low'] = list_rmse_low
        result[j]['rmse_high'] = list_rmse_high
        result[j]['avg_win_rmse'] = rmse(y_pred = list_y_pred, y_true = list_y_actual)
        result[j]['avg_win_rmse_low'] = rmse(y_pred = list_y_hat_low, y_true = list_y_actual)
        result[j]['avg_win_rmse_high'] = rmse(y_pred = list_y_hat_high, y_true = list_y_actual)
        result[j]['max_abs_error'] = np.max(list_abs_error)
        
        
    return(result)



################# SEASONALITY ADJUSTMENT INNER FUNCTION #################
def seasadj(df, col_list):
    import warnings
    warnings.simplefilter('ignore')
    """
    Seasonally adjust cols specified in col_list with x13 ARIMA. Returns a dataframe with date, ID and the adjusted cols.
    May need to merged with unadjusted variables afterwards.
    
    Parameters:
    ===========
    df: Dataframe with cols and a date col
    col_list: Col names that needs to be adjusted
    
    Example:
    ========
       seasadj(df=df_analysis, col_list = col_list)
    """
    #Empty container
    ID_dict = {}
    
    #Splitting df in df for each ID
    for ID in df['ID'].unique():
        ID_dict[ID] = df[df['ID'] == ID].set_index('date').copy()
    
    #For each ID and for each var - seasonally adjust
    for ID in ID_dict.keys():
        for col in col_list:
            try:
                ID_dict[ID][col] = x13(ID_dict[ID][col], x12path='x13as/x13as.exe').seasadj.round(2)
            except:
                print('error at '+str(col) + ' ' + str(ID))
                ID_dict[ID][col] = ID_dict[ID][col].round(2).copy()
    
    #Concatting
    df_result = pd.concat(ID_dict).reset_index().drop('level_0', axis=1).copy()
    
    #Sorting
    df_result.sort_values(by = ['ID' ,'date'], inplace = True)
    
    #Merging variables from the input dataframe
#     df_result = pd.merge(left=df_result, right = df.drop(col_list, axis=1), how = 'left', on = ['ID', 'date'])
    
    return df_result


######################################## ML MODELS ###########################################################


# Test/train split MONTHLY
def test_train_split(df, window, testsize, valsize, y_col, rolling_window = False, df_output = False, jump = 1, geo = "dk"):
    """
    Creates test/train/val split of a dataframe that preserves the underlying time dimension.
    Splits can either be done by rolling window (rolling_window=True) or expanding window (rolling_window=False).
    Returns six dicts of arrays (or DF's) intented as input for sklearn: X_train, X_val, X_test, y_train, y_val, y_test, where X are features and y is the target.
    
    Parameters:
    ===========
    df: Dataframe that contains only feature columns (X) and target column (y) and a time column (named 'date') that will be removed.
    window: Initial window size for the feature split. Must be `int`>=1
    testsize: Window size for the test split. Determines number of columns in the test arrays. Must be `int`>=1
    valsize: Window size for validation split.
    y_col: Column name for the target column from df. Must be `str`
    rolling_window = False: Determines whether or not the train window size is fixed (rolling) or expanding. Must be `int`
    jump: Determines how many time periods to move the test window for each iteration. Should always be 1(?).
    geo: countries to look at. Take the values, "dk", "se", "no" and all
    
    Example:
    ========
       X_train, X_val, X_test, y_train, y_val, y_test = test_train_split(df = df_analysis, window = 36, testsize=1, valsize = 1,  y_col='target_actual', rolling_window = True, df_output= True)
    """
        
    #PARAMETERS
    start = df['date'].min()
    rolling_start = df['date'].min()
    
    #KEY START
    j=0
    
    
    #EMPTY CONTAINERS
    X_train = {}
    X_val = {}
    X_test = {}
    

    y_train = {}
    y_val = {}
    y_test = {}
    
    test_dates = {}
    
    if rolling_window == False:
        ########### EXPANDING WINDOW ###############

        for i in range(window, len(df['date'].unique()), jump):
            #KEYS
            j+= 1

            #TRAIN
            X_train[j] = df[df['date'] < start + pd.DateOffset(months = i)]
            y_train[j] = df[y_col][df['date'] < start + pd.DateOffset(months = i)]
            
            #VALIDATION
            X_val[j] = df[(start + pd.DateOffset(months = i) <= df['date']) &  (df['date'] < start + pd.DateOffset(months = i + valsize))]
            y_val[j] = df[y_col][(start + pd.DateOffset(months = i) <= df['date']) &  (df['date'] < start + pd.DateOffset(months = i + valsize))]

            #TEST
            X_test[j] = df[(start + pd.DateOffset(months = i+valsize) <= df['date']) &  (df['date'] < start + pd.DateOffset(months = i + valsize + testsize))]
            y_test[j] = df[y_col][(start + pd.DateOffset(months = i + valsize) <= df['date']) &  (df['date'] < start + pd.DateOffset(months = i + valsize + testsize))]
            
            #DATES FOR y_test
            test_dates[j] = X_test[j][['date']].copy()
            
            #DROPPING COLS FROM X_TRAIN AND X_TEST - REMEMBER TO ADD 'date'
            droplist = [y_col, 'date']
            X_train[j] = X_train[j].drop(droplist, axis = 1).copy()
            X_val[j] = X_val[j].drop(droplist, axis = 1).copy()
            X_test[j] = X_test[j].drop(droplist, axis = 1).copy()
            
                
                        
            #ARRAY OUTPUT
            if df_output == False:
                X_train[j] = np.array(X_train[j])
                X_val[j] = np.array(X_val[j])
                X_test[j] = np.array(X_test[j])                
                
                y_train[j] = np.array(y_train[j])
                y_val[j] = np.array(y_val[j])
                y_test[j] = np.array(y_test[j])
            
            #Dropping empty DF's for the tail
            if X_test[j].shape[0] != global_id(geo = geo).shape[0] * testsize:
                del X_train[j]
                del X_val[j]
                del X_test[j]
                del test_dates[j]
                del y_test[j]
                del y_val[j]
    
    else:
        ########### ROLLING WINDOW ###############
        for i in range(window, len(df['date'].unique()), jump):
            #KEYS
            j+= 1

            
            #TRAIN
            X_train[j] = df[(rolling_start <= df['date']) &  (df['date'] < start + pd.DateOffset(months = i))]
            y_train[j] = df[y_col][(rolling_start <= df['date']) &  (df['date'] < start + pd.DateOffset(months = i))]
            
            #VALIDATION
            X_val[j] = df[(start + pd.DateOffset(months = i) <= df['date']) &  (df['date'] < start + pd.DateOffset(months = i + valsize))]
            y_val[j] = df[y_col][(start + pd.DateOffset(months = i) <= df['date']) &  (df['date'] < start + pd.DateOffset(months = i + valsize))]

            #TEST
            X_test[j] = df[(start + pd.DateOffset(months = i+valsize) <= df['date']) &  (df['date'] < start + pd.DateOffset(months = i + valsize + testsize))]
            y_test[j] = df[y_col][(start + pd.DateOffset(months = i + valsize) <= df['date']) &  (df['date'] < start + pd.DateOffset(months = i + valsize + testsize))]
            
            #DATES FOR y_test
            test_dates[j] = X_test[j][['date']].copy()
            
            #DROPPING COLS FROM X_TRAIN AND X_TEST - REMEMBER TO ADD 'date'
            droplist = [y_col, 'date']
            X_train[j] = X_train[j].drop(droplist, axis = 1).copy()
            X_val[j] = X_val[j].drop(droplist, axis = 1).copy()
            X_test[j] = X_test[j].drop(droplist, axis = 1).copy()
                        
            #ARRAY OUTPUT
            if df_output == False:
                X_train[j] = np.array(X_train[j])
                X_val[j] = np.array(X_val[j])
                X_test[j] = np.array(X_test[j])                
                
                y_train[j] = np.array(y_train[j])
                y_val[j] = np.array(y_val[j])
                y_test[j] = np.array(y_test[j])
            
            #UPDATING ROLLING START
            rolling_start += pd.DateOffset(months = jump)
            
            #Dropping empty DF's for the tail
            if X_test[j].shape[0] != global_id(geo = geo).shape[0] * testsize:
                del X_train[j]
                del X_val[j]
                del X_test[j]
                del test_dates[j]
                del y_test[j]
                del y_val[j]

    return X_train, X_val, X_test, y_train, y_val, y_test, test_dates



# Test/train split - QUARTERLY
def test_train_split_Q(df, window, testsize, valsize, y_col, rolling_window = False, df_output = False, jump = 1, geo_count = 43):
    """
    Creates test/train/val split of a dataframe that preserves the underlying time dimension.
    Splits can either be done by rolling window (rolling_window=True) or expanding window (rolling_window=False).
    Returns six dicts of arrays (or DF's) intented as input for sklearn: X_train, X_val, X_test, y_train, y_val, y_test, where X are features and y is the target.
    
    Parameters:
    ===========
    df: Dataframe that contains only feature columns (X) and target column (y) and a time column (named 'date') that will be removed.
    window: Initial window size for the feature split. Must be `int`>=1
    testsize: Window size for the test split. Determines number of columns in the test arrays. Must be `int`>=1
    valsize: Window size for validation split.
    y_col: Column name for the target column from df. Must be `str`
    rolling_window = False: Determines whether or not the train window size is fixed (rolling) or expanding. Must be `int`
    jump: Determines how many time periods to move the test window for each iteration. Should always be 1(?).
    geo: countries to look at. Take the values, "dk", "se", "no" and all
    
    Example:
    ========
       X_train, X_val, X_test, y_train, y_val, y_test = test_train_split(df = df_analysis, window = 36, testsize=1, valsize = 1,  y_col='target_actual', rolling_window = True, df_output= True)
    """
        
    #PARAMETERS
    start = df['date'].min()
    rolling_start = df['date'].min()
    
    #KEY START
    j=0
    
    
    #EMPTY CONTAINERS
    X_train = {}
    X_val = {}
    X_test = {}
    

    y_train = {}
    y_val = {}
    y_test = {}
    
    test_dates = {}
    
    if rolling_window == False:
        ########### EXPANDING WINDOW ###############

        for i in range(window, len(df['date'].unique()), jump):
            #KEYS
            j+= 3

            #TRAIN
            X_train[j] = df[df['date'] < start +  i]
            y_train[j] = df[y_col][df['date'] < start + i]
            
            #VALIDATION
            X_val[j] = df[(start +  i <= df['date']) &  (df['date'] < start + i + valsize)]
            y_val[j] = df[y_col][(start + i <= df['date']) &  (df['date'] < start +  i + valsize)]

            #TEST
            X_test[j] = df[(start +  i+valsize <= df['date']) &  (df['date'] < start +  i + valsize + testsize)]
            y_test[j] = df[y_col][(start +  i + valsize <= df['date']) &  (df['date'] < start +i + valsize + testsize)]
            
            #DATES FOR y_test
            test_dates[j] = X_test[j][['date']].copy()
            
            #DROPPING COLS FROM X_TRAIN AND X_TEST - REMEMBER TO ADD 'date'
            droplist = [y_col, 'date']
            X_train[j] = X_train[j].drop(droplist, axis = 1).copy()
            X_val[j] = X_val[j].drop(droplist, axis = 1).copy()
            X_test[j] = X_test[j].drop(droplist, axis = 1).copy()
            
                
                        
            #ARRAY OUTPUT
            if df_output == False:
                X_train[j] = np.array(X_train[j])
                X_val[j] = np.array(X_val[j])
                X_test[j] = np.array(X_test[j])                
                
                y_train[j] = np.array(y_train[j])
                y_val[j] = np.array(y_val[j])
                y_test[j] = np.array(y_test[j])
            
            #Dropping empty DF's for the tail
            if X_test[j].shape[0] != (len([col for col in df if col.startswith('ID_')]) +1) * testsize:
                del X_train[j]
                del X_val[j]
                del X_test[j]
                del test_dates[j]
                del y_test[j]
                del y_val[j]
    
    else:
        ########### ROLLING WINDOW ###############
        for i in range(window, len(df['date'].unique()), jump):
            #KEYS
            j+= 1

            
            #TRAIN
            X_train[j] = df[(rolling_start <= df['date']) &  (df['date'] < start +  i)]
            y_train[j] = df[y_col][(rolling_start <= df['date']) &  (df['date'] < start + i)]
            
            #VALIDATION
            X_val[j] = df[(start + i <= df['date']) &  (df['date'] < start +  i + valsize)]
            y_val[j] = df[y_col][(start + i <= df['date']) &  (df['date'] < start +  i + valsize)]

            #TEST
            X_test[j] = df[(start +  i+valsize <= df['date']) &  (df['date'] < start + i + valsize + testsize)]
            y_test[j] = df[y_col][(start + i + valsize <= df['date']) &  (df['date'] < start + i + valsize + testsize)]
            
            #DATES FOR y_test
            test_dates[j] = X_test[j][['date']].copy()
            
            #DROPPING COLS FROM X_TRAIN AND X_TEST - REMEMBER TO ADD 'date'
            droplist = [y_col, 'date']
            X_train[j] = X_train[j].drop(droplist, axis = 1).copy()
            X_val[j] = X_val[j].drop(droplist, axis = 1).copy()
            X_test[j] = X_test[j].drop(droplist, axis = 1).copy()
                        
            #ARRAY OUTPUT
            if df_output == False:
                X_train[j] = np.array(X_train[j])
                X_val[j] = np.array(X_val[j])
                X_test[j] = np.array(X_test[j])                
                
                y_train[j] = np.array(y_train[j])
                y_val[j] = np.array(y_val[j])
                y_test[j] = np.array(y_test[j])
            
            #UPDATING ROLLING START
            rolling_start +=  jump
            
            #Dropping empty DF's for the tail
            if X_test[j].shape[0] != geo_count * testsize: #global_id(geo = geo).shape[0] * testsize:
                del X_train[j]
                del X_val[j]
                del X_test[j]
                del test_dates[j]
                del y_test[j]
                del y_val[j]

    return X_train, X_val, X_test, y_train, y_val, y_test, test_dates



##################### Bootstrap splits ######################

def bootstrap_sample(X_train, X_test, y_train, y_test, sample_size = 5, df_output = True):
    """
    Creates a bootstrap sample of the chosen size for one train/test split
    
    Parameters:
    ===========
    X_train, X_test, y_train, y_test: One test/train split (one window)
    sample_size: The number of regions in each sample. This can maximum be the original number of regions
    df_output: If True the function will return the bootstrap sample as a dataframe. Otherwise this will be an numpy array
    
    
    Example:
    ========
       
    """

    #define initial dataframes
    X_train_boot_sample = pd.DataFrame()
    X_test_boot_sample = pd.DataFrame()

    # define geoareas
    geo = [col for col in X_train if col.startswith('ID_')]

    if(sample_size > len(geo)+1):
        raise ValueError('sample_size shoud be lower or equal to number of regions')
    
    geo_boot = np.random.randint(low=0, high=len(geo)+1, size=(sample_size,))
    #print(geo_boot)
    for value in geo_boot:
        # Retrive the correct data for boot element
        if value == max(geo_boot):
            X_train_boot_element = X_train[(X_train[geo].sum(axis=1)== 0)]
            X_test_boot_element = X_test[(X_test[geo].sum(axis=1)== 0)]
        else:
            X_train_boot_element = X_train.loc[X_train[geo[value]] == 1]
            X_test_boot_element = X_test.loc[X_test[geo[value]] == 1]
        # concate dataframe
        X_train_boot_sample = pd.concat([X_train_boot_sample,  X_train_boot_element])
        X_test_boot_sample = pd.concat([X_test_boot_sample,  X_test_boot_element])

        # retrive y_train and y_val by matching index
        index_train = X_train_boot_sample.index.values.tolist()
        index_test = X_test_boot_sample.index.values.tolist()
        
    # Create the y _rain and y_test by merging on hte selected index' from X_train, X_test
    y_train_boot_sample = pd.DataFrame({'index':index_train})
    y_train_boot_sample = y_train_boot_sample.merge(pd.DataFrame(y_train), left_on = 'index', right_index = True, how = 'left').set_index('index')
    y_train_boot_sample = y_train_boot_sample.iloc[:,0]
    
    y_test_boot_sample = pd.DataFrame({'index':index_test})
    y_test_boot_sample = y_test_boot_sample.merge(pd.DataFrame(y_test), left_on = 'index', right_index = True, how = 'left').set_index('index')
    y_test_boot_sample =y_test_boot_sample.iloc[:,0]
        
    #return the four dataframe in the one bootstrap
    if df_output == True:
        return(X_train_boot_sample, X_test_boot_sample, y_train_boot_sample, y_test_boot_sample)
    else:
        return(np.array(X_train_boot_sample), np.array(X_test_boot_sample), np.array(y_train_boot_sample), np.array(y_test_boot_sample))

    
#BOOTSTRAPPING
def bootstrap_n_samples(X_train, X_test, y_train, y_test, sample_size = 1, n_samples = 10, df_output = True):
    """
    Creates n bootstrap samples of the chosen samplesize for one train/test split and returns this as a dictionary with each n samples as keys.
    
    Parameters:
    ===========
    X_train, X_test, y_train, y_test: One test/train split (one window)
    sample_size: The number of regions in each sample. This can maximum be the original number of regions
    n_samples: The number of bootstrap samples for one window
    df_output: If True the function will return the bootstrap sample as a dataframe. Otherwise this will be an numpy array
    
    
    Example:
    ========
       
    """
    
    #define key start
    key = 0

    #define empty dictionaries
    X_train_boot = {}
    X_test_boot = {}
    y_train_boot = {}
    y_test_boot = {}

    # retrive n bootstrap samples 
    for sample in range(n_samples):    
        # create new key
        key+= 1
        X_train_boot_sample,X_test_boot_sample, y_train_boot_sample, y_test_boot_sample = bootstrap_sample(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, sample_size = sample_size, df_output = df_output)

        X_train_boot[key] = X_train_boot_sample
        X_test_boot[key] = X_test_boot_sample                            
        y_train_boot[key] = y_train_boot_sample
        y_test_boot[key] = y_test_boot_sample
    
    return(X_train_boot, X_test_boot, y_train_boot, y_test_boot)


def bootstrap_all_windows(X_train, X_test, y_train, y_test, sample_size = 5, n_samples = 10, df_output = True):
    """
    Creates n bootstrap samples of the chosen samplesize for all train/test split and returns this as a nested dictionary with each window as keys and nested with the n samples as keys.
    
    Parameters:
    ===========
    X_train, X_test, y_train, y_test: The dictionary wit all test/train splits
    sample_size: The number of regions in each sample. This can maximum be the original number of regions
    n_samples: The number of bootstrap samples for one window
    df_output: If True the function will return the bootstrap sample as a dataframe. Otherwise this will be an numpy array
    
    
    Example:
    ========
    X_train, X_val, X_test, y_train, y_val, y_test = test_train_split(df = df_analysis, window = 36, testsize=1, valsize = 1,  y_col='target_actual', rolling_window = False, df_output= True)
    X_train_boot, X_test_boot, y_train_boot, y_test_boot = bootstrap_all_windows(X_train, X_test, y_train, y_test, sample_size = 1, n_samples = 10, df_output = False)   
    """
    
    
    X_train_all = {}
    X_test_all = {}
    y_train_all = {}
    y_test_all = {}
    
    for window in tqdm(X_train.keys(), desc = 'Bootstrap sampling'):

        #retrive dictionary with bootstrap samples for each window
        X_train_boot, X_test_boot, y_train_boot, y_test_boot = bootstrap_n_samples(X_train[window], X_test[window], y_train[window], y_test[window], sample_size = sample_size, n_samples = n_samples,  df_output = df_output)

        #append to key in window
        X_train_all[window] = X_train_boot
        X_test_all[window] = X_test_boot
        y_train_all[window] =  y_train_boot
        y_test_all[window] = y_test_boot
        
    return(X_train_all, X_test_all, y_train_all, y_test_all)



# LASSO / RIDGE / ELASTIC /RF/ XGBOOST TUNING MODEL OVER PARAMS FOR A SINGLE WINDOW

def model_tuning(X_fit, y_fit, X_test, y_test, params, n_components, model_str, round = 2):
    """
    Tunes a Lasso/Ridge/Elastic Net/Random Forest/XGboost model over a list of parameter sets/tuples. Will fit a Lasso/Ridge/Elastic Net/Random Forest/XGboost model if only one set of parameters are passed.
    Returns a dict with the scores for each parameter set, the best score with associated parameters, and the predicted values.
    
    Parameters:
    ===========
    X_fit, y_fit, X_test, y_test: Arrays with fit (training) data and the test (validation) data on which the scores are calculated
    params: 2-dimensional parameter space to tune over
    model_str: Model to be run. Must either be 'lasso', 'ridge' or 'elastic net' or randomforest' or 'xgboost'
    
    Example:
    ========
    results = model_tuning(X_fit=X_train[1], y_fit=y_train[1], X_test=X_val[1], y_test = y_val[1], params=params, model_str = 'LASSO')
    """
    #RMSE
    par_rmse = {}
    
    #Max-error
    par_maxerror = {}
    
    #R2-score
    par_r2 = {}
    
    
    #COMMON
    y_actual_dict = {}
    y_pred_dict = {}
    pca_components = {}
    pca_explainedvariance = {}
    
    #lOOPING OVER PARAMETER SPACE TO FIT MODELS
        #lOOPING OVER PARAMETER SPACE TO FIT MODELS
    if (model_str.lower() == 'ols'):

        #RUNNING THE MODEL
        model_reg = LinearRegression()
        model_reg.fit(X_fit, y_fit)
        
        # Coefficienter
        coef = list(model_reg.coef_)
        intercept = model_reg.intercept_
        result_coef = np.append(intercept,coef)
        
        y_pred = model_reg.predict(X_test)
        y_pred_dict[('OLS')] = y_pred.round(round)

        par_rmse['OLS'] = rmse(y_pred.round(round), y_true= y_test)


        #RETURNING ONLY FOR BEST SCORES
        best_par = min(par_rmse, key = par_rmse.get)
        best_y_pred = y_pred_dict[best_par]


        #RESULTS DICT ITEMS
        results = {}
        results['coef'] = result_coef
        results['par_rmse_dict'] = par_rmse
        results['best'] = best_par, par_rmse[best_par]
        results['y_actual'] = y_test
        results['y_pred_dict'] = y_pred_dict
        results['best_y_pred'] = best_y_pred

        return results
    
    elif (model_str.lower() == 'lasso' or model_str.lower() =='ridge'):

        for n in n_components:
            pca = PCA(n_components = n, random_state=1)
            X_fit_pca = pca.fit_transform(X_fit)
            X_test_pca = pca.transform(X_test)
        
            for alpha in params:

                #Storing PCA INFO
                pca_components[(n, alpha)] = pca.components_
                pca_explainedvariance[(n, alpha)] = pca.explained_variance_ratio_

                #RUNNING THE MODEL
                if model_str.lower() == 'lasso':
                    model_reg = Lasso(alpha = alpha,
                                      max_iter = -1, random_state = 1)
                    model_reg.fit(X_fit_pca, y_fit)

                    y_pred = model_reg.predict(X_test_pca)

                    y_pred_dict[(n, alpha)] = y_pred.round(round)


                    par_rmse[(n, alpha)] = rmse(y_pred.round(round), y_true= y_test)
                    par_maxerror[(n, alpha)] = max_error(y_true = y_test, y_pred = y_pred.round(round))
                    par_r2[(n, alpha)] = r2_score(y_true = y_test, y_pred = y_pred.round(round))

                elif model_str.lower() == 'ridge':
                    model_reg = Ridge(alpha = alpha,
                                      max_iter = -1, random_state = 1)
                    model_reg.fit(X_fit_pca, y_fit)

                    y_pred = model_reg.predict(X_test_pca)
                    y_pred_dict[(n, alpha)] = y_pred.round(round)

                    par_rmse[(n, alpha)] = rmse(y_pred.round(round), y_true= y_test)
                    par_maxerror[(n, alpha)] = max_error(y_true = y_test, y_pred = y_pred.round(round))
                    par_r2[(n, alpha)] = r2_score(y_true = y_test, y_pred = y_pred.round(round))

                else:
                    raise ValueError('Model must either be: "LASSO" or "RIDGE" or "ELASTICNET"')
    elif model_str.lower() == 'elasticnet':
        
        for n in n_components:
            pca = PCA(n_components = n, random_state=1)
            X_fit_pca = pca.fit_transform(X_fit)
            X_test_pca = pca.transform(X_test)
        
            for alpha, ratio in params:

                #Storing PCA INFO
                pca_components[(n, alpha, ratio)] = pca.components_
                pca_explainedvariance[(n, alpha, ratio)] = pca.explained_variance_ratio_

                #RUNNING THE MODEL
                model_reg = ElasticNet(alpha = alpha,
                                    l1_ratio = ratio,
                                    max_iter = -1, 
                                    random_state = 1)
                model_reg.fit(X_fit_pca, y_fit)

                y_pred = model_reg.predict(X_test_pca)
                y_pred_dict[(n, alpha, ratio)] = y_pred.round(round)

                par_rmse[(n, alpha, ratio)] = rmse(y_pred.round(round), y_true= y_test)
                par_maxerror[(n, alpha, ratio)] = max_error(y_true = y_test, y_pred = y_pred.round(round))
                par_r2[(n, alpha, ratio)] = r2_score(y_true = y_test, y_pred = y_pred.round(round))
                
    elif model_str.lower() =='randomforest':
        
        for n in n_components:
            pca = PCA(n_components = n, random_state=1)
            X_fit_pca = pca.fit_transform(X_fit)
            X_test_pca = pca.transform(X_test)
        
            for n_estimators,  max_depth, max_features, min_samples_split, min_samples_leaf in params:
                #PCA
                pca = PCA(n_components = n, random_state=1)
                X_fit_pca = pca.fit_transform(X_fit)
                X_test_pca = pca.transform(X_test)
                
                #Finding number of PCA corresponding to 
                #n_components = len(pca.explained_variance_ratio_)

                #Storing PCA INFO
                pca_components[(n, n_estimators,  max_depth, max_features, min_samples_split, min_samples_leaf)] = pca.components_
                pca_explainedvariance[(n, n_estimators,  max_depth, max_features, min_samples_split, min_samples_leaf)] = pca.explained_variance_ratio_

                #RUNNING THE MODEL
                model_reg = RandomForestRegressor(n_estimators = n_estimators,
                                                max_depth = max_depth,
                                                max_features =  max_features,
                                                min_samples_split = min_samples_split,
                                                min_samples_leaf = min_samples_leaf,
                                                bootstrap=True,
                                                n_jobs=-1,
                                                random_state=1)
                model_reg.fit(X_fit, y_fit)

                y_pred = model_reg.predict(X_test)
                y_pred_dict[(n, n_estimators,  max_depth, max_features, min_samples_split, min_samples_leaf)] = y_pred.round(round)

                par_rmse[(n, n_estimators,  max_depth, max_features, min_samples_split, min_samples_leaf)] = rmse(y_pred.round(round), y_true= y_test)
                par_maxerror[(n, n_estimators,  max_depth, max_features, min_samples_split, min_samples_leaf)] = max_error(y_true = y_test, y_pred = y_pred.round(round))
                par_r2[(n, n_estimators,  max_depth, max_features, min_samples_split, min_samples_leaf)] = r2_score(y_true = y_test, y_pred = y_pred.round(round))
            
    elif(model_str.lower() =='xgboost'):
        
        for n in n_components:
            pca = PCA(n_components = n, random_state=1)
            X_fit_pca = pca.fit_transform(X_fit)
            X_test_pca = pca.transform(X_test)
        
        
            for n_estimators,  max_depth, colsample_bytree, gamma, subsample, min_child_weight in params:
                
                #Finding number of PCA corresponding to 
                #n_components = len(pca.explained_variance_ratio_)

                #Storing PCA INFO
                pca_components[(n, n_estimators,  max_depth, colsample_bytree, gamma, subsample, min_child_weight)] = pca.components_
                pca_explainedvariance[(n, n_estimators,  max_depth, colsample_bytree, gamma, subsample, min_child_weight)] = pca.explained_variance_ratio_
                
                #RUNNING THE MODEL
                model_reg = XGBRegressor(objective ='reg:squarederror', 
                                        colsample_bytree = colsample_bytree, 
                                        learning_rate = 0.1,
                                        max_depth = max_depth,
                                        n_estimators = n_estimators,
                                        gamma = gamma,
                                        subsample = subsample,
                                        min_child_weight = min_child_weight,
                                        n_jobs = -1,
                                        seed = 1)
                
                model_reg.fit(X_fit, y_fit)

                y_pred = model_reg.predict(X_test)
                y_pred_dict[(n, n_estimators,  max_depth, colsample_bytree, gamma, subsample, min_child_weight)] = y_pred.round(round)

                par_rmse[(n, n_estimators,  max_depth, colsample_bytree, gamma, subsample, min_child_weight)] = rmse(y_pred.round(round), y_true= y_test)
                par_maxerror[(n, n_estimators,  max_depth, colsample_bytree, gamma, subsample, min_child_weight)] = max_error(y_true = y_test, y_pred = y_pred.round(round))
                par_r2[(n, n_estimators,  max_depth, colsample_bytree, gamma, subsample, min_child_weight)] = r2_score(y_true = y_test, y_pred = y_pred.round(round))
        
    else:
        raise ValueError('Model must either be: "lasso" or "ridge" or "elasticnet" or "randomforest" or "xgboost"')
    
              

    
    #RMSE
    best_par_rmse = min(par_rmse, key = par_rmse.get)
    best_y_pred_rmse = y_pred_dict[best_par_rmse]
    
    #Identify keys with min value

    min_val = min(par_rmse.values())
    min_keys = [k for k, v in par_rmse.items() if v==min_val]

    if (len(min_keys) > 0) and ((len(min_keys) % 2) == 0): 
        key_number = int(np.floor(len(min_keys)/2)) -1
    else:
        key_number = int(np.floor(len(min_keys)/2))
 
    best_par_rmse = min_keys[key_number]

    best_y_pred_rmse = y_pred_dict[best_par_rmse]  
       

    #Max-error
    best_par_maxerror = min(par_maxerror, key = par_maxerror.get)
    best_y_pred_maxerror = y_pred_dict[best_par_maxerror]
    
    #R2-score
    best_par_r2 = max(par_r2, key = par_r2.get)
    best_y_pred_r2 = y_pred_dict[best_par_r2]     
       
    #RETURNING ONLY FOR BEST SCORES
    best_pca_components = pca_components[best_par_rmse]
    best_pca_explainedvariance = pca_explainedvariance[best_par_rmse]
    
    #RESULTS DICT ITEMS
    results = {}
    
    #RMSE
    # results['par_rmse_dict'] = par_rmse
    results['best_rmse'] = best_par_rmse, par_rmse[best_par_rmse]
    results['best_y_pred_rmse'] = best_y_pred_rmse
    
    #Max-error
    # results['par_maxerror_dict'] = par_maxerror
    results['best_maxerror'] = best_par_maxerror, par_maxerror[best_par_maxerror]
    results['best_y_pred_maxerror'] = best_y_pred_maxerror
    
    #R2-score
    # results['par_r2_dict'] = par_r2
    results['best_r2'] = best_par_r2, par_r2[best_par_r2]
    results['best_y_pred_r2'] = best_y_pred_r2
    
    #COMMON ACROSS SCORES
    # results['y_pred_dict'] = y_pred_dict
    results['y_actual'] = y_test
    results['pca_components'] = best_pca_components
    results['pca_explainedvariance'] = best_pca_explainedvariance
    
    return results

# TUNING ACROSS WINDOWS - INNER LOOPS
def tuning_window(X_fit, y_fit, X_test, y_test, params, n_components, model_str):
    """
    Tunes a model across windows and returns nested dict with results.
    
    Parameters:
    ===========
    X_fit, y_fit, X_test, y_test: Arrays with fit (training) data and the test (validation) data on which the scores are calculated
    params: 2-dimensional parameter space to tune over
    model_str: Model to be run. Must either be 'lasso' or 'ridge'
    
    Example:
    ========
    results_lasso = tuning_window(X_fit = X_train, y_fit = y_train, X_test = X_val, y_test = y_val, params = params, model_str = 'lasso')  
    """
    #CONTAINER WITH RESULTS
    results_dict = {}
    
    #LOOPING OVER EACH WINDOW
    for win in tqdm(X_fit.keys(), desc = 'Tuning params for window'):
        results_dict[win] = model_tuning(X_fit=X_fit[win], y_fit=y_fit[win], X_test=X_test[win], y_test = y_test[win], params=params, n_components = n_components, model_str = model_str)
    
    return results_dict

# TUNING ACROSS WINDOWS - INNER LOOPS - MULTIPROCESSING VERSION
def tuning_window_mp(X_fit, y_fit, X_test, y_test, params, n_components, model_str):
    """
    Tunes a model across windows and returns nested dict with results. Multiprocessing version.
    
    Parameters:
    ===========
    X_fit, y_fit, X_test, y_test: Arrays with fit (training) data and the test (validation) data on which the scores are calculated
    params: 2-dimensional parameter space to tune over
    model_str: Model to be run. Must either be 'lasso' or 'ridge'
    
    Example:
    ========
    results_lasso_mp = tuning_window_mp(X_fit = X_train, y_fit = y_train, X_test = X_val, y_test = y_val, params = params, model_str = 'lasso', func = model_tuning)  
    """
   
    #Iterable to loop over
    iterables = X_fit.keys()
            
    #Use all CPUs
    p = mp.Pool(mp.cpu_count())
            
    #Results dict
    results_mp = {win : p.apply_async(func = model_tuning, args=(X_fit[win], y_fit[win], X_test[win], y_test[win], params, n_components, model_str)).get() for win in tqdm(iterables)}

    p.close()

        
    return results_mp   



# TUNING ACROSS WINDOWS - INNER LOOPS WITH DIFFERENTIATED PARAMS SPACES
def tuning_window_bestpar(X_fit, y_fit, X_test, y_test, results_random, model_str, num=200, factor=10):
    """
    Tunes a model across windows and returns nested dict with results.
    
    Parameters:
    ===========
    X_fit, y_fit, X_test, y_test: Arrays with fit (training) data and the test (validation) data on which the scores are calculated
    results_random: Results dict from a random search from `tuning_window`
    model_str: Model to be run. Must either be 'lasso' or 'ridge' or 'randomforest' or 'xgboost'
    num: Number of floats around the best parameter from the random search
    factor: Multiplier that determines width/range around the best paramter from random search
    
    Example:
    ========
    results_lasso_opt = tuning_window_bestpar(X_fit = X_train, y_fit = y_train, X_test = X_val, y_test = y_val, results_random=results_lasso, model_str = 'lasso', num = 25, factor=2) 
    """
    #CONTAINER WITH RESULTS
    results_dict = {}
    
    #LOOPING OVER EACH WINDOW
    for win in tqdm(X_fit.keys(), desc = 'Tuning best params for window'):
        if model_str.lower() == 'randomforest':
            
            n_components = [0.90]           
            
            n_estimators = results_random[win]['best_rmse'][0][1]
            n_estimators = [n_estimators-8, n_estimators-6, n_estimators-4, n_estimators-2, n_estimators, n_estimators+2, n_estimators+4, n_estimators+6, n_estimators+8]
            
            # -75, -50, -25, result, +25, + 50, + 75
            max_depth = results_random[win]['best_rmse'][0][2]
            if max_depth is not None:
                max_depth = [max_depth-8, max_depth-6, max_depth-4, max_depth-2, max_depth, max_depth+2, max_depth+4, max_depth+6, max_depth+8]
            else: 
                max_depth = [max_depth]
            
            max_features =  [results_random[win]['best_rmse'][0][3]]
            
            #one lower and one higher than the optimal (keep only psitive)
            #min_samples_split = results_random[win]['best'][0][4]
            #min_samples_split = [min_samples_split-1, min_samples_split,min_samples_split+1]
            #min_samples_split = [x for x in min_samples_split if x > 1]
            min_samples_split = [2]
            
            #one lower and one higher than the optimal (keep only psitive)
            #min_samples_leaf = results_random[win]['best'][0][5]
            #min_samples_leaf = [min_samples_leaf-1, min_samples_leaf, min_samples_leaf+1]
            #min_samples_leaf = [x for x in min_samples_leaf if x > 0]
            min_samples_leaf =[1]
            
            d = {'n_compenents' : n_components,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'max_features': max_features,
                'min_samples_split': min_samples_split,
                'min_samples_leaf':min_samples_leaf
                }
            #print(d.values())

            params = list(d.values())
            params = list(itertools.product(*params))
            print('Number of params to be tested: ' + str(len(params)))
            
            print('Parameter values: ')
            print(n_components)
            print(n_estimators)
            print(max_features)
            print(max_depth)
            print(min_samples_split)
            print(min_samples_leaf)
        
        elif model_str.lower() == 'xgboost':
            
            n_components = [0.6,0.7,0.8,0.9]
            
            
            n_estimators = results_random[win]['best_rmse'][0][1]
            n_estimators = [n_estimators-8, n_estimators-6, n_estimators-4, n_estimators-2, n_estimators, n_estimators+2, n_estimators+4, n_estimators+6, n_estimators+8]
            
            max_depth = results_random[win]['best_rmse'][0][2]
            max_depth = [max_depth-8, max_depth-6, max_depth-4, max_depth-2, max_depth, max_depth+2, max_depth+4, max_depth+6, max_depth+8]
            max_depth = [x for x in max_depth if x >0]
            
            #one lower and one higher than the optimal (keep only psitive)
            colsample_bytree = results_random[win]['best_rmse'][0][3]
            colsample_bytree = [colsample_bytree-0.1, colsample_bytree, colsample_bytree+0.1]
            colsample_bytree = [x for x in colsample_bytree if x >= 0]
            colsample_bytree = [x for x in colsample_bytree if x <= 1]
            
            d = {'n_compenents' : n_components,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'colsample_bytree': colsample_bytree
                }
            #print(d.values())

            params = list(d.values())
            params = list(itertools.product(*params))
            print('Number of params to be tested: ' + str(len(params)))
            
            print('Parameter values: ')
            print(n_components)
            print(n_estimators)
            print(max_depth)
            print(colsample_bytree)
            
        
                    
            
        else:
        
            #Creates new grid for best pararmeters in the window
            alphas = grid_bestpar(results_random[win]['best'][0][0], num=num, factor = factor)
            n_components = [0.90]
            params = [(alpha, n) for alpha in alphas for n in n_components]

            if model_str.lower() == 'elasticnet':
                l1_ratio = list(np.arange(0.01,0.99,0.05)) ## 0.0 cannot be included due to a bug in the code
                params = [(alpha, n, ratio) for alpha in alphas for n in n_components for ratio in l1_ratio]

        
        results_dict[win] = model_tuning(X_fit=X_fit[win], y_fit=y_fit[win], X_test=X_test[win], y_test = y_test[win], params=params, model_str = model_str)
    
    return results_dict

#FITTING THE FINAL MODEL FOR EACH WINDOW WITH THE BEST PARAMETERS
def final_model(inner_results, X_fit, y_fit, X_test, y_test, model_str, score='rmse'):
    """
    Fits a model with the best/tuned parameters from `inner_results`
    
    Parameters:
    ===========
    inner_results: Dict with results of inner tuning loop. See `tuning window` func.
    X_fit, y_fit, X_test, y_test: Arrays with fit (training) data and the test (validation) data on which the scores are calculated
    params: 2-dimensional parameter space to tune over
    model_str: Model to be run. Must either be 'lasso' or 'ridge'
    
    Example:
    ========
    results_final = final_model(inner_results=results_lasso, X_fit = X_train, y_fit = y_train, X_test = X_test, y_test = y_test, model_str = 'lasso')
    """
    #CONTAINER
    results_outer = {}
    
    #FITTING FOR EACH WINDOW WITH BEST PARAMETERS FROM inner_results
    for win in tqdm(X_fit.keys()):

        n_components = [inner_results[win]['best_'+score.lower()][0][0]]
        results_outer[win] = model_tuning(X_fit = X_fit[win], y_fit=y_fit[win], X_test = X_test[win], y_test = y_test[win], params=[inner_results[win]['best_'+score.lower()][0][1:]], n_components= n_components, model_str = model_str)
    
    return results_outer


#FITTING OVER BOOTSTRAP TO GET CONFIDENCE INTERVALS
def final_model_boot(inner_results, model_str, sample_size, n_samples, window, testsize, valsize, rolling_window, df):
    """
    Fits all bootstrap sample with the best/tuned parameters from `inner_results`. Used to create confidence intervals on the scores from `final_model
    
    Parameters:
    ===========
    inner_results: Dict with results of inner tuning loop. See `tuning window` func.
    model_str: Model to be run. Must either be 'lasso' or 'ridge'
    sample_size: The number of regions in each sample. This can maximum be the original number of regions
    n_samples: The number of bootstrap samples for one window
    
    ### THESE PARAMETERS MUST BE THE SAMLE AS IN `final_model`:
    window: Initial window size for the feature split. Must be `int`>=1
    testsize: Window size for the test split. Determines number of columns in the test arrays. Must be `int`>=1
    valsize: Window size for validation split.
    rolling_window = False: Determines whether or not the train window size is fixed (rolling) or expanding. Must be `int`
    
    Example:
    ========
    results_boot = final_model_boot(inner_results=results_lasso, model_str='lasso', sample_size=5, n_samples=10, window = window, testsize = testsize, valsize = valsize, rolling_window = rolling_window)
    """
    
    #Resampling as DF's
    X_train, X_val, X_test, y_train, y_val, y_test, y_dates= test_train_split(df = df, window = window, testsize=testsize, valsize = valsize,
                                                                      y_col='target_actual', rolling_window = rolling_window, df_output= True)

    # #Concatting TRAIN and VAL
    for win in X_train.keys():
        X_train[win] = pd.concat([X_train[win], X_val[win]], axis = 0).copy()
        y_train[win] = pd.concat([y_train[win], y_val[win]], axis = 0).copy()

    #BOOTSTRAPPING SAMPLES ACROSS WINDOWS
    X_train_boot, X_test_boot, y_train_boot, y_test_boot = bootstrap_all_windows(X_train=X_train, X_test = X_test, y_train=y_train, y_test=y_test,
                                                                                 sample_size = sample_size, n_samples = n_samples, df_output = False)
    #Standardising for each window's boot sample
    for win in X_train_boot.keys():
        for sample in X_train_boot[win].keys():
            sc = StandardScaler()
            X_train_boot[win][sample] = sc.fit_transform(X_train_boot[win][sample])
            X_test_boot[win][sample] = sc.transform(X_test_boot[win][sample])
    
    #FITTING FINAL MODEL OVER BOOT SAMPLES
    results_boot = {}
    for win in tqdm(X_train_boot.keys(), desc='Fitting over windows'):
        results_boot[win] = {}
        for sample in X_train_boot[win].keys():
            results_boot[win][sample] =  model_tuning(X_fit = X_train_boot[win][sample], y_fit=y_train_boot[win][sample],
                                                        X_test = X_test_boot[win][sample], y_test = y_test_boot[win][sample], params=[inner_results[win]['best'][0]], model_str = model_str)
    return results_boot



def weight_gen(n_models, samples):
    """
    Returns sets of random (from dirichlet) weights that sum to 1 - including corner weights and equal weights.
    
    Parameters:
    n_models: Number of models, i.e. number of weights in each set
    samples: Number of random weight samples to draw
    ===========
        
    Example:
    ========
        weights = weight_gen(5, 10)
    """    
    #seed    
    np.random.seed(1)
    
    #random weights that sum to 1 - we only keep unique sets
    weights = [np.random.dirichlet(np.ones(n_models), size = 1).round(2) for i in range(0, samples + 1)]
    weights = [l[0].tolist() for l in weights]
    weights = list(set(tuple(x) for x in weights))
    
    #Adding corner solution weights
    a = [0 for i in range(1, n_models)]
    a.append(1)
    
    combinations = list(itertools.permutations(a, n_models))
    combinations = list(set(x for x in combinations))
    
    for i in combinations:
        weights.append(i)
    
    #Equal weights
    weights.append(tuple([round(1/n_models,2) for i in range(0, n_models)]))
    
    return weights


###########################################################################################################
######################################## SCRAPE AND IMPORT ################################################
###########################################################################################################

# Jobindex
def retrieve_job_post_data(areas = ['region-hovedstaden', 'region-sjaelland', 'region-midtjylland', 'region-syddanmark', 'region-nordjylland'], 
                           years = [*range(2004, 2020)], months = [*range(1, 13)], base_url = "https://www.jobindex.dk/jobsoegning/", country = 'DK', quarterly = False):
    """
    
    Parameters:
    ===========
    areas: A list of all the requested areas in the same format as the specific area is displayed in the url 
    years:  The period from which the data is retrieved
    months: The months to retrive
    base_url = The base url
    country = 'DK' if you want to retrieve data from jobindex.dk. 'SE' if you want to retrive data from jobsafari.se. 'NO' if you want to retrieve data from jobsafari.no
    
    
    
    Example:
    ========
       retrieve_data_jobindex(areas = ['region-hovedstaden', 'region-sjaelland', 'region-midtjylland', 'region-syddanmark', 'region-nordjylland'],
       years = [*range(2004, 2020)], months = [*range(1, 13)], base_url = "https://www.jobindex.dk/jobsoegning/", country = 'DK')
    """
    
    # define start and end date
    startdate = []
    for year in years:
        for month in months:
            if year == 2019 and month == datetime.now().month:
                break
            if month < 10:
                month = str(0) +str(month)
            startdate.append(str(year)+str(month)+str(0)+str(1))
            
    
    if quarterly == True: 
        months = [3, 6, 9, 12]
        
    enddate = []
    for year in years:
        for month in months:
            lastday = monthrange(year, month)[1]
            if year == 2019 and month == datetime.now().month:
                break
            if month < 10:
                month = str(0) +str(month)
            enddate.append(str(year)+str(month)+str(lastday))

    # define empty lists for later
    date_list = []
    full_area_list = []
    jobs_list = []
    sector_list = []


    for i in tqdm(range(0,len(startdate))): #for each date
        for j in range(0,len(areas)):  #for each municipality

            time.sleep(np.random.uniform(low = 0.5, high = 1.5)) # sleep"

            #create url
            url = base_url + areas[j] +'?jobage=archive&maxdate=' + enddate[i] + '&mindate=' + startdate[i]

             # send request
            r = requests.get(url = url) 
            soup = BeautifulSoup(r.text,'lxml')

            # retrive number of hits 
            try: 
                result = soup.findAll('h2', attrs={'class':'search-title'})[0].get_text()
                
                if country == 'DK':
                    result = result.split(' ')[0]
                    result = int(result.replace('.', ''))
                
                else:
                    result = [int(s) for s in result.split() if s.isdigit()]
                    result =  "".join(str(x) for x in result)

            except: 
                result = np.nan

            # Retrieve municipality
            area_list = []

            try:
                for k in range(len(soup.findAll('div', attrs={'class':'filterarea'})[0].findAll('a', attrs={'class':'title collapsed'}))):
                    area_name = soup.findAll('div', attrs={'class':'filterarea'})[0].findAll('a', attrs={'class':'title collapsed'})[k].get_text()
                    area_hits = int(re.match('.*?([0-9]+)$', area_name).group(1))
                    area_name = area_name.translate(str.maketrans('', '', digits))

                    #append to list 
                    area_list.append((area_name, area_hits))
            except: 
                area_list.append((np.nan, np.nan))

            # append to lists for later use 
            date_list.append(startdate[i])
            full_area_list.append(areas[j])
            jobs_list.append(result)
            sector_list.append([area_list])

    df_jobindex = pd.DataFrame({'date' :  date_list,
                             'area' : full_area_list,
                             'jobs' : jobs_list,
                             'sectors' : sector_list},
                            columns=['date' ,'area' , 'jobs', 'sectors'])
    return(df_jobindex)


# PYTRENDS

def trends_fetch(geo_codes, terms, global_id = global_id(), sleep = 10):
    """
    Returns dataframe with Google Trends indicies for the list of `terms` for a given set of geo-codes.
    
    Parameters:
    ===========
    geo_codes: A dict of geo-codes fetched from `Pytrends.interest_by_region`. geo_codes.keys() must be geo-codes and geo_codes.values() must be the strings.
    terms: List of search terms for Google Trends.
    global_id=global_id(): Merge dataframe with unique, shared ID's.
    Example:
    ========
       dfTrends_DK = trends_fetch(geo_codes=geo_regions_DK, terms=term_list(geo='DK'))
    """
    #IMPORTS
    from pytrends.request import TrendReq #pip install pytrends
    
    
    #Chunkifying the terms
    kw_list = list(chunks(terms, 1))
    print('\n Total number of terms: ' + str(len(terms)) +
          '\n Number of chunks: ' + str(len(kw_list)) +
         '\n Number of queries needed: ' + str(len(kw_list)*len(geo_codes.keys()))
         )
    
        #LOGIN
    proxies = [] #list of https proxies if blocked by google
    hl = 'DK'
    tz = 300 #Timezone - need to figure out DK
    retries = 10
    backoff_factor = 1

    # Login to Google. Only need to run this once, the rest of requests will use the same session.
    pt = TrendReq(hl=hl, tz=tz, timeout=(2, 5), retries=retries, backoff_factor=backoff_factor)
    
        #BUILDING THE PAYLOAD
    #Common parameters and initiating empty containers
    timeframe = 'all'

    #Sleep in seconds between queries
    # sleep = 10 #np.random.uniform(low = 0.5, high = 1.5) # 60 seconds ensures no blockrate limit
    print('\n Estimated time with sleep at ' + str(sleep) + ' seconds is ' + 
          str(round((sleep + 0.8) * len(terms) * len(geo_codes.keys()) * ((1/60)**2), 2)) + ' hours'
         )

    #Empty dict of df's container
    df = {}

    #Terms not fetched due to KeyError:
    error = []
    
        #QUERY LOOP
    for geo in tqdm(geo_codes.keys(), desc = 'Keyterms for each geo-code'):

        #Creating the dimensions of each dataframe by fetching DUMMY
        pt.build_payload(kw_list=['playstation'], timeframe=timeframe, geo = geo)
        df[geo] = pt.interest_over_time()
        df[geo]['geo'] = str(geo)
        df[geo].reset_index(level = 0, inplace = True)

        #Fetching terms for each geo
        for term in kw_list:

            try: #fast sleep before reached limit
                #Payload per geo
                pt.build_payload(kw_list=term, timeframe=timeframe, geo = geo)

                #Randomized sleeper
                time.sleep(sleep)
                #Tempory fetch to be appended
                temp = pt.interest_over_time()
                #Adding geo-code
                temp['geo'] = str(geo)

                #Resetting index
                temp.reset_index(level=0, inplace=True)

                #Appending col to dataframe               
                df[geo][term] = temp[term]

            except requests.ConnectTimeout: #Sleep must be 60 seconds
                print('Got blocked at ' + str(term) + ' for geo: ' + str(geo))

                #Randomized sleeper - long
                time.sleep(60.5)

                #Payload per geo
                pt.build_payload(kw_list=term, timeframe=timeframe, geo = geo)

                #Tempory fetch to be appended
                temp = pt.interest_over_time()
                #Adding geo-code
                temp['geo'] = str(geo)

                #Resetting index
                temp.reset_index(level=0, inplace=True)

                #Appending col to dataframe
                df[geo][term] = temp[term]


            #For KeyErrors (no data on searched term)
            except KeyError:
                print('KeyError for chunk: ' + str(term) + ' for geo: ' + str(geo))
                error.append(term)

                #Add np.nan col to df[geo][term[i]] ? Or just remove from kw_list?

        #CLEANING BEFORE EXPORT
    #Concatting the df's
    dfTrends = pd.concat(df, ignore_index=True, sort = False)
    
    #Data check
    if len(terms)==dfTrends.shape[1]-4:
        print('Number of search terms matches number of columns in the dataframe')
    else:
        print('Something went wrong!')
        
    #Adding ID for DK regions
#     global_id = global_id

    dfTrends = dfTrends.merge(global_id[['ID', 'trends']], how = 'left', left_on = 'geo', right_on = 'trends').copy()
    
    #Removing unneccesary cols
    dfTrends.drop(['playstation', 'isPartial'], axis = 1, inplace = True)

    return dfTrends


def trends_fetch_cross(dates, terms, global_id = global_id(), sleep = 10):
    """
    Returns dataframe with Google Trends indicies for the list of `terms` for a given set of geo-codes.
    
    Parameters:
    ===========
    geo_codes: A dict of geo-codes fetched from `Pytrends.interest_by_region`. geo_codes.keys() must be geo-codes and geo_codes.values() must be the strings.
    terms: List of search terms for Google Trends.
    global_id=global_id(): Merge dataframe with unique, shared ID's.
    Example:
    ========
       dfTrends_DK = trends_fetch(geo_codes=geo_regions_DK, terms=term_list(geo='DK'))
    """
    #IMPORTS
    from pytrends.request import TrendReq #pip install pytrends
    
    
    #Chunkifying the terms
    kw_list = list(chunks(terms, 1))
#     print('\n Total number of terms: ' + str(len(terms)) +
#           '\n Number of chunks: ' + str(len(kw_list)) +
#          '\n Number of queries needed: ' + str(len(kw_list)*len(dates))
#          )
    
        #LOGIN
    proxies = [] #list of https proxies if blocked by google
    hl = 'DK'
    tz = 300 #Timezone - need to figure out DK
    retries = 10
    backoff_factor = 1

    # Login to Google. Only need to run this once, the rest of requests will use the same session.
    pt = TrendReq(hl=hl, tz=tz, timeout=(2, 5), retries=retries, backoff_factor=backoff_factor)
    
        #BUILDING THE PAYLOAD
    #Common parameters and initiating empty containers
#     timeframe = 'all'

    #Sleep in seconds between queries
    # sleep = 10 #np.random.uniform(low = 0.5, high = 1.5) # 60 seconds ensures no blockrate limit
#     print('\n Estimated time with sleep at ' + str(sleep) + ' seconds is ' + 
#           str(round((sleep + 0.8) * len(terms) * len(geo_codes.keys()) * ((1/60)**2), 2)) + ' hours'
#          )

    #Empty dict of df's container
    df = {}

    #Terms not fetched due to KeyError:
    error = []
    
        #QUERY LOOP
    for date in tqdm(dates, desc = 'Keyterms for each date'):

        #Creating the dimensions of each dataframe by fetching DUMMY
        pt.build_payload(kw_list=['playstation'], timeframe=date, geo = 'DK')
        df[date] = pt.interest_by_region(resolution='CITY', inc_low_vol=True, inc_geo_code=True)
        df[date]['date'] = str(date)[0:10]
        df[date].reset_index(level = 0, inplace = True)

        #Fetching terms for each geo
        for term in kw_list:

            try: #fast sleep before reached limit
                #Payload per geo
                pt.build_payload(kw_list=term, timeframe=date, geo = 'DK')

                #Randomized sleeper
                time.sleep(sleep)
                #Tempory fetch to be appended
                temp = pt.interest_by_region(resolution='CITY', inc_low_vol=True, inc_geo_code=True)
                #Adding geo-code
                temp['date'] = str(date)[0:10]

                #Resetting index
                temp.reset_index(level=0, inplace=True)

                #Appending col to dataframe               
                df[date][term] = temp[term]

            except requests.ConnectTimeout: #Sleep must be 60 seconds
                print('Got blocked at ' + str(term) + ' for date: ' + str(date))

                #Randomized sleeper - long
                time.sleep(60.5)

                #Payload per geo
                pt.build_payload(kw_list=term, timeframe=date, geo = 'DK')

                #Tempory fetch to be appended
                temp = pt.interest_by_region(resolution='CITY', inc_low_vol=True, inc_geo_code=True)
                #Adding geo-code
                temp['date'] = str(date)[0:10]

                #Resetting index
                temp.reset_index(level=0, inplace=True)

                #Appending col to dataframe               
                df[date][term] = temp[term]


            #For KeyErrors (no data on searched term)
            except KeyError:
                print('KeyError for chunk: ' + str(term) + ' for date: ' + str(date))
                error.append(term)

                #Add np.nan col to df[geo][term[i]] ? Or just remove from kw_list?
            
            #General error (?)
            except:
                df[date][term] = np.nan

        #CLEANING BEFORE EXPORT
    #Concatting the df's
    dfTrends = pd.concat(df, ignore_index=True, sort = False)
    
    #Data check
    if len(terms)==dfTrends.shape[1]-4:
        print('Number of search terms matches number of columns in the dataframe')
    else:
        print('Something went wrong!')
        
    #Adding ID for DK regions
#     global_id = global_id

#     dfTrends = dfTrends.merge(global_id[['ID', 'trends']], how = 'left', left_on = 'geo', right_on = 'trends').copy()
    
    #Removing unneccesary cols
    dfTrends.drop(['playstation'], axis = 1, inplace = True)

    return dfTrends


######################################## Results ########################################
