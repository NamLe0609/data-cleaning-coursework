import pandas as pd
import numpy as np
import re

def cleanCategorical(df, categoricalData):
    for column in categoricalData:
        df[column] = df[column].str.casefold().str.strip()
        df[column] = df[column].str.extract(r'([a-zA-Z0-9\s\-\/,&.]+)')
        df[column] = df[column].fillna('NA')
    return df

# Replaces missing data with the string 'NaN and 
# extract only numerical data while ignoring strings
# also removes commas from numbers
def cleanNum(df, columnName):
    if df[columnName].dtypes != 'float64':
        df[columnName] = df[columnName].replace(',','', regex=True)
        df[columnName] = df[columnName].str.extract(r'([-+]?\d*\.?\d+)').astype(float)
    df[columnName] = df[columnName].fillna(0)
    return df[columnName]

def cleanAllNum(df, numericalData):
    for column in numericalData:
        df[column] = cleanNum(df, column)
    return df

def tbToGBHDD(row):
    return row * 1024 if row <= 8 else row

def cleanHDD(df):
    df['harddisk'] = df["harddisk"].apply(tbToGBHDD)
    return df

def mhzToGhzCPU(row):
    return round(row / 1000, 1) if row > 10 else round(row, 1)

def cleanCPUSpeed(df):
    df['cpu_speed'] = df['cpu_speed'].apply(mhzToGhzCPU)
    return df

def extractColor(row):
    colorMap = {
        r'black|dark (?:metallic|side)|carbon|balck': 'black',
        r'silver|platinum|aluminum|sliver|midnight|mercury': 'silver',
        r'gr[ae]y|gary|lunar|graphite|ash': 'grey',
        r'blue|cobalt|sky|teal': 'blue',
        r'red': 'red',
        r'white|light': 'white',
        r'almond|dune|beige': 'brown',
        r'yellow|gold|apollo': 'yellow',
        r'green|mint|sage|moss': 'green',
        r'pink|electro': 'pink',
    }
    for regex, color in colorMap.items():
        if re.search(regex, row):
            return color
    return 'NA'

def cleanColor(df):
    df['color'] = df['color'].str.split(r'[,/]')
    df = df.explode('color')
    df['color'] = df['color'].str.strip()
    df['color'] = df['color'].apply(extractColor)
    return df

def cleanRam(df):
    df["ram"] = df["ram"].round()
    return df

def standardizeOS(row):
    osMapping = {
        r'windows 10|win 10': 'windows10',
        r'windows 11|win 11': 'windows11',
        r'windows 8|win 8': 'windows8',
        r'windows 7|win 7': 'windows7',
        r'windows': 'windows',
        r'chrome os': 'chromeos',
        r'linux': 'linux',
        r'mac os|macos': 'macos',
    }
    for regex, os in osMapping.items():
        if re.search(regex, row):
            return os
    return 'NA'

def cleanOS(df):
    df['os'] = df['os'].apply(standardizeOS)
    return df

def standardizeFeatures(row):
    sfMapping = {
        r'anti-? ?glare|anti[ -]?(?:gla|reflection)' : 'anti-glare',
        r'backlit|backlight': 'backlit_keyboard',
        r'edge|thin|narrow|bezel': 'thin_bezel', #https://www.tomshardware.com/news/dell-infinityedge-oled-monitors,30854.html https://linustechtips.com/topic/1242078-what-the-hell-is-nanoedge-by-asus/
        r'stylus|pen|stylus': 'stylus',
        r'audio': 'hd_audio',
        r'fingerprint': 'fingerprint_reader',
        r'speakers|stereo': 'stereo_speakers',
        r'wifi & bluetooth': 'wifi_and_bluetooth',
        r'resistant|water|dishwasher': 'water_resistant',
        r'gorilla': 'corning_gorilla_glass',
        r'keypad': 'numeric_keypad',
        r'chiclet': 'chiclet_keyboard',
        r'touch[ -]?screen': 'touch-screen',
        r'multi[ -]?touch': 'multi-touch',
        r'alexa': 'alexa',
        r'light and compact|narrow|space saving|portable': 'lightweight',
        r'information not available|and play on a fast|work|create|high quality|built for entertainment|premium business-class notebook': 'NA',
    }
    updated = set()
    for item in row:
        if item == '':
            continue
        item = item.strip()
        notFound = True
        for regex, feature in sfMapping.items():
            if re.search(regex, item):
                updated.add(feature)
                notFound = False
                break
        if notFound:
            item = item.replace(' ','_')
            updated.add(item)
                
    if 'NA' in updated:
        updated.remove('NA')
        
    return tuple(sorted(updated))
                
def cleanSpecialFeatures(df):
    df['special_features'] = df['special_features'].str.split(',')
    df['special_features'] = df['special_features'].apply(standardizeFeatures)
    return df

def extractGPU(df):
    mask = df['graphics'].isin(['integrated', 'dedicated', 'NA']) == False
    null_mask_graphics_co = df['graphics_coprocessor'].str.contains('NA')
    search_mask = mask & null_mask_graphics_co
    df.loc[search_mask, 'graphics_coprocessor'] = df.loc[search_mask, 'graphics']
    df.loc[mask, 'graphics'] = 'NA'
    return df
    
def standardizeGPU(row):
    gpuExtract = {
        r'(?P<gpuBrand>nvidia[ _]?(?:quadro rtx|quadro|rtx|gtx)?)[ _]?(?:intel)?[ _]?(?P<gpuModel>\d{4}[ _]?(ti)?\s?(?:ada)?|[kpat]\d{4}[m]?|([ktpa]|mx)?\d{3}m?)?', # Nvidia
        r'(?P<gpuBrand>intel[ _]?(?:iris|u?hd))[ _]?(?P<gpuModel>\d{3,4})?', # Intel iris, hd and uhd
        r'^(integrated)?\s?(?P<gpuBrand>intel[ ]?(celeron|arc)?)\s?(integrated|dedicated|(?:processor|integrated)?)?\s?(?P<gpuModel>a\d{3}m)?$', # Intel, ICeleron and IArc
        r'(?P<gpuBrand>amd)\s?(?P<gpuModel>(?:(?:mobility|\s?radeon)+)?\s?(?:(?:\s|wx|rx|vega|pro|r[457]|hd|athlon|silver|integrated|m|gl)+)?\s?(?:\d{1,4}m?)?)', # AMD this also works r'(amd)\s?((?:(?!rtx).)*)'
        r'(?P<gpuBrand>apple)\s?(?P<gpuModel>m1\s?(?:pro)?)?', # Apple
        r'(?P<gpuBrand>mediatek)', # Mediatek
        r'(?P<gpuBrand>arm)\s?(?P<gpuModel>mali-g\d{2}\s?(?:mp3|2ee mc2))', # Arm
    }
    
    useless = r'xps9300-7909slv-pus|inter core i7-8650u'
    
    for regex in gpuExtract:
        if match := re.search(regex, row):
            if match.groupdict().get('gpuModel'):
                return match.group('gpuBrand').strip()+ ' ' + match.group('gpuModel').strip()
            else:
                return match.group('gpuBrand').strip()
    
    if re.search(useless, row):
        return 'NA'
    
    return row

# Set graphics column value based on graphics_coprocessor column. Then remove all 'dedicated' and 'integrated' from 
def fillInGraphics(df):
    df['graphics'] = 'NA'
    integrated = r'integrated|intel|mediatek|powervr|arm|adreno|athlon|mobility|6[18]0m|vega|r4|r5|r7' # https://www.notebookcheck.net/AMD-Radeon-610M-GPU-Benchmarks-and-Specs.654293.0.html
    dedicated = r'dedicated|nvidia|560|rx'
    mask = df['graphics_coprocessor'].str.contains(integrated)
    df.loc[mask, 'graphics'] = 'integrated'
    mask = df['graphics_coprocessor'].str.contains(dedicated)
    df.loc[mask, 'graphics'] = 'dedicated'
    return df

def cleanGPU(df):
    gpuMapping = {
        r'iris x[e]?|intel xe': 'intel iris',
        r'nvidia geforce[r]?|geforce': 'nvidia',
        r'nvidia (?:trx|rtx)|nvidia intel rtx': 'nvidia rtx',
        r'(?<!nvidia)quadro|qn20-m1-r': 'nvidia quadro', # https://forums.lenovo.com/t5/ThinkPad-P-and-W-Series-Mobile-Workstations/NVIDIA-QN20-M1-R/m-p/5165568 and https://www.reddit.com/r/laptops/comments/wxlz4p/anyone_heard_of_a_nvidia_qn20m1r_graphics_card/
        r'\bgt\b': 'gtx',
        r'ati': 'amd', # https://www.networkworld.com/article/735534/data-center-amd-says-goodbye-to-the-ati-brand.html
        r'620u': 'uhd 620',
        r'^t550$': 'nvidia quadro t550',
        r'^t1200$': 'nvidia quadro t1200',
        r'(?<!\w )nvidia t': 'nvidia quadro t',
        r'(?<!\w )rtx': 'nvidia rtx',
        r'(?<!\w )radeon': 'amd radeon',
        r'^nvidia 3050$': 'nvidia rtx 3050',
        r'(?<!\w )uhd': 'intel uhd',
        r'(?<!\w )(?<!u)hd|gt2': 'intel hd', # https://www.techpowerup.com/gpu-specs/intel-haswell-gt2.g591
        r'(?<!rtx\s)a3000': 'rtx a3000',
        r'(?<!apple\s)m1': 'apple m1',
        r'(?<!arm\s)mali': 'arm mali',
        r'(?<!powervr\s)gx6250': 'powervr gx6250',
        r'integrated[ _]?graphics|embedded|intergrated|integreted': 'integrated',
        r'dedicated|integrated[ ,]+dedicated': 'dedicated',
        r' graphic[s]?': '',
        r'^amd radeon 5$': 'amd radeon r5',
        r'^amd radeon 7$': 'amd radeon r7',
    }
    
    df = extractGPU(df)
    
    for regex, gpu in gpuMapping.items():
        df['graphics_coprocessor'] = df['graphics_coprocessor'].str.replace(regex, gpu, regex=True)
    
    df['graphics_coprocessor'] = df['graphics_coprocessor'].apply(standardizeGPU)
    
    df = fillInGraphics(df)
    
    dedicatedIntegratedMapping = {
        r'^dedicated$|^integrated$': 'NA',
        r' integrated| dedicated': '',
    }
    
    for regex, replacement in dedicatedIntegratedMapping.items():
        df['graphics_coprocessor'] = df['graphics_coprocessor'].str.replace(regex, replacement, regex=True)
    
    df[['gpuBrand', 'gpuModel']] = df['graphics_coprocessor'].str.split(n=1, expand=True)
    df['gpuModel'] = df['gpuModel'].fillna('NA')
    df = df.drop(columns=['graphics_coprocessor'], axis = 1)
    
    return df

def standardizeCPU(row):
    cpuBrand = None
    
    cpuExtract = {
        r'(?P<cpuBrand>amd)?\s?(?P<cpuModel>(?:ryzen|(?:[ra]\s|a-)series|athlon|silver|kabini|a4|a10)+(?:(?:\s|[a]?\d{1}|\d{4}|[umxhk]|-)+)?)', # AMD
        r'(?P<cpuBrand>intel)?[ ]?(?P<cpuModel>(?:celeron|core|pentium|atom|xeon|mobile)+[ ]?(?:[imd](?:\d{1})?-?)?[ ]?(?:\d{3,5}[ugxmhktyq]+(?:\d{1})?e?|[nzp](?:\d{4})?|5y10|extreme|2 quad)?)', # Intel
        r'(?P<cpuModel>(?:cortex) (?:a\d{1,2}))', # Arm
        r'(?P<cpuModel>snapdragon)' # Qualcomm
    }
    
    cpuBrandMap = {
        r'ryzen|a[- ]series|athlon|a10|kabini|a4': 'amd',
        r'celeron|core|pentium|atom|xeon|mobile': 'intel',
        r'cortex': 'arm',
        r'snapdragon': 'qualcomm',
    }
    
    for regex in cpuExtract:
        if match := re.search(regex, row):
            if not match.groupdict().get('cpuBrand'):
                for regex, brand in cpuBrandMap.items():
                    if re.search(regex, row):
                        cpuBrand = brand 
                        break
            else:
                cpuBrand = match.group('cpuBrand').strip()
            
            if match.groupdict().get('cpuModel'):
                return cpuBrand + ' ' + match.group('cpuModel').strip().replace('-',' ')
            else:
                return cpuBrand
    
    return row
    

def cleanCPU(df):
    cpuMapping = {
        r'corei7-10750h': 'core i7-10750h',
        r'unknown|others':'NA',
        r' dual-core| cpu| family| other| processor': '',
    }
    
    for regex, replacement in cpuMapping.items():
        df['cpu'] = df['cpu'].str.replace(regex, replacement, regex=True)
        
    df['cpu'] = df['cpu'].apply(standardizeCPU)
    
    df[['cpuBrand', 'cpuModel']] = df['cpu'].str.split(n=1, expand=True)
    df['cpuModel'] = df['cpuModel'].fillna('NA')
    df = df.drop(columns=['cpu'], axis = 1)

    return df

def removeBrandInModel(row, brands):
    for brand in brands:
        if brand in row['model']:
            row['brand'] = brand
            row['model'] = row['model'].replace(brand, '').strip()
            row['model'] = re.sub('  +', ' ', row['model'])
            break
    return row

def cleanModel(df):
    
    
    brands = df['brand'].unique()
    df = df.apply(removeBrandInModel, axis=1, brands=brands)
    return df

def main():
    pd.set_option("display.max_rows", None)
    fileName = 'amazon_laptop_2023.xlsx'

    df = pd.read_excel(fileName)
    df = df.dropna(axis=1, how='all') # Drop any column with all missing data

    # We dont know a computer's model, so cant recommend it
    # IF this was a task analysing computers available on the market then
    # maybe we wont need the model name
    df = df.dropna(axis=0, subset=['model'])
    
    df = df.drop_duplicates(ignore_index=True, keep='first') # Drop rows which are exact duplicates
    # df.drop_duplicates(subset=['model', 'screen_size', 'color', 'harddisk', 'cpu', 'ram'], keep='first', inplace=True)

    # Standardize column names (Like making OS lower case)
    df.columns = df.columns.str.lower().str.strip()
    categoricalData = ['brand', 'model', 'color', 'cpu', 'os', 'special_features', 'graphics', 'graphics_coprocessor']
    df = cleanCategorical(df, categoricalData)

    # Rename columns which has units that needs to be standardized 
    # and to put those into numerical feature list instead of categorical
    numericalData = ['harddisk', 'ram', 'screen_size', 'cpu_speed', 'rating', 'price']
    df = cleanAllNum(df, numericalData)
    
    
    """ df = cleanHDD(df) # Multiply TB values (less than 8) by 1024 to make it GB
    df = cleanCPUSpeed(df) # Divide MHz values (more than 10) by 1000 to make it GHz
    df = cleanRam(df) # Ram is only integer amount. Round in case value is not integer
    df = cleanColor(df) # Clean color to remove non-standard values
    df = cleanOS(df) # Clean OS by simplifying it to OS type, and version
    df = cleanSpecialFeatures(df) # Clean special features by standardising features which are the same
    df = cleanGPU(df) # Clean GPU by standardizing all values and splitting them into gpu brand and gpu model. Also fill graphics column based on co_processor column
    df = cleanCPU(df) # Clean CPU by standardizing all values and splitting them into cpu brand and cpu model
    df = df.drop_duplicates(ignore_index=True, keep='first') # Drop rows which are exact duplicates """
    
    df = cleanModel(df)
    #print(df['model'].value_counts())
    
    df = df.rename(columns={
        "harddisk": "harddisk_gb", 
        "ram": "ram_gb", 
        "screen_size": "screen_size_in", 
        "cpu_speed": "cpu_speed_ghz", 
        "price": "price_dollar"
    })
    df.to_excel('amazon_laptop_2023_cleaned.xlsx')
   
main()
