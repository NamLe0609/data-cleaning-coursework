import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import re

FILEPATH = 'images/'

# Replaces missing data with the string 'NA and 
# extract only alphanumeric and 'normal' characters
def cleanCategorical(df, categoricalData):
    for column in categoricalData:
        df[column] = df[column].str.casefold().str.strip()
        df[column] = df[column].str.extract(r'([a-zA-Z0-9\s\-\/,&.]+)')
        df[column] = df[column].fillna('NA')
    return df

# Replaces missing data with 0 and 
# extract only numerical data while ignoring strings
# also removes commas from numbers
def cleanNum(df, columnName):
    if df[columnName].dtypes != 'float64':
        df[columnName] = df[columnName].replace(',','', regex=True)
        df[columnName] = df[columnName].str.extract(r'([-+]?\d*\.?\d+)').astype(float)
    df[columnName] = df[columnName].fillna(0) # 0 instead of NaN as it is easier to process
    return df[columnName]

# Apply function to list of numerical data
def cleanAllNum(df, numericalData):
    for column in numericalData:
        df[column] = cleanNum(df, column)
    return df

# If number is below 8, it must be in TB https://techfident.co.uk/how-much-storage-do-i-need-on-my-laptop/
def tbToGBHDD(row):
    return row * 1024 if row <= 8 else row

def cleanHDD(df):
    df['harddisk'] = df["harddisk"].apply(tbToGBHDD)
    return df

# If number is greater than 10 GHz, it must be in MHz https://www.lenovo.com/gb/en/glossary/what-is-processor-speed/
def mhzToGhzCPU(row):
    return round(row / 1000, 1) if row > 10 else round(row, 1)

def cleanCPUSpeed(df):
    df['cpu_speed'] = df['cpu_speed'].apply(mhzToGhzCPU)
    return df

# Standardize the colors to remove things such as 'Darkside of the moon'
def standardizeColor(row):
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

# Split colors by [,] and [/] (which means multiple choice of colors), and explode it into multiple rows
def cleanColor(df):
    df['color'] = df['color'].str.split(r'[,/]')
    df = df.explode('color')
    df['color'] = df['color'].str.strip()
    df['color'] = df['color'].apply(standardizeColor)
    return df

# Round the ram (as they cannot be a decimal number)
def cleanRam(df):
    df["ram"] = df["ram"].round()
    return df

# Remove unnecessary information and only extract OS
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

# Standardize spelling and meaning for standard features
# Then sort them alphabetically and convert it into a tuple of features
def standardizeFeatures(row):
    sfMapping = {
        r'anti-? ?glare|anti[ -]?(?:gla|reflection)' : 'anti-glare',
        r'backlit|backlight': 'backlit keyboard',
        r'edge|thin|narrow|bezel': 'thin bezel', #https://www.tomshardware.com/news/dell-infinityedge-oled-monitors,30854.html https://linustechtips.com/topic/1242078-what-the-hell-is-nanoedge-by-asus/
        r'stylus|pen|stylus': 'stylus',
        r'audio': 'hd audio',
        r'fingerprint': 'fingerprint reader',
        r'speakers|stereo': 'stereo speakers',
        r'wifi & bluetooth': 'wifi and bluetooth',
        r'resistant|water|dishwasher': 'water resistant',
        r'gorilla': 'corning gorilla glass',
        r'keypad': 'numeric keypad',
        r'chiclet': 'chiclet keyboard',
        r'touch[ -]?screen': 'touch-screen',
        r'multi[ -]?touch': 'multi-touch',
        r'alexa': 'alexa',
        r'light and compact|narrow|space saving|portable': 'lightweight',
        r'ruggedized': 'rugged',
        r'2[ -]in[ -]1': '2-in-1',
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
            updated.add(item)
                
    if 'NA' in updated:
        updated.remove('NA')
        
    return tuple(sorted(updated))
    
# Get special features located in model name
def getSpecialFeature(row):
    specialFeaturesPattern = r'detachable 2[ -]in[ -]1|2[ -]in[ -]1|rugged|multi-touch'
    specialFeatures = re.findall(specialFeaturesPattern, row['model'])
    if specialFeatures:
        for feature in specialFeatures:
            row['model'] = row['model'].replace(feature, '').strip()
        row['special_features'] += specialFeatures
    return row
                
def cleanSpecialFeatures(df):
    df['special_features'] = df['special_features'].str.split(',')
    df = df.apply(getSpecialFeature, axis=1)
    df['special_features'] = df['special_features'].apply(standardizeFeatures)
    return df

# Get the GPU in graphics column and move to graphics_coprocessor if its empty
# Uses masks to make it much easier
def extractGPU(df):
    mask = df['graphics'].isin(['integrated', 'dedicated', 'NA']) == False
    null_mask_graphics_co = df['graphics_coprocessor'].str.contains('NA')
    search_mask = mask & null_mask_graphics_co
    df.loc[search_mask, 'graphics_coprocessor'] = df.loc[search_mask, 'graphics']
    df.loc[mask, 'graphics'] = 'NA'
    return df
    
# Extract GPU into gpuBrand and gpuModel using regex
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

# Set graphics column value based on graphics_coprocessor column
def fillInGraphics(df):
    df['graphics'] = 'NA'
    integrated = r'integrated|intel|mediatek|powervr|arm|adreno|athlon|mobility|6[18]0m|vega|r4|r5|r7' # https://www.notebookcheck.net/AMD-Radeon-610M-GPU-Benchmarks-and-Specs.654293.0.html
    dedicated = r'dedicated|nvidia|560|rx'
    mask = df['graphics_coprocessor'].str.contains(integrated)
    df.loc[mask, 'graphics'] = 'integrated'
    mask = df['graphics_coprocessor'].str.contains(dedicated)
    df.loc[mask, 'graphics'] = 'dedicated'
    return df

# Standardize GPU names for easier extraction
# Remove all 'dedicated' and 'integrated' after Graphics column filled in 
# Split the GPU brand and model into two column
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

# Standardize CPU into brand and column
# If no brand was found, infer it based on cpu model 
def standardizeCPU(row):
    cpuBrand = None
    
    cpuExtract = {
        r'(?P<cpuBrand>amd)?\s?(?P<cpuModel>(?:ryzen|(?:[ra]\s|a-)series|athlon|silver|kabini|a4|a10)+(?:(?:\s|[a]?\d{1}|\d{4}|[umxhk]|-)+)?)', # AMD
        r'(?P<cpuBrand>intel)?[ ]?(?P<cpuModel>(?:celeron|core|pentium|atom|xeon|mobile)+[ ]?(?:[imd](?:\d{1})?-?)?[ ]?(?:\d{3,5}[ugxmhktyq]+(?:\d{1})?e?|[nzp](?:\d{4})?|5y10|extreme|2 quad)?)', # Intel
        r'(?P<cpuModel>(?:cortex) (?:a\d{1,2}))', # Arm
        r'(?P<cpuModel>snapdragon)' # Qualcomm
    }
    
    cpuBrandMap = {
        r'ryzen|a[- ]series|athlon|a10|kabini|a4': 'amd', # https://www.amd.com/en/products/specifications/processors
        r'celeron|core|pentium|atom|xeon|mobile': 'intel', # https://ark.intel.com/content/www/us/en/ark.html
        r'cortex': 'arm', # https://www.arm.com/products/silicon-ip-cpu
        r'snapdragon': 'qualcomm', # https://www.qualcomm.com/snapdragon/overview
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
    
# Tidy up CPU for easier extraction
# Then split cpu into cpu brand and model
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

# Move the brand which are actually models to the model column
def moveBrand(row):
    modelInBrand = ['alienware', 'latitude', 'toughbook', 'jtd']
    for model in modelInBrand:
        if model in row['brand'] and model not in row['model']:
            row['model'] = model + ' ' + row['model']
            break
    return row

# Standardize brands which are the same
def cleanBrand(row):
    brandMapping = {
        r'mac': 'apple',
        r'toughbook': 'panasonic',
        r'alienware|latitude': 'dell',
    }
    for regex, brand in brandMapping.items():
        if re.search(regex, row):
            return brand
    return row

# If the brand is in the model name, remove it
def removeBrandInModel(row, brands):
    for brand in brands:
        if brand in row['model']:
            row['brand'] = brand
            row['model'] = row['model'].replace(brand, '').strip()
            break
    return row

# From the model name, infer the brand
def fillBrandFromModel(row):
    brandMapping = {
        r'mac': 'apple',
        r'toughbook': 'panasonic',
        r'alienware|latitude|precision|e6520': 'dell',
        r'zephyrus|fire': 'asus',
    }
    for regex, brand in brandMapping.items():
        if re.search(regex, row['model']):
            row['brand'] = brand
            break
    return row

# Fix typo and remove unnecessary text from model name
def removeUnnecessaryFromModel(df):
    replaceMap = {
        r'lititude': 'latitude',
        r'laptop|newest|flagship|commercial| pc|mobile workstation': '',
    }
    for regex, replace in replaceMap.items():
        df['model'] = df['model'].replace(regex, replace, regex=True)
    
    return df

# Remove double spaces and trailing whitespace
def cleanup(row):
    row = row.strip()
    row = re.sub('  +', ' ', row)
    return row

# Apply all previously mentioned function to the brand nad model column
def cleanModelAndBrand(df):
    df = df.apply(moveBrand, axis=1)
    df['brand'] = df['brand'].apply(cleanBrand)
    brands = df['brand'].unique()
    df = df.apply(removeBrandInModel, axis=1, brands=brands)
    df = df.apply(fillBrandFromModel, axis=1)
    df = removeUnnecessaryFromModel(df)
    df['model'] = df['model'].apply(cleanup)
    return df

# Apply all column cleaning, and do some preprocessing/postprocessing
def cleanData():
    fileName = 'amazon_laptop_2023.xlsx'

    df = pd.read_excel(fileName)
    df = df.dropna(axis=1, how='all') # Drop any column with all missing data

    # We dont know a computer's model, so cant recommend it
    # IF this was a task analysing computers available on the market then
    # maybe we wont need the model name
    df = df.dropna(axis=0, subset=['model'])
    
    df = df.drop_duplicates(ignore_index=True, keep='first') # Drop rows which are exact duplicates

    # Standardize column names (Like making OS lower case)
    df.columns = df.columns.str.lower().str.strip()
    categoricalData = ['brand', 'model', 'color', 'cpu', 'os', 'special_features', 'graphics', 'graphics_coprocessor']
    df = cleanCategorical(df, categoricalData)

    # Rename columns which has units that needs to be standardized 
    # and to put those into numerical feature list instead of categorical
    numericalData = ['harddisk', 'ram', 'screen_size', 'cpu_speed', 'rating', 'price']
    df = cleanAllNum(df, numericalData)
    
    # Clean all columns
    df = cleanHDD(df) # Multiply TB values (less than 8) by 1024 to make it GB
    df = cleanCPUSpeed(df) # Divide MHz values (more than 10) by 1000 to make it GHz
    df = cleanRam(df) # Ram is only integer amount. Round in case value is not integer
    df = cleanColor(df) # Clean color to remove non-standard values
    df = cleanOS(df) # Clean OS by simplifying it to OS type, and version
    df = cleanSpecialFeatures(df) # Clean special features by standardising features which are the same
    df = cleanGPU(df) # Clean GPU by standardizing all values and splitting them into gpu brand and gpu model. Also fill graphics column based on co_processor column
    df = cleanCPU(df) # Clean CPU by standardizing all values and splitting them into cpu brand and cpu model
    df = cleanModelAndBrand(df) # Remove data which doesnt belong in the column. Move then to correct place
    
    # Drop rows which are exact duplicates
    df = df.drop_duplicates(ignore_index=True, keep='first') 
    # Dropping NA and pd.nan in model
    df = df[df['model'] != 'NA']
    df = df.dropna(axis=0, subset=['model'])
    
    # Rename column to make more sense
    df = df.rename(columns={
        "harddisk": "harddisk_gb", 
        "ram": "ram_gb", 
        "screen_size": "screen_size_in", 
        "cpu_speed": "cpu_speed_ghz", 
        "price": "price_dollar"
    })
    
    # Change the order of the columns
    newOrder = ['brand', 'model', 'screen_size_in', 'color', 'harddisk_gb',
             'cpuBrand', 'cpuModel', 'ram_gb', 'os', 'special_features',
             'graphics', 'gpuBrand', 'gpuModel', 'cpu_speed_ghz', 'rating', 'price_dollar']
    df = df[newOrder]
    
    new_data_types = {
        'harddisk_gb': 'int64',
        'ram_gb': 'int64',
    }
    df = df.astype(new_data_types)
    
    name = ['ram_screen_hdd_outlier', 'brand_color_os_pregrouping', 'hdd_prebin']
    plotGraphsClean(df, name)
    
    df = cleanPostVisualize(df)
    
    newOrder = ['brand', 'model', 'screen_size_in', 'color', 'harddisk_gb', 'harddisk_range_gb',
             'cpuBrand', 'cpuModel', 'ram_gb', 'os', 'special_features',
             'graphics', 'gpuBrand', 'gpuModel', 'rating', 'price_dollar']
    df = df[newOrder]
    
    name = ['ram_screen_hdd_nooutlier', 'brand_color_os_postgrouping', 'hdd_postbin']
    plotGraphsClean(df, name)
    
    df.to_excel('amazon_laptop_2023_cleaned.xlsx', index=False)

# Plot data to show and remove outliers
def plotOutlier(laptops, name):
    # Create a figure and subplots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

    # Plot the boxplot for RAM, Screen Size, HDD
    sns.boxplot(data=laptops, x='ram_gb', flierprops={"marker": "x"}, ax=axes[0, 0])
    axes[0, 0].set(xlabel='GB', ylabel='RAM', title='Distribution of RAM')
    axes[0, 0].set_ylabel(axes[0, 0].get_ylabel(), rotation=0, labelpad=10)

    sns.boxplot(data=laptops, x='screen_size_in', flierprops={"marker": "x"}, ax=axes[0, 1])
    axes[0, 1].set(xlabel='In', ylabel='Size', title='Distribution of Screen Size')
    axes[0, 1].set_ylabel(axes[0, 1].get_ylabel(), rotation=0, labelpad=15)

    sns.boxplot(data=laptops, x='harddisk_gb', flierprops={"marker": "x"}, ax=axes[0, 2])
    axes[0, 2].set(xlabel='GB', ylabel='HD', title='Distribution of Hard Disk')
    axes[0, 2].set_ylabel(axes[0, 2].get_ylabel(), rotation=0, labelpad=10)
    
    # Plot histograms for RAM, Screen Size, and HDD in the second row
    sns.histplot(data=laptops, x='ram_gb', ax=axes[1, 0])
    axes[1, 0].set(xlabel='GB', ylabel='Freq')

    sns.histplot(data=laptops, x='screen_size_in', ax=axes[1, 1])
    axes[1, 1].set(xlabel='In', ylabel='Freq')

    sns.histplot(data=laptops, x='harddisk_gb', ax=axes[1, 2])
    axes[1, 2].set(xlabel='GB', ylabel='Freq')
    
    plt.tight_layout()
    plt.savefig(FILEPATH + name + '.png')

# Plot data to group into less parts
def plotGroupCount(laptops, name):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 8))
    
    # Plot countplots for brand, color, OS
    rotation = 45
    label = ['Brand', 'Color', 'Operating System']
    if laptops['brand'].nunique() > 15:
        rotation = 77
        for i in range(len(label)):
            label[i] += ' (More than 10)'
    sns.countplot(data=laptops, x='brand', order=laptops['brand'].value_counts().index, ax=axes[0])
    axes[0].set(xlabel=label[0], ylabel='Count', title='Distribution of Brand')
    axes[0].set_ylabel(axes[0].get_ylabel(), rotation=0, labelpad=20)
    axes[0].set_xlabel(axes[0].get_xlabel(), rotation=0, labelpad=10)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=rotation)
    
    sns.countplot(data=laptops, x='color', order=laptops['color'].value_counts().index, ax=axes[1])
    axes[1].set(xlabel=label[1], ylabel='Count', title='Distribution of Color')
    axes[1].set_ylabel(axes[1].get_ylabel(), rotation=0, labelpad=20)
    axes[1].set_xlabel(axes[1].get_xlabel(), rotation=0, labelpad=20)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
    
    sns.countplot(data=laptops, x='os', order=laptops['os'].value_counts().index, ax=axes[2])
    axes[2].set(xlabel=label[2], ylabel='Count', title='Distribution of Operating System')
    axes[2].set_ylabel(axes[2].get_ylabel(), rotation=0, labelpad=20)
    axes[2].set_xlabel(axes[2].get_xlabel(), rotation=0, labelpad=10)
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(FILEPATH + name + '.png')

# Plot cpu speed to show why to drop it
def plotDropCount(laptops):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    
    laptops['cpu_speed_ghz'] = laptops['cpu_speed_ghz'].fillna(0)
    sns.countplot(data=laptops, x='cpu_speed_ghz', order=laptops['cpu_speed_ghz'].value_counts().index, ax=axes)
    axes.set(xlabel='CPU Speed (GHz)', ylabel='Count', title='Distribution of CPU speed')
    axes.set_ylabel(axes.get_ylabel(), rotation=0, labelpad=20)
    axes.set_xlabel(axes.get_xlabel(), rotation=0, labelpad=20)

    plt.tight_layout()
    plt.savefig(FILEPATH + 'cpu_speed_sparce.png')

# Plot hard disk to show why to bin values
def plotBins(laptops, name):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
        
    col = 'harddisk_gb'
    data = laptops[(laptops['harddisk_gb'] >= 0) & (laptops['harddisk_gb'] <= 2048)]
    if 'harddisk_range_gb' in laptops:
        col = 'harddisk_range_gb'
        laptops[col] = laptops[col].astype('category')
        data = laptops
        
    sns.countplot(data=data, x=col, ax=axes)
    axes.set(xlabel='Hard Disk (GB)', ylabel='Count', title='Distribution of Hard Disk')
    axes.set_ylabel(axes.get_ylabel(), rotation=0, labelpad=20)
    axes.set_xlabel(axes.get_xlabel(), rotation=0, labelpad=20)
    axes.set_xticklabels(axes.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(FILEPATH + name + '.png')

def plotGraphsClean(df, name = ['a', 'b', 'c']):
    sns.set_theme()
    laptops = df
    numericalData = ['screen_size_in', 'harddisk_gb', 'ram_gb', 'rating', 'price_dollar'] #, 'cpu_speed_ghz']
    laptops[numericalData] = laptops[numericalData].replace(0, np.nan)
    
    plotOutlier(laptops, name[0])
    plotGroupCount(laptops, name[1])
    if 'cpu_speed_ghz' in laptops:
        plotDropCount(laptops)
    plotBins(laptops, name[2])
    
# Further clean data from visualization
def cleanPostVisualize(df):
    # Remove outliers
    df = df[df['ram_gb'] <= 70]
    df = df[df['screen_size_in'] <= 20]
    df = df[df['harddisk_gb'] <= 2048]
    
    # Reduce brand and color category
    df.loc[df.groupby('brand').brand.transform('count').lt(11), 'brand'] = 'others'
    df.loc[df.groupby('color').color.transform('count').lt(11), 'color'] = 'others'
    df.loc[df.groupby('os').os.transform('count').lt(11), 'os'] = 'others'
    
    # Drop CPU speed column
    df = df.drop(columns=['cpu_speed_ghz'], axis = 1)
    
    # Bin HDD
    df.loc[df['harddisk_gb'] == 65, 'harddisk_gb'] = 64
    df.loc[df['harddisk_gb'] == 120, 'harddisk_gb'] = 128
    df.loc[df['harddisk_gb'] == 250, 'harddisk_gb'] = 256
    df.loc[df['harddisk_gb'] == 500, 'harddisk_gb'] = 512
    df.loc[df['harddisk_gb'] == 1000, 'harddisk_gb'] = 1024
    df.loc[df['harddisk_gb'] == 2000, 'harddisk_gb'] = 2048
    
    bins = [16, 32, 64, 128, 256, 512, 1024, 2048, np.inf]
    df['harddisk_range_gb'] = pd.cut(df['harddisk_gb'], bins=bins, right=False)

    return df
    
cleanData()