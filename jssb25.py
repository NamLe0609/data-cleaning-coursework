import pandas as pd
import numpy as np
import re

def cleanCategorical(df, categoricalData):
    for column in categoricalData:
        df[column] = df[column].str.casefold().str.strip()
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
        r'black|dark|carbon|balck': 'black',
        r'silver|platinum|aluminum|sliver|midnight|mercury': 'silver',
        r'gr[ae]y|gary|lunar|graphite': 'grey',
        r'blue|cobalt|sky': 'blue',
        r'red': 'red',
        r'white|light': 'white',
        r'almond|dune|beige': 'brown',
        r'yellow|gold|apollo': 'yellow',
        r'green|mint|sage': 'green',
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

def extractOS(row):
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
    df['os'] = df['os'].apply(extractOS)
    return df

def main():
    pd.set_option("display.max_rows", None)
    fileName = 'amazon_laptop_2023.xlsx'

    df = pd.read_excel(fileName)

    # Drop any column with all missing data
    df = df.dropna(axis=1, how='all')

    # We dont know a computer's model, so cant recommend it
    # IF this was a task analysing computers available on the market then
    # maybe we wont need the model name
    df = df.dropna(axis=0, subset=['model'])

    # Drop rows which are exact duplicates
    df = df.drop_duplicates(ignore_index=True, keep='first')
    # df.drop_duplicates(subset=['model', 'screen_size', 'color', 'harddisk', 'cpu', 'ram'], keep='first', inplace=True)

    # Standardize column names (Like making OS lower case)
    df.columns = df.columns.str.lower().str.strip()

    categoricalData = ['brand', 'model', 'color', 'cpu', 'os', 'special_features', 'graphics', 'graphics_coprocessor']
    df = cleanCategorical(df, categoricalData)

    # Rename columns which has units that needs to be standardized 
    # and to put those into numerical feature list instead of categorical

    numericalData = ['harddisk', 'ram', 'screen_size', 'cpu_speed', 'rating', 'price']
    df = cleanAllNum(df, numericalData)
    
    # Multiply TB values (less than 8) by 1024 to make it GB
    df = cleanHDD(df)
    
    # Divide MHz values (more than 10) by 1000 to make it GHz
    df = cleanCPUSpeed(df)

    # Ram is only integer amount. Round in case value is not integer
    df = cleanRam(df)
    
    # Clean color to remove non-standard values
    df = cleanColor(df)
    
    # Clean OS by simplifying it to OS type, and version
    df = cleanOS(df)

    df = df.rename(columns={
        "harddisk": "harddisk_gb", 
        "ram": "ram_gb", 
        "screen_size": "screen_size_in", 
        "cpu_speed": "cpu_speed_ghz", 
        "price": "price_dollar"
    })
    #print(df.dtypes)
    #print(df.describe())
    df.to_excel('amazon_laptop_2023_cleaned.xlsx')

main()
