import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import time
import os

def q25(x):
  return x.quantile(0.25)
def q50(x):
  return x.quantile(0.5)
def q75(x):
  return x.quantile(0.75)
def q99(x):
  return x.quantile(0.99)
def q1(x):
  return x.quantile(0.1)
quantiles = [q1, q25, q50, q75, q99]
quantilenames = ['q1', 'q25', 'q50', 'q75', 'q99']
quantile_values = [0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99]

def getDiffToPrev(aDataFrame, aType, aTypeName, aColumnFilter, aColumnCompare, shift=1):
  if shift < 0:
    myRelation = 'Next'
  else:
    myRelatio = 'Prev'
  myColumn = aTypeName + 'Column'
  myRelColumn = myRelation + myColumn
  myDiffColumn = 'DiffTo' + myRelation + aTypeName
  aDataFrame[myColumn] = aDataFrame[aColumnFilter].where(aType)
  aDataFrame[myRelColumn] = aDataFrame[myColumn].ffill().shift(shift).fillna(0)
  aDataFrame[myDiffColumn] = aDataFrame[aColumnCompare] - aDataFrame[myRelColumn]
  return aDataFrame

{
  'Name': 'Name',
  'ColumnFilter': 'ColumnFilter',
  'ColumnCompare': 'ColumnCompare',
  'Relation': 1,
  'Filters': 
  [
    'OR',
    [
      [
        'AND',
        [
          {
            'Name': 'ColumnName',
            'Op': 'GEQ',
            'Value': 12
          }
        ],

        [
          {
            'Name': 'ColumnName2',
            'Op': 'EQ',
            'Value': 15
          }
        ]
      ]
    ]
  ]
}

def fetchFilter(myDf, myFilter):
  if myFilter['Op'] == 'LT':
    return (myDf[myFilter['Name']] < myFilter['Value'])
  if myFilter['Op'] == 'LEQ':
    return (myDf[myFilter['Name']] <= myFilter['Value'])
  if myFilter['Op'] == 'GT':
    return (myDf[myFilter['Name']] > myFilter['Value'])
  if myFilter['Op'] == 'GEQ':
    return (myDf[myFilter['Name']] >= myFilter['Value'])
  if myFilter['Op'] == 'EQ':
    return (myDf[myFilter['Name']] == myFilter['Value'])
  if myFilter['Op'] == 'NEQ':
    return (myDf[myFilter['Name']] != myFilter['Value'])
  if myFilter['Op'] == 'ISNA':
    return (myDf[myFilter['Name']].isna() == True)
  if myFilter['Op'] == 'NISNA':
    return (myDf[myFilter['Name']].isna() == False)
  
def decodeFilterInfo(myDf, myFilter):
  if len(myFilter) > 1:
    myOp = myFilter[0]
    myNewDf = pd.DataFrame()
    for i, myValue in enumerate(myFilter[1]):
      if i == 0:
        if myOp == 'NOT':
          myNewDf = ~(decodeFilterInfo(myDf, myValue))
        else:
          myNewDf = (decodeFilterInfo(myDf, myValue))
      else:
        if myOp == 'OR':
          myNewDf = (myNewDf) | (decodeFilterInfo(myDf, myValue))
        else:
          myNewDf = (myNewDf) & (decodeFilterInfo(myDf, myValue))
    return myNewDf
  return fetchFilter(myDf, myFilter[0])

def getSortedPerGroupInfo(myDfs, myDiffs, myTypes, myGroupColumn):
  myDfs.fillna(0, inplace=True)
  myDfsSorted = {}
  for myGroup in myDfs[myGroupColumn].unique():
    print(f'Running group: {myGroup}')
    myDfsSorted[myGroup] = myDfs[myDfs[myGroupColumn] == myGroup].sort_values(['timestamp', 'offset'])
    for myValue in myDiffs:
      print(myValue) 
      myDfsSorted[myGroup] = getDiffToPrev(myDfsSorted[myGroup], decodeFilterInfo(myDfsSorted[myGroup], myValue['Filters']), myValue['Name'], myValue['ColumnFilter'], myValue['ColumnCompare'], myValue['Relation'])
    for myValue in myTypes:
      print(myValue)
      myDfsSorted[myGroup].loc[decodeFilterInfo(myDfsSorted[myGroup], myValue['Filters']), 'Type'] = myValue['Name']
  return pd.concat(myDfsSorted.values())

def getPassedCount(myDf, myFilter):
  myNewDf = myDf
  myNewDf['Passed'] = decodeFilterInfo(myDf, myFilter)
  myNewDf['PassedNum'] = myNewDf['Passed'].astype(int)
  myNewDf = pd.DataFrame(myNewDf.gropuby(['Type']).agg({'Passed': 'sum', 'PassedNum': 'mean', 'offset': 'count'}))
  myNewDf.sort_values('PassedNum', ascending = False, inplace=True)
  myNewDf.rename(columns={'PassedNum': 'PassedPercent', 'Passed': 'PassedCount', 'offset': 'TotalCount'})
  return myNewDf
  
def getDf(myDfs, merge):
  myNewDf = myDfs[merge]
  myNewDf = myNewDf.merge(myDfs['ds'], on='SUBJID')
  myNewDf = myNewDf[(myNewDf['DS_DATA_REFUS'] != 1)]
  myNewDf = myNewDf.merge(myDfs['cpc'], on='SUBJID')
  myNewDf = myNewDf.merge(myDfs['bras_rando'], on='SUBJID')
  myNewDf = getType(myNewDf)
  return myNewDf

def getType(myNewDf):
  myCpc1Filter = {
    'Name': 'CPC_SC3',
    'Op': 'EQ', 
    'Value': 1
  }
  myCpc2Filter = {
    'Name': 'CPC_SC3',
    'Op': 'EQ', 
    'Value': 2
  }
  myHypothermiaFilter = {
    'Name': 'groupe',
    'Op': 'EQ',
    'Value': 1
  }
  myNormothermiaFilter = {
    'Name': 'groupe',
    'Op': 'EQ',
    'Value': 0
  }
  myCpc12Filter = ['OR', [[myCpc1Filter], [myCpc2Filter]]]
  myCpc12HypothermiaFilter = ['AND', [myCpc12Filter, [myHypothermiaFilter]]]
  myCpc12NormothermiaFilter = ['AND', [myCpc12Filter, [myNormothermiaFilter]]]
  myNotCpc12NormothermiaFilter = ['AND', [['NOT', [myCpc12Filter]], [myNormothermiaFilter]]]
  myNotCpc12HypothermiaFilter = ['AND', [['NOT', [myCpc12Filter]], [myHypothermiaFilter]]]
  myNewDf['Type'] = 'NotCondideredInStudy'
  myCpc12HypMap = decodeFilterInfo(myNewDf, myCpc12HypothermiaFilter)
  myNewDf.loc[myCpc12HypMap, 'Type'] = 'CPC12HYpothermia'
  myCpc12NotHypMap = decodeFilterInfo(myNewDf, myCpc12NormothermiaFilter)
  myNewDf.loc[myCpc12NotHypMap, 'Type'] = 'CPC12Normothermia'
  myNotCpc12HypMap = decodeFilterInfo(myNewDf, myNotCpc12HypothermiaFilter)
  myNewDf.loc[myNotCpc12HypMap, 'Type'] = 'NotCPC12Hypothermia'
  myNotCpc12NotHypMap = decodeFilterInfo(myNewDf, myNotCpc12NormothermiaFilter)
  myNewDf.loc[myNotCpc12NotHypMap, 'Type'] = 'NotCPC12Normothermia'
  return myNewDf

def readDfs():
  read_path = 'formatteddata\\'
  myDfs = {}
  for file in os.listdir(read_path):
    myDfs[file.split('.')[0]] = pd.read_csv(read_path + file)
  return myDfs