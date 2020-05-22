import re
import json
import math
import sys
import numpy as np
from scipy import stats
from pyspark import SparkContext

sc = SparkContext()
pathToJson = sys.argv[1]
mainRDD = sc.textFile(pathToJson)
regex_pattern = r'((?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))'
re_obj = sc.broadcast(re.compile(regex_pattern))

def wordCount(review):
  words = []
  review = json.loads(review)
  try:
    rev = review['reviewText']
    if rev:
      words = re.findall("((?:[\.,!?;\"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))", rev.lower())
  except:
    return []
  return words

wordFreqRDD = mainRDD.flatMap(lambda review: wordCount(review)).map(lambda word: (word, 1)).reduceByKey(lambda a,b:a +b)

frequentWords = wordFreqRDD.top(1000, key=lambda x: x[1])

frequentWords = sc.broadcast(frequentWords)

def totalWordCount(rev):
  words = re.findall("((?:[\.,!?;\"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))", rev.lower())
  return len(words)

def relFreqCount(review):
  rel = []
  try:
    parsedReview = json.loads(review)
    totalWords = wordCount(review)
    for word in frequentWords.value:
      count = totalWords.count(word[0])
      if count > 0:
        rel.append((str(word[0]), ((count/len(totalWords)), int(parsedReview['verified']), parsedReview['overall'])))
      else:
        rel.append((str(word[0]), (0.0, int(parsedReview['verified']), parsedReview['overall'])))
  except:
     return []
  return rel

wordsRDD = mainRDD.flatMap(lambda review: relFreqCount(review)).groupByKey().map(lambda x : (x[0], list(x[1])))

def retNormalised(v):
  return ((v- np.mean(v))/np.std(v))

def linear_reg(r, controlVariable=False):
  records = list(r[1])
  word = r[0]
  XVal = []
  YVal = []
  XVal1 = []

  for record in records:
    XVal.append(record[0])
    YVal.append(record[2])

  if controlVariable:
    XVal1.append(record[2])

  XVal_ = retNormalised(XVal)
  YVal_ = retNormalised(YVal)
  if controlVariable:
    XVal1_ = retNormalised(XVal1)
  
  m = np.shape(records)[0]
  if controlVariable:
    X = np.matrix([np.ones(m), XVal_, XVal1_]).T
  else:
    X = np.matrix([np.ones(m), XVal_]).T
  y = np.matrix(YVal_).T

  beta_mat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

  # Get predictions
  if controlVariable:
    YPred = np.array(beta_mat[0] + beta_mat[1] * XVal_ + beta_mat[1] * XVal1_)
  else:
    YPred = np.array(beta_mat[0] + beta_mat[1] * XVal_)
  # RSS  
  rss = np.sum(np.square(YPred - YVal_))

  # DF
  df = len(XVal_) - (2)
  # s_sqaure
  s2 = rss/df
  # Calculate t_value
  mean = np.mean(XVal_)
  denom = np.sum(np.square(XVal_ - mean))
  t = beta_mat[1]/math.sqrt(s2/denom)
  # Calculate p_value
  res = stats.t.sf(np.abs(t), df)*1000
  return (word, beta_mat[1][0, 0], res[0][0])

finalResWithoutControl = wordsRDD.map(lambda word_rec: linear_reg(word_rec))

finalResWithoutControl.sortBy(lambda x: x[1]).saveAsTextFile('neg_corr_no_control')

finalResWithoutControl.sortBy(lambda x: -x[1]).saveAsTextFile('pos_corr_no_control')

finalResWithControl = wordsRDD.map(lambda word_rec: linear_reg(word_rec, controlVariable=True))

finalResWithControl.sortBy(lambda x: x[1]).saveAsTextFile('neg_corr_with_control')

finalResWithControl.sortBy(lambda x: -x[1]).saveAsTextFile('pos_corr_with_control')