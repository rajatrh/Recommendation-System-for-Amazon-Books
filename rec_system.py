import re
import ast
import json
import math
import sys
from datetime import datetime
from pyspark import SparkContext

sc = SparkContext()
pathToJson = sys.argv[1]
tProd = ast.literal_eval(sys.argv[2])
itemId = sc.broadcast(tProd)

mainRDD = sc.textFile(pathToJson)

def filter_based_on_asin_user(review):
  res = ()
  try:
    review = json.loads(review)
    d = review['unixReviewTime']
    res = ((review['asin'], review['reviewerID']), [(int(d), review['overall'])])
  except:
    return res
  return res

def normalizeRatings(r):
  s = 0
  res = []
  c = len(r[1])
  for usersRatings in r[1]:
    s += usersRatings[1]
  nr =(s/c)

  for usersRatings in r[1]:
    rating = usersRatings[1]-nr
    res.append((usersRatings[0], usersRatings[1], rating))
  
  return (r[0], res)

def calculateSimilarity(prodA):
  res = []
  
  sumA = 0
  for u in prodA[1]:
      sumA += (u[2] * u[2])
  
  for prodB in item_normalised_ratings.value:
    dot_product = 0
    count = 0
    sumB = 0
    for u in prodB[1]:
      sumB += (u[2] * u[2])
    for uA in prodA[1]:
      for uB in prodB[1]:
        if uB[0] in uA[0]:
          dot_product += (uB[2] * uA[2])
          count += 1
    if count >= 2:
      try:
        sim = dot_product / (math.sqrt(sumA) * math.sqrt(sumB))
        if (sim >0):
          for uA in prodA[1]:
            res.append(((uA[0], prodB[0]),(prodA[0], uA[1], sim)))
      except:
        pass
  return res

def predictRating(userData):
  user = userData[0][0]
  res = []
  for prodB in item_normalised_ratings.value:
    targetProduct = prodB[0]
    if targetProduct == userData[0][1]:
      s = []
      ut = []
      top = 0
      bottom = 0
      alreadyRated = False
      for prodA in userData[1]:
        if (prodA[0] == targetProduct):
          alreadyRated = True
        s.append(prodA[2])
        ut.append(prodA[1])

      if len(s) >= 2:
        for i in range(len(s)):
          top += (s[i] * ut[i])
          bottom += s[i]
        if bottom != 0 and not alreadyRated:
          res.append((prodB[0], user, round(top/bottom,3)))
  return res

# Group by asin and user
step1 = mainRDD.map(lambda review: filter_based_on_asin_user(review)).groupByKey().mapValues(list)

# Take only the latest reviews
step2 = step1.sortBy(lambda a: -a[1][0][0][0])

step3_items = step2.map(lambda x : (x[0][0],(x[0][1], x[1][0][0][1])))\
.groupByKey().map(lambda x : (x[0], list(x[1]), len(list(x[1]))))\
.filter(lambda x: x[2] >= 25)

step4_products = step3_items.flatMap(lambda y: [((value[0]),(y[0], value[1])) for value in y[1]])

step5_products = step4_products.groupByKey().map(lambda x : (x[0], list(x[1]), len(list(x[1]))))\
.filter(lambda x: x[2] >= 5)

ut_matrix = step5_products.flatMap(lambda y: [((value[0]),(y[0], value[1])) for value in y[1]])

norm_ut_mat = ut_matrix.groupByKey().mapValues(list).map(lambda x : normalizeRatings(x))

item_normalised_ratings = sc.broadcast(norm_ut_mat.filter(
    lambda x: x[0] in itemId.value)
    .collect())

sim_norm_ut_mat = norm_ut_mat.flatMap(lambda x: calculateSimilarity(x))

neigh_user_sim_norm_ut_mat = sim_norm_ut_mat.groupByKey().mapValues(list).map(lambda x: (x[0], sorted(x[1], key=lambda x: x[2], reverse=True)[:50]))

finalRes = neigh_user_sim_norm_ut_mat.flatMap(lambda x: predictRating(x))

finalRes.saveAsTextFile('rec_sys')





