Code cell <s_MpWP8JcQtv>
# %% [code]
!pip install pyspark
!pip install -U -q PyDrive
!apt install openjdk-8-jdk-headless -qq
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"



from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext
import pandas as pd

# create the Spark Session
spark = SparkSession.builder.getOrCreate()

# create the Spark Context
sc = spark.sparkContext


Code cell <0B1ildmzhdIi>
# %% [code]
#@title Allow access to Google Drive client in order to download the data
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client
# Make sure to follow the interactive instructions
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)



# Download the file
id='1lfe9w6wPAh_n7J1hsQmdAgJOzv3-7Xun'
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('friends.txt')

# read in your file
txt = sc.textFile("friends.txt", 1)
def allPermutations(usr):
  n=len(usr)
  ans=[]
  for i in range(n):
    for j in range(n):
      if(i==j):
        continue
      ans.append((usr[i],usr[j]))

  return ans;


from operator import add
friAdj= txt.map(lambda line: line.split('\t')).filter(lambda x: len(x)>1).map(lambda x: (int(x[0]), [int(y) for y in x[1].split(',') if y]))
friPairs=friAdj.flatMap(lambda x: [(x[0],j) for j in x[1] ]) #they are already friend
multualPairs=friAdj.flatMap(lambda x: [(j,1) for j in allPermutations(x[1])]).subtractByKey(friPairs).reduceByKey(add).map(lambda usr_friend_idx: (usr_friend_idx[0][0], (usr_friend_idx[0][1], usr_friend_idx[1]))).sortBy(lambda x: (-x[1][1], x[1][0])).map(lambda x: (x[0], x[1][0])).groupByKey().map(lambda x: (x[0],list(x[1])[:10]))


multualPairs.filter(lambda x: x[0]==11).collect()

#noFriend = txt.filter(lambda x: len(x.split('\t')) == 1).map(lambda x: x.split('\t')[0])  #they got no friend
#multualPairs =multualPairs.union(noFriend)

targetList=[924, 8941, 8942, 9019, 9020, 9021, 9022, 9990, 9992, 9993]
tarPairs=multualPairs.filter(lambda x:len(x)>1 and x[0] in targetList).collect()
from google.colab import drive
#drive.mount('/content/drive')
output_path = '/content/drive/My Drive/mutual_pairs_recommendations.txt'

with open(output_path, 'w') as file:
    for user, recommendations in tarPairs:
        recommendations_str = ','.join(str(rec) for rec in recommendations)
        file.write(f"{user}\t{recommendations_str}\n")



#cross product version, tooooooo slowwwww
import copy
from operator import add
# get the adjacent dictionary for every user
friAdj=txt.map(lambda x: x.split('\t')).filter(lambda x: len(x)>1).map(lambda x: (int(x[0]),[int(y) for y in x[1].split(',') if y !='']))
dicts=dict(friAdj.collect())


#sc.parallelize(friAdj).map(lambda x: (x[0],[(i,1) for i in [ dicts.get(y) for y in x[1]] if i not in x[1]])).collect()[0]
tmp=(friAdj.map(lambda x: (x[0],[(i,1) for k in [ dicts.get(y) for y in x[1]] for i in k if k not in x[1]])).collect())
ans=[(l[0],sorted(sc.parallelize(l[1]).reduceByKey(add).collect(),key = lambda x: x[1])[0:10]) for l in tmp]
#res=sc.parallelize(ans).reduceByKey(add).collect()
#sc.parallelize(friAdj).map(lambda x: (x[0],[(i,1) for k in [ dicts.get(y) for y in x[1]] for i in k if k not in x[1]])).collect()[0]

#print(dicts[0])

Text cell <Bj8gbFxkWjqV>
# %% [markdown]




