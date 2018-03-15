
'''
Created on Feb 13, 2018

@author: MSURES56
'''

import os
import sys
from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans

app = Flask(__name__)


@app.route('/uberdata')
def get_uber_data():
	spark = SparkSession\
        .builder\
        .appName("Uber")\
        .getOrCreate()

	dataset = spark.read.csv('uberdata.csv', inferSchema =True, header="True")
	assembler = VectorAssembler(inputCols=["Lat", "Lon"],outputCol="features")
	dataset=assembler.transform(dataset)
	(training, testdata) = dataset.randomSplit([0.7, 0.3], seed = 5043)
	kmeans = KMeans().setK(8)
	model = kmeans.fit(dataset)
	cat=model.transform(testdata)
	data=cat.groupBy("prediction").count().orderBy("prediction")
	return jsonify(data.toJSON().collect())
	
def get_port():
    port = os.getenv("PORT")
    if type(port) == str:
        return port
    return 8080	
	
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=get_port())
