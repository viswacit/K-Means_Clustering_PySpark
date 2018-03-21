import os
from flask import Flask, request, jsonify
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/uberdata')
def get_uber_data():
	spark = SparkSession\
        .builder\
        .appName("Uber Dataset")\
        .getOrCreate()
	cluster_count = int(request.args.get('cluster_count'))	
	dataset = spark.read.csv('uberdata.csv', inferSchema =True, header="True")
	assembler = VectorAssembler(inputCols=["Lat", "Lon"],outputCol="features")
	dataset=assembler.transform(dataset)
	(training, testdata) = dataset.randomSplit([0.7, 0.3], seed = 5043)
	kmeans = KMeans().setK(cluster_count)
	model = kmeans.fit(dataset)
	transformed=model.transform(testdata).withColumnRenamed("prediction","cluster_id")
	transformed.createOrReplaceTempView("data_table")
	transformed.cache()
	centerList=list()
	cluster_centers = model.clusterCenters()
	count=int()
	for center in cluster_centers:
		centersIndList=list()
		centersIndList.append(format(center[0], '.8f'))
		centersIndList.append(format(center[1], '.8f'))
		centersIndList.append(count)
		centerList.append(centersIndList)
		count=count+1		
	centers=spark.createDataFrame(centerList)
	centers.createOrReplaceTempView("centers")
	resultsDFF = spark.sql("SELECT centers._1 as Longitude, centers._2 as Latitude FROM data_table, centers WHERE data_table.cluster_id=centers._3")
	data=resultsDFF.groupBy("Longitude", "Latitude").count()
	return jsonify(data.toJSON().collect())
	
def get_port():
	port = os.getenv("PORT")
	if type(port) == str:
		return port
	return 8080	
	
if __name__ == "__main__":
	app.run(host="0.0.0.0", port=get_port())