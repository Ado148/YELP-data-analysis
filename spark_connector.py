from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, desc

windows_ip = "127.0.0.1"
input_uri = f"mongodb://{windows_ip}:27017/data0.analysis"
output_uri = f"mongodb://{windows_ip}:27017/data0.results"

myspark = (
    SparkSession.builder
        .appName("ApplicationForDataAnalysis")
        # Pre spark-submit je lepsie nechat packages v konfiguracii, aby to fungovalo
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0")
        .config("spark.mongodb.read.connection.uri", input_uri)
        .config("spark.mongodb.write.connection.uri", output_uri)
        # Fix pre Java 17 - pridaj pre istotu
        .config("spark.driver.extraJavaOptions", "--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED")
        .getOrCreate()
)

# POZOR: Pre MongoDB V10 konektor nepridavaj znova option("uri") pri .read()
# Staci, ked je to nastavene v SparkSession.builder
df_business = (
    myspark.read
        .format("mongodb")
        .load()
)

print("\n--- SCHÉMA ÚDAJOV ---")
df_business.printSchema()
print("\n--- PRVÉ RIADKY ---")
df_business.show(5)



print("\n--- Statistics: 1. Top 20 categories ---")
# select rows with the 'categories', rows without it delete
categories_df = df_business.select("categories").dropna()

# split categories string by , to array ['A', 'B', ...]
split_categories_df = categories_df.withColumn(
    "category_arr",
    split(col("categories"), ", ")
)

# create new row for each element in the array
exploded_df = split_categories_df.withColumn(
    "category",
    explode(col("category_arr"))
)

# group by by categories
category_counts = exploded_df.groupBy("category").count()

# order results from biggest to smallest nad show only TOP 20
top_20_categories = category_counts.orderBy(desc("count"))

top_20_categories.show(20)


