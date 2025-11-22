from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, desc, round, avg, count, sum, to_timestamp, date_format

windows_ip = "127.0.0.1"
input_uri = f"mongodb://{windows_ip}:27017/data0.analysis"
output_uri = f"mongodb://{windows_ip}:27017/data0.results"

myspark = (
    SparkSession.builder
        .appName("ApplicationForDataAnalysis")
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0")
        .config("spark.mongodb.read.connection.uri", input_uri)
        .config("spark.mongodb.write.connection.uri", output_uri)
        # fix for java 17
        .config("spark.driver.extraJavaOptions", "--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED")
        .getOrCreate()
)

df_data = (
    myspark.read
        .format("mongodb")
        .load()
)

# print("\n--- SCHÉMA ÚDAJOV ---")
# df_business.printSchema()
# print("\n--- PRVÉ RIADKY ---")
# df_business.show(5)


print("\n###### Statistics: 1. Top 20 categories ######")
# select rows with the 'categories', without deletion
categories_df = df_data.select("categories").dropna()

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
# group by categories
category_counts = exploded_df.groupBy("category").count()
# order results from biggest to smallest nad show only TOP 20
top_20_categories = category_counts.orderBy(desc("count"))

#top_20_categories.show(20)
top_20_categories.toPandas().head(20).to_csv("outputs/stat_1.csv", index=False)



print("\n###### Statistics: 2. Geographic Distribution (Avg Rating & Count) ######")
# group by state and city, .agg() for multiple metrics in parallel
geo_distribution_df = df_data.groupBy("state", "city").agg(
    avg("stars").alias("average_stars"),
    count("business_id").alias("business_count")
)
geo_distribution_df = geo_distribution_df.withColumn(
    "average_stars", 
    round(col("average_stars"), 2)
)

result_geo_df = geo_distribution_df.orderBy("state", "city")
result_geo_df.show(20)
result_geo_df.toPandas().head(20).to_csv("outputs/stat_2.csv", index=False)



print("\n###### Statistics: 3. Open vs Closed Businesses Percentage by City ######")
# group by city
city_open_stats = df_data.groupBy("city").agg(
    sum("is_open").alias("open_count"),
    count("business_id").alias("total_count")
)

# count the % 
result_open_stats = city_open_stats.withColumn(
    "open_percentage",
    round((col("open_count") / col("total_count")) * 100, 2)
)

result_open_stats.orderBy(desc("total_count")).show()
result_open_stats.toPandas().head(20).to_csv("outputs/stat_3.csv", index=False)



print("\n###### Statistics: 4. Restaurant Price Distribution ######")
#find where the Restaurants in the categories are
restaurants_df = df_data.filter(col("categories").contains("Restaurants"))

# filter out null
price_df = restaurants_df.select(col("attributes.RestaurantsPriceRange2").alias("price_range")
                                 ).filter(col("price_range").isNotNull() & (col("price_range") != "None"))

# count how many restaurants are in the category
price_distribution = price_df.groupBy("price_range").count()
result_price_dist = price_distribution.orderBy("price_range")
result_price_dist.show()
result_price_dist.toPandas().head(20).to_csv("outputs/stat_4.csv", index=False)



print("\n###### Statistics: 5. User Activity Over Time (Reviews per Month) ######")

df_reviews = df_data.filter(col("review_id").isNotNull())

# process the date and convert date (string) to date format and get YYYY-MM
df_reviews_dates = df_reviews.withColumn(
    "date_ts",
    to_timestamp(col("date"))
).withColumn(
    "year_month",
    date_format(col("date_ts"), "yyyy-MM")    
)
    
# count how many reviews there were in the each minth    
time_series_df = df_reviews_dates.groupBy("year_month").count().orderBy("year_month")

time_series_df.show(20)
time_series_df.toPandas().head(20).to_csv("outputs/stat_5.csv", index=False)



print("\n--- Statistics: 6. Top 100 Users by Review Count ---")

# connect to the given collection
users_output_uri = f"mongodb://{windows_ip}:27017/data0.user"

df_users = (
    myspark.read
    .format("mongodb")
    .option("uri", users_output_uri)
    .load()
)

df_real_users = df_users.filter(col("user_id").isNotNull() & col("business_id").isNull())

top_users_list = df_real_users.select("name", "review_count", "user_id") \
    .orderBy(desc("review_count"))

top_users_list.show(100, truncate=False)
top_users_list.toPandas().head(100).to_csv("outputs/stat_6.csv", index=False)