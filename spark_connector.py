import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, desc, round, avg, count, sum, to_timestamp, date_format

# --- Config ---
WINDOWS_IP = "127.0.0.1"
DB_NAME = "data0"
MONGO_URI_BASE = f"mongodb://{WINDOWS_IP}:27017/{DB_NAME}"



# --- Creating connection ---
def get_spark_session():
    return ( SparkSession.builder
        .appName("YelpDataAnalysis")
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0")
        # fix for java 17
        .config("spark.driver.extraJavaOptions", "--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED")
        .getOrCreate())



# --- Helper functions ---
def read_collection(spark, collection_name):
    # read the right collection from the DB
    return (
        spark.read.format("mongodb")
            .option("spark.mongodb.read.connection.uri", f"mongodb://{WINDOWS_IP}:27017")
            .option("spark.mongodb.read.database", DB_NAME)
            .option("spark.mongodb.read.collection", collection_name)
            .load()
    )

def save_output(df, filename):
    # Save only first 20 rows of the data
    try:
        # save it to the dir "outputs"
        df.toPandas().head(20).to_csv(f"outputs/{filename}.csv", index=False)
    except Exception as e:
        print(f"[ERROR] Unable to save data to CSV format: {e}")



# --- I. STATISTICS EXPERIMENTS ---
def task_stat_1(spark):
    print("\n###### Statistics: 1. Top 20 categories ######")
    df = read_collection(spark, "business")
    
    # Rozdelenie stringu na pole a explózia na riadky
    df_exploded = df.select(explode(split(col("categories"), ", ")).alias("category"))
    
    category_counts = df_exploded.groupBy("category").count()
    result = category_counts.orderBy(desc("count"))
    
    result.show(20)
    save_output(result, "stat_1_top_categories")
    
def task_stat_2(spark):
    print("\n###### Statistics: 2. Geographic Distribution (Avg Rating & Count) ######")
    df = read_collection(spark, "business")
    
    # get rid of NULL and empty cities cols
    df = df.filter(col("city").isNotNull() & col("state").isNotNull())
    
    result = df.groupBy("state", "city").agg(
        round(avg("stars"), 2).alias("average_stars"),
        count("business_id").alias("business_count")
    ).orderBy("state", "city")
    
    result.show(20)
    save_output(result, "stat_2_geo_dist")   
    
def task_stat_3(spark):
    print("\n###### Statistics: 3. Open vs Closed Businesses Percentage by City ######")
    df = read_collection(spark, "business")
    
    # filter out the NULL cities
    df = df.filter(col("city").isNotNull())
    
    city_stats = df.groupBy("city").agg(
        sum("is_open").alias("open_count"),
        count("business_id").alias("total_count")
    )
    
    # count the % 
    result = city_stats.withColumn(
        "open_percentage",
        round((col("open_count") / col("total_count")) * 100, 2)
    ).orderBy(desc("total_count"))
    
    result.show(20)
    save_output(result, "stat_3_open_closed")

def task_stat_4(spark):
    print("\n--- Úloha 4: Distribúcia cien reštaurácií ---")
    df = read_collection(spark, "business")
    
    # filter only correct restaurants
    df_rest = df.filter(col("categories").contains("Restaurants"))
    df_prices = df_rest.select(col("attributes.RestaurantsPriceRange2").alias("price_range")) \
                       .filter(col("price_range").isNotNull() & (col("price_range") != "None"))
    
    result = df_prices.groupBy("price_range").count().orderBy("price_range")
    
    result.show()
    save_output(result, "stat_4_price_dist")

def task_stat_5(spark):
    print("\n--- Úloha 5: Aktivita používateľov v čase (Recenzie) ---")
    df = read_collection(spark, "review")
    
    # convert date (string) to the date format 
    df = df.withColumn("year_month", date_format(to_timestamp(col("date")), "yyyy-MM"))
    
    result = df.groupBy("year_month").count().orderBy("year_month")
    
    result.show(20)
    save_output(result, "stat_5_user_activity")

def task_stat_6(spark):
    print("\n--- Úloha 6: Top 100 najaktívnejších používateľov ---")
    df = read_collection(spark, "user")
    
    if "business_id" in df.columns:
        df = df.filter(col("business_id").isNull())
        
    result = df.select("name", "review_count", "user_id").orderBy(desc("review_count"))
    
    result.show(20)
    
    try:
        result.toPandas().head(100).to_csv("outputs/stat_6_top_users.csv", index=False)
        print(f"[INFO] Výsledok uložený do outputs/stat_6_top_users.csv")
    except Exception as e:
        print(e)
  
def main():
    spark = get_spark_session()

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("Error: must pass a task number as argument.")
        sys.exit(1)

    if choice == '1': task_stat_1(spark)
    elif choice == '2': task_stat_2(spark)
    elif choice == '3': task_stat_3(spark)
    elif choice == '4': task_stat_4(spark)
    elif choice == '5': task_stat_5(spark)
    elif choice == '6': task_stat_6(spark)
    elif choice == '0':
        print("Exiting.")
    else:
        print("Invalid choice")

    spark.stop()

if __name__ == "__main__":
    main()

