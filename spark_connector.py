import sys, os
from pyspark.sql import SparkSession
from pyspark.sql.functions import size, length, regexp_replace, when, col, split, explode, desc, round, avg, count, sum, to_timestamp, date_format
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import PipelineModel, Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle, json, gc
from pymongo import MongoClient
import pandas as pd

# --- Config ---
WINDOWS_IP = "127.0.0.1"
DB_NAME = "data0"
MONGO_URI_BASE = f"mongodb://{WINDOWS_IP}:27017/{DB_NAME}"



# --- Creating connection ---
def get_spark_session():
    return ( SparkSession.builder
        .appName("YelpDataAnalysis")
        .master("local[*]") # use all CPU cores
        .config("spark.driver.memory", "8g") # 5GB RAM for Driver
        .config("spark.executor.memory", "8g")# 5GB RAM for Executor
        .config("spark.sql.adaptive.enabled", "true")
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
        if not os.path.exists("outputs"):
            os.makedirs("outputs")
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
    print("\n###### Statistics: 4. Restaurant Price Distribution ######")
    df = read_collection(spark, "business")
    
    # filter only correct restaurants
    df_rest = df.filter(col("categories").contains("Restaurants"))
    df_prices = df_rest.select(col("attributes.RestaurantsPriceRange2").alias("price_range")) \
                       .filter(col("price_range").isNotNull() & (col("price_range") != "None"))
    
    result = df_prices.groupBy("price_range").count().orderBy("price_range")
    
    result.show()
    save_output(result, "stat_4_price_dist")

def task_stat_5(spark):
    print("\n###### Statistics: 5. User Activity Over Time (Reviews per Month) ######")
    df = read_collection(spark, "review")
    
    # convert date (string) to the date format 
    df = df.withColumn("year_month", date_format(to_timestamp(col("date")), "yyyy-MM"))
    
    result = df.groupBy("year_month").count().orderBy("year_month")
    
    result.show(20)
    save_output(result, "stat_5_user_activity")

def task_stat_6(spark):
    print("\n--- Statistics: 6. Top 100 Users by Review Count ---") 
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
  
  
  
#--- II. Research EXPERIMENTS (ML) ---

def task_start_71(spark):
    print("\n###### Research: 1. Sentiment Analysis (ML Prediction) ######")

    model_path = "models/sentiment_model"
    # --- 1. Global fetching and redistribution of the data ---
    df = read_collection(spark, "review").select("text", "stars")
    df = df.filter(col("text").isNotNull())
    
    df = df.sample(withReplacement=False, fraction=0.1, seed=42)
    df = df.limit(500000)
    print("\n\n\n\n\n\n\n##########################################################")
    pocet_riadkov = df.count()
    print(f"Počet riadkov po sample a limit: {pocet_riadkov}")
    print("\n\n\n\n\n\n\n##########################################################")

    #df = df.limit(100000)
    (trainingModel, testModel) = df.randomSplit([0.8, 0.2], seed=42) 
    
    # --- 2. Models ---
    if os.path.exists(model_path):
        print(f"[INFO] Model found in the '{model_path}'. Loading...")
        model = PipelineModel.load(model_path)
        
        # model = PipelineModel.load(model_path)
        
        # # load data for the testing
        # print("[INFO] Loading data...")
        # df = read_collection(spark, "review").select("text", "stars")
        # df = df.filter(col("text").isNotNull())
        
        # # divide the data same as on the model training
        # (_, testData) = df.randomSplit([0.8, 0.2], seed=42)
    
    else:
        print("[INFO] Loading data and training new model...")
        # df = read_collection(spark, "review").select("text", "stars")
        # df = df.filter(col("text").isNotNull())
        
        # prepare data by using Tokenizer, which divides sentence to words
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        # get rid of the useless words (a, the, and, an ...)
        remover = StopWordsRemover(inputCol="words", outputCol="filtered")
        # HashingTF + IDF: converts words to numbers and vectors (vecotr is set to be 10000) to prevent collsions
        hashingtf = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        # ML algo. We try to guess the stars (labelCol) based on the text
        lr = LogisticRegression(
            maxIter=100, 
            regParam=0.001, 
            labelCol="stars", 
            featuresCol="features"
        )
        
        # create pipeline for learning and testing the model
        pipeline = Pipeline(stages=[tokenizer, remover, hashingtf, idf, lr])
        # 80% training, 20% testing the model
        (trainingModel, testModel) = df.randomSplit([0.8, 0.2], seed=42)
        
        print("     -> Model training...")
        model = pipeline.fit(trainingModel)
        
        print(f"   -> Saving the model into the'{model_path}'...")
        model.write().overwrite().save(model_path)
        
    # --- 3. Predictions and evaluation of the model ---
    predictions = model.transform(testModel) # prediction on testing data
    print("     -> Evaluating accuracy...")
    predictions.select("text", "stars", "prediction").show(10, truncate=50)
        
    # evaluate accurancy
    evaluator = MulticlassClassificationEvaluator(
        labelCol="stars", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    
    print(f"\n============================================")
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    print(f"============================================")
    save_output(predictions.select("stars", "prediction"), "research_1_sentiment_predictions")

def task_start_7():
    print("\n###### Research: 7. Sentiment Analysis (TensorFlow GPU) - Direct MongoDB ######")
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        import numpy as np
        from sklearn.model_selection import train_test_split
        print("✅ Everything imported correctly.")
    except ImportError as e:
        print(f"❌ Error on import: {e}")
        return

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    prediction_successful = 0
    
    # Check the GPU
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"✅ GPU avaible: {len(gpu_devices)}")
        try:
            tf.config.experimental.set_memory_growth(gpu_devices[0], True)
        except Exception as e:
            print(f"⚠️ Unable to set memory growth: {e}")
    else:
        print("⚠️  GPU unavaible, using CPU")
        
        
    # --- 0. Verify the existence of the previous model ---
    model_path = 'models/sentiment_model_tensorflow_gpu.keras'
    tokenizer_path = 'models/tokenizer.pickle'
    config_path = 'models/model_config.json'
    
    if os.path.exists(model_path) and os.path.exists(tokenizer_path) and os.path.exists(config_path):
        print("✅ Existing model found! Loading model for prediction...")
        
        try:
            # Load the trained model
            model = keras.models.load_model(model_path)
            
            # Load tokenizer
            with open(tokenizer_path, 'rb') as handle:
                tokenizer = pickle.load(handle)
                
            # Load config
            with open(config_path, 'r') as f:
                config = json.load(f)
            max_length = config['max_length']
            
            print("✅ Model, tokenizer and config loaded successfully!")
            
            # --- PREDICTION WITH EXISTING MODEL ---
            print("\n[INFO] Using existing model for prediction on new MongoDB data...")
            
            # Load new data from MongoDB for prediction
            try:
                client = MongoClient("mongodb://127.0.0.1:27017/")
                db = client.data0
                collection = db.review
                
                total_count = collection.count_documents({"text": {"$ne": None}, "stars": {"$ne": 3}})
                print(f"All the rows: {total_count}")

                batch_size = 500000
                skip = 0
                all_results = []
                
                while skip < total_count:
                    batch_cursor = collection.find(
                        {"text": {"$ne": None}, "stars" : {"$ne": 3}},
                        {"text": 1, "stars": 1}
                    
                    ).skip(skip).limit(batch_size)
                    
                    batch_data = list(batch_cursor)
                    if len(batch_data) == 0:
                        break
                    df = pd.DataFrame(batch_data)
                    batch_num = skip / batch_size + 1
                    print(f"Processing batch {skip//batch_size + 1}: {len(df)} records")
                    
                    # Prepare data for prediction
                    texts = df['text'].astype(str).values
                    true_labels = df['stars'].apply(lambda x: 1 if x > 3 else 0).values
                
                    # Tokenize and pad the new data
                    sequences = tokenizer.texts_to_sequences(texts)
                    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
                
                    # Make predictions
                    print("[INFO] Making predictions...")
                    predictions = model.predict(padded_sequences, batch_size=256, verbose=1)
                    predicted_binary = (predictions > 0.5).astype(int).flatten()

                    batch_results = pd.DataFrame({
                        'text': texts,
                        'true_stars': df['stars'],
                        'true_sentiment': true_labels,
                        'predicted_sentiment': predicted_binary,
                        'correct': predicted_binary == true_labels
                    })
                    
                    all_results.append(batch_results) # append to the final result
                    
                    # memory cleanup after one batch
                    del batch_data, df, texts, true_labels, sequences, padded_sequences, predictions, predicted_binary, batch_results
                    gc.collect()
                    
                    skip += batch_size
                    
                    # save the predictions for the current batch
                    if batch_num % 5 == 0 >= total_count:
                        if all_results:
                            final_results = pd.concat(all_results, ignore_index=True)
                            final_results.to_csv("outputs/predictions_with_existing_model.csv", index=False)
                            print(f"✅ Checkpoint saved: {len(final_results)} predictions")
                
                client.close()
                
                # connect all the results onto 1 file
                if all_results:
                    final_results = pd.concat(all_results, ignore_index=True)     
                   # accuracy = final_results['correct'].mean()
                   
                    # Calculate accuracy on new data
                    accuracy = np.mean(predicted_binary == true_labels)

                    # total_predictions = len(final_results)
                    # positive_predictions = final_results['predicted_sentiment'].sum()
                    # negative_predictions = total_predictions - positive_predictions
                    
                    print(f"\n" + "="*60)
                    print(f"PREDICTION RESULTS WITH EXISTING MODEL:")
                    print(f"Accuracy on new data: {accuracy * 100:.2f}%")
                    # print(f"Total predictions: {len(total_predictions)}")
                    # print(f"Positive predictions: {(positive_predictions)}")
                    # print(f"Negative predictions: {negative_predictions}")
                    print("="*60)

                    final_results.to_csv("outputs/predictions_with_existing_model.csv", index=False)
                    print("✅ Predictions saved to 'outputs/predictions_with_existing_model.csv'")
                
                    # Show some examples
                    print("\nSample predictions:")
                    sample_df = final_results.head(10)
                    for idx, row in sample_df.iterrows():
                        text_preview = row['text'][:60] + "..." if len(row['text']) > 60 else row['text']
                        true_label = "Positive" if row['true_sentiment'] == 1 else "Negative"
                        pred_label = "Positive" if row['predicted_sentiment'] == 1 else "Negative"
                        correct_symbol = "✅" if row['correct'] else "❌"
                        print(f"{correct_symbol} True: {true_label:8} | Pred: {pred_label:8} | Stars: {row['true_stars']} | Text: {text_preview}")
                    
                    prediction_successful = 0

                    return  # Exit function after prediction
                else:
                    print("❌ No data was processed!")
                
            except Exception as e:
                print(f"❌ Error loading data from MongoDB: {e}")
                prediction_successful = 0
        except Exception as e:
            print(f"❌ Error loading existing model: {e}")
            print("Will train a new model...")
    
    if prediction_successful == 0:
        print("❌ No existing model found or prediction failed. Training new model...")
    print("❌ No existing model found or loading failed. Training new model...")
   
    # --- 1. Loading data directly fromt he MongoDB ---
    print("[INFO] Loading data directly from the MongoDB...")
    try:
        client = MongoClient("mongodb://127.0.0.1:27017/")
        db = client.data0
        collection = db.review
        
        # take only 4,5 stars - positive, 1,2 stars -negative reviews
        pipeline = [
            {"$match": {
                "text": {"$ne": None},
                "stars": {"$ne": 3} # take out 3 stars reviews
            }},
            {"$sample": {"size": 1000000}},
            {"$project": {"text": 1, "stars": 1}}
        ]
        
        print("[INFO] Downloading...")
        cursor = collection.aggregate(pipeline)
        data = list(cursor)
        df = pd.DataFrame(data)
        client.close()
        
        print(f"✅ Loading {len(df)} rows from MongoDB")
        
    except Exception as e:
        print(f"❌ Error loading from MongoDB: {e}")
        return

    # --- 2. Data preparation ---
    print("[INFO] Data preparation...")
    
    # change 4,5 stars -> 1 (positive), 1,2 -> 0 (negative)
    df['label'] = df['stars'].apply(lambda x: 1 if x > 3 else 0)
    
    texts = df['text'].astype(str).values
    labels = df['label'].values

    # Divide testing and training data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"[INFO] Training data: {len(X_train)} samples")
    print(f"[INFO] Test data: {len(X_test)} samples")

    # Tokenization of text
    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # padding of sequences
    max_length = 200
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

    # --- 3. Creating model ---
    def create_tf_model():
        model = keras.Sequential([
            layers.Embedding(20000, 64, input_length=max_length),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)), # Bidirectional je silnejší
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            # ZMENA: 1 výstupný neurón + sigmoid pre binárnu klasifikáciu
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            loss='binary_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.0001), # slower learning rate for stability
            metrics=['accuracy']
        )
        return model

    model = create_tf_model()
    print("✅ Model created")
    print(model.summary())

    # --- 4. Training ---
    print("[INFO] Starting training...")
    
    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
    ]
    
    history = model.fit(
        X_train_pad, y_train,
        batch_size=256,
        epochs=15,
        validation_data=(X_test_pad, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # --- 5. Evaluation ---
    test_loss, test_accuracy = model.evaluate(X_test_pad, y_test, verbose=0)

    print(f"\n" + "="*60)
    print(f"TensorFlow GPU Model accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Number of training data: {len(X_train)}")
    print(f"Number of testing data: {len(X_test)}")
    print("="*60)

    model.save('models/sentiment_model_tensorflow_gpu.keras')
    print("✅ Model saved to 'models/sentiment_model_tensorflow_gpu'")

    with open('models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("✅ Tokenizer saved to 'models/tokenizer.pickle'")

    with open('models/model_config.json', 'w') as f:
        json.dump({'max_length': max_length}, f)
    print("✅ Model configuration saved")

def task_start_8(spark):
    print("\n###### Research 8: Attribute Correlation (WiFi & Parking vs Stars) ######")

    # Pick establishments with wifi attrib and clean it
    df = read_collection(spark, "business")
    
    print("\n[INFO] Analysis of WiFi influence on the rating...")
    df_wifi = df.select(
        col("stars"),
        regexp_replace(col("attributes.WiFi"), "^u'|'$|'", "").alias("wifi_cleaned") # clear the wifi attr of trash
    ).filter(
        col("wifi_cleaned").isNotNull() & (col("wifi_cleaned") != "None")
    )
    
    # aggregation
    wifi_stats = df_wifi.groupBy("wifi_cleaned").agg(
        round(avg("stars"), 2).alias("avg_rating"),
        count("stars").alias("count")
    ).orderBy(desc("avg_rating"))
    
    print("   -> Results for WiFi:")
    wifi_stats.show()
    save_output(wifi_stats, "research_2_wifi_correlation")
    
    print("\n[INFO] Analysis of parking lot...")
    df_parking = df.filter(col("attributes.BusinessParking").isNotNull())
    df_parking = df_parking.withColumn("has_lot_parking", # create new col has_lot_parking which will be yes or no based on a lot info
        when(col("attributes.BusinessParking").contains("'lot': True"), "Yes")
        .otherwise("No")
    )
    
    # aggregation
    parking_stats = df_parking.groupBy("has_lot_parking").agg(
        round(avg("stars"), 2).alias("avg_rating"),
        count("stars").alias("count")
    ).orderBy(desc("avg_rating"))
    
    print("   -> Results for parking lot:")
    parking_stats.show()
    save_output(parking_stats, "research_2_parking_correlation")

def task_start_9(spark):
    print("\n###### Research 9: Hidden Dependencies (Review Length vs Stars) ######")

    df = read_collection(spark, "review")
    
    df_clean = df.filter(col("text").isNotNull()) # filter out empty reviews
    
    df_len = df_clean.withColumn("review_length", length(col("text"))) # get the length of the review
    
    # aggregation based on the stars
    # get the avg len for each category (1 star, 2 stars ...)
    result = df_len.groupBy("stars").agg(
        round(avg("review_length"),2).alias("avg_length"),
        count("review_id").alias("review_count")
    ).orderBy("stars")
    
    print("[INFO] AVG length of the review based on the stars: ")
    result.show()
    save_output(result, "research_3_length_vs_stars")
    
    correlation = df_len.stat.corr("stars", "review_length")
    print(f"\n   -> Linear correlation (Stars vs Length): {correlation:.4f}")
    
    print("-" * 50)
    if abs(correlation) < 0.1:
        print("[INFO]: Low linear correlation showing that relationship is not rectilinear")
        print("Check the table above - we are looking for a 'U' shape (long texts on the edges).")
    else:
        print("[INFO]: There is a correlation")

        
    
def task_start_10(spark):
    print("\n###### Research 10: Predictive Task (Random Forest - Predicting Fans) ######")
    
    # load the data
    print("[INFO] Loading and preparing data...")
    df_user = read_collection(spark, "user")
    
    # prepare the data
    df_clean = df_user.select(
        col("review_count").cast("double"),
        col("average_stars").cast("double"),
        col("fans").cast("double").alias("label"),
        size(split(col("friends"), ",")).alias("friend_count")
    ).filter(
        (col("review_count") > 0) & 
        (col("review_count") <= 500) &  # filtering out extremes
        col("average_stars").isNotNull() &
        (col("fans") <= 200)  # filtering out extremes
    ).dropna()
    
    print(f"[DEBUG] Number of rows after filtering: {df_clean.count()}")
    
    # vector assembler
    feature_columns = ["review_count", "average_stars", "friend_count"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df_ml = assembler.transform(df_clean).select("features", "label")
    
    # split
    df_sampled = df_ml.sample(False, 0.5, seed=42) 
    print(f"[DEBUG] Počet vzorky: {df_sampled.count()}")
    
    (training_data, test_data) = df_sampled.randomSplit([0.8, 0.2], seed=42)
    
    # Using cache for saving up memory
    training_data = training_data.coalesce(4).cache()
    test_data = test_data.coalesce(2).cache()
    training_data.count() 
    test_data.count()     
    
    # model training
    print("[INFO] Training Random Forest Regressor...")
    rf = RandomForestRegressor(
        featuresCol="features", 
        labelCol="label", 
        numTrees=50,    
        maxDepth=25, 
        seed=42
    )
    model = rf.fit(training_data)
    
    # evaluation
    print("[INFO] Evaluation...")
    predictions = model.transform(test_data)
    
    evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse") # average error
    evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2") # model quality
    
    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    
    print("\n" + "="*60)
    print(f"RESULTS OF RANDOM FOREST (Fans Prediction):")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R2 (model accuracy): {r2:.4f}")
    print("="*60)
    
    print("\nAttribute importance (Feature Importance):")
    importances = model.featureImportances.toArray()
    
    importance_data = []
    for i, imp in enumerate(importances):
        name = feature_columns[i]
        print(f"  {name:15}: {imp:.4f} ({imp*100:.1f}%)")
        importance_data.append((name, float(imp)))
    
    imp_df = spark.createDataFrame(importance_data, ["feature", "importance"])
    save_output(imp_df, "research_4_feature_importance")
    
    predictions.select("label", "prediction").show(5)
    save_output(predictions.select("label", "prediction"), "research_4_predictions_sample")
    
    # clean cache
    training_data.unpersist()
    test_data.unpersist()
            
def task_start_11(spark):
    print("\n###### Research 11: Anomaly Detection (Statistical & Semantic) ######")
    
    # --- Statistical anomalies ---
    df_bus = read_collection(spark, "business")
    
    # definition of the suspicious establishment
    # Too much reviews but low rating ?
    # Too little reviews but perfect rating ?
    
    suspicious_bad = df_bus.filter((col("review_count") > 500) & (col("stars") < 2.0))
    suspicious_good = df_bus.filter((col("review_count") < 5) & (col("stars") == 5.0))
    
    print(f"   ->Number of popular but bad rated establishments: {suspicious_bad.count()}")
    print(f"   ->Number of unpopular but good rated establishments: {suspicious_good.count()}")
    save_output(suspicious_bad, "research_5_statistical_anomalies")
    
    print("\n[INFO] Searching reviews which does not corresponds to the rating (TensorFlow)...")
    
    model_path = 'models/sentiment_model_tensorflow_gpu.keras'
    tokenizer_path = 'models/tokenizer.pickle'
    config_path = 'models/model_config.json'
    
    if not os.path.exists(model_path):
        print("❌Model does not exist, please run the task number 7")
        return
    
    try:
        # load model
        model = keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        # load data from mongodb
        #collection = read_collection(spark, "review")
        client = MongoClient("mongodb://127.0.0.1:27017/")
        db = client.data0 
        collection = db.review
        
        pipeline = [
            {"$match": {"stars": {"$in": [1, 5]}, "text": {"$ne": None}}},
            {"$sample": {"size": 1000000}},
            {"$project": {"text": 1, "stars": 1, "business_id": 1}}
        ]
        print("   -> Downloading data for the analysis...")
        df_reviews = pd.DataFrame(list(collection.aggregate(pipeline)))
        
        # predictions
        texts = df_reviews['text'].astype(str).values
        with open(config_path, 'r') as f:
            config = json.load(f)
        max_length = config['max_length'] # same as for traingn the model
        
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        
        preds = model.predict(padded, verbose=0)
        df_reviews['sentiment_score'] = preds
        
        # searching for anomalies
        anomalies_pos_star_neg_text = df_reviews[ # good rating but model predict negative rating
            (df_reviews['stars'] == 5) & (df_reviews['sentiment_score'] < 0.2)
        ]
        
        anomalies_neg_star_pos_text = df_reviews[ # bad rating but model predict positive rating
            (df_reviews['stars'] == 1) & (df_reviews['sentiment_score'] > 0.8)
        ]
        
        print(f"\n   -> Found {len(anomalies_pos_star_neg_text)} anomalies (5* ratings, negative text)")
        if not anomalies_pos_star_neg_text.empty:
            print("      Examples:")
            for txt in anomalies_pos_star_neg_text['text'].head(3):
                print(f"      - {txt[:80]}...")

        print(f"\n   -> Found {len(anomalies_neg_star_pos_text)} anomalies (1* ratings, positive text)")
        if not anomalies_neg_star_pos_text.empty:
            print("      Examples:")
            for txt in anomalies_neg_star_pos_text['text'].head(3):
                print(f"      - {txt[:80]}...")
        
        # save the results
        all_anomalies = pd.concat([anomalies_pos_star_neg_text, anomalies_neg_star_pos_text])
        all_anomalies.to_csv("outputs/research_5_semantic_anomalies.csv", index=False)
        print("✅ Anomalies saved to the file.")
    except Exception as e:
        print(f"❌ Error occured while analyzing: {e}")
    
 
    
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
    # data science
    elif choice == '71': task_start_71(spark)
    elif choice == '7': task_start_7()
    elif choice == '8': task_start_8(spark)
    elif choice == '9': task_start_9(spark)
    elif choice == '10': task_start_10(spark)
    elif choice == '11': task_start_11(spark)

    elif choice == '0':
        print("Exiting.")
    else:
        print("Invalid choice")

    spark.stop()

if __name__ == "__main__":
    main()

