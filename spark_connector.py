import sys, os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, desc, round, avg, count, sum, to_timestamp, date_format
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import PipelineModel, Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator



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

# --- II. TensorFlow GPU EXPERIMENT (OPTIMIZED FOR 6GB VRAM) ---
def task_start_7():
    print("\n###### Research: 1. Sentiment Analysis (TensorFlow GPU) - Direct MongoDB ######")
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        import numpy as np
        from pymongo import MongoClient
        import pandas as pd
        from sklearn.model_selection import train_test_split
        import pickle
        import json
        print("✅ Everything imported correctly.")
    except ImportError as e:
        print(f"❌ Error on import: {e}")
        return

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Check the GPU
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"✅ GPU avaible: {len(gpu_devices)}")
        try:
            tf.config.experimental.set_memory_growth(gpu_devices[0], True)
        except Exception as e:
            print(f"⚠️ Unable to set memory growth: {e}")
    else:
        print("ℹ️  GPU unavaible, using CPU")
        
        
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
                
                # Get different data for prediction (not the same as training)
                batch_size = 1000000
                total_processed = 0
                pipeline = [
                    {"$match": {
                        "text": {"$ne": None},
                        "stars": {"$ne": 3}  # Remove 3-star reviews
                    }},
                    {"$limit": batch_size}, # go throught the batches
                    {"$project": {"text": 1, "stars": 1}} # which values are interesting to us
                ]
                
                print("[INFO] Loading new data from MongoDB for prediction...")
                cursor = collection.aggregate(pipeline)
                data = list(cursor)
                df = pd.DataFrame(data)
                client.close()
                
                print(f"✅ Loaded {len(df)} new reviews for prediction")
                
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
                
                # Calculate accuracy on new data
                accuracy = np.mean(predicted_binary == true_labels)
                
                print(f"\n" + "="*60)
                print(f"🎯 PREDICTION RESULTS WITH EXISTING MODEL:")
                print(f"Accuracy on new data: {accuracy * 100:.2f}%")
                print(f"Total predictions: {len(predicted_binary)}")
                print(f"Positive predictions: {np.sum(predicted_binary)}")
                print(f"Negative predictions: {len(predicted_binary) - np.sum(predicted_binary)}")
                print("="*60)
                
                # Save predictions
                results_df = pd.DataFrame({
                    'text': texts,
                    'true_stars': df['stars'],
                    'true_sentiment': true_labels,
                    'predicted_sentiment': predicted_binary,
                    'correct': predicted_binary == true_labels
                })
                
                total_processed += len(df)
                print("\n\n\n\n\n\n")
                print(f"="*60 + "Processed {total_processed} documents...")
                
                results_df.to_csv("outputs/predictions_with_existing_model.csv", index=False)
                print("✅ Predictions saved to 'outputs/predictions_with_existing_model.csv'")
                
                # Show some examples
                print("\nSample predictions:")
                sample_df = results_df.head(10)
                for idx, row in sample_df.iterrows():
                    text_preview = row['text'][:60] + "..." if len(row['text']) > 60 else row['text']
                    true_label = "Positive" if row['true_sentiment'] == 1 else "Negative"
                    pred_label = "Positive" if row['predicted_sentiment'] == 1 else "Negative"
                    correct_symbol = "✅" if row['correct'] else "❌"
                    print(f"{correct_symbol} True: {true_label:8} | Pred: {pred_label:8} | Stars: {row['true_stars']} | Text: {text_preview}")
                
                return  # Exit function after prediction
                
            except Exception as e:
                print(f"❌ Error loading prediction data from MongoDB: {e}")
                # Continue to training if prediction fails
        except Exception as e:
            print(f"❌ Error loading existing model: {e}")
            print("Will train a new model...")
    
    print("❌ No existing model found or loading failed. Training new model...")
   
   
    
    
    

    # --- 1. Loading data directly fromt he MongoDB ---
    print("[INFO] Loading data directly from the MongoDB...")
    try:
        client = MongoClient("mongodb://127.0.0.1:27017/")
        db = client.data0
        collection = db.review
        
        # use aggregation for the random pick 
        # pipeline = [
        #     {"$match": {"text": {"$ne": None}}},
        #     {"$sample": {"size": 1000000}}, 
        #     {"$project": {"text": 1, "stars": 1}}
        # ]
        
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

    # print(f"Training data shape: {X_train_pad.shape}")
    # print(f"Testing data shape: {X_test_pad.shape}")

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

    elif choice == '0':
        print("Exiting.")
    else:
        print("Invalid choice")

    spark.stop()

if __name__ == "__main__":
    main()

