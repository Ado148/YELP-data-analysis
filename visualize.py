from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Nastavenie štýlu
sns.set_theme(style="whitegrid")
OUTPUT_DIR = "outputs"
IMAGE_DIR = "report_images"

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def plot_stat_1():
    print("Visualizing Task 1: Top Categories")
    df = pd.read_csv(f"{OUTPUT_DIR}/stat_1_top_categories.csv")
    # plt.figure(figsize=(12, 8))
    # sns.barplot(data=df.head(20), x='count', y='category', palette='viridis')
    # plt.title('Top 20 Business Categories on Yelp')
    # plt.xlabel('Number of Businesses')
    # plt.tight_layout()
    # plt.savefig(f"{IMAGE_DIR}/task1_top_categories.png")
    # plt.close()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Graf 1: Bar Chart
    sns.barplot(data=df, x='count', y='category', ax=ax1, palette="Blues_r")
    ax1.set_title("TOP 15 Categories (Absolute Count)")
    
    # Graf 2: Pie Chart (Podiel TOP 15)
    ax2.pie(df.head(15)['count'], labels=df.head(15)['category'], autopct='%1.1f%%')
    ax2.set_title("Market Share of Top 15 Categories")
    
    plt.tight_layout()
    plt.savefig(f"{IMAGE_DIR}/task1_combined.png")

def plot_stat_2():
    print("Visualizing Task 2: Geo Distribution")
    df = pd.read_csv(f"{OUTPUT_DIR}/stat_2_geo_dist.csv")
    top_cities = df.sort_values(by='business_count', ascending=False).head(15)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Graf A: Horizontálny bar chart pre mestá
    sns.barplot(data=top_cities, x='business_count', y='city', hue='state', ax=ax1, palette="muted")
    ax1.set_title('Top 15 Cities by Business Density')
    
    # Graf B: Distribúcia podľa štátov (agregované z pôvodných dát)
    state_counts = df.groupby('state')['business_count'].sum().sort_values(ascending=False).head(8)
    ax2.pie(state_counts, labels=state_counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set3"))
    ax2.set_title('Market Share by State (Top 8)')
    
    plt.tight_layout()
    plt.savefig(f"{IMAGE_DIR}/task2_geo_final.png")
    plt.close()

def plot_stat_3():
    print("Visualizing Task 3: Open vs Closed")
    df = pd.read_csv(f"{OUTPUT_DIR}/stat_3_open_closed.csv")
    top_cities = df.head(12) # Vezmeme top 12 pre prehľadnosť
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Graf A: Percentuálny podiel otvorených (pôvodný)
    sns.barplot(data=top_cities, x='open_percentage', y='city', ax=ax1, color='skyblue')
    ax1.axvline(x=100, color='red', linestyle='--')
    ax1.set_title('Survival Rate: Percentage of Open Businesses')
    
    # Graf B: Absolútne počty (Stacked bar)
    # Musíme vypočítať 'closed' z total_count a is_open
    top_cities_plot = top_cities.copy()
    top_cities_plot['closed_count'] = top_cities_plot['total_count'] - top_cities_plot['open_count']
    
    top_cities_plot[['city', 'open_count', 'closed_count']].set_index('city').plot(
        kind='barh', stacked=True, ax=ax2, color=['#2ecc71', '#e74c3c']
    )
    ax2.set_title('Absolute Count of Open vs Closed Businesses')
    ax2.legend(["Open", "Closed"])
    
    plt.tight_layout()
    plt.savefig(f"{IMAGE_DIR}/task3_status_combined.png")
    plt.close()

def plot_stat_4():
    print("Visualizing Task 4: Price Range")
    df = pd.read_csv(f"{OUTPUT_DIR}/stat_4_price_dist.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Graf A: Koláčový graf (proporcie)
    ax1.pie(df['count'], labels=df['price_range'], autopct='%1.1f%%', 
            colors=sns.color_palette('pastel'), startangle=140)
    ax1.set_title('Proportional Distribution of Price Ranges')
    
    # Graf B: Bar chart (porovnanie veľkosti)
    sns.barplot(data=df, x='price_range', y='count', ax=ax2, palette='muted')
    ax2.set_title('Absolute Number of Restaurants per Price Range')
    ax2.set_yscale('log') # Logaritmická mierka ak sú veľké rozdiely
    
    plt.tight_layout()
    plt.savefig(f"{IMAGE_DIR}/task4_price_combined.png")
    plt.close()

def plot_stat_5():
    print("Visualizing Task 5: User Activity Over Time")
    df = pd.read_csv(f"{OUTPUT_DIR}/stat_5_user_activity.csv")
    # Pre lepšiu čitateľnosť vezmeme len posledných 5 rokov alebo vzorku
    df['year_month'] = pd.to_datetime(df['year_month'])
    df = df.sort_values('year_month')
    plt.figure(figsize=(14, 6))
    plt.plot(df['year_month'], df['count'], marker='o', color='teal', linewidth=1, markersize=2)
    plt.title('Monthly Review Activity Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Reviews')
    plt.savefig(f"{IMAGE_DIR}/task5_activity_time.png")
    plt.close()

def plot_stat_6():
    print("Visualizing Task 6: Top 100 Users (Showing top 20)")
    df = pd.read_csv(f"{OUTPUT_DIR}/stat_6_top_users.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Graf A: TOP 20 Influencerov
    sns.barplot(data=df.head(20), x='review_count', y='name', ax=ax1, palette="rocket")
    ax1.set_title('Top 20 Power Users (by Review Count)')
    
    # Graf B: Distribúcia (Boxplot alebo Violin plot pre top 100)
    sns.boxenplot(x=df['review_count'], ax=ax2, color="orange")
    ax2.set_title('Statistical Distribution of Top 100 Users Activity')
    
    plt.tight_layout()
    plt.savefig(f"{IMAGE_DIR}/task6_users_final.png")
    plt.close()
    
    
    
def plot_sentiment_research_results(history=None):
    print("Visualizing Research Task 7: Sentiment Analysis Results")
    IMAGE_DIR = "report_images"
    
    # GRAF 1: História učenia (Loss a Accuracy)
    # Ak máš objekt 'history' z model.fit(), použi ho. Ak nie, tento blok preskoč.
    if history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_title('Model Accuracy Progress')
        ax1.legend()
        
        ax1.plot(history.history['loss'], label='Train Loss')
        ax1.plot(history.history['val_loss'], label='Val Loss')
        ax2.set_title('Model Loss Progress')
        ax2.legend()
        plt.savefig(f"{IMAGE_DIR}/research7_training_history.png")
    
    # GRAF 2: Distribúcia predikcií (z tvojho 3.3GB CSV)
    # Načítame len stĺpce, ktoré potrebujeme, a to po častiach (chunking)
    print("Processing large CSV for sentiment distribution...")
    positive_count = 0
    negative_count = 0
    correct_count = 0
    total_processed = 0

    # Čítame po 500k riadkoch, aby sme nezhodili RAM
    for chunk in pd.read_csv("outputs/predictions_with_existing_model.csv", 
                             usecols=['predicted_sentiment', 'correct'], chunksize=500000):
        positive_count += chunk['predicted_sentiment'].sum()
        negative_count += (len(chunk) - chunk['predicted_sentiment'].sum())
        correct_count += chunk['correct'].sum()
        total_processed += len(chunk)

    # Vizualizácia výsledkov
    plt.figure(figsize=(10, 6))
    labels = ['Positive Sentiment', 'Negative Sentiment']
    counts = [positive_count, negative_count]
    
    sns.barplot(x=labels, y=counts, palette=['#2ecc71', '#e74c3c'])
    plt.title(f'Overall Sentiment Distribution (Total: {total_processed} reviews)')
    plt.ylabel('Number of Reviews')
    
    # Pridáme text s presnosťou (Accuracy) priamo do grafu
    accuracy = (correct_count / total_processed) * 100
    plt.text(0.5, max(counts)*0.9, f"Model Accuracy: {accuracy:.2f}%", 
             ha='center', fontsize=15, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig(f"{IMAGE_DIR}/research7_sentiment_dist.png")
    plt.close()
    
def plot_confusion_matrix_large():
    print("Generating Confusion Matrix from large dataset (cleaning NaNs)...")
    y_true = []
    y_pred = []
    
    # Čítame po kúskoch
    chunk_iterator = pd.read_csv(
        "outputs/predictions_with_existing_model.csv", 
        usecols=['true_sentiment', 'predicted_sentiment'], 
        chunksize=500000
    )

    for chunk in chunk_iterator:
        # 1. ODSTRÁNENIE CHÝBAJÚCICH HODNÔT (NaN)
        # Toto vyhodí riadky, kde chýba true alebo predikovaný sentiment
        clean_chunk = chunk.dropna(subset=['true_sentiment', 'predicted_sentiment'])
        
        # 2. PRETYPOVANIE na int (aby tam neboli floaty ako 1.0)
        y_true.extend(clean_chunk['true_sentiment'].astype(int).values)
        y_pred.extend(clean_chunk['predicted_sentiment'].astype(int).values)
    
    print(f"Data cleaned. Total valid records for matrix: {len(y_true)}")

    # Výpočet matice
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    # Pridáme aj percentá pre lepší vizuál
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Heatmapa: annot=True napíše počty, cm_percent pridá farbu podľa úspešnosti
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    
    plt.title('Confusion Matrix: Sentiment Analysis (Bi-LSTM)')
    plt.ylabel('Actual Label (Ground Truth)')
    plt.xlabel('Predicted Label (Model)')
    plt.tight_layout()
    plt.savefig("report_images/research7_confusion_matrix_final.png")
    plt.close()
    print("✅ Confusion matrix saved to report_images/research7_confusion_matrix_final.png")
    
    
    
def plot_stat_8():
    df_wifi = pd.read_csv(f"{OUTPUT_DIR}/research_2_wifi_correlation.csv")
    df_parking = pd.read_csv(f"{OUTPUT_DIR}/research_2_parking_correlation.csv")

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Graf 1: WiFi vs Rating
    sns.barplot(data=df_wifi, x='wifi_cleaned', y='avg_rating', palette='coolwarm', ax=ax1)
    ax1.set_title('Impact of WiFi on Average Rating')
    ax1.set_ylim(3.0, 4.0)  # Zoom pre lepšiu viditeľnosť rozdielov
    for p in ax1.patches:
        ax1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    # Graf 2: Parking vs Rating
    sns.barplot(data=df_parking, x='has_lot_parking', y='avg_rating', palette='viridis', ax=ax2)
    ax2.set_title('Impact of Parking Lot on Average Rating')
    ax2.set_ylim(3.0, 4.0)
    for p in ax2.patches:
        ax2.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(f"{IMAGE_DIR}/research_8_attributes_final.png")
    
def plot_stat_9():
    df = pd.read_csv(f"{OUTPUT_DIR}/research_3_length_vs_stars.csv")

    # Vytvorenie grafu
    plt.figure(figsize=(10, 6))
    sns.barplot(x='stars', y='avg_length', data=df, palette='magma')

    # Nastavenie názvov a popisov
    plt.title('Relationship between review length and number of stars', fontsize=14)
    plt.xlabel('Number of stars', fontsize=12)
    plt.ylabel('Average text length (number of characters)', fontsize=12)

    # Pridanie hodnôt nad stĺpce
    for p in plt.gca().patches:
        plt.gca().annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{IMAGE_DIR}/research_9_length_analysis.png")
    
def plot_stat_10():
    # Načítanie výsledkov z tvojich Spark úloh
    df_imp = pd.read_csv(f"{OUTPUT_DIR}/research_4_feature_importance.csv")
    df_pred = pd.read_csv(f"{OUTPUT_DIR}/research_4_predictions_sample.csv")

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Graf 1: Dôležitosť atribútov
    df_imp = df_imp.sort_values(by='importance', ascending=False)
    sns.barplot(data=df_imp, x='importance', y='feature', palette='magma', ax=ax1)
    ax1.set_title('Feature prediction (Predicting fans count)')

    # Graf 2: Skutočnosť vs Predikcia (Scatter plot)
    # Zoberieme vzorku 2000 bodov pre prehľadnosť
    sample = df_pred.sample(min(len(df_pred), 2000))
    sns.scatterplot(data=sample, x='label', y='prediction', alpha=0.5, ax=ax2, color='teal')
    # Ideálna čiara (x=y)
    max_v = max(sample['label'].max(), sample['prediction'].max())
    ax2.plot([0, max_v], [0, max_v], color='red', linestyle='--')
    ax2.set_title('Actual vs predicted Fans')

    plt.tight_layout()
    plt.savefig(f"{IMAGE_DIR}/research_10_rf_fans_final.png")
    
def plot_stat_11():
    df_statistical = pd.read_csv(f"{OUTPUT_DIR}/research_5_statistical_anomalies.csv")
    df_semantic = pd.read_csv(f"{OUTPUT_DIR}/research_5_semantic_anomalies.csv")

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Graf 1: Sémantické anomálie (Nesúlad Sentimentu)
    df_semantic['anomaly_type'] = df_semantic.apply(
        lambda x: "5 stars Rating / Negative Text" if x['stars'] == 5 else "1 star Rating / Positive Text", axis=1
    )
    sns.countplot(data=df_semantic, x='anomaly_type', palette=['#e74c3c', '#2ecc71'], ax=ax1)
    ax1.set_title('Semantic Anomalies: Sentiment mismatch')
    ax1.set_ylabel('Number of reviews')

    # Graf 2: Štatistické anomálie (Top podozrivé podniky)
    top_bad = df_statistical.sort_values(by='review_count', ascending=False).head(10)
    sns.barplot(data=top_bad, x='review_count', y='name', palette='Reds_r', ax=ax2)
    ax2.set_title('Top statistical anomalies (Review count > 500, Stars < 2)')
    ax2.set_xlabel('Number of reviews')

    plt.tight_layout()
    plt.savefig(f"{IMAGE_DIR}/research_11_anomalies_final.png")

if __name__ == "__main__":
    plot_stat_1()
    plot_stat_2()
    plot_stat_3()
    plot_stat_4()
    plot_stat_5()
    plot_stat_6()
    #plot_sentiment_research_results()   # 7
    #plot_confusion_matrix_large()       # 7
    #plot_stat_8()
    #plot_stat_9()
    #plot_stat_10()
    plot_stat_11()
    
    print(f"\n✅ All plots saved to '{IMAGE_DIR}' folder!")