# YELP dataset analysis

**Dataset used in this project:** https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset

#### How to execute the script:
python3 spark_connector.py [num]

- num is a number of the task we want to process (from 1-12)



#### Tasks:

**Statistical tasks (Business Intelligence)**

1. Analysis of top categories: Display the TOP 20 most frequent business categories (from the categories field in business.json) in the entire dataset.

2. Geographic distribution of ratings: Calculate the average rating (stars) and total number of businesses (business_id) for each state and city.

3. Analysis of open businesses: Determining the percentage of businesses that are still open (is_open = 1) compared to closed ones (is_open = 0), grouped by city.

4. Restaurant price distribution: Analysis of the distribution of price categories (attribute RestaurantsPriceRange2) for all businesses in the "Restaurants" category.

5. User activity over time: Create a time series showing the total number of reviews written (from review.json) by month and year of creation (date field).

6. Identification of the most active users: Compilation of a TOP 100 user ranking (from user.json) based on their total number of reviews written (review_count).

**Research tasks (Data Science)**

1. Predictive task (Sentiment Analysis): Create a machine learning model that predicts ratings (stars in review.json, e.g., 1-5) based solely on the text of the review (text).

2. Searching for hidden dependencies (Attribute Correlation): Statistical analysis of whether and to what extent specific attributes (from attributes in business.json) influence the final rating (stars). For example:

Do businesses with WiFi = 'free' have a higher average rating than those with WiFi = 'no'?

	Do businesses with BusinessParking = 'lot' (private parking) have higher ratings?

3. Predictive task (user impact prediction): Create a regression model to predict the number of "fans" (fans in user.json) a user will have. The input variables can be review_count, average_stars, and the number of friends (length of the friends list).

4. Missing data recovery / Anomaly detection: Identification of businesses that have a suspiciously high number of reviews (review_count) but a very low average rating (stars) – and vice versa. (This may indicate, for example, purchased reviews or an anomaly in the data).
 
5. Social network analysis (Social Mining): (This area is directly mentioned in your assignment). Use the list of friends (friends in user.json) to create a social graph. Subsequently, identify "communities" or "bubbles" (clustering) and analyze whether these communities prefer the same categories of businesses.
