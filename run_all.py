"""
run_all.py
Runs all four RestaurantIQ modules end-to-end and prints a combined summary.
"""
import rating_prediction
import cuisine_classification
import location_analysis
import recommender
from preprocess import load_data, clean_data

if __name__ == "__main__":
    print("=" * 60)
    print("1. RATING PREDICTION")
    print("=" * 60)
    rating_prediction.train_and_evaluate()

    print("\n" + "=" * 60)
    print("2. RESTAURANT RECOMMENDATION")
    print("=" * 60)
    raw = clean_data(load_data())
    df, matrix, tfidf = recommender.build_content_matrix(raw)
    print(recommender.recommend("Italian", city="New Delhi", price_range=2,
                                 top_n=5, df=df, matrix=matrix).to_string(index=False))

    print("\n" + "=" * 60)
    print("3. CUISINE CLASSIFICATION")
    print("=" * 60)
    cuisine_classification.train_and_evaluate()

    print("\n" + "=" * 60)
    print("4. LOCATION-BASED ANALYSIS")
    print("=" * 60)
    df2 = clean_data(load_data())
    summary = location_analysis.city_summary(df2)
    print(summary.head(10).to_string())
    location_analysis.build_map(df2)

    print("\nAll tasks completed. Check the /outputs folder for saved artifacts.")
