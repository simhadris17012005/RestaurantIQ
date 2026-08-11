"""
recommender.py
Content-based restaurant recommendation engine based on
cuisine preference, price range, and city.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import load_data, clean_data


def build_content_matrix(df):
    """Combine cuisine + city + price range into one text profile per restaurant."""
    df = df.copy()
    df['profile'] = (
        df['Cuisines'].astype(str) + ' ' +
        df['City'].astype(str) + ' ' +
        ('price_' + df['Price range'].astype(str))
    )
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(df['profile'])
    return df, matrix, tfidf


def recommend(cuisine, city=None, price_range=None, top_n=5, df=None, matrix=None):
    """Return top_n restaurants matching a user's preference profile."""
    if df is None or matrix is None:
        raw = clean_data(load_data())
        df, matrix, _ = build_content_matrix(raw)

    query_parts = [cuisine]
    if city:
        query_parts.append(city)
    if price_range:
        query_parts.append(f'price_{price_range}')
    query = ' '.join(query_parts)

    tfidf = TfidfVectorizer(stop_words='english')
    combined = df['profile'].tolist() + [query]
    tfidf_matrix = tfidf.fit_transform(combined)
    sims = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    df = df.copy()
    df['similarity'] = sims
    results = df.sort_values('similarity', ascending=False).head(top_n)
    return results[['Restaurant Name', 'City', 'Cuisines', 'Price range',
                     'Aggregate rating', 'similarity']]


if __name__ == "__main__":
    raw = clean_data(load_data())
    df, matrix, tfidf = build_content_matrix(raw)

    print("Sample recommendation -> Italian food, New Delhi, price range 2")
    recs = recommend("Italian", city="New Delhi", price_range=2, top_n=5, df=df, matrix=matrix)
    print(recs.to_string(index=False))

    print("\nSample recommendation -> Chinese food, price range 3")
    recs2 = recommend("Chinese", price_range=3, top_n=5, df=df, matrix=matrix)
    print(recs2.to_string(index=False))
