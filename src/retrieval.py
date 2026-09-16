import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HistoricalRetriever:
    """
    TF-IDF based retriever for historical Apple Support conversations.
    """

    def __init__(self, texts):
        self.texts = list(texts)

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=100_000,
            sublinear_tf=True,
        )

        self.matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query, top_k=5):
        """
        Retrieve the most similar historical customer messages.

        Returns a list of dictionaries containing:
        - index
        - customer_text
        - similarity
        """

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []

        for idx in top_indices:
            results.append(
                {
                    "index": int(idx),
                    "customer_text": self.texts[idx],
                    "similarity": float(similarities[idx]),
                }
            )

        return results

    def best_similarity(self, query):
        """
        Return the similarity score of the best historical match.
        """

        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

        return float(similarities.max())
