import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

class SentimentAnalysisAgent:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of a given text.

        Args:
            text (str): The text to be analyzed.

        Returns:
            dict: A dictionary containing the sentiment scores.
        """
        sentiment_scores = self.sia.polarity_scores(text)
        return sentiment_scores

    def classify_sentiment(self, text):
        """
        Classify the sentiment of a given text as positive, negative, or neutral.

        Args:
            text (str): The text to be classified.

        Returns:
            str: The sentiment classification.
        """
        sentiment_scores = self.analyze_sentiment(text)
        compound_score = sentiment_scores['compound']

        if compound_score >= 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    def analyze_text_sentiment(self, text):
        """
        Analyze the sentiment of a given text and print the results.

        Args:
            text (str): The text to be analyzed.
        """
        sentiment_scores = self.analyze_sentiment(text)
        print("Sentiment Scores:")
        print(f"Positive: {sentiment_scores['pos']}")
        print(f"Negative: {sentiment_scores['neg']}")
        print(f"Neutral: {sentiment_scores['neu']}")
        print(f"Compound: {sentiment_scores['compound']}")

        sentiment_classification = self.classify_sentiment(text)
        print(f"Sentiment Classification: {sentiment_classification}")

# Example usage:
if __name__ == "__main__":
    agent = SentimentAnalysisAgent()
    text = "I love this product! It's amazing."
    agent.analyze_text_sentiment(text)
