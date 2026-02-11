from collections import ChainMap
from collections import Counter
from typing import Callable, Dict, Set
import re

import pandas as pd


class FeatureMap:
    name: str

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        pass

    @classmethod
    def prefix_with_name(cls, d: Dict) -> Dict[str, float]:
        """just a handy shared util function"""
        return {f"{cls.name}/{k}": v for k, v in d.items()}


class BagOfWords(FeatureMap):
    name = "bow"
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        words = text.lower().split()
        features = {}
        for w in words:
            if w not in cls.STOP_WORDS:
                features[w] = 1.0
        return cls.prefix_with_name(features)


class BagOfWordsCounts(FeatureMap):
    name = "bow_counts"
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        words = text.lower().split()
        features = {}
        for w in words:
            if w not in cls.STOP_WORDS:
                if w in features:
                    features[w] += 1.0
                else:
                    features[w] = 1.0
        return cls.prefix_with_name(features)


class Bigrams(FeatureMap):
    name = "bigram"
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        words = text.lower().split()
        words = [w for w in words if w not in cls.STOP_WORDS]
        
        features = {}
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            features[bigram] = 1.0
        return cls.prefix_with_name(features)


class Trigrams(FeatureMap):
    name = "trigram"
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        words = text.lower().split()
        words = [w for w in words if w not in cls.STOP_WORDS]
        
        features = {}
        for i in range(len(words) - 2):
            trigram = f"{words[i]}_{words[i+1]}_{words[i+2]}"
            features[trigram] = 1.0
        return cls.prefix_with_name(features)


class SentenceLength(FeatureMap):
    name = "len"

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        length = len(text.split())
        
        features = {}
        if length < 5:
            features["very_short"] = 1.0
        elif length < 10:
            features["short"] = 1.0
        elif length < 20:
            features["medium"] = 1.0
        elif length < 30:
            features["long"] = 1.0
        else:
            features["very_long"] = 1.0
        
        features["num_words"] = float(length)
        
        return cls.prefix_with_name(features)


class SentimentWords(FeatureMap):
    name = "sentiment"
    
    POSITIVE_WORDS = {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
        'love', 'best', 'awesome', 'perfect', 'beautiful', 'brilliant',
        'outstanding', 'superb', 'magnificent', 'enjoy', 'enjoyed', 'happy',
        'delightful', 'impressive', 'compelling', 'entertaining', 'fun',
        'liked', 'remarkable', 'terrific', 'fabulous', 'extraordinary',
        'finest', 'pleasure', 'recommend', 'refreshing', 'solid', 'smart',
        'charming', 'powerful', 'moving', 'engaging', 'clever', 'strong',
        'vibrant', 'rich', 'effective', 'gorgeous', 'admirable', 'pleasing'
    }
    
    NEGATIVE_WORDS = {
        'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'boring',
        'dull', 'poor', 'weak', 'disappointing', 'waste', 'stupid', 'mediocre',
        'pointless', 'predictable', 'annoying', 'ridiculous', 'mess', 'fails',
        'failed', 'poorly', 'lacking', 'unfortunately', 'unbearable', 'shallow',
        'empty', 'flat', 'lifeless', 'awkward', 'clumsy', 'tedious', 'tiresome',
        'incoherent', 'confused', 'muddled', 'worthless', 'useless', 'pathetic',
        'forgettable', 'bland', 'uninspired', 'disappoints', 'flawed'
    }

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        words = set(text.lower().split())
        
        features = {}
        
        pos_count = len(words & cls.POSITIVE_WORDS)
        neg_count = len(words & cls.NEGATIVE_WORDS)
        
        features["pos_words"] = float(pos_count)
        features["neg_words"] = float(neg_count)
        
        if pos_count + neg_count > 0:
            features["sentiment_ratio"] = (pos_count - neg_count) / (pos_count + neg_count)
        
        if pos_count > 0:
            features["has_positive"] = 1.0
        if neg_count > 0:
            features["has_negative"] = 1.0
            
        return cls.prefix_with_name(features)


class Punctuation(FeatureMap):
    name = "punct"

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        features = {}
        
        features["exclamation"] = float(text.count('!'))
        features["question"] = float(text.count('?'))
        features["quotes"] = float(text.count('"') + text.count("'"))
        
        words = text.split()
        caps_count = sum(1 for w in words if w.isupper() and len(w) > 1)
        features["all_caps_words"] = float(caps_count)
        
        return cls.prefix_with_name(features)


class Capitalization(FeatureMap):
    name = "caps"

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        features = {}
        
        words = text.split()
        if not words:
            return cls.prefix_with_name(features)
        
        cap_words = sum(1 for w in words if w and w[0].isupper())
        features["capitalized_ratio"] = cap_words / len(words)
        
        if words[0] and words[0][0].isupper():
            features["starts_capital"] = 1.0
        
        return cls.prefix_with_name(features)


class WordStatistics(FeatureMap):
    name = "wordstats"

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        features = {}
        
        words = text.split()
        if not words:
            return cls.prefix_with_name(features)
        
        avg_word_len = sum(len(w) for w in words) / len(words)
        features["avg_word_length"] = avg_word_len
        
        long_words = sum(1 for w in words if len(w) > 6)
        features["long_words"] = float(long_words)
        
        short_words = sum(1 for w in words if len(w) < 3)
        features["short_words"] = float(short_words)
        
        return cls.prefix_with_name(features)


FEATURE_CLASSES_MAP = {
    c.name: c for c in [
        BagOfWords, 
        BagOfWordsCounts,
        Bigrams,
        Trigrams,
        SentenceLength, 
        SentimentWords,
        Punctuation,
        Capitalization,
        WordStatistics
    ]
}


def make_featurize(
    feature_types: Set[str],
) -> Callable[[str], Dict[str, float]]:
    featurize_fns = [FEATURE_CLASSES_MAP[n].featurize for n in feature_types]

    def _featurize(text: str):
        f = ChainMap(*[fn(text) for fn in featurize_fns])
        return dict(f)

    return _featurize


__all__ = ["make_featurize"]
