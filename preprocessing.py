import nltk
import re

def convertShortForms(article):
    article = article.replace("won't", "will not")
    article = article.replace("can't", "cannot")
    article = article.replace("i'm", "i am")
    article = article.replace("ain't", "is not")
    article = article.replace("(\w+)'ll", "\g<1> will")
    article = article.replace("(\w+)'ve", "\g<1> have")
    article = article.replace("(\w+)'re", "\g<1> are")
    article = article.replace("(\w+)'d", "\g<1> would")
    return article

def convertAbbreviations(article):
    article = article.replace("U.S.", "United States")
    article = article.replace("U.K.", "United Kingdom")
    article = article.replace("etc.", "and so on")
    article = article.replace("e.g.", "for example")
    article = article.replace("i.e.", "more precisely")
    return article

def convertNumbers(article):
    article = re.sub(r'(\d+) million', r'\g<1>000000', article)
    article = re.sub(r'(\d+) billion', r'\g<1>000000000', article)
    article = re.sub(r'(\d+) trillion', r'\g<1>000000000000', article)
    return article

def convertDates(article):
    article = re.sub(r'(\d+)/(\d+)/(\d+)', r'\g<3>-\g<1>-\g<2>', article)
    article = re.sub(r'(\d+)-(\d+)-(\d+)', r'\g<3>-\g<1>-\g<2>', article)
    return article

def convertMoney(article):
    article = re.sub(r'\$(\d+)', r'\g<1> dollars', article)
    return article

def removeStopWords(article):
    article = article.split()
    stopWords = set(nltk.corpus.stopwords.words('english'))
    article = [word for word in article if word not in stopWords]
    article = ' '.join(article)
    return article

def stemWords(article):
    stemmer = nltk.stem.PorterStemmer()
    article = article.split()
    article = [stemmer.stem(word) for word in article]
    article = ' '.join(article)
    return article

def lemmatizeWords(article):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    article = article.split()
    article = [lemmatizer.lemmatize(word) for word in article]
    article = ' '.join(article)
    return article

def removePunctuation(article):
    article = re.sub(r'[^\w\s]', '', article)
    return article

def processArticle(article):
    article = article.lower()
    article = convertShortForms(article)
    article = convertAbbreviations(article)
    article = convertNumbers(article)
    article = convertDates(article)
    article = convertMoney(article)
    article = removePunctuation(article)
    article = removeStopWords(article)
    article = stemWords(article)
    article = lemmatizeWords(article)
    return article