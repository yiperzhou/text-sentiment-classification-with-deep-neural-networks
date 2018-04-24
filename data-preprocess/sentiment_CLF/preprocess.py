def label_sentiment_category(row):
    """
    split reviews into two categories, pos, labled by "1", neg, labeled by "0"
    [0, 1, 2] is neg, [4,5] is pos, 
    [3] is neutral, it needs to be deleted since we only do two-class classification
    """
    if row["score"] in [0, 1, 2]:
        return 0
    if row["score"] in [3]:
        return -1
    if row["score"] in [4, 5]:
        return 1


def clean_punc_and_marks(row):
    """
    remove punctuation, including ???????
    """
    words = nltk.word_tokenize(row["review"])

    words=[word.lower() for word in words if word.isalpha()]
    words = words[:250]
    return " ".join(words)


def cleanSentences(string):
    """
    removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
    """
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


if __name__ == "__main__":
    # connect to the db
    # client = MongoClient()
    # db = client.sentimentAnalysis
    filePath = "/home/yi/Desktop/csv-zusammenfuehren.de_r922bdrm.csv"
    # with open(filePath) as datafile:dd
    #     rawdata = json.load(datafile)
    data = pd.read_csv(filePath)
    
    # data = pd.read_json(filePath)

    # # take Barcelona city hotel reviews as example
    # city = "barcelonaTripadvisor"
    # data = getCollection(collet = city)

    data['sentiment'] = data.apply(lambda row: label_sentiment_category(row),axis=1)
    print("data size : ", data.shape)

    # remove row where sentiment is neutral
    data = data[data.sentiment != -1]

    print("data shape after remove neutral reviews : ", data.shape)
    data['review'] = data.apply(lambda row: clean_punc_and_marks(row),axis=1)

    X = data["review"]
    y = data["sentiment"]



    assert len(X) == len(y)
    print("check the consistent size of reviews and sentiment : ", "review size : ", len(X), "sentiment size: ", len(y))
    
    XY = pd.DataFrame({"review": X, "sentiment":y})
    
    savePath = "/home/yi/Desktop/csv-zusammenfuehren.de_r922bdrm_XY.csv"
    if not os.path.exists(savePath):
        XY.to_csv(savePath)
    
