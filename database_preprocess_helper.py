from pymongo import MongoClient


def _connect_mongo(host="localhost", username="", password="", db="5gopt"):
    '''
    connect to mongodb in localhost and access database and collections
    :return:
    '''
    port = 27017

    if username and password:
        mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)

    return conn[db]


def collection_read_mongo(collection, query={}, no_id=True):
    '''
    get specific colelction, and remove the "_id" column
    :param collection:
    :param query:
    :param no_id:
    :return:
    '''
    db = _connect_mongo()
    cursor = db[collection].find(query)
    df = pd.DataFrame(list(cursor))

    if no_id:
        try:
            del df["_id"]
        except:
            pass
    return df


def change_mongo_collection_name(db=MongoClient().test):
    '''
    changing mongo collection name,
    :param db: mongoDB database, default "test" database
    :return:
    '''
    # 35 cities name
    citiesName = ["Amsterdam", "Athens", "Barcelona", "Berlin", "Helsinki",
                  "Paris", "Stockholm", "London", "Manchester", "Roma",
                  "Prague", "Edinburgh", "Vienna", "Lisbon", "Budapest",
                  "Madrid", "Warsaw", "Brussels", "Dublin", "Munich",
                  "Riga", "hamburg","Vilnius", "Tallinn", "Reykjavik",
                  "Frankfurt", "Zurich", "liverpool", "Minsk", "Oslo",
                  "Kiev", "Bucharest", "Sofia", "Cologne", "Kharkiv"]

    collNames = db.collection_names()
    for i in collNames:
        subNameList = i.split("_")
        if len(subNameList) > 1:
            if subNameList[1] in ["tripadvisor", "Tripadvisor"]:
                newName = subNameList[0].lower() + "Tripadvisor"
            if subNameList[1] in ["Booking", "booking"]:
                newName = subNameList[0].lower() + "Booking"
            else:#                 using original nameri
                newName = "_".join(subNameList)
            db[i].rename(newName)
        else:
            pass
    return db.collection_names()

