# sentimentAnalysis

## data stored in MongoDB command
* show collection statistics  
db.Amsterdam2_tripadvisor_hotels_reviews.stats()
* rename collection name  
db.Amsterdam3_tripadvisor_hotels_reviews.renameCollection("amsterdamTripadvisor")
 
## Data Sample
* tripadvisor review data sample  
```json

{
    "hotelLocation": "Raadhuis Straat 51, 1016 DD Amsterdam, The Netherlands",
    "hotelName": " Hotel Nadia",
    "hotelStars": "2.0",
    "hotelUrl": "https://www.tripadvisor.com/Hotel_Review-g188590-d232493-Reviews-Hotel_Nadia-Amsterdam_North_Holland_Province.html",
    "title": "Overnight stay.",
    "review": "This is a very small hotel. Staff are very friendly.But you have to be quite fit to go up all the stairs with your luggage. I also missed breakfast as I was 5 mins too late.The room was small but clean and the balcony was nice.The hotel is in a very good location.",
    "url": "https://www.tripadvisor.com/ShowUserReviews-g188590-d232493-r507832414-Hotel_Nadia-Amsterdam_North_Holland_Province.html",
    "score": 4,
    "userId": "nigcastle"，
    "date": "August 1, 2017"
}



```
* booking.com review data sample  
```json
{
    "title": "Amazing hotel in amazing location - definitely would stay again!",
    "url": "https://www.booking.com/reviews/fi/hotel/seurahuone/review/39ebcbf8c6394036.html",
    "hotelUrl": "https://www.booking.com/hotel/fi/seurahuone.html",
    "hotelStars": "4 stars",
    "hotelLocation": "Kaivokatu 12, Eteläinen Suurpiiri, 00100 Helsinki, Finland",
    "hotelName": "\nHotel Seurahuone Helsinki\n",
    "score": "\n7.5\n",
    "negative_content": "We liked everything. Was a little expensive, but worth the extra.",
    "date": "\nJune 26, 2016\n",
    "positive_content": "Amazing hotel full of character, large comfortable room with great facilities, really nice friendly and helpful staff, great location right in the centre of the city, dining hall was spectacular, 24 hour shop across the road"
}
```
## raw review data statistics
### Booking.com

### tripadvisor.com


## review data statistics after removing records which contain absent or null value
### Booking.com

### Tripadvisor.com

# Experiment Environment
ubuntu 16.10  
keras version: '2.1.6'  
(backend) tensorflow version: 1.7.0 with GPU version  
CUDA version: 9.0.176  
cudnn version: 7.1  
python 3.5  

# algos
## VADER
VADER Sentiment Analysis. VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and 
rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media, 
and works well on texts from other domains.
https://github.com/cjhutto/vaderSentiment

## SVM, LR
reference paper: Thumbs up? Sentiment Classification using Machine Learning
Techniques, link, https://arxiv.org/pdf/cs/0205070.pdf

## RNN with LSTM, CNN
LSTM  
tensorflow实现基于LSTM的文本分类方法, 博客链接， https://blog.csdn.net/u010223750/article/details/53334313,
源码： https://github.com/luchi007/RNN_Text_Classify  

CNN  
Implementing a CNN for Text Classification in TensorFlow, http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/  
code, https://github.com/dennybritz/cnn-text-classification-tf 

## CNN，RNN中实验的具体设置：
filter的宽度等于词向量的长度,
flag.number_filters = flag.word_embedding

