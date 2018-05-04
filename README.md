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
    "title": "Outstanding",
    "url": "https://www.tripadvisor.com/ShowUserReviews-g188590-d189389-r145091651-Sofitel_Legend_The_Grand_Amsterdam-Amsterdam_North_Holland_Province.html",
    "hotelUrl": "https://www.tripadvisor.com/Hotel_Review-g188590-d189389-Reviews-Sofitel_Legend_The_Grand_Amsterdam-Amsterdam_North_Holland_Province.html",
    "review": "I rate this hotel as the best I've stayed in. It occupies a beautiful, historic building sandwiched between two canals in the heart of old 'Dam. It's at the bottom of the Red Light district, but don't let that put you off - this is the heart of the old centre, and the hotel's locality south of the Damstraat bridge which traverses O.Voorburgwal is actually quite peaceful at night. Our room was large and well - appointed with a canal view. The bed, oh the bed. The most comfortable bed I've had the pleasure to sleep in. Hermes toiletries. Spotlessly clean. Service and food are outstanding and what you expect of a hotel of this calibre. Concierge was excellent.In summary, I cannot recommend this hotel highly enough. A hotel which richly deserves it's 5-star rating.",
    "hotelStars": "5.0",
    "userId": "000Glenn",
    "hotelLocation": "Oudezijds Voorburgwal 197, 1012 EX Amsterdam, The Netherlands",
    "hotelName": " Sofitel Legend The Grand Amsterdam ",
    "score": 5,
    "date": null
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