## Environment
### TUT ThinkStation
* Python 3.6.4 (anaconda)  
* cudatoolkit  9.0  -- install with conda-forge community version  
* tensorflow-gpu 1.8.0  
### XPS laptop
* Python 3.5.5(anaconda)




## sentimentAnalysis
### data folder
* ./data/  


 
### Data Sample
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
### clean review data statistics
 
 data source                       |       number    
 ----------------------------------|-----------------
 booking.com                       |   xxx  
 tripadvisor.com                   |   xxx



## algos
order | algorithms               |  details      | accuracy
------| -------------------------|---------------|------------------- 
1     | VADER                    |  [1]          | 
2     | SVM(LR)                  |  [2]          |  
3     | RNN with LSTM            |  [3]          |   
4     | CNN                      |  [4]          |   
  
  
  
[1] https://github.com/cjhutto/vaderSentiment
[2]Thumbs up? Sentiment Classification using Machine Learning  |https://arxiv.org/pdf/cs/0205070.pdf
[3]tensorflow实现基于LSTM的文本分类方法, 博客链接， https://blog.csdn.net/u010223750/article/details/53334313; https://github.com/luchi007/RNN_Text_Classify  
[4]Implementing a CNN for Text Classification in TensorFlow, http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/; https://github.com/dennybritz/cnn-text-classification-tf 