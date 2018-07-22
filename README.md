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

### other related files are stored in google cloud. below is the link.
https://drive.google.com/drive/folders/0B9MedSkTfH80ZEtQOVg4YzlSdWs?usp=sharing
 
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

{
    "hotelStars": 3,
    "hotelUrl": "https://www.tripadvisor.com/Hotel_Review-g186338-d189750-Reviews-Premier_Inn_London_Kensington_Olympia_Hotel-London_England.html",
    "hotelName" : "Premier Inn London Kensington (Olympia) Hotel",
    "hotelStars":	3,
    "hotelUrl" : " https://www.tripadvisor.com/Hotel_Review-g186338-d189750-Reviews-Premier_Inn_London_Kensington_Olympia_Hotel-London_England.html",
    "review" : "We stayed in this hotel while on a family trip to London (myself, wife and a baby) and our first experience was not a positive one. As we arrived there is a flight of stairs on the main entrance, and as we were carrying our luggage and baby stroller, we would of course have some trouble to carry it. We ringed the reception and asked for some help or to use the disabled elevator on the stairs to carry the stroller. For our amazement we were greeted by the receptionist who informed us the elevator was only for disabled people and denied the help.After a hard time carrying everything to the reception we checked in, only to discover that our reservation for a room with a baby cot had been ignored and that all rooms were booked, so there was no room large enough to fit a baby cot. This was obviously a mistake in the booking system as we had the receipt where we specified that need, however, that didn't prevent the receptionist from rudely advising us to be more careful with the reservations in the future. After this, as a workaround, she offered us two rooms. One that was that was "too small to possibly fit the cot" on the ground floor with only a couple of stairs to carry the stroller. And another that was bigger but had a few more stairs. We were inclined for the first one because it would be impossible to carry the bags and the baby stroller through a flight of stairs, but she advised us to take the second one as there was no big difference on the number of stairs. We accepted the advice and asked for help to at least carry the bags to the room. After a couple of minutes waiting for help we were surprised to see that she had called a couple of housekeepers (women) to carry our bags. They promptly refused to carry them stating that it wasn't their duties to carry bags, they could help only with the stroller as a special favor. Having no alternative we had to carry everything again ourselves, thats when we realized that the "couple of stairs" she referred meant the room was in an entirely different floor. The housekeepers told us it would be completely impossible for us to carry everything everyday and told us the receptionist was crazy (yes, we had already realized that fact). They offered to solve the situation themselves talking to the manager and switching our room for one in the ground floor. I couldn't believe that a woman doing housekeeping had a lot more knowledge of how to handle a customers needs than a receptionist.To be honest, this unfortunate event was the only bad experience on this hotel (apart from another episode with another receptionist that couldn't advise us on how to get to Heathrow and, although she nearly make us miss the flight, she wouldn't stop treating us in a rude patronizing way) the rest of the hotel was very clean, the staff at the breakfast and the housekeepers were competent and friendly and it was inline with what we might expect from this kind of hotel. However, they should specially careful selecting the staff that they put at the front desk, as they are the privileged point of contact with the customers and will contribute enormously for the first impression one gets from an hotel.. in this case, it was a terrible way to start.",
    "score" : 3,
    "title" :  ".. A bad way to start", 
    "url" : "https://www.tripadvisor.com/ShowUserReviews-g186338-d189750-r99850654-Premier_Inn_London_Kensington_Olympia_Hotel-London_England.html",
    "userId"	: "fprata",
    "CNN_glove_50dims" : 1,
    "CNN_word2vec_300dims" : 1,
    "liblinear_SVM" :	5,
    "rnn_word2vec_300dims" : 1,	
    "VADER" :	5,
    "deviation" :	1.959591794
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
  

## template projects
1. pytorch-sentiment-classification
    ### sentiment-classification
    LSTM and CNN sentiment analysis in PyTorch
    
    The sentiment model is trained on Stanford Sentiment Treebank (i.e. SST2).
    
    #### Requires
    - torch 
    - torchtext 
    - tdqm
    - torchwordemb  `pip install torchwordemb`
  
2. Character-level Convolutional Networks for Text Classification
    NIPS 2015 paper
    
  
[1] https://github.com/cjhutto/vaderSentiment
[2]Thumbs up? Sentiment Classification using Machine Learning  |https://arxiv.org/pdf/cs/0205070.pdf
[3]tensorflow实现基于LSTM的文本分类方法, 博客链接， https://blog.csdn.net/u010223750/article/details/53334313; https://github.com/luchi007/RNN_Text_Classify  
[4]Implementing a CNN for Text Classification in TensorFlow, http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/; https://github.com/dennybritz/cnn-text-classification-tf 