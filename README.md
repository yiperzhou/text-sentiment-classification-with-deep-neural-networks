## Environment

### XPS laptop
* Python 3.5.5(anaconda)
* PyTorch 0.4 with cuda 9.0



## sentimentAnalysis

### other related files are stored in google cloud. below is the link.
https://drive.google.com/drive/folders/0B9MedSkTfH80ZEtQOVg4YzlSdWs?usp=sharing

### Thesis in overleaf
https://v1.overleaf.com/15464915nsfzbmsffzzs#/75103087/

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

### clean review data statistics
 
 data source                       |       number    
 ----------------------------------|-----------------
 tripadvisor.com                   |   xxx



## algos
order | algorithms               |  details      | accuracy
------| -------------------------|---------------|------------------- 
1     | VADER                    |  [1]          | 
      |   Machine learning method|               |                         
2     | SVM(LR)                  |  [2]          |                        
      | Deep learning method     |                                      
3     | Word CNN                 |  [5]          |                         
4     | CNN_Text_Model           |  [4]           |                         
5     | BiLSTMConv               |   [3]          |                              
6     | VDCNN                    |   [6]         |                  



  
 
# template project   
ToxicCommentClassification-pytorch, https://github.com/keithyin/ToxicCommentClassification-pytorch

[1] https://github.com/cjhutto/vaderSentiment

[2] Thumbs up? Sentiment Classification using Machine Learning  |https://arxiv.org/pdf/cs/0205070.pdf

[3] tensorflow实现基于LSTM的文本分类方法, 博客链接， https://blog.csdn.net/u010223750/article/details/53334313; https://github.com/luchi007/RNN_Text_Classify,   

[4] Implementing a CNN for Text Classification in TensorFlow, http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/; https://github.com/dennybritz/cnn-text-classification-tf 

[5] Johnson, Rie, and Tong Zhang. "Convolutional neural networks for text categorization: Shallow word-level vs. deep character-level." arXiv preprint arXiv:1609.00718 (2016).

[6] Conneau, Alexis, et al. "Very deep convolutional networks for text classification." arXiv preprint arXiv:1606.01781 (2016). https://github.com/threelittlemonkeys/vdcnn-pytorch, https://github.com/ArdalanM/nlp-benchmarks

[7] 
