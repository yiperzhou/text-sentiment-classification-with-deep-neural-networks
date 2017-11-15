# senti_wordnet class, using senti_wordnet 3.0 
fswn = open(#open sentiment wordnet 3.0 file
            "/home/yi/Music/Sentiment-Analysis/SentiWordNet_3.0/swn/www/admin/dump/SentiWordNet_3.0.0_20130122.txt")


swn_lines = list()
for line in fswn.readlines():
    if line.startswith("#"):
        pass
    else:
        swn_lines.append(line.strip())
fswn.close()

from collections import defaultdict     # don't need to check whether key exists
temp = defaultdict(list)


for i, line in enumerate(swn_lines):
    data = line.split("\t")
    score = (data[2], data[3])
    line_words = data[4].strip().split(" ")
    for word in line_words:
        w_n = word.split("#")
        w_n[0] += "#" + data[0]
        index = int(w_n[1]) - 1
        if len(temp[w_n[0]]) > 0:
            if len(temp[w_n[0]]) <= index:
                tmp_lst1 = temp[w_n[0]]
                # update the word score.
                tmp_lst2 = tmp_lst1 + [(0.0,0.0)]*(index-len(tmp_lst1) + 1)
                temp[w_n[0]] = tmp_lst2
            temp[w_n[0]][index] = score
        else:
            tmp_lst3 = [(0.0, 0.0)] * index
            tmp_lst3.append(score)
            temp[w_n[0]] = tmp_lst3
        




    print("end")