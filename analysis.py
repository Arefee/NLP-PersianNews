import json
import numpy as np
from hazm import *
import math
import jsonlines

result_1 = []
result_2 = []
result_22 = []
result_3 = []
def logfrequency(vocab , word ,count):
    idf = np.ones(len(vocab[0]))
    for i in range(len(vocab)):
        for j in range(len(vocab[i])):
            if vocab[i][j] > 1:
                idf[j] = idf[j]+vocab[i][j]
    wtd = np.zeros((len(vocab), len(vocab[0])))
    for i in range(len(vocab)):
        for j in range(len(vocab[i])):
            if vocab[i][j] > 0:
                wtd[i][j] = (1 + math.log2(vocab[i][j])) * (math.log2(count / idf[j]))
            else:
                wtd[i][j] = 0
    wtd_test = np.zeros((len(word)))
    for j in range(len(word)):
        wtd_test[j] = (math.log2(count/idf[j]))
    length_normalization(wtd, wtd_test)

def length_normalization(wtd, wtd_test):
    len_normaliz = np.zeros((len(wtd), len(wtd[0])))
    for i in range(len(wtd)):
        for j in range(len(wtd[i])):
            m=0
            if wtd[i][j] > 0:
                for k in range(len(wtd[i])):
                    m = m+(math.pow(wtd[i][k], 2))
                len_normaliz[i][j] = wtd[i][j]/math.sqrt(m)
            else:
                len_normaliz[i][j] = 0
    len_normTest = np.zeros(len(wtd_test))
    for i in range(len(wtd_test)):
            m = 0
            if wtd_test[i] > 0:
                for k in range(len(wtd_test)):
                    m = m+(math.pow(wtd_test[k], 2))
                len_normTest[i] = wtd_test[i]/math.sqrt(m)
            else:
                len_normTest[i] = 0
    if len(len_normaliz) == 14:
         cosine(len_normaliz, len_normTest, 0)
    else:
        cosine(len_normaliz, len_normTest, 1)

def cosine(len_normaliz, len_normTest, flag):
    cos = np.zeros(len(len_normaliz))
    for i in range(len(len_normaliz)):
        for j in range(len(len_normTest)):
            cos[i] = cos[i]+(len_normTest[j]*len_normaliz[i][j])

    if flag==0:
        max1 = np.argmax(cos)
        if np.max(cos) > 0:
            func_result1(max1)
        else:
            func_result1(-1)
        cos[max1] = 0

        max2 = np.argmax(cos)
        if np.max(cos)> 0:
            func_result2(max2)
        else:
            func_result2(max2)
        cos[max2] = 0

        max3 = np.argmax(cos)
        if np.max(cos) > 0:
            func_result22(max3)
        else:
            func_result22(-1)

    if flag==1:
        max = np.argmax(cos)
        if np.max(cos) > 0:
            func_result3(max)
        else:
            func_result3(-1)



def func_result1(argmaxcos):
    if argmaxcos == -1:
        result_1.append(None)
    elif argmaxcos == 0:
        result_1.append('اقتصادی')
    elif argmaxcos == 1:
        result_1.append('فرهنگی/هنری')
    elif argmaxcos == 2:
        result_1.append('اجتماعی')
    elif argmaxcos == 3:
        result_1.append('بین الملل')
    elif argmaxcos == 4:
        result_1.append('ورزشی')
    elif argmaxcos == 5:
        result_1.append('عمومی')
    elif argmaxcos == 6:
        result_1.append('سلامت')
    elif argmaxcos == 7:
        result_1.append('خواندنی ها و دیدنی ها')
    elif argmaxcos == 8:
        result_1.append('عصرايران دو')
    elif argmaxcos == 9:
        result_1.append('فناوری')
    elif argmaxcos == 10:
        result_1.append('حوادث')
    elif argmaxcos == 11:
        result_1.append('سرگرمی')
    elif argmaxcos == 12:
        result_1.append('سیاست خارجی')
    elif argmaxcos == 13:
        result_1.append('علمی')


def func_result2(argmaxcos):
    if argmaxcos == -1:
        result_2.append(None)
    elif argmaxcos == 0:
        result_2.append('اقتصادی')
    elif argmaxcos == 1:
        result_2.append('فرهنگی/هنری')
    elif argmaxcos == 2:
        result_2.append('اجتماعی')
    elif argmaxcos == 3:
        result_2.append('بین الملل')
    elif argmaxcos == 4:
        result_2.append('ورزشی')
    elif argmaxcos == 5:
        result_2.append('عمومی')
    elif argmaxcos == 6:
        result_2.append('سلامت')
    elif argmaxcos == 7:
        result_2.append('خواندنی ها و دیدنی ها')
    elif argmaxcos == 8:
        result_2.append('عصرايران دو')
    elif argmaxcos == 9:
        result_2.append('فناوری')
    elif argmaxcos == 10:
        result_2.append('حوادث')
    elif argmaxcos == 11:
        result_2.append('سرگرمی')
    elif argmaxcos == 12:
        result_2.append('سیاست خارجی')
    elif argmaxcos == 13:
        result_1.append('علمی')


def func_result22(argmaxcos):
    if argmaxcos == -1:
        result_22.append(None)
    elif argmaxcos == 0:
        result_22.append('اقتصادی')
    elif argmaxcos == 1:
        result_22.append('فرهنگی/هنری')
    elif argmaxcos == 2:
        result_22.append('اجتماعی')
    elif argmaxcos == 3:
        result_22.append('بین الملل')
    elif argmaxcos == 4:
        result_22.append('ورزشی')
    elif argmaxcos == 5:
        result_22.append('عمومی')
    elif argmaxcos == 6:
        result_22.append('سلامت')
    elif argmaxcos == 7:
        result_22.append('خواندنی ها و دیدنی ها')
    elif argmaxcos == 8:
        result_22.append('عصرايران دو')
    elif argmaxcos == 9:
        result_22.append('فناوری')
    elif argmaxcos == 10:
        result_22.append('حوادث')
    elif argmaxcos == 11:
        result_22.append('سرگرمی')
    elif argmaxcos == 12:
        result_22.append('سیاست خارجی')
    elif argmaxcos == 13:
        result_1.append('علمی')


def func_result3(argmaxcos):
    if argmaxcos == -1:
        result_3.append(None)
    elif argmaxcos == 0:
        result_3.append('AsrIran')
    elif argmaxcos == 1:
        result_3.append('Fars')

def term_frequency(word , count):
    vocab = np.zeros((14, len(word)))
    for tf_s in word[:]:
        for tf_d in train[:, 1]:
            for newspath in tf_d:
                j = 0
                if newspath == 'اقتصادی':
                    for word_d in train[:, 2]:
                        for wd in word_d:
                            j = j + 1
                            if wd == tf_s:
                                vocab[0, word.index(tf_s)] = vocab[0, word.index(tf_s)] + 1
                elif newspath == 'فرهنگی/هنری':
                    for word_d in train[:, 2]:
                        for wd in word_d:
                            j = j + 1
                            if wd == tf_s:
                                vocab[1, word.index(tf_s)] = vocab[1, word.index(tf_s)] + 1
                elif newspath == 'اجتماعی':
                    for word_d in train[:, 2]:
                        for wd in word_d:
                            j = j + 1
                            if wd == tf_s:
                                vocab[2, word.index(tf_s)] = vocab[2, word.index(tf_s)] + 1
                elif newspath == 'بین الملل':
                    for word_d in train[:, 2]:
                        for wd in word_d:
                            j = j + 1
                            if wd == tf_s:
                                vocab[3, word.index(tf_s)] = vocab[3, word.index(tf_s)] + 1
                elif newspath == 'ورزشی':
                    for word_d in train[:, 2]:
                        for wd in word_d:
                            j = j + 1
                            if wd == tf_s:
                                vocab[4, word.index(tf_s)] = vocab[4, word.index(tf_s)] + 1
                elif newspath == 'عمومی':
                    for word_d in train[:, 2]:
                        for wd in word_d:
                            j = j + 1
                            if wd == tf_s:
                                vocab[5, word.index(tf_s)] = vocab[5, word.index(tf_s)] + 1
                elif newspath == 'سلامت':
                    for word_d in train[:, 2]:
                        for wd in word_d:
                            j = j + 1
                            if wd == tf_s:
                                vocab[6, word.index(tf_s)] = vocab[6, word.index(tf_s)] + 1
                elif newspath == 'خواندنی ها و دیدنی ها':
                    for word_d in train[:, 2]:
                        for wd in word_d:
                            j = j + 1
                            if wd == tf_s:
                                vocab[7, word.index(tf_s)] = vocab[7, word.index(tf_s)] + 1
                elif newspath == 'عصرايران دو':
                    for word_d in train[:, 2]:
                        for wd in word_d:
                            j = j + 1
                            if wd == tf_s:
                                vocab[8, word.index(tf_s)] = vocab[8, word.index(tf_s)] + 1
                elif newspath == 'فناوری':
                    for word_d in train[:, 2]:
                        for wd in word_d:
                            j = j + 1
                            if wd == tf_s:
                                vocab[9, word.index(tf_s)] = vocab[9, word.index(tf_s)] + 1
                elif newspath == 'حوادث':
                    for word_d in train[:, 2]:
                        for wd in word_d:
                            j = j + 1
                            if wd == tf_s:
                                vocab[10, word.index(tf_s)] = vocab[10, word.index(tf_s)] + 1
                elif newspath == 'سرگرمی':
                    for word_d in train[:, 2]:
                        for wd in word_d:
                            j = j + 1
                            if wd == tf_s:
                                vocab[11, word.index(tf_s)] = vocab[11, word.index(tf_s)] + 1
                elif newspath == 'سیاست خارجی':
                    for word_d in train[:, 2]:
                        for wd in word_d:
                            j = j + 1
                            if wd == tf_s:
                                vocab[12, word.index(tf_s)] = vocab[12, word.index(tf_s)] + 1
                elif newspath == 'علمی':
                    for word_d in train[:, 2]:
                        for wd in word_d:
                            j = j + 1
                            if wd == tf_s:
                                vocab[13, word.index(tf_s)] = vocab[13, word.index(tf_s)] + 1
    logfrequency(vocab, word, count)

def term_frequency_agency(word,count):
    vocab = np.zeros((2, len(word)))
    for tf_s in word[:]:
        for agency in train_total[:, 0]:
                j = 0
                if agency== 'AsrIran':
                    for word_d in train_total[:, 2]:
                        for wd in word_d:
                            j = j + 1
                            if wd == tf_s:
                                vocab[0, word.index(tf_s)] = vocab[0, word.index(tf_s)] + 1
                elif agency == 'Fars':
                    for word_d in train_total[:, 2]:
                        for wd in word_d:
                            j = j + 1
                            if wd == tf_s:
                                vocab[1, word.index(tf_s)] = vocab[1, word.index(tf_s)] + 1
    logfrequency(vocab, word, count)

if __name__ == '__main__':
    data = []
    data_dictionary = {}
    test_data = []
    test_dictionary = {}
    stop = []
    stop_dic = {}
    normalizer = Normalizer()
    with open('stopwords.json', encoding='utf-8') as stop:
        stop = json.load(stop)
    stopwords = np.empty((len(stop)), dtype='object')
    for line in stop:
        stop_dic = line
        stopwords[stop.index(line)]= stop_dic['stop']

    # with open('data.json', encoding='utf-8') as data:
    #     data = json.load(data)
    with open('asrirandata.jsonl', encoding='utf-8') as reader:
        for line in reader:
            line = json.loads(line.encode('utf8'))
            data.append(line)
    train = np.empty((len(data), 3), dtype='object')

    for line in data:
        data_dictionary = line
        train[data.index(line), 0] = data_dictionary['NewsAgency']
        train[data.index(line), 1] = data_dictionary['newsPathLinks']
        train[data.index(line), 1] = [*train[data.index(line), 1]]
        for newslink in train[data.index(line), 1]:
            for nl in newslink[:]:
                if nl == 'صفحه نخست':
                    del newslink[newslink.index[nl]]
        train[data.index(line), 2] = word_tokenize(normalizer.normalize(data_dictionary['body']))
    for word in train[:, 2]:
        for w in word[:]:
            for stop in stopwords:
                if w == stop:
                    del word[word.index(w)]
    with open('testtask1.jsonl', encoding='utf-8') as reader:
        for line in reader:
            line = json.loads(line.encode('utf8'))
            test_data.append(line)
    # with open('test.json', encoding='utf-8') as test_data:
    #     test_data = json.load(test_data)
    test = np.empty((len(test_data), 3), dtype='object')
    for line in test_data:
        test_dictionary = line
        test[test_data.index(line), 0] = test_dictionary['NewsAgency']
        test[test_data.index(line), 1] = test_dictionary['newsPathLinks']
        test[test_data.index(line), 1] = [*test[test_data.index(line), 1]]
        test[test_data.index(line), 2] = word_tokenize(normalizer.normalize(test_dictionary['body']))
    for word in test[:, 2]:
        for w in word[:]:
            for stop in stopwords:
                if w == stop:
                    del word[word.index(w)]
    print('Number of Test for Task 1 & 2:',len(test))
    count = 0
    for word in train[:, 2]:
        for w in word[:]:
            count=count+1
    for word in test[:, 2]:
        term_frequency(word, count)


    test_data_agency = []
    with open('testtask3.jsonl', encoding='utf-8') as reader:
        for line in reader:
            line = json.loads(line.encode('utf8'))
            test_data_agency.append(line)
    test_agency = np.empty((len(test_data_agency), 3), dtype='object')
    for line in test_data_agency:
        test_dictionary = line
        test_agency[test_data_agency.index(line), 0] = test_dictionary['NewsAgency']
        test_agency[test_data_agency.index(line), 1] = 0
        test_agency[test_data_agency.index(line), 2] = word_tokenize(normalizer.normalize(test_dictionary['body']))
    for word in test_agency[:, 2]:
         for w in word[:]:
             for stop in stopwords:
                 if w == stop:
                    del word[word.index(w)]
    test_total = np.concatenate((test, test_agency), axis=0)
    print('Number of Test for Task 3:', len(test_total))
    data_agency = []
    # with open('data2.json', encoding='utf-8') as data_agency:
    #     data_agency = json.load(data_agency)
    with open('farsnewsdata.jsonl', encoding='utf-8') as reader:
        for line in reader:
            line = json.loads(line.encode('utf8'))
            data_agency.append(line)
    train_agency = np.empty((len(data_agency), 3), dtype='object')
    for line in data_agency:
        data_dictionary = line
        train_agency[data_agency.index(line), 0] = data_dictionary['NewsAgency']
        train_agency[data_agency.index(line), 1] = 0
        train_agency[data_agency.index(line), 2] = word_tokenize(normalizer.normalize(data_dictionary['body']))
    for word in train_agency[:, 2]:
        for w in word[:]:
            for stop in stopwords:
                if w == stop:
                    del word[word.index(w)]
    train_total = np.concatenate((train, train_agency), axis=0)
    for word in train_agency[:, 2]:
        for w in word[:]:
            count = count+1
    for word in test_total[:, 2]:
        term_frequency_agency(word, count)

    print(result_1)
    print(result_2)
    print(result_22)
    print(result_3)
    tp1 = 0
    fp1 = 0
    fn1 = 0
    tn1 = 0
    j = 0
    for newspathlink in result_1:
        for list in test[:, 1]:
            for i in list:
                if newspathlink == i:
                    tp1 = tp1 + 1
                elif newspathlink != i:
                    fp1 = fp1 + 1
                elif newspathlink == None:
                    fn1 = fn1 + 1
                elif i == None:
                    tn1 = tn1 + 1
            j = j+1
    acc1 = (tp1 + tn1) / (tp1 + tn1 + fp1 + tn1)
    pre1 = (tp1) / (tp1 + fp1)
    rec1 = (tp1) / (tp1 + fn1)
    f1m1 = (2 * pre1 * rec1) / (pre1 + rec1)
    print('Task 1:')
    print('\t Accuracy: ', acc1)
    print('\t Precision: ', pre1)
    print('\t Recall: ', rec1)
    print('\t F1 measure: ', f1m1)
    for newspathlink in result_2:
        for list in test[:, 1]:
            for i in list:
                if newspathlink == i:
                    tp1 = tp1 + 1
                elif newspathlink != i:
                    fp1 = fp1 + 1
                elif newspathlink == None:
                    fn1 = fn1 + 1
                elif i == None:
                    tn1 = tn1 + 1
    for newspathlink in result_22:
        for list in test[:, 1]:
            for i in list:
                if newspathlink == i:
                    tp1 = tp1 + 1
                elif newspathlink != i:
                    fp1 = fp1 + 1
                elif newspathlink == None:
                    fn1 = fn1 + 1
                elif i == None:
                    tn1 = tn1 + 1
    acc1 = (tp1 + tn1) / (tp1 + tn1 + fp1 + tn1)
    pre1 = (tp1) / (tp1 + fp1)
    rec1 = (tp1) / (tp1 + fn1)
    f1m1 = (2 * pre1 * rec1) / (pre1 + rec1)
    print('Task 2:')
    print('\t Accuracy: ', acc1)
    print('\t Precision: ', pre1)
    print('\t Recall: ', rec1)
    print('\t F1 measure: ', f1m1)
    tp3 = 0
    fp3 = 0
    fn3 = 0
    tn3 = 0
    j = 0
    for newspathlink in result_3:
        for i in test_total[:, 0]:
            if newspathlink == i:
                tp3 = tp3 + 1
            elif newspathlink != i:
                fp3 = fp3 + 1
            elif newspathlink == None:
                fn3 = fn3 + 1
            elif i == None:
                tn3 = tn3 + 1
        j = j + 1
    acc3 = (tp3 + tn3) / (tp3 + tn3 + fp3 + tn3)
    pre3 = (tp3) / (tp3 + fp3)
    rec3 = (tp3) / (tp3 + fn3)
    f1m3 = (2 * pre3 * rec3) / (pre3 + rec3)
    print('Task 3:')
    print('\t Accuracy: ', acc3)
    print('\t Precision: ', pre3)
    print('\t Recall: ', rec3)
    print('\t F1 measure: ', f1m3)