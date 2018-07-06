#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import csv


# get the tested features
def readFeatures():
    csv_reader = csv.reader(open('../data/stat.csv'))
    switcher_emotion = {
        "neutral": [1, 0, 0, 0, 0, 0, 0],
        "surprise": [0, 1, 0, 0, 0, 0, 0],
        "sadness": [0, 0, 1, 0, 0, 0, 0],
        "disgust": [0, 0, 0, 1, 0, 0, 0],
        "fear": [0, 0, 0, 0, 1, 0, 0],
        "happiness": [0, 0, 0, 0, 0, 1, 0],
        "anger": [0, 0, 0, 0, 0, 0, 0],
    }
    switcher_race = {
        "Asian": [1, 0, 0],
        "ASIAN": [1, 0, 0],
        "White": [0, 1, 0],
        "WHITE": [0, 1, 0],
        "Black": [0, 0, 1],
        "BLACK": [0, 0, 1],
    }
    switcher_adName = {
        "snack": [1, 0, 0, 0],
        "ikea": [0, 1, 0, 0],
        "cosm": [0, 0, 1, 0],
        "lancome": [0, 0, 0, 1],
    }
    switcher_label = {
        0: [1, 0, 0, 0, 0, 0],
        1: [0, 1, 0, 0, 0, 0],
        2: [0, 0, 1, 0, 0, 0],
        3: [0, 0, 0, 1, 0, 0],
        4: [0, 0, 0, 0, 1, 0],
        5: [0, 0, 0, 0, 0, 1],
    }
    row_tensor = []
    label_tensor = []
    for row in csv_reader:
        row_value = [float(row[i]) for i in range(0, 4)]
        row_value.extend(switcher_emotion.get(row[4]))
        row_value.extend(switcher_race.get(row[5]))
        row_value.extend([float(row[i]) for i in range(6, 15)])
        row_value.extend(switcher_adName.get(row[15]))
        row_tensor.append(row_value)
        label_tensor.append(int(row[16]))

    # print row_tensor
    return row_tensor, label_tensor


# init the predictor from storage
def predictor(test_row):
    saver = tf.train.import_meta_graph(
        '../model/predictor-3999.meta')
    initer = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(initer)
        saver.restore(sess,
                      tf.train.latest_checkpoint("../model/"))
        graph = tf.get_default_graph()
        predict = graph.get_operation_by_name("predict")
        x = graph.get_tensor_by_name("holdingplace:0")
        feed_dict = {x: test_row}

    return predict, feed_dict


# get videos'scores
def calScore(test_row):
    initer = tf.initialize_all_variables()

    test_row = [test_row]
    # init the model

    with tf.Session() as sess:
        sess.run(initer)
        saver = tf.train.import_meta_graph(
            '../model/predictor-3999.meta')
        saver.restore(sess,
                      tf.train.latest_checkpoint("../model/"))
        graph = tf.get_default_graph()
        predict = graph.get_tensor_by_name("predict:0")
        x = graph.get_tensor_by_name("holdingplace:0")
        feed_dict = {x: test_row}
        score = sess.run(predict, feed_dict)

    return score


# find the max scores
def findMax():
    row_tensor, label = readFeatures()
    np.set_printoptions(suppress=True)
    # confusion matrix
    v1Score = [[], [], [], [], [], [], [], [], []]
    v2Score = [[], [], [], [], [], [], [], [], []]
    v3Score = [[], [], [], [], [], [], [], [], []]
    v4Score = [[], [], [], [], [], [], [], [], []]

    for i in range(len(row_tensor)):
        maxResult = 0.0
        predictResult = calScore(row_tensor[i])
        for k in range(0, 6):
            l = predictResult[0][k]
            if l > maxResult:
                maxResult = l
                max = k
        if row_tensor[i][23] == 1:
            print (str(i) + ":snack predict score is: " + str(max) + ". The real score is :" + str(label[i]))
            v1Score = judgeStat(max, label[i], v1Score)
        if row_tensor[i][24] == 1:
            print (str(i) + ":ikea predict score is: " + str(max) + ". The real score is :" + str(label[i]))
            v2Score = judgeStat(max, label[i], v2Score)
        if row_tensor[i][25] == 1:
            print (str(i) + ": cosm predict score is: " + str(max) + ". The real score is :" + str(label[i]))
            v3Score = judgeStat(max, label[i], v3Score)
        if row_tensor[i][26] == 1:
            print (str(i) + ": lancome predict score is: " + str(max) + ". The real score is :" + str(label[i]))
            v4Score = judgeStat(max, label[i], v4Score)

    v1Count = [len(v1Score[0]), len(v1Score[1]), len(v1Score[2]), len(v1Score[3]), len(v1Score[4]), len(v1Score[5]), len(v1Score[6]), len(v1Score[7]), len(v1Score[8])]
    print (calStat(v1Count))
    v2Count = [len(v2Score[0]), len(v2Score[1]), len(v2Score[2]), len(v2Score[3]), len(v2Score[4]), len(v2Score[5]),
               len(v2Score[6]), len(v2Score[7]), len(v2Score[8])]
    print (calStat(v2Count))
    v3Count = [len(v3Score[0]), len(v3Score[1]), len(v3Score[2]), len(v3Score[3]), len(v3Score[4]), len(v3Score[5]),
               len(v3Score[6]), len(v3Score[7]), len(v3Score[8])]
    print (calStat(v3Count))
    v4Count = [len(v4Score[0]), len(v4Score[1]), len(v4Score[2]), len(v4Score[3]), len(v4Score[4]), len(v4Score[5]),
               len(v4Score[6]), len(v4Score[7]), len(v4Score[8])]
    print (calStat(v4Count))

def judgeStat(max, label, v1Score):
    if max == 5 or max == 4:
        if label == 5 or label == 4:
            v1Score[0].append(1)
        if label == 3 or label == 2:
            v1Score[1].append(1)
        if label == 1 or label == 0:
            v1Score[2].append(1)
    if max == 3 or max == 2:
        if label == 3 or label == 2:
            v1Score[4].append(1)
        if label == 5 or label == 4:
            v1Score[3].append(1)
        if label == 1 or label == 0:
            v1Score[5].append(1)
    if max == 1 or max == 0:
        if label == 5 or label == 4:
            v1Score[7].append(1)
        if label == 3 or label == 2:
            v1Score[8].append(1)
        if label == 1 or label == 0:
            v1Score[6].append(1)
    return v1Score


def calStat(v1Count):
    v1AC = float((v1Count[0] + v1Count[4] + v1Count[6]) )/ float(v1Count[0] + v1Count[1] + v1Count[2] + v1Count[3] + v1Count[4] + v1Count[6] + v1Count[7] + v1Count[8] + v1Count[5]) if v1Count[0] + v1Count[1] + v1Count[2] + v1Count[3] + v1Count[4] + v1Count[6] + v1Count[7] + v1Count[8] + v1Count[5] != 0 else 0
    v1TPR = float(v1Count[0]) / float(v1Count[0] + v1Count[3] + v1Count[8]) if (v1Count[0] + v1Count[3] + v1Count[8]) != 0 else 0
    v1TSR = float(v1Count[4]) / float(v1Count[1] + v1Count[4] + v1Count[7])if (v1Count[1] + v1Count[4] + v1Count[7]) != 0 else 0
    v1USR = float(v1Count[1]) / float(v1Count[1] + v1Count[4] + v1Count[7]) if (v1Count[1] + v1Count[4] + v1Count[7]) != 0 else 0
    v1DSR = float(v1Count[7]) / float(v1Count[1] + v1Count[4] + v1Count[7]) if (v1Count[1] + v1Count[4] + v1Count[7]) != 0 else 0
    v1TNR = float(v1Count[6]) / float(v1Count[2] + v1Count[5] + v1Count[6]) if (v1Count[2] + v1Count[5] + v1Count[6]) != 0 else 0
    v1Score = [v1AC, v1TPR, v1TSR, v1USR, v1DSR, v1TNR]
    return v1Score


if __name__ == "__main__":
    findMax()

