import tensorflow as tf
import numpy as np
import csv


# get the tested features
def readFeatures():
    csv_reader = csv.reader(open('../data/consumer.txt'))
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
        "White": [0, 1, 0],
        "Black": [0, 0, 1],
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


    return row_tensor



# init the predictor from storage
def predictor(test_row):
    saver = tf.train.import_meta_graph('../model/predictor-3999.meta')
    initer = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(initer)
        saver.restore(sess, tf.train.latest_checkpoint("../model/"))
        graph = tf.get_default_graph()
        predict = graph.get_operation_by_name("predict")
        x = graph.get_tensor_by_name("holdingplace:0")
        feed_dict = {x:test_row}

    return predict, feed_dict



# get videos'scores
def calScore(row_tensor,ad_name):


    initer = tf.initialize_all_variables()

    test_row = row_tensor[0]

    for i in range(0, 4):
        test_row.pop()


    switcher_adName = {
        "snack": [1, 0, 0, 0],
        "ikea": [0, 1, 0, 0],
        "cosm": [0, 0, 1, 0],
        "lancome": [0, 0, 0, 1],
    }
    ad_row = switcher_adName.get(ad_name)
    test_row.extend(ad_row)
    test_row = [test_row]
    # init the model

    input_tensor = tf.convert_to_tensor(test_row)

    with tf.Session() as sess:
        sess.run(initer)
        saver = tf.train.import_meta_graph(
            '../predictor-3999.meta')
        saver.restore(sess, tf.train.latest_checkpoint("../model/"))
        graph = tf.get_default_graph()
        predict = graph.get_tensor_by_name("predict:0")
        x = graph.get_tensor_by_name("holdingplace:0")
        feed_dict = {x: test_row}
        score = sess.run(predict, feed_dict)


        

    return score




# find the max scores
def findMax():
    ad_list = ["snack","ikea","cosm","lancome"]
    maxScore = 0
    maxAd = "none"
    row_tensor = readFeatures()
    np.set_printoptions(suppress=True)
    for ad_name in ad_list:
        l = calScore(row_tensor, ad_name)[0]
        predictResult = [l[0], l[1], l[2], l[3], l[4], l[5]]

        maxResult = 0.0
        for i in range(1,6):

            if predictResult[i] > maxResult:
                maxResult = predictResult[i]
                result = i
        #print predictResult
        #print ("video " + ad_name + "'s score is: " + str(result))
        if result >= maxScore :
            maxScore = result
            maxAd = ad_name

    return maxAd





if __name__ == "__main__":
    max = findMax()
    print(max)

