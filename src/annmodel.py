#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import csv

class annmodel:
    # initializing the weights
    def weights_variable(self,shape):
        return tf.Variable(tf.random_normal(shape,stddev=0.01))

    def bias_variable(self,shape):
        return tf.Variable(tf.zeros(shape))

    # visualize the data distribute
    def variable_summaries(self,var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean',mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
                tf.summary.scalar('stddev',stddev)
            tf.summary.scalar('max',tf.reduce_max(var))
            tf.summary.scalar('min',tf.reduce_min(var))
            tf.summary.histogram('histogram',var)

    # define a new layer
    def nn_layer(self,input_tensor,input_dim,output_dim,layer_name,act=tf.nn.sigmoid):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = self.weights_variable([input_dim,output_dim])
                self.variable_summaries(weights)
            with tf.name_scope('bias'):
                bias = self.bias_variable([output_dim])
                self.variable_summaries(bias)
            with tf.name_scope('wx_plus_b'):
                preactivate = tf.matmul(input_tensor,weights)+bias
                tf.summary.histogram('pre_activations',preactivate)
            if act == None:
                return preactivate
            else:
                activations = act(preactivate,name='activation')
                tf.summary.histogram('activations', activations)
                return activations

    # read csv file
    def read_csv(self):
        csv_reader = csv.reader(open('../data/test.csv'))
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
            label_tensor.append(switcher_label.get(int(row[16])))
        input_tensor = tf.convert_to_tensor(row_tensor)
        label = tf.convert_to_tensor(label_tensor)
        input_features = input_tensor.shape[1]
        input_number = input_tensor.shape[0]
        return input_features, input_number, row_tensor, label_tensor

    def run_model(self,hidden1_size,hidden2_size,hidden3_size):
        input_features,input_number,input_tensor,label = self.read_csv()
        # define the layer 1
        training_data = tf.placeholder(tf.float32,shape= [None,int(input_features)],name = "holdingplace")
        training_label = tf.placeholder(tf.float32,shape =[None, int(len(label[0]))],name = "label")
        #print("input_features  %s , hidden1_size %s"%(input_features,hidden1_size))
        l1 = self.nn_layer(training_data,int(input_features),int(hidden1_size),"layer1")
        l2 = self.nn_layer(l1, hidden1_size, hidden2_size, "layer2")
        l3 = self.nn_layer(l2, hidden2_size, hidden3_size, "layer3")
        output = self.nn_layer(l3,hidden3_size,len(label[0]),"layer_output",act = None)
        prediction = tf.nn.softmax(output,name="predict")





        cross_entropy = -tf.reduce_sum(training_label * tf.log(prediction))

        train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.arg_max(prediction, 1),tf.arg_max(training_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        tf.summary.scalar('loss', cross_entropy)
        tf.summary.scalar('accuracy', accuracy)
        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=1)
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('../train_visual',
                                                sess.graph )
            sess.run(tf.global_variables_initializer())

            for i in range(4000):
                if i % 100 == 0:
                    train_accuracy = sess.run(accuracy, feed_dict = {training_data:input_tensor,training_label:label})
                    loss_value = sess.run(cross_entropy,feed_dict = {training_data:input_tensor,training_label:label })
                    print("setp %s , train accuracy %s, loss %s"%(i,train_accuracy,loss_value))
                summary,_=sess.run([merged,train_step],feed_dict={training_data:input_tensor,training_label:label})
                train_writer.add_summary(summary,i)
                if i == 3999:
                    saver.save(sess, "../model/predictor",global_step=i)



if __name__ == "__main__":
    annmodel = annmodel()
    annmodel.run_model(24,10,10)