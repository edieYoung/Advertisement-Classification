# Advertisement-Classification

## Training Model
This deep learning model is based on the tensorflow framework. It can process with the input data and output the predict result which tells whether this advertisement fits the audience well with such kinds of facial features. And in this case I simply used ANN model for training, because the input is only a series of data but not with a complex structure. Maybe it can be improved with mode input data and mode complex layers structure.
During the training process, it will save the model in the end, as well as the visual losses provided by tensorflow itself.

## Forward Predict
This is a process of prediction. It will load the model which is saved in the process of annmodel.py. Given with the input, it will do a forward calculate to give the predict result, which is the process of how computer can decide which ad should be delivered to the viewer.

## Statistic
This is for verifying the model and prediction results. The indexes are as follow


Symbol | Description
------- | -------
TP0| 将意向反应正确判断为意向反应的样本个数
UP1| 将情感反应错误判断为意向反应的样本个数
UP2| 将认知反应错误判断为意向反应的样本个数
DS| 将意向反应错误判断为情感反应的样本个数
TS| 将情感反应正确判断为情感反应的样本个数
US| 将认知反应错误判断为情感反应的样本个数
TN| 将认知反应正确判断为认知反应的样本个数
DN1| 将情感反应错误判断为认知反应的样本个数
DN2| 将意向反应错误判断为认知反应的样本个数

## Data
In the data fold, you can find the training data as well as the data for statistic and prediction test. The training data was collected by the application for advertisement rating with users' facial features. And this application is also in my repos, the git link is :[edieYoung/face_detect_for_advertising](https://github.com/edieYoung/face_detect_for_advertising)





