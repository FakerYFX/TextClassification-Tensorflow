# Method
## 1-EMNLP2014 Convolutional Neural Networks for Sentence Classification

## 2-AAAI2015 Recurrent Convolutional Neural Networks for Text Classification

## 3-NAACL2016 Hierarchical Attention Networks for Document Classification

## 4-ICLR2017 A Structured Self-attentive Sentence Embedding

# Experiments 
python 2.7   
TensorFlow-gpu 1.4  
CUDA 8.0+  
numpy 1.12.1+  
glove.6B.100d.txt  

|      实验数据集      |    实现框架   |Train|Test|Class_Num|Max_Sentence_Length|
|:--------------------:|:--------------:|:------:|:----:|:--------:|:--------:|
|Yahoo!Answers-Health|TensorFlow|144,784|18,016|21|16|

1.EMNLP2014(CNN)  
```
Epoch_Total = 10     
Epoch 0: cost: 0.0585964461979, acc: 0.515306122449  
Epoch 0: dev acc: 0.534413854352  
Epoch 1: cost: 0.0528805645145, acc: 0.530612244898  
Epoch 1: dev acc: 0.537910746004  
Epoch 2: cost: 0.0506094549688, acc: 0.553571428571  
Epoch 2: dev acc: 0.537855239787  
Epoch 3: cost: 0.0498712082603, acc: 0.55612244898  
Epoch 3: dev acc: 0.537744227353  
Epoch 4: cost: 0.0518160111898, acc: 0.545918367347  
Epoch 4: dev acc: 0.537910746004  
Epoch 5: cost: 0.0494907867848, acc: 0.553571428571  
Epoch 5: dev acc: 0.53779973357  
Epoch 6: cost: 0.0528369571786, acc: 0.544642857143  
Epoch 6: dev acc: 0.538021758437  
Epoch 7: cost: 0.05352423112, acc: 0.529336734694  
Epoch 7: dev acc: 0.538021758437  
Epoch 8: cost: 0.050898156604, acc: 0.544642857143  
Epoch 8: dev acc: 0.538077264654  
Epoch 9: cost: 0.0502259273614, acc: 0.5625  
Epoch 9: dev acc: 0.53779973357  
dev acc的最大值在0.5380左右 
```
2.AAAI2015(RCNN)   
(1)BiLSTM (text_rnn_main.py)
```
Epoch 0: train cost: 0.0590172300533, acc: 0.49362244898  
Epoch 0: test acc: 0.535024422735  
Epoch 1: train cost: 0.0543372389309, acc: 0.531887755102  
Epoch 1: test acc: 0.54335035524  
Epoch 2: train cost: 0.0514704325065, acc: 0.543367346939  
Epoch 2: test acc: 0.544571492007  
Epoch 3: train cost: 0.0507184206223, acc: 0.553571428571  
Epoch 3: test acc: 0.54468250444  
Epoch 4: train cost: 0.0537773793449, acc: 0.543367346939  
Epoch 4: test acc: 0.544960035524  
Epoch 5: train cost: 0.0495000013283, acc: 0.563775510204  
Epoch 5: test acc: 0.545348579041  
Epoch 6: train cost: 0.0531619136431, acc: 0.552295918367  
Epoch 6: test acc: 0.544960035524  
Epoch 7: train cost: 0.0554935609808, acc: 0.524234693878  
Epoch 7: test acc: 0.545071047957  
Epoch 8: train cost: 0.0507630480795, acc: 0.547193877551  
Epoch 8: test acc: 0.545126554174  
Epoch 9: train cost: 0.0534029142285, acc: 0.544642857143  
Epoch 9: test acc: 0.545237566607  
dev acc的最大值在0.5453左右  
```
(2)BiGRU + Max_Pool(text_rcnn_main.py) 
``` 
Epoch 0: cost: 0.0608407126702, acc: 0.492346938776   
Epoch 0: dev acc: 0.534302841918  
Epoch 1: cost: 0.0536251258181, acc: 0.519132653061  
Epoch 1: dev acc: 0.541629662522  
Epoch 2: cost: 0.0529526417353, acc: 0.535714285714  
Epoch 2: dev acc: 0.540130994671  
Epoch 3: cost: 0.0517433822459, acc: 0.543367346939  
Epoch 3: dev acc: 0.540019982238  
Epoch 4: cost: 0.0536590512006, acc: 0.52806122449  
Epoch 4: dev acc: 0.540019982238  
Epoch 5: cost: 0.0490315490383, acc: 0.5625  
Epoch 5: dev acc: 0.539964476021  
dev acc的最大值在0.5416左右 
```

(3)LSTM + Convolution + Max_Pool (text_rnn_cnn_main.py) 
```
Epoch 0: cost: 0.0561928252633, acc: 0.517857142857  
Epoch 0: dev acc: 0.539631438721  
Epoch 1: cost: 0.0515705027751, acc: 0.542091836735  
Epoch 1: dev acc: 0.546458703375  
Epoch 2: cost: 0.0494501282062, acc: 0.55612244898  
Epoch 2: dev acc: 0.546292184725  
dev acc的最大值在0.5464左右 
```
3.NAACL2016(Attention)  
(1)BiGRU + Attention(text_rnn_att_main.py)
```
Epoch 0: cost: 0.0573329023865, acc: 0.521683673469
Epoch 0: dev acc: 0.539575932504
Epoch 1: cost: 0.0507515909112, acc: 0.531887755102
Epoch 1: dev acc: 0.546403197158
Epoch 2: cost: 0.0493788933572, acc: 0.549744897959
Epoch 2: dev acc: 0.546902753108
dev acc的最大值在0.5469左右
```
(2)BiGRU + Multi-Attention(text_multi_att_main.py)
```
Epoch 1 training ...  
Epoch 0: cost: 0.0595334453546, acc: 0.498724489796  
Epoch 0: dev acc: 0.529529307282  
Epoch 2 training ...  
Epoch 1: cost: 0.0518820048595, acc: 0.539540816327  
Epoch 1: dev acc: 0.540464031972  
Epoch 3 training ...  
Epoch 2: cost: 0.051048655443, acc: 0.533163265306  
Epoch 2: dev acc: 0.540741563055  
Epoch 4 training ...  
Epoch 3: cost: 0.0502704293752, acc: 0.567602040816  
Epoch 3: dev acc: 0.540797069272  
Epoch 5 training ...  
Epoch 4: cost: 0.0536497159272, acc: 0.545918367347  
Epoch 4: dev acc: 0.541185612789  
Epoch 6 training ...  
Epoch 5: cost: 0.0486801341936, acc: 0.557397959184  
Epoch 5: dev acc: 0.541407637655  
dev acc的最大值在0.5414左右
```
4.ICLR2017(Structured Self-Attention)
```
Epoch 1 training ...  
Epoch 0: cost: 0.49992824209, acc: 0.498724489796  
Epoch 0: dev acc: 0.522646536412  
Epoch 2 training ...  
Epoch 1: cost: 0.47787049656, acc: 0.52806122449  
Epoch 1: dev acc: 0.533470248668  
Epoch 3 training ...  
Epoch 2: cost: 0.479035496712, acc: 0.533163265306  
Epoch 2: dev acc: 0.533525754885  
dev acc的最大值在0.5335左右
```







