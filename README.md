# cnn_chinese_hw

A convolutional neural network using Keras for recognising Chinese 
(Simplified/Traditional) and Japanese Kanji. Because it uses Tomoe data,
I have put it under the same license (LGPL 2.1). Unlike CASIA and some other 
publicly available Chinese handwriting datasets, this data can be used for 
commercial purposes. An advantage over many other open-source engines is 
that strokes can be drawn out-of-order. 

It is still under heavy development and not ready for general use (pre-alpha). 

It augments the Tomoe data: distorting from the center, randomising the points, 
rotating the characters and strokes to a degree to increase the likelihood of 
recognition. 

When the correct candidate isn't always the first one, it usually 
is in the top few. Adding 
[dropout](https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/) 
to the dense (fully connected) layers and 
[batch normalization](https://www.kdnuggets.com/2018/09/dropout-convolutional-networks.html) 
to the convolutional 2d layers significantly improved results.

Because the data was drawn by only a few people, it may have trouble 
recognising some people's handwriting, although I think it provides pretty good
results. I have added a few hundred characters which I have drawn myself, 
many of them with incorrect numbers of strokes/more or less curves so as to 
increase the likelihood of the CNN being able to recognise different 
people's handwriting, including non-native speakers. 

