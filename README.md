# cnn_chinese_hw

# Introduction

A convolutional neural network using Keras for recognising Chinese 
(Simplified/Traditional) and Japanese Kanji. An advantage over many 
other open-source engines is that strokes can be drawn out-of-order. 

It is intended to be used as an input method, and accepts x,y coordinate 
points of strokes as input parameters. This differs from some other
engines which are trained to recognise Kanji/Hanzi drawn on physical 
paper with a brush or pen.

# Status

It is still under heavy development and not ready for general use (alpha)
although it is showing promising results.

# License

Because it uses [Tomoe](https://sourceforge.net/projects/tomoe/) data, 
I have put this project and my supplemental data under the same license 
(LGPL 2.1). Compared to the license of some other publicly 
available Chinese handwriting datasets, the LGPL is quite permissive 
and allows for commercial use as well as for use in research.

[KanjiVG](https://kanjivg.tagaini.net/) data is also included for 
validation purposes. This data is not combined when recognizing due 
it being under the Creative Commons Attribution-ShareAlike 3.0 
license. 

# Comments on Implementation

It augments the Tomoe data: distorting from the center, randomizing the points, 
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
increase the likelihood of the CNN being able to recognize different 
people's handwriting, including non-native speakers. 
