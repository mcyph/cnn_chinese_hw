# cnn_chinese_hw

A convolutional neural network using Keras for recognising Chinese 
(Simplified/Traditional) and Japanese Kanji.

The main advantage over engines such as Tomoe and Zinnia is that strokes can be drawn 
out-of-order. Currently, the performance of this engine will not be as high as these
engines for those who can write the 

It is still under heavy development. It uses data from Tomoe (data/handwriting*.xml)
under the GNU LGPL. I have added some data which I have drawn myself, many of them
with incorrect numbers of strokes/more or less curves so as to increase the 
likelihood of the CNN being able to recognise different people's handwriting,
including non-native speakers.
