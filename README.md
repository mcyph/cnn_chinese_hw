# cnn_chinese_hw

# Introduction

A convolutional neural network using Keras for recognising Chinese 
(Simplified/Traditional) and Japanese Kanji licensed under the LGPL 2.1. 
An advantage over many other open-source engines is that strokes can 
be drawn out-of-order. 

It is intended to be used as an input method, and accepts x,y coordinate 
points of strokes as input parameters. This differs from some other
engines which are trained to recognise Kanji/Hanzi drawn on physical 
paper with a brush or pen.

# Install

``cnn_chinese_hw`` requires git lfs to download the trained tensorflow 
lite model file when using ``git`` directly: 
https://github.com/git-lfs/git-lfs/wiki/Installation.
Alternatively, you could download this file from the GitHub web interface at 
https://github.com/mcyph/cnn_chinese_hw/blob/master/cnn_chinese_hw/data/hw_quant_model.tflite.

Then, type:

    pip3 install git+https://github.com/mcyph/cnn_chinese_hw/cnn_chinese_hw.git

If you just want to get predictions, you don't need a full `tensorflow` install -
you can use the instructions at https://www.tensorflow.org/lite/guide/python
to install just the tensorflow lite runtime.

In order to train the model however, you will need `tensorflow` at least version 2.0+.
pip3 may need to be updated to do this.

    pip3 install --upgrade pip
    pip3 install tensorflow

# Recognize characters

    from cnn_chinese_hw.recognizer.TFLiteRecognizer import TFLiteRecognizer
    rec = TFLiteRecognizer()
    print rec.get_L_candidates(
        [[(208, 0), (199, 119), (94, 341)],
         [(0, 461), (781, 520), (915, 520), (999, 479)],
         [(189, 167), (213, 209), (238, 826), (268, 934), (203, 910)],
         [(303, 514), (94, 766)],
         [(462, 17), (497, 586), (522, 688), (646, 886), (796, 1000)],
         [(716, 628), (462, 916)],
         [(696, 101), (771, 155), (835, 251)]]
    )

This should recognize `我`.

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


    Copyright (C) 2020  Dave Morrissey
    
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License 2.1 as published by the Free Software Foundation.
    
    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.
    
    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
    USA
