# -*- coding: utf-8 -*-
import wx
import os
from cnn_chinese_hw.gui.HandwritingRecognizerCanvas import HandwritingRecogniserCanvas
os.chdir('../recognizer')


class RecogiserFrame(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, -1, 'Handwriting Demo', size=(1000,1100))
        Szr = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(Szr)

        canvas = self.canvas = HandwritingRecogniserCanvas(self)
        button_clear = wx.Button(self, label="Clear")
        button_clear.Bind(wx.EVT_BUTTON, self.on_button_clear)

        Szr.Add(canvas)
        Szr.Add(button_clear)
        Szr.Fit(self)

    def on_button_clear(self, evt):
        self.canvas.clear()


if __name__ == '__main__':
    App = wx.PySimpleApp()
    Frame = RecogiserFrame(None)
    Frame.Show()
    App.MainLoop()
