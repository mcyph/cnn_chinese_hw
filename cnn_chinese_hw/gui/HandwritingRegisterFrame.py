# -*- coding: utf-8 -*-
import wx
import os
from cnn_chinese_hw.gui.HandwritingRegisterCanvas import HandwritingRegisterCanvas
from cnn_chinese_hw.gui.HandwritingRegister import HandwritingRegister
os.chdir('../recognizer')


class HandwritingRegisterFrame(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, -1, 'Handwriting Demo', size=(1000,1100))
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)

        canvas = self.canvas = HandwritingRegisterCanvas(self)
        button_clear = wx.Button(self, label="Clear")
        button_next = wx.Button(self, label="Next")
        button_add = wx.Button(self, label="Add")
        button_clear.Bind(wx.EVT_BUTTON, self.on_button_clear)
        button_next.Bind(wx.EVT_BUTTON, self.on_button_next)
        button_add.Bind(wx.EVT_BUTTON, self.on_button_add)
        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(canvas)
        sizer2.Add(button_clear)
        sizer2.Add(button_next)
        sizer2.Add(button_add)
        sizer.Add(sizer2)
        sizer.Fit(self)

        self.hw_register = HandwritingRegister()
        self.ord_ = self.hw_register.get_next_char()
        print(f"Please draw: {chr(self.ord_)}")
    
    def on_button_clear(self, evt):
        self.canvas.clear()
    
    def on_button_next(self, evt):
        #self.hw_register.add_char(self.ord_, self.canvas.get_L_lines())
        self.canvas.clear()
        self.ord_ = self.hw_register.get_next_char()
        print(f"Please draw: {chr(self.ord_)}")
    
    def on_button_add(self, evt):
        self.hw_register.add_char(self.ord_, self.canvas.get_L_lines())
        self.canvas.clear()


if __name__ == '__main__':
    App = wx.PySimpleApp()
    Frame = HandwritingRegisterFrame(None)
    Frame.Show()
    App.MainLoop()
