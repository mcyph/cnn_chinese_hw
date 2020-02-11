import wx
from abc import ABC, abstractmethod
from cnn_chinese_hw.stroke_tools.get_vertex import get_vertex
from cnn_chinese_hw.stroke_tools.points_to_plot import points_to_plot


BUFFERED = False


class HandwritingCanvas(wx.ScrolledWindow):
    def __init__(self, parent, id=-1, size=(250, 250)):
        wx.ScrolledWindow.__init__(self, parent, id, (0, 0), size=size)

        self.lines = []
        self.LConvLines = []

        self.maxWidth = 1000
        self.maxHeight = 1000
        self.x = self.y = 0
        self.curLine = []
        self.drawing = False

        self.SetBackgroundColour("WHITE")
        self.SetCursor(wx.StockCursor(wx.CURSOR_PENCIL))

        self.SetVirtualSize((self.maxWidth, self.maxHeight))
        #self.SetScrollRate(20,20)

        if BUFFERED:
            # Initialize the buffer bitmap.  No real DC is needed at this point.
            self.buffer = wx.EmptyBitmap(self.maxWidth, self.maxHeight)
            dc = wx.BufferedDC(None, self.buffer)
            dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
            dc.Clear()
            self.do_drawing(dc)

        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_button_event)
        self.Bind(wx.EVT_LEFT_UP,   self.on_left_button_event)
        self.Bind(wx.EVT_MOTION,    self.on_left_button_event)
        self.Bind(wx.EVT_PAINT, self.on_paint)

    def get_L_lines(self):
        return self.lines

    def clear(self):
        self.lines = []
        self.curLine = []
        self.Refresh()

    def get_width(self):
        return self.maxWidth

    def get_height(self):
        return self.maxHeight

    def on_paint(self, event):
        if BUFFERED:
            # Create a buffered paint DC.  It will create the real
            # wx.PaintDC and then blit the bitmap to it when dc is
            # deleted.  Since we don't need to draw anything else
            # here that's all there is to it.
            dc = wx.BufferedPaintDC(self, self.buffer, wx.BUFFER_VIRTUAL_AREA)
        else:
            dc = wx.PaintDC(self)
            self.PrepareDC(dc)
            # since we're not buffering in this case, we have to
            # paint the whole window, potentially very time consuming.
            self.do_drawing(dc)

    def do_drawing(self, dc, printing=False):
        #dc.BeginDrawing()
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.SetPen(wx.Pen(wx.Colour(0xFF, 0x20, 0xFF), 1, wx.SOLID))
        self.draw_saved_lines(dc)
        #dc.EndDrawing()

    def draw_saved_lines(self, dc):
        dc.SetPen(wx.Pen('MAGENTA', 4))
        for line in self.lines:
            for x, coords in enumerate(line):
                if x:
                    prev_coords = line[x-1]
                else:
                    prev_coords = coords

                dc.DrawLine(*prev_coords+coords)

        dc.SetPen(wx.Pen('BLACK', 3))
        #print self.LConvLines
        for line in self.LConvLines:
            line = points_to_plot([line])[0]
            for coords in line:
                coords = coords[0]/4, coords[1]/4
                dc.DrawLine(*coords+coords)

    def set_xy(self, event):
        self.x, self.y = self.convert_event_coords(event)

    def convert_event_coords(self, event):
        newpos = self.CalcUnscrolledPosition(event.GetX(), event.GetY())
        return newpos

    def on_left_button_event(self, event):
        if event.LeftDown():
            self.SetFocus()
            self.set_xy(event)
            self.curLine = []
            self.CaptureMouse()
            self.drawing = True

        elif event.Dragging() and self.drawing:
            if BUFFERED:
                # If doing buffered drawing we'll just update the
                # buffer here and then refresh that portion of the
                # window, then that portion of the buffer will be
                # redrawn in the EVT_PAINT handler.
                dc = wx.BufferedDC(None, self.buffer)
            else:
                # otherwise we'll draw directly to a wx.ClientDC
                dc = wx.ClientDC(self)
                self.PrepareDC(dc)

            dc.SetPen(wx.Pen('MEDIUM FOREST GREEN', 4))
            coords = (self.x, self.y)
            self.curLine.append(coords)
            dc.DrawLine(*coords+coords)
            self.set_xy(event)

            if BUFFERED:
                # figure out what part of the window to refresh
                x1,y1, x2,y2 = dc.GetBoundingBox()
                x1,y1 = self.CalcScrolledPosition(x1, y1)
                x2,y2 = self.CalcScrolledPosition(x2, y2)
                # make a rectangle
                rect = wx.Rect()
                rect.SetTopLeft((x1,y1))
                rect.SetBottomRight((x2,y2))
                rect.Inflate(2,2)
                # refresh it
                self.RefreshRect(rect)

        elif event.LeftUp() and self.drawing:
            if self.curLine:
                self.lines.append(self.curLine)
            self.curLine = []
            self.ReleaseMouse()
            self.drawing = False
            #print self.lines

            def limit(i):
                # HACK: Prevent offscreen drawing!
                if i < 0: i = 0
                if i > 1000: i = 1000
                return i

            # Remove noise in the line data
            # (only leaving important significant changes in direction)
            # mirroring the original Tomoe data
            LVertices = [get_vertex(i) for i in self.lines]
            #print(LVertices)

            self.process_lines(LVertices)
            self.Refresh()

    @abstractmethod
    def process_lines(self, LVertices):
        pass

