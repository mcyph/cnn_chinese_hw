def iter_tomoe_data(path):
    """
    Open the Tomoe handwriting data at path
    """
    cur_char = None
    strokes_list = []
    LCurStroke = []

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith('<utf8>'):
                if cur_char:
                    if LCurStroke:
                        strokes_list.append(LCurStroke)
                        LCurStroke = []

                    yield ord(cur_char), strokes_list
                    strokes_list = []

                if '<utf8>&#x' in line:
                    char = line.split('<utf8>&#x')[1].split(';')[0]
                    char = int(char, 16)
                else:
                    char = ord(line.split('<utf8>')[1].split('<')[0])
                cur_char = chr(char)

            elif line.startswith('<stroke>'):
                if LCurStroke:
                    strokes_list.append(LCurStroke)
                    LCurStroke = []

            elif line.startswith('<point'):
                x = line.split('x="')[1].split('"')[0]
                y = line.split('y="')[1].split('"')[0]
                x = int(x)
                y = int(y)
                LCurStroke.append((x, y))

    if cur_char:
        if LCurStroke:
            strokes_list.append(LCurStroke)

        yield ord(cur_char), strokes_list
