class re_encoder:
    def __init__(self,text):
        self.text = text
    def run(self):
        l = []
        for i in range(len(self.text)-1,-1,-1):
            l.append(self.text[i])
        return''.join(l)