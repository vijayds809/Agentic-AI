class di_encoder:
    """direct usage of indexing[::-1]"""
    def __init__(self,text):
        self.text = text
    def run(self):
        return self.text[::-1]