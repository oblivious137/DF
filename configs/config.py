class _CFG(dict):
    def __init__(self, *argw, **karg):
        super().__init__(*argw, **karg)
    
    def __setattr__(self, name: str, value) -> None:
        self[name]=value
    
    def __getattr__(self, name: str):
        if name in self:
            return self[name]
        return None
    
    def __getstate__(self):
        return dict(**self)
    
    def __setstate__(self, state):
        self.__init__(**state)