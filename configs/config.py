class _CFG(dict):
    def __init__(self, *argw, **karg):
        super().__init__(*argw, **karg)
    
    def __setattr__(self, name: str, value) -> None:
        self[name]=value
    
    def __getattr__(self, name: str):
        if name in self:
            return self[name]
        return None