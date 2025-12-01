CONTENT_REG, PROV_REG, PROP_REG, AGGR_REG, POLICY_REG = {}, {}, {}, {}, {}

def register(registry, name: str):
    """装饰器式注册机制"""
    def deco(cls):
        registry[name] = cls
        return cls
    return deco
