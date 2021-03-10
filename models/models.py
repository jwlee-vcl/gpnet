
def create_model_gpnet(opt, _isTrain):
    model = None
    from .gpnet_model import GPNet
    model = GPNet(opt, _isTrain)
    print("model [%s] was created" % (model.name()))
    return model

    