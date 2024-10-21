def normalize_01(pic):
    return (pic - pic.min())/(pic.max() - pic.min())

def normalize_11(pic):
    return 2*(pic - pic.min())/(pic.max() - pic.min()) - 1

def normalize_white(pic):
    return (pic - pic.mean())/pic.std()