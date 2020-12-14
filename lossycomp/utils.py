"""Model functions"""

def decay_schedule(epoch, lr):
    # decay by 0.1 every 5 epochs; use `% 1` to decay after each epoch
    if (epoch % 10 == 0) and (epoch != 0):
        lr = lr * 0.1
    return lr

