import numpy as np


def early_stop_gl(val_loss):
    var_opt = min(val_loss) + 1e-6
    var_va = val_loss[-1] + 1e-6
    gl = 100 * (var_va / var_opt - 1)
    if gl > 2:
        return 1
    else:
        return 0


def early_stop_pq(epoch, train_loss, val_loss):
    k = 5
    if epoch >= k:
        sum_train = np.sum(train_loss[-1:-k:-1])
        p_k = 1000 * (sum_train / (k * np.min(train_loss[-1:-k:-1])) - 1)
        gl = early_stop_gl(val_loss)
        p_q = gl / p_k
        if p_q > 0.5:
            return True
        else:
            return False


def early_stop_up(epoch, val_loss):
    s = 3
    k = 5
    count = 0
    if epoch > s * k:
        for i in range(s):
            if val_loss[-1 - (i - 1) * k] > val_loss[-k - (i - 1) * k]:
                count += 1
        if count == s:
            return True
        else:
            return False