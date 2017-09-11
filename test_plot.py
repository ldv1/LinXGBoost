import matplotlib.pyplot as plt

def test_subplot(test_func_name, train_X, train_Y, test_X, test_Y, pred_Y, method_name, fontsize='xx-large'):
    plt.plot(train_X, train_Y, 'bo', markersize=16, label='training data') # 16
    plt.plot(test_X, test_Y, 'b-', linewidth=8, label='ground truth') # 8
    plt.plot(test_X, pred_Y, 'r-', linewidth=8, label=method_name+' predictions')
    if test_func_name == "linear":
        plt.ylim(-2, 1.5)
    elif test_func_name == "sine":
        plt.ylim(-3, 3)
    elif test_func_name == "heavysine":
        plt.ylim(-7, 5)
    plt.legend(fontsize = fontsize)

def test_plot3a(test_func_name, train_X, train_Y, test_X, test_Y, pred1_Y, pred1_name, pred2_Y, pred2_name, pred3_Y, pred3_name, fontsize='xx-large', savefig=False):
    plt.figure(figsize=(50, 16))
    plt.subplot(131)
    test_subplot(test_func_name, train_X, train_Y, test_X, test_Y, pred1_Y, pred1_name, fontsize)
    plt.subplot(132)
    test_subplot(test_func_name, train_X, train_Y, test_X, test_Y, pred2_Y, pred2_name, fontsize)
    plt.subplot(133)
    test_subplot(test_func_name, train_X, train_Y, test_X, test_Y, pred3_Y, pred3_name, fontsize)
    if savefig:
        plt.savefig('foo.png', bbox_inches='tight')
    plt.show()

def test_plot3b(test_func_name,
                train1_X, train1_Y, test1_X, test1_Y, pred1_Y, pred1_name,
                train2_X, train2_Y, test2_X, test2_Y, pred2_Y, pred2_name,
                train3_X, train3_Y, test3_X, test3_Y, pred3_Y, pred3_name,
                fontsize='xx-large', savefig=False):
    plt.figure(figsize=(50, 16))
    plt.subplot(131)
    test_subplot(test_func_name, train1_X, train1_Y, test1_X, test1_Y, pred1_Y, pred1_name, fontsize)
    plt.subplot(132)
    test_subplot(test_func_name, train2_X, train2_Y, test2_X, test2_Y, pred2_Y, pred2_name, fontsize)
    plt.subplot(133)
    test_subplot(test_func_name, train3_X, train3_Y, test3_X, test3_Y, pred3_Y, pred3_name, fontsize)
    if savefig:
        plt.savefig('foo.png', bbox_inches='tight')
    plt.show()

def test_plot2(test_func_name, train_X, train_Y, test_X, test_Y, pred1_Y, pred1_name, pred2_Y, pred2_name, fontsize='xx-large', savefig=False):
    plt.figure(figsize=(50, 16))
    plt.subplot(121)
    test_subplot(test_func_name, train_X, train_Y, test_X, test_Y, pred1_Y, pred1_name, fontsize)
    plt.subplot(122)
    test_subplot(test_func_name, train_X, train_Y, test_X, test_Y, pred2_Y, pred2_name, fontsize)
    if savefig:
        plt.savefig('foo.png', bbox_inches='tight')
    plt.show()
