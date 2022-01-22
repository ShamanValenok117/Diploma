def make_plots(model, name='Test', save=False):
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(15,5))
    fig.suptitle(name, fontsize=20)

    fig.add_subplot(121)
    plt.plot(model.history.history['loss'],'b--',label='train')
    plt.plot(model.history.history['val_loss'],'g-',label='validation')
    plt.legend(loc='best')
    plt.ylabel('loss')
    x_right_limit = len(model.history.history['loss'])
    tick = int((x_right_limit+1)/10)
    plt.xticks(range(0,x_right_limit+1,tick))
    plt.grid()
    plt.xlabel('epochs')
    plt.title('Loss_vs_epoch');

    fig.add_subplot(122)
    plt.plot(model.history.history['sparse_categorical_accuracy'],'b--',label='train')
    plt.plot(model.history.history['val_sparse_categorical_accuracy'],'g-',label='validation')
    plt.legend(loc='best')
    plt.ylabel('accuracy')
    plt.xticks(range(0,x_right_limit+1,tick))
    plt.grid()
    plt.xlabel('epochs')
    plt.title('Accuracy_vs_epoch');
    
    if save: fig.savefig(name)