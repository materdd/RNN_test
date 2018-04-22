import matplotlib.pyplot as plt
import json

def plot():
    with open('result/log') as f:
        logs = json.load(f)
    
    # 結果を整理
    loss_train = [ log['main/loss'] for log in logs ]
    loss_test  = [ log['validation/main/loss'] for log in logs ]
    
    # プロット
    plt.plot(loss_train, label='train')
    plt.plot(loss_test,  label='test')
    plt.legend()
    #plt.show()

    plt.savefig("./result/accuracy.png")

if __name__ == '__main__':
    plot()