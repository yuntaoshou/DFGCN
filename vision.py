import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

#tsne可视化
def TwoD_Tsne(best_labels, best_feature, best_umask, dataset, hidden_size):
    # 假设你的特征数据的形状是 [32, 74, 1024]
    # 将特征张量重塑为 [32 * 74, 1024]
    labels = []
    feature_data = []
    # 读取有效特征, best_feature=[dialog/batch_size, batch_size, seq_len, hidden],best_umask=[dialog/batch_size, batch_size, seq_len]
    index = 0
    for i in range(len(best_umask)):
        for j in range(len(best_umask[i])):
            for k in range(len(best_umask[i][j])):
                if best_umask[i][j][k]==1:
                    labels.append(best_labels[index])
                    feature_data.append(best_feature[i][j][k])
                index += 1
    labels = np.array(labels)
    features_np = [feature.cpu().detach().numpy() for feature in feature_data]
    # TSNE降维
    tsne = TSNE(n_components=2, learning_rate=1000.0, init="random")
    features_2d = tsne.fit_transform(features_np)
    plt.figure(figsize=(10, 8))
    if dataset == 'IEMOCAP':
        # 绘图
        # ['happy', 'sadness', 'neutral', 'anger', 'excited', 'frustrated']
        colors = ['r', 'g', 'b', 'y', 'c', 'm']
        # colors = ['r', 'g', 'b', 'y', 'r', 'g']
        for i in range(6):
            idx = labels == i
            plt.scatter(features_2d[idx, 0], features_2d[idx, 1], c=colors[i])
        plt.legend(['happy', 'sadness', 'neutral', 'anger', 'excited', 'frustrated'])
        plt.title('{} - 6 Class 2D TSNE Visualization'.format(dataset))

    elif dataset=='MELD':
        colors = ['r', 'g', 'b', 'y', 'c', 'm',  'orange' ]
        for i in range(7):
            idx = labels == i
            plt.scatter(features_2d[idx, 0], features_2d[idx, 1], c=colors[i])
        plt.legend(['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'])
        plt.title('{} - 7 Class 2D TSNE Visualization'.format(dataset))

    else:
        colors = ['r', 'g', 'b', 'y', 'c', 'm',  'orange' ]
        for i in range(7):
            idx = labels == i
            plt.scatter(features_2d[idx, 0], features_2d[idx, 1], c=colors[i])

        plt.legend(['happiness', 'neutral', 'anger', 'sadness', 'fear', 'surprise', 'digust'])
        plt.title('{} - 7 Class 2D TSNE Visualization'.format(dataset))

    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# 样本分布可视化
def sampleCount(dataset, dataloader, datatype):
    if dataset == 'MELD':
        num_classes = 7
        numName_classes = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
    elif dataset == 'IEMOCAP':
        num_classes = 6   #  0         1          2         3         4           5
        numName_classes = ['happy', 'sadness', 'neutral', 'anger', 'excited', 'frustrated']
    else:
        num_classes = 7
        numName_classes = ['happiness', 'neutral', 'anger', 'sadness', 'fear', 'surprise', 'digust']
    label_count = [0] * num_classes

    for data in dataloader:
        textf, visuf, acouf, qmask, umask, labels = data[:-3]
        valid_labels = []  # 放在循环内部
        for l, m in zip(labels, umask):
            valid_label = []
            for ll, mm in zip(l, m):
                if mm == 1:
                    valid_label.append(ll.item())
            valid_labels.append(valid_label)
        # 统计数量
        for label in valid_labels:
            for l in label:
                label_count[l] += 1
    # 计算比例
    label_count = np.array(label_count)
    label_ratio = label_count / label_count.sum()
    # 绘制条形图
    plt.figure()
    plt.bar(numName_classes,  label_ratio)
    plt.xlabel('Label')
    plt.ylabel('Ratio')
    plt.title('{}_Label Distribution'.format(datatype))
    plt.show()
    print('{} Label count:'.format(datatype), label_count)

# 混淆矩阵可视化
def confuPLT(confusion, dataset):
    # 设置标签类别
    if dataset=='MELD':
        class_names = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
    elif dataset=='IEMOCAP':
        class_names = ['happy', 'sadness', 'neutral', 'anger', 'excited', 'frustrated']
    else:
        class_names = ['happiness', 'neutral', 'anger', 'sadness', 'fear', 'surprise', 'digust']
    # 创建热图
    plt.figure(figsize=(9, 7))
    sns.set(font_scale=1.2)  # 设置字体大小
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig("./fig/dednet.png", dpi=600)
    # plt.savefig("dednet_conf.png", dpi=600)
