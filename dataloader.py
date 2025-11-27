import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy as np

class IEMOCAPDataset(Dataset):
    def __init__(self, path, windows, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)
        self.windows = windows

    def get_semantic_adj(self, data):
        semantic_adj = []
        max_len = 0
        # 获取一个batch中最大的对话长度
        for i in range(len(data)):
            if max_len < len(data[i][3]):
                max_len = len(data[i][3])
        batch_speakers = []
        for i in range(len(data)):
            batch_speakers.append(data[i][3].tolist())
        # 每个speakers是一个dialog
        for speaker in batch_speakers:  # 遍历每个对话 对应的说话人列表（非去重）
            s = torch.zeros(max_len, max_len, dtype=torch.long)  # （N,N） 0 表示填充部分 没有语义关系
            for i in range(len(speaker)):  # 每个utterance 的说话人 和 其他 utterance 的说话人 是否相同
                for j in range(i-self.windows, i+self.windows+1):
                    if j < 0:
                        continue
                    elif j == len(speaker):
                        break
                    else:
                        if speaker[i] == speaker[j]:
                            if i == j:
                                s[i, j] = 1  # 对角线  self
                            elif i < j:
                                s[i, j] = 2  # self-future
                            else:
                                s[i, j] = 3  # self-past
                        else:
                            if i < j:
                                s[i, j] = 4  # inter-future
                            elif i > j:
                                s[i, j] = 5  # inter-past
            semantic_adj.append(s)
        return torch.stack(semantic_adj)

    def  getSelf_semantic_adj(self, data):
        Self_semantic_adj = []
        max_len = 0
        # 获取一个batch中最大的对话长度
        for i in range(len(data)):
            if max_len < len(data[i][3]):
                max_len = len(data[i][3])
        batch_speakers = []
        for i in range(len(data)):
            batch_speakers.append(data[i][3].tolist())
        # 每个speakers是一个dialog
        for speaker in batch_speakers:  # 遍历每个对话 对应的说话人列表（非去重）
            s = torch.zeros(max_len, max_len, dtype=torch.long)  # （N,N） 0 表示填充部分 没有语义关系
            for i in range(len(speaker)):  # 每个utterance 的说话人 和 其他 utterance 的说话人 是否相同
                for j in range(i-self.windows, i+self.windows+1):
                    if j < 0:
                        continue
                    elif j == len(speaker):
                        break
                    else:
                        if speaker[i] == speaker[j]:
                            if i == j:
                                s[i, j] = 1
                            elif i > j:
                                s[i, j] = 2
                            else:
                                s[i, j] = 3
            Self_semantic_adj.append(s)
        return torch.stack(Self_semantic_adj)

    def getCross_semantic_adj(self, data):
        Cross_semantic_adj = []
        max_len = 0
        # 获取一个batch中最大的对话长度
        for i in range(len(data)):
            if max_len < len(data[i][3]):
                max_len = len(data[i][3])
        batch_speakers = []
        for i in range(len(data)):
            batch_speakers.append(data[i][3].tolist())
        # 每个speakers是一个dialog
        for speaker in batch_speakers:  # 遍历每个对话 对应的说话人列表（非去重）
            s = torch.zeros(max_len, max_len, dtype=torch.long)  # （N,N） 0 表示填充部分 没有语义关系
            for i in range(len(speaker)):  # 每个utterance 的说话人 和 其他 utterance 的说话人 是否相同
                for j in range(i-self.windows, i+self.windows+1):
                    if j < 0:
                        continue
                    elif j == len(speaker):
                        break
                    else:
                        if speaker[i] == speaker[j]:
                            if i == j:
                                s[i, j] = 1  # 对角线  self
                        else:
                            if i < j:
                                s[i, j] = 4  # inter-future
                            elif i > j:
                                s[i, j] = 5  # inter-past
                            # s[i, j] = 3

            Cross_semantic_adj.append(s)
        return torch.stack(Cross_semantic_adj)

    def __getitem__(self, index):
        vid = self.keys[index]
        # vid = self.keys[3]
        return torch.FloatTensor(np.array(self.videoText[vid])),\
               torch.FloatTensor(np.array(self.roberta2[vid])),\
               torch.FloatTensor(np.array(self.roberta3[vid])),\
               torch.FloatTensor(np.array(self.roberta4[vid])),\
               torch.FloatTensor(np.array(self.videoVisual[vid])),\
               torch.FloatTensor(np.array(self.videoAudio[vid])),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid])

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        Self_semantic_adj = self.getSelf_semantic_adj(data)
        Cross_semantic_adj = self.getCross_semantic_adj(data)
        Semantic_adj = self.get_semantic_adj(data)
        # print(semantic_adj.shape)
        data = [pad_sequence(dat[i]) if i < 7 else pad_sequence(dat[i], True) if i < 9 else dat[i].tolist() for i in
                dat]
        data.append(torch.LongTensor(Self_semantic_adj))
        data.append(torch.LongTensor(Cross_semantic_adj))
        data.append(torch.LongTensor(Semantic_adj))
        return data
    
class MELDDataset(Dataset):
    def __init__(self, path, windows, train=True):
        videoIDs, videoSpeakers, videoLabels, videoText, \
        roberta2, roberta3, roberta4, \
        videoAudio, videoVisual, videoSentence, trainVid, \
        testVid,_ = pickle.load(open(path, 'rb'), encoding='latin1')

        self.videoIDs = videoIDs
        self.roberta2 = roberta2
        self.roberta3 = roberta3
        self.roberta4 = roberta4
        self.videoSpeakers = videoSpeakers
        self.videoLabels = videoLabels
        self.videoText = videoText
        self.videoAudio = videoAudio
        self.videoVisual = videoVisual
        self.videoSentence = videoSentence
        self.trainVid = trainVid
        self.testVid = testVid
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.windows = windows
        self.len = len(self.keys)

    def __getitem__(self, index):
        # dialog索引
        vid = self.keys[index]
        # vid = self.keys[37]
        return torch.FloatTensor(np.array(self.videoText[vid])),\
               torch.FloatTensor(np.array(self.roberta2[vid])),\
               torch.FloatTensor(np.array(self.roberta3[vid])),\
               torch.FloatTensor(np.array(self.roberta4[vid])),\
               torch.FloatTensor(np.array(self.videoVisual[vid])),\
               torch.FloatTensor(np.array(self.videoAudio[vid])),\
               torch.IntTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid])

    def __len__(self):
        return self.len

    def get_semantic_adj(self, data):
        semantic_adj = []
        max_len = 0
        # 获取一个batch中最大的对话长度
        for i in range(len(data)):
            if max_len<len(data[i][3]):
                max_len=len(data[i][3])
        batch_speakers = []
        for i in range(len(data)):
            batch_speakers.append(data[i][3].tolist())
        # 每个speakers是一个dialog
        for speaker in batch_speakers:  # 遍历每个对话 对应的说话人列表（非去重）
            s = torch.zeros(max_len, max_len, dtype=torch.long)  # （N,N） 0 表示填充部分 没有语义关系
            for i in range(len(speaker)):  # 每个utterance 的说话人 和 其他 utterance 的说话人 是否相同
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        if i == j:
                            s[i, j] = 1  # 对角线  self
                        elif i < j:
                            s[i, j] = 2  # self-future
                        else:
                            s[i, j] = 3  # self-past
                    else:
                        if i < j:
                            s[i, j] = 4  # inter-future
                        elif i > j:
                            s[i, j] = 5  # inter-past
            semantic_adj.append(s)
        return torch.stack(semantic_adj)

    # 相同说话者子图
    def getSelf_semantic_adj(self, data):
        Self_semantic_adj = []
        max_len = 0
        # 获取一个batch中最大的对话长度
        for i in range(len(data)):
            if max_len < len(data[i][3]):
                max_len = len(data[i][3])
        batch_speakers = []
        for i in range(len(data)):
            batch_speakers.append(data[i][3].tolist())
        # 每个speakers是一个dialog
        for speaker in batch_speakers:  # 遍历每个对话 对应的说话人列表（非去重）
            s = torch.zeros(max_len, max_len, dtype=torch.long)  # （N,N） 0 表示填充部分 没有语义关系
            for i in range(len(speaker)):  # 每个utterance 的说话人 和 其他 utterance 的说话人 是否相同
                for j in range(i-self.windows, i+self.windows+1):
                    if j < 0:
                        continue
                    elif j == len(speaker):
                        break
                    else:
                        if speaker[i] == speaker[j]:
                            if i == j:
                                s[i, j] = 1
                            elif i > j:
                                s[i, j] = 2 # intra-past
                            else:
                                s[i, j] = 3 # intra-future
            Self_semantic_adj.append(s)
        return torch.stack(Self_semantic_adj)

    # 不同说话者子图
    def getCross_semantic_adj(self, data):
        Cross_semantic_adj = []
        max_len = 0
        # 获取一个batch中最大的对话长度
        for i in range(len(data)):
            if max_len < len(data[i][3]):
                max_len = len(data[i][3])
        batch_speakers = []
        for i in range(len(data)):
            batch_speakers.append(data[i][3].tolist())
        # 每个speakers是一个dialog
        for speaker in batch_speakers:  # 遍历每个对话 对应的说话人列表（非去重）
            s = torch.zeros(max_len, max_len, dtype=torch.long)  # （N,N） 0 表示填充部分 没有语义关系
            for i in range(len(speaker)):  # 每个utterance 的说话人 和 其他 utterance 的说话人 是否相同
                for j in range(i-self.windows, i+self.windows+1):
                    if j < 0:
                        continue
                    elif j == len(speaker):
                        break
                    else:
                        if speaker[i] == speaker[j]:
                            if i == j:
                                s[i, j] = 1  # 对角线  self
                        else:
                            if i < j:
                                s[i, j] = 4  # inter-future
                            elif i > j:
                                s[i, j] = 5  # inter-past
                            # s[i, j] = 3

            Cross_semantic_adj.append(s)
        return torch.stack(Cross_semantic_adj)

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label += self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        Self_semantic_adj = self.getSelf_semantic_adj(data)
        Cross_semantic_adj = self.getCross_semantic_adj(data)
        Semantic_adj = self.get_semantic_adj(data)
        # print(semantic_adj.shape)
        data = [pad_sequence(dat[i]) if i < 7 else pad_sequence(dat[i], True) if i < 9 else dat[i].tolist() for i in
                dat]
        data.append(torch.LongTensor(Self_semantic_adj))
        data.append(torch.LongTensor(Cross_semantic_adj))
        data.append(torch.LongTensor(Semantic_adj))
        return data
