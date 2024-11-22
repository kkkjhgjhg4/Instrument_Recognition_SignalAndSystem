import os
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter

# 标签映射（缩写到完整名称）
label_mapping = {
    'cel': 'cello',
    'cla': 'clarinet',
    'flu': 'flute',
    'gac': 'acoustic_guitar',
    'gel': 'electric_guitar',
    'org': 'organ',
    'pia': 'piano',
    'sax': 'saxophone',
    'tru': 'trumpet',
    'vio': 'violin',
    'voi': 'vocal'
}


def extract_label_from_folder(folder_name, label_mapping):
    """
    从文件夹名称中提取标签，并映射到完整的乐器名称。

    Args:
        folder_name (str): 文件夹名称。
        label_mapping (dict): 标签缩写到完整名称的映射字典。

    Returns:
        str: 完整的乐器名称，如果未找到映射则返回 'unknown'。
    """
    label_abbr = folder_name.lower()
    return label_mapping.get(label_abbr, 'unknown')


def get_file_paths_and_labels(root_dir, label_mapping):
    """
    递归遍历目录，收集所有.wav文件路径及其对应的标签。

    Args:
        root_dir (str): 根目录路径。
        label_mapping (dict): 标签缩写到完整名称的映射字典。

    Returns:
        list, list: 文件路径列表和对应的标签列表。
    """
    file_paths = []
    labels = []
    for subdir, dirs, files in os.walk(root_dir):
        # 提取当前子目录的标签（假设标签在子目录名称）
        folder_name = os.path.basename(subdir)
        label = extract_label_from_folder(folder_name, label_mapping)
        if label == 'unknown':
            continue  # 跳过未知标签的文件夹
        for filename in files:
            if filename.lower().endswith('.wav'):
                file_path = os.path.join(subdir, filename)
                file_paths.append(file_path)
                labels.append(label)
    return file_paths, labels


# 定义数据集类
class IRMASDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, max_len=44100 * 3):
        """
        Args:
            file_paths (list): 音频文件路径列表。
            labels (list): 对应的标签列表。
            transform (callable, optional): 对输入数据的变换。
            max_len (int): 固定音频长度（采样点数）。超过的部分截断，不足的部分填充。
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.max_len = max_len
        self.unique_labels = sorted(list(set(labels)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        # 加载音频，转换为单声道，采样率为44.1kHz
        y, sr = librosa.load(audio_path, sr=44100, mono=True)

        # 数据增强（可选）
        # 可以在此处添加数据增强方法，例如时间伸缩、频率掩蔽等
        # 示例：
        # if np.random.rand() > 0.5:
        #     y = librosa.effects.time_stretch(y, rate=1.1)
        # if np.random.rand() > 0.5:
        #     y = librosa.effects.pitch_shift(y, sr, n_steps=2)

        # 固定长度
        if len(y) > self.max_len:
            y = y[:self.max_len]
        else:
            y = np.pad(y, (0, max(0, self.max_len - len(y))), 'constant')

        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = librosa.util.fix_length(mfcc, size=40, axis=1)  # 固定时间帧数

        # 转置以适应CNN输入（channels, height, width）
        mfcc = mfcc.T  # shape: (time_frames, n_mfcc)
        mfcc = np.expand_dims(mfcc, axis=0)  # shape: (1, time_frames, n_mfcc)

        if self.transform:
            mfcc = self.transform(mfcc)
        else:
            mfcc = torch.tensor(mfcc, dtype=torch.float32)

        label = self.label_to_idx[label]
        label = torch.tensor(label, dtype=torch.long)

        return mfcc, label


# 定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.3)

        # 假设MFCC的时间帧数为40，经过3次池化（每次除以2），时间帧数约为5
        # n_mfcc = 40，经过3次池化后约为5
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # [batch, 32, H/2, W/2]
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # [batch, 64, H/4, W/4]
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # [batch, 128, H/8, W/8]
        x = x.view(x.size(0), -1)  # 展平
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def main():
    # 设置路径和参数
    train_dir = './IRMAS/IRMAS-TrainingData'  # 替换为实际训练数据路径


    batch_size = 32
    num_workers = 4  # 可以尝试减少，尤其是在调试时
    num_epochs = 30

    # 获取文件路径和标签
    train_files, train_labels = get_file_paths_and_labels(train_dir, label_mapping)

    # 打印调试信息
    print(f'训练集文件数量: {len(train_files)}')
    if len(train_files) == 0:
        print("错误：训练集文件数量为0。请检查文件路径和文件夹名称是否正确。")
        exit(1)

    print('训练集示例:')
    for i in range(min(5, len(train_files))):
        print(f'文件: {train_files[i]}, 标签: {train_labels[i]}')

    # 划分训练集和验证集
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files,
        train_labels,
        test_size=0.2,
        stratify=train_labels,
        random_state=42
    )

    print(f'训练集实际文件数量: {len(train_files)}')
    print(f'验证集文件数量: {len(val_files)}')

    # 创建数据集
    train_dataset = IRMASDataset(train_files, train_labels)
    val_dataset = IRMASDataset(val_files, val_labels)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 检查DataLoader是否正常
    if len(train_loader.dataset) == 0:
        print("错误：训练数据集为空。")
        exit(1)
    if len(val_loader.dataset) == 0:
        print("错误：验证数据集为空。")
        exit(1)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'使用设备: {device}')

    # 初始化模型
    num_classes = len(train_dataset.unique_labels)
    print(f'类别数: {num_classes}')
    model = SimpleCNN(num_classes=num_classes).to(device)

    # 定义损失函数和优化器
    # 处理类别不平衡问题，可以使用加权损失函数
    label_counts = Counter(train_labels)
    class_weights = 1. / np.array([label_counts[label] for label in train_dataset.unique_labels])
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 定义学习率调度器（可选）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # 训练和验证循环
    best_val_acc = 0.0
    patience = 10  # 早停耐心
    trigger_times = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)

        # 验证
        model.eval()
        val_running_loss = 0.0
        val_all_preds = []
        val_all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_all_preds.extend(preds.cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_all_labels, val_all_preds)
        val_f1 = f1_score(val_all_labels, val_all_preds, average='weighted')
        val_precision = precision_score(val_all_labels, val_all_preds, average='weighted')
        val_recall = recall_score(val_all_labels, val_all_preds, average='weighted')

        print(f'Epoch [{epoch + 1}/{num_epochs}] '
              f'Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} '
              f'Val Precision: {val_precision:.4f} Val Recall: {val_recall:.4f} Val F1: {val_f1:.4f}')

        # 调整学习率
        scheduler.step(val_acc)

        # 早停逻辑
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trigger_times = 0
            torch.save(model.state_dict(), 'best_irmas_cnn.pth')
            print(f'最佳模型已保存，验证准确率: {best_val_acc:.4f}')
        else:
            trigger_times += 1
            print(f'Early stopping trigger: {trigger_times}/{patience}')
            if trigger_times >= patience:
                print('早停触发，停止训练')
                break

    # 加载最佳模型进行评估
    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('best_irmas_cnn.pth'))
    model.eval()

    def evaluate(model, loader):
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Evaluating"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        print(f'Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    # 评估验证集
    evaluate(model, val_loader)


if __name__ == '__main__':
    main()