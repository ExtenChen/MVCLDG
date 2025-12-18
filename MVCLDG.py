import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional


class MVCLDGModel(nn.Module):
    """
    Multi-View Contrastive Learning Domain Generalization (MVCLDG)
    多视图对比学习域泛化模型
    
    核心创新点：
    1. 多视图特征提取：融合原始EEG信号和希尔伯特变换相位信息
    2. 域不变表示学习：通过域对齐和对比学习约束
    3. Inception-to-Inception架构：编码器投影器同构但增加非线性
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int = 2, 
                 num_domains: int = 4, config: Optional[Dict] = None):
        """
        初始化MVCLDG模型
        
        Args:
            input_shape: (channels_in, EEG_channels, time_points)
            num_classes: 分类类别数
            num_domains: 域数量
            config: 配置参数字典
        """
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        
        # 默认配置
        self.config = config or {
            'filters_per_branch': 12,
            'dropout_rate': 0.25,
            'temperature': 0.07,
            'tradeoff_align': 1.0,
            'tradeoff_contrast': 0.5,
            'projection_dim': 32
        }
        
        # 多视图特征提取模块
        self.raw_view_encoder = MultiViewEncoder(input_shape, 'raw')
        self.ht_view_encoder = MultiViewEncoder(input_shape, 'ht')
        self.fusion_encoder = MultiViewEncoder(input_shape, 'fusion')
        
        # 投影器（Inception-to-Inception架构）
        self.raw_projection = InceptionProjection(self.raw_view_encoder.feature_dim)
        self.ht_projection = InceptionProjection(self.ht_view_encoder.feature_dim)
        self.fusion_projection = InceptionProjection(self.fusion_encoder.feature_dim)
        
        # 分类器（每个视图一个）
        self.raw_classifier = nn.Linear(self.raw_view_encoder.feature_dim, num_classes)
        self.ht_classifier = nn.Linear(self.ht_view_encoder.feature_dim, num_classes)
        self.fusion_classifier = nn.Linear(self.fusion_encoder.feature_dim, num_classes)
        
        # 域对齐模块
        self.domain_aligner = DomainAlignmentModule(num_domains)
        
        # 对比学习模块
        self.contrastive_learner = MultiViewContrastiveLearner(
            temperature=self.config['temperature'],
            projection_dim=self.config['projection_dim']
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, channels, time_points]
            
        Returns:
            dict: 包含各视图的输出
        """
        # 分离原始视图和HT视图
        batch_size = x.shape[0]
        n_channels = self.input_shape[1]
        
        # 假设输入已经包含原始和HT视图的拼接
        raw_view = x[:, :n_channels//2, :]  # 原始视图
        ht_view = x[:, n_channels//2:, :]   # HT视图
        
        # 提取特征
        raw_features = self.raw_view_encoder(raw_view)
        ht_features = self.ht_view_encoder(ht_view)
        
        # 融合视图（拼接原始和HT特征）
        fusion_features = self.fusion_encoder(x)
        
        # 投影（用于对比学习）
        raw_projection = self.raw_projection(raw_features)
        ht_projection = self.ht_projection(ht_features)
        fusion_projection = self.fusion_projection(fusion_features)
        
        # 分类
        raw_logits = self.raw_classifier(raw_features)
        ht_logits = self.ht_classifier(ht_features)
        fusion_logits = self.fusion_classifier(fusion_features)
        
        return {
            'raw': {
                'logits': raw_logits,
                'features': raw_features,
                'projection': F.normalize(raw_projection, dim=1)
            },
            'ht': {
                'logits': ht_logits,
                'features': ht_features,
                'projection': F.normalize(ht_projection, dim=1)
            },
            'fusion': {
                'logits': fusion_logits,
                'features': fusion_features,
                'projection': F.normalize(fusion_projection, dim=1)
            }
        }


class MultiViewEncoder(nn.Module):
    """多视图特征提取模块"""
    
    def __init__(self, input_shape: Tuple[int, int, int], view_type: str = 'raw'):
        super().__init__()
        self.view_type = view_type
        n_channels = input_shape[1]
        time_points = input_shape[2]
        
        # 多尺度Inception块
        self.inception_blocks = nn.ModuleList([
            InceptionBlock(1, 12, kernel_sizes=[time_points//4, time_points//8]),
            InceptionBlock(12, 24, kernel_sizes=[n_channels//2, n_channels//4]),
        ])
        
        # 自适应池化
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 计算特征维度
        self.feature_dim = self._compute_feature_dim(input_shape)
    
    def _compute_feature_dim(self, input_shape: Tuple[int, int, int]) -> int:
        """计算特征维度"""
        # 通过前向传递计算维度
        x = torch.randn(1, *input_shape)
        with torch.no_grad():
            for block in self.inception_blocks:
                x = block(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
        return x.shape[1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 添加通道维度
        x = x.unsqueeze(1)  # [B, 1, C, T]
        
        # 通过多尺度Inception块
        for block in self.inception_blocks:
            x = block(x)
        
        # 池化和扁平化
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        return x


class InceptionBlock(nn.Module):
    """多尺度Inception块"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int]):
        super().__init__()
        self.branches = nn.ModuleList()
        
        # 多个并行的卷积分支
        for kernel_size in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), 
                         padding='same'),
                nn.BatchNorm2d(out_channels),
                nn.ELU(),
                nn.Dropout(0.25)
            )
            self.branches.append(branch)
        
        # 融合层
        self.fusion = nn.Conv2d(len(kernel_sizes) * out_channels, out_channels, 
                               kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        branch_outputs = [branch(x) for branch in self.branches]
        concatenated = torch.cat(branch_outputs, dim=1)
        fused = self.fusion(concatenated)
        return fused


class InceptionProjection(nn.Module):
    """Inception-to-Inception投影器"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        
        # 同构但增加非线性的投影架构
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.3),
            
            # 多尺度特征提取（模拟Inception结构）
            ParallelLinear(hidden_dim, hidden_dim // 2, [1, 2, 4]),
            
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.projection(x)


class ParallelLinear(nn.Module):
    """并行线性层（模拟Inception结构）"""
    
    def __init__(self, input_dim: int, output_dim: int, dilation_rates: List[int]):
        super().__init__()
        self.branches = nn.ModuleList()
        
        for dilation in dilation_rates:
            branch = nn.Sequential(
                nn.Linear(input_dim, output_dim // len(dilation_rates)),
                nn.BatchNorm1d(output_dim // len(dilation_rates)),
                nn.ELU()
            )
            self.branches.append(branch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        branch_outputs = [branch(x) for branch in self.branches]
        return torch.cat(branch_outputs, dim=1)


class DomainAlignmentModule(nn.Module):
    """域对齐模块"""
    
    def __init__(self, num_domains: int):
        super().__init__()
        self.num_domains = num_domains
        
    def forward(self, features: List[torch.Tensor], domain_labels: torch.Tensor) -> torch.Tensor:
        """
        计算域对齐损失
        
        Args:
            features: 各域的特征列表
            domain_labels: 域标签
            
        Returns:
            torch.Tensor: 域对齐损失
        """
        if len(features) != self.num_domains:
            raise ValueError(f"Expected {self.num_domains} domains, got {len(features)}")
        
        total_loss = 0
        num_pairs = 0
        
        # 计算所有域对之间的对齐损失
        for i in range(self.num_domains):
            for j in range(i + 1, self.num_domains):
                loss = self.coral_loss(features[i], features[j])
                total_loss += loss
                num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else total_loss
    
    def coral_loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """CORAL域对齐损失"""
        source_mean = source.mean(0, keepdim=True)
        target_mean = target.mean(0, keepdim=True)
        mean_loss = F.mse_loss(source_mean, target_mean)
        
        # 协方差对齐
        source_cov = self._compute_covariance(source)
        target_cov = self._compute_covariance(target)
        cov_loss = F.mse_loss(source_cov, target_cov)
        
        return mean_loss + cov_loss
    
    def _compute_covariance(self, x: torch.Tensor) -> torch.Tensor:
        """计算协方差矩阵"""
        x_centered = x - x.mean(0, keepdim=True)
        return torch.mm(x_centered.t(), x_centered) / (x.size(0) - 1)


class MultiViewContrastiveLearner(nn.Module):
    """多视图对比学习模块"""
    
    def __init__(self, temperature: float = 0.07, projection_dim: int = 32):
        super().__init__()
        self.temperature = temperature
        self.projection_dim = projection_dim
    
    def forward(self, projections: Dict[str, torch.Tensor], 
                labels: torch.Tensor, domain_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算多视图对比学习损失
        
        Args:
            projections: 各视图的投影字典
            labels: 样本标签
            domain_labels: 域标签（用于跨域对比学习）
            
        Returns:
            Dict[str, torch.Tensor]: 各视图的对比损失
        """
        losses = {}
        
        # 计算每个视图的对比损失
        for view_name, projection in projections.items():
            # 创建正样本对（简单数据增强）
            projection_aug = self._simple_augmentation(projection)
            
            # 组织特征对
            features = torch.stack([projection, projection_aug], dim=1)
            
            # 计算对比损失
            if domain_labels is not None:
                # 跨域对比学习
                loss = self._cross_domain_contrastive_loss(features, labels, domain_labels)
            else:
                # 域内对比学习
                loss = self._intra_domain_contrastive_loss(features, labels)
            
            losses[view_name] = loss
        
        return losses
    
    def _simple_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """简单的特征增强"""
        lam = np.random.uniform(0.9, 1.0)
        shuffled = x[torch.randperm(x.size(0))]
        return lam * x + (1 - lam) * shuffled
    
    def _intra_domain_contrastive_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """域内对比学习损失"""
        batch_size = features.shape[0]
        num_views = features.shape[1]
        
        # 合并视图
        features = features.view(batch_size * num_views, -1)
        labels = labels.repeat(num_views)
        
        # 计算相似度矩阵
        similarity = torch.mm(features, features.t()) / self.temperature
        
        # 创建正样本掩码
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # 移除自相似性
        self_mask = torch.eye(batch_size * num_views, device=features.device)
        mask = mask * (1 - self_mask)
        
        # 计算对比损失
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
        
        # 只对正样本对计算
        mean_log_prob = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob.mean()
        
        return loss
    
    def _cross_domain_contrastive_loss(self, features: torch.Tensor, labels: torch.Tensor, 
                                       domain_labels: torch.Tensor) -> torch.Tensor:
        """跨域对比学习损失"""
        batch_size = features.shape[0]
        num_views = features.shape[1]
        
        # 合并视图
        features = features.view(batch_size * num_views, -1)
        labels = labels.repeat(num_views)
        domain_labels = domain_labels.repeat(num_views)
        
        # 计算相似度矩阵
        similarity = torch.mm(features, features.t()) / self.temperature
        
        # 创建正样本掩码（同类别但不同域）
        same_class = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        different_domain = (domain_labels.unsqueeze(0) != domain_labels.unsqueeze(1)).float()
        mask = same_class * different_domain
        
        # 移除自相似性
        self_mask = torch.eye(batch_size * num_views, device=features.device)
        mask = mask * (1 - self_mask)
        
        # 计算对比损失
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
        
        # 只对正样本对计算
        mean_log_prob = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob.mean()
        
        return loss


class MVCLDGTrainer:
    """MVCLDG训练器"""
    
    def __init__(self, model: nn.Module, device: torch.device, config: Dict):
        self.model = model
        self.device = device
        self.config = config
        
        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
    
    def train_step(self, data_loader: DataLoader, domain_labels: torch.Tensor) -> Dict[str, float]:
        """训练步骤"""
        self.model.train()
        total_losses = {
            'total': 0,
            'ce_raw': 0, 'ce_ht': 0, 'ce_fusion': 0,
            'align_raw': 0, 'align_ht': 0, 'align_fusion': 0,
            'contrast_raw': 0, 'contrast_ht': 0, 'contrast_fusion': 0
        }
        
        for batch_idx, batch in enumerate(data_loader):
            self.optimizer.zero_grad()
            
            # 准备数据
            inputs, labels = self._prepare_batch(batch)
            
            # 前向传播
            outputs = self.model(inputs)
            
            # 计算损失
            losses = self._compute_losses(outputs, labels, domain_labels)
            
            # 总损失
            total_loss = (losses['ce_total'] + 
                         self.config['tradeoff_align'] * losses['align_total'] +
                         self.config['tradeoff_contrast'] * losses['contrast_total'])
            
            # 反向传播
            total_loss.backward()
            self.optimizer.step()
            
            # 记录损失
            for key in total_losses:
                if key in losses:
                    total_losses[key] += losses[key].item()
            total_losses['total'] += total_loss.item()
        
        # 计算平均损失
        for key in total_losses:
            total_losses[key] /= len(data_loader)
        
        return total_losses
    
    def _prepare_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备批次数据"""
        inputs, labels = batch
        inputs = inputs.float().to(self.device)
        labels = labels.long().to(self.device)
        return inputs, labels
    
    def _compute_losses(self, outputs: Dict, labels: torch.Tensor, 
                        domain_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算所有损失"""
        batch_size = labels.size(0)
        num_domains = self.model.num_domains
        domain_size = batch_size // num_domains
        
        # 初始化损失字典
        losses = {}
        
        # 对每个视图计算损失
        for view in ['raw', 'ht', 'fusion']:
            view_output = outputs[view]
            
            # 分割域
            logits_split = torch.chunk(view_output['logits'], num_domains, dim=0)
            features_split = torch.chunk(view_output['features'], num_domains, dim=0)
            labels_split = torch.chunk(labels, num_domains, dim=0)
            
            # 交叉熵损失
            ce_loss = 0
            for domain_logits, domain_labels in zip(logits_split, labels_split):
                ce_loss += self.ce_loss(domain_logits, domain_labels)
            ce_loss /= num_domains
            
            # 域对齐损失
            align_loss = self.model.domain_aligner(features_split, domain_labels[:domain_size])
            
            # 对比学习损失
            contrast_losses = self.model.contrastive_learner(
                {view: view_output['projection']}, 
                labels, 
                domain_labels
            )
            contrast_loss = contrast_losses[view]
            
            # 记录损失
            losses[f'ce_{view}'] = ce_loss
            losses[f'align_{view}'] = align_loss
            losses[f'contrast_{view}'] = contrast_loss
        
        # 总损失
        losses['ce_total'] = sum(losses[f'ce_{view}'] for view in ['raw', 'ht', 'fusion']) / 3
        losses['align_total'] = sum(losses[f'align_{view}'] for view in ['raw', 'ht', 'fusion']) / 3
        losses['contrast_total'] = sum(losses[f'contrast_{view}'] for view in ['raw', 'ht', 'fusion']) / 3
        
        return losses
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = self._prepare_batch(batch)
                outputs = self.model(inputs)
                
                # 使用融合视图的分类结果
                fusion_logits = outputs['fusion']['logits']
                predictions = fusion_logits.argmax(dim=1)
                
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        accuracy = total_correct / total_samples
        return {'accuracy': accuracy}


class EEGDatasetWithHT(Dataset):
    """带希尔伯特变换的EEG数据集"""
    
    def __init__(self, data_path: str, dataset_id: int, include_ht: bool = True):
        self.data_path = data_path
        self.dataset_id = dataset_id
        self.include_ht = include_ht
        
        # 加载数据
        self.data, self.labels, self.domain_labels = self._load_data()
        
        # 如果需要，添加HT视图
        if self.include_ht:
            self.data = self._add_hilbert_transform(self.data)
        
        # 标准化
        self.data = self._standardize_data(self.data)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]
    
    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """加载数据（示例实现）"""
        # 在实际实现中，这里会加载真实的EEG数据
        n_samples = 1000
        n_channels = 64
        n_timepoints = 256
        
        # 生成示例数据
        data = np.random.randn(n_samples, n_channels, n_timepoints).astype(np.float32)
        labels = np.random.randint(0, 2, n_samples)
        
        # 模拟不同域的数据
        domain_labels = np.concatenate([
            np.zeros(n_samples // 4),
            np.ones(n_samples // 4),
            np.full(n_samples // 4, 2),
            np.full(n_samples // 4, 3)
        ])
        
        return torch.FloatTensor(data), torch.LongTensor(labels), torch.LongTensor(domain_labels)
    
    def _add_hilbert_transform(self, data: torch.Tensor) -> torch.Tensor:
        """添加希尔伯特变换（示例实现）"""
        # 在实际实现中，这里会计算真实的希尔伯特变换
        n_samples, n_channels, n_timepoints = data.shape
        
        # 模拟HT变换结果
        ht_data = np.random.randn(n_samples, n_channels, n_timepoints).astype(np.float32)
        ht_data = torch.FloatTensor(ht_data)
        
        # 拼接原始数据和HT数据
        combined = torch.cat([data, ht_data], dim=1)
        
        return combined
    
    def _standardize_data(self, data: torch.Tensor) -> torch.Tensor:
        """标准化数据"""
        mean = data.mean(dim=2, keepdim=True)
        std = data.std(dim=2, keepdim=True)
        return (data - mean) / (std + 1e-8)
    
    def get_domain_labels(self) -> torch.Tensor:
        """获取域标签"""
        return self.domain_labels


def main():
    """主函数：MVCLDG模型训练示例"""
    
    # 配置参数
    config = {
        'num_domains': 4,
        'num_classes': 2,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'tradeoff_align': 1.0,
        'tradeoff_contrast': 0.5,
        'temperature': 0.07,
        'projection_dim': 32,
        'num_epochs': 10
    }
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据准备
    input_shape = (1, 64, 256)  # (channels_in, EEG_channels, time_points)
    
    # 创建数据集
    dataset = EEGDatasetWithHT('path/to/data', dataset_id=1, include_ht=True)
    data_loader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    
    # 获取域标签
    domain_labels = dataset.get_domain_labels()
    
    # 创建模型
    model = MVCLDGModel(
        input_shape=input_shape,
        num_classes=config['num_classes'],
        num_domains=config['num_domains'],
        config=config
    )
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = MVCLDGTrainer(model, device, config)
    
    # 训练循环
    print("开始训练...")
    for epoch in range(config['num_epochs']):
        # 训练
        train_losses = trainer.train_step(data_loader, domain_labels)
        
        # 评估
        eval_metrics = trainer.evaluate(data_loader)
        
        # 输出进度
        print(f"Epoch {epoch+1}/{config['num_epochs']}:")
        print(f"  总损失: {train_losses['total']:.4f}")
        print(f"  融合视图准确率: {eval_metrics['accuracy']:.4f}")
        
        # 输出详细损失（每5个epoch）
        if (epoch + 1) % 5 == 0:
            print(f"  详细损失:")
            print(f"    交叉熵: RAW={train_losses['ce_raw']:.4f}, "
                  f"HT={train_losses['ce_ht']:.4f}, "
                  f"Fusion={train_losses['ce_fusion']:.4f}")
            print(f"    域对齐: RAW={train_losses['align_raw']:.4f}, "
                  f"HT={train_losses['align_ht']:.4f}, "
                  f"Fusion={train_losses['align_fusion']:.4f}")
            print(f"    对比学习: RAW={train_losses['contrast_raw']:.4f}, "
                  f"HT={train_losses['contrast_ht']:.4f}, "
                  f"Fusion={train_losses['contrast_fusion']:.4f}")
    
    print("训练完成！")


if __name__ == "__main__":
    main()
