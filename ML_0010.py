import collections
import torch
import torch.utils.data as data
from torch import nn
from d2l import torch as d2l
import math
import jieba


"""生成数据集"""


# 训练数据处理

def train_data_iter(batch_size, num_steps, num_examples):
    # 读取数据文件
    def read_data_nmt():
        with open('train.txt', 'r', encoding='utf-8') as f:
            return f.read()
    # 格式调整

    def preprocess_nmt(text):
        def no_space(char, prev_char):
            return char in set(',.!?') and prev_char != ' '

        text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
        out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
               for i, char in enumerate(text)]
        return ''.join(out)
    # 分词并把中英句子分开，分别放到两个数组里面

    def tokenize_nmt(text, _num_examples=None):
        source, target = [], []
        for i, line in enumerate(text.split('\n')):
            if _num_examples and i > _num_examples:
                break
            parts = line.split('\t')
            if len(parts) == 2:
                source.append(parts[0].split(' '))
                target.append(jieba.lcut(parts[1]))
                # 引入jieba库分词
        return source, target
    # 定义一个Vocab类，将次元依据出现频率变成数字

    class Vocab:
        """文本词表"""

        def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
            if tokens is None:
                tokens = []
            if reserved_tokens is None:
                reserved_tokens = []
            # 按出现频率排序
            counter = count_corpus(tokens)
            self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                       reverse=True)
            # 未知词元的索引为0
            self.idx_to_token = ['<unk>'] + reserved_tokens
            self.token_to_idx = {token: idx
                                 for idx, token in enumerate(self.idx_to_token)}
            for token, freq in self._token_freqs:
                if freq < min_freq:
                    break
                if token not in self.token_to_idx:
                    self.idx_to_token.append(token)
                    self.token_to_idx[token] = len(self.idx_to_token) - 1

        def __len__(self):
            return len(self.idx_to_token)
        # 词元转索引

        def __getitem__(self, tokens):
            if not isinstance(tokens, (list, tuple)):
                return self.token_to_idx.get(tokens, self.unk)
            return [self.__getitem__(token) for token in tokens]
        # 索引转词元

        def to_tokens(self, indices):
            if not isinstance(indices, (list, tuple)):
                return self.idx_to_token[indices]
            return [self.idx_to_token[index] for index in indices]

        @property
        def unk(self):  # 未知词元的索引为0
            return 0

        @property
        def token_freqs(self):
            return self._token_freqs

    def count_corpus(tokens):
        # 统计次元概率
        # 这里的tokens是1维列表或2维列表
        if len(tokens) == 0 or isinstance(tokens[0], list):
            # 将词元列表展平成一个列表
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)

    def truncate_pad(line, _num_steps, padding_token):
        # 截断或填充文本序列
        if len(line) > _num_steps:
            return line[:_num_steps]  # 截断
        return line + [padding_token] * (_num_steps - len(line))  # 填充

    def build_array_nmt(lines, vocab, _num_steps):
        # 将机器翻译的文本序列转换成小批量
        lines = [vocab[line] for line in lines]
        lines = [line + [vocab['<eos>']] for line in lines]
        array = torch.tensor([truncate_pad(line, _num_steps, vocab['<pad>']) for line in lines])
        valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
        return array, valid_len

    def load_array(data_arrays, _batch_size, is_train=True):
        # 构造一个PyTorch数据迭代器。
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, _batch_size, shuffle=is_train)

    def load_data_nmt(_batch_size, _num_steps, _num_examples=16000):
        # 返回翻译数据集的迭代器和词表
        text = preprocess_nmt(read_data_nmt())
        source, target = tokenize_nmt(text, _num_examples)
        src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
        tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
        # <pad>表示填充 <bos>标志开始 <eos>标志结束
        src_array, src_valid_len = build_array_nmt(source, src_vocab, _num_steps)
        tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, _num_steps)
        data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
        data_iter_ = load_array(data_arrays, _batch_size)
        return data_iter_, src_vocab, tgt_vocab

    return load_data_nmt(batch_size, num_steps, num_examples)


# 测试数据处理
def test_data_load(num_examples):
    def read_test_data_nmt():
        with open('test.txt', 'r', encoding='utf-8') as f:
            return f.read()

    def preprocess_nmt(text_):
        def no_space(char, prev_char):
            return char in set(',.!?') and prev_char != ' '

        text_ = text_.replace('\u202f', ' ').replace('\xa0', ' ').lower()
        out = [' ' + char if i > 0 and no_space(char, text_[i - 1]) else char
               for i, char in enumerate(text_)]
        return ''.join(out)

    def test_tokenize_nmt(text_, _num_examples=None):
        source, target = [], []
        for i, line in enumerate(text_.split('\n')):
            if _num_examples and i > _num_examples:
                break
            parts = line.split('\t')
            if len(parts) == 2:
                source.append(parts[0])
                target.append(jieba.lcut(parts[1]))
        return source, target

    text = preprocess_nmt(read_test_data_nmt())
    return test_tokenize_nmt(text, num_examples)


"""定义模型"""

# 先写出基本框架，这里用的seq2seq，用gru作循环层


class Encoder(nn.Module):
    # 编码器
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, x, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    # 解码器
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, x, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    # 合并编码器解码器，构建模型
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_x, dec_x, *args):
        enc_outputs = self.encoder(enc_x, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_x, dec_state)

# 重写模型


class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, x, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        x = self.embedding(x)
        # 在循环神经网络模型中，第一个轴对应于时间步
        # 交换batch_size与num_steps
        x = x.permute(1, 0, 2)
        output, state = self.rnn(x)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state


def sequence_mask(X, valid_len, value=0):
    # 在序列中屏蔽不相关的项,也就是pad
    max_len = X.size(1)
    mask = torch.arange((max_len), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    # 生成bool值tensor
    X[~mask] = value
    # 把X对应的mask值为false的地方赋值0
    return X


class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        # 为了得到输出，做一个num_hiddens到vocab的全连接层
    def init_state(self, enc_outputs, *args):
        # return enc_outputs[1]
        return (enc_outputs[1], enc_outputs[1][-1])

    def forward(self, X, state):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)
        # new
        encode = state[1]
        # 编码器生成的上下文C，这样处理是为了避免预测时忽略C
        state = state[0]
        # new end
        x_and_context = torch.cat((X, context), 2)
        # 拼接C与中文编码，作为第二个循环神经网络的输入
        output, state = self.rnn(x_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        # return output, state
        return output, (state, encode)

# 下面是一个修改后的交叉熵损失，因为前面文本处理时对句子进行了填充，这里不希望对这些地方进行计算


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)
        # 调用父类的交叉熵损失函数
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        # 忽略<pad>位置的数据，乘以0来消除它
        return weighted_loss


"""训练"""


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):

    # 初始化模型参数
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    # 使用GPU训练
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            # 去掉每个句子的<eos>再加上<bos>
            Y_hat, _ = net(X, dec_input, X_valid_len)
            loss_ = loss(Y_hat, Y, Y_valid_len)
            loss_.sum().backward()      # 损失函数的标量进行“反向传播”
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss_.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
        print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} 'f'tokens/sec on {str(device)}')
    d2l.plt.show()


embed_size, num_hiddens, num_layers, dropout = 32, 512, 2, 0.1
batch_size, num_steps, num_examples = 64, 8, 16000
lr, num_epochs, device = 0.001, 100, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = train_data_iter(batch_size, num_steps, num_examples=num_examples)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


"""预测以及bleu评分"""

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq))


def bleu(pred_seq, label_seq, k):
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


source_, target_ = test_data_load(1000)
# 取出测试集数据

for english, chinese in zip(source_, target_):
    translation = predict_seq2seq(net, english, src_vocab, tgt_vocab, num_steps, device)
    print(f'{english} => {translation}, bleu {bleu(translation, chinese, k=2):.3f}')
