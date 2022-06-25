from binascii import a2b_hex
from numpy import imag
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuestionFeatureExtractor(nn.Module):
    """
    The hierarchical representation extractor as in (Lu et al, 2017) paper Sec. 3.2.
    """
    def __init__(self, word_inp_size, embedding_size, dropout=0.5):
        super().__init__()
        self.embedding_layer = nn.Linear(word_inp_size, embedding_size)

        self.phrase_unigram_layer = nn.Conv1d(embedding_size, embedding_size, 1, 1, 0)
        self.phrase_bigram_layer = nn.Conv1d(embedding_size, embedding_size, 2, 1, 1)
        self.phrase_trigramm_layer = nn.Conv1d(embedding_size, embedding_size, 3, 1, 1)

        self.dropout = nn.Dropout(p=dropout)

        self.lstm = nn.LSTM(embedding_size, embedding_size,
                            num_layers=1, dropout=dropout, batch_first=True)

    def forward(self, Q):
        """
        Inputs:
            Q: question_encoding in a shape of B x T x word_inp_size
        Outputs:
            qw: word-level feature in a shape of B x T x embedding_size
            qs: phrase-level feature in a shape of B x T x embedding_size
            qt: sentence-level feature in a shape of B x T x embedding_size
        """
        # word level
        Qw = torch.tanh(self.embedding_layer(Q))
        Qw = self.dropout(Qw) # B x T x E

        # phrase level
        Qw_bet = Qw.permute(0, 2, 1) # B x E x T
        Qp1 = self.phrase_unigram_layer(Qw_bet) # B x E x Tish
        # print("shape of Qp1 is: ", Qp1.shape)
        Qp2 = self.phrase_bigram_layer(Qw_bet)[:, :, 1:]
        Qp3 = self.phrase_trigramm_layer(Qw_bet)
        Qp = torch.stack([Qp1, Qp2, Qp3], dim=-1)
        Qp, _ = torch.max(Qp, dim=-1) # B x E
        # print("phrase level embedding size (B x E)?: ", Qp.shape)
        Qp = torch.tanh(Qp).permute(0, 2, 1)
        Qp = self.dropout(Qp) # B x T x E

        # sentence level
        Qs, (_, _) = self.lstm(Qp)

        return Qw, Qp, Qs


class AlternatingCoAttention(nn.Module):
    """
    The Alternating Co-Attention module as in (Lu et al, 2017) paper Sec. 3.3.
    """
    def __init__(self, d=512, k=512, dropout=0.5):
        super().__init__()
        self.d = d
        self.k = k

        self.Wx1 = nn.Linear(d, k)
        self.whx1 = nn.Linear(k, 1)

        self.Wx2 = nn.Linear(d, k)
        self.Wg2 = nn.Linear(d, k)
        self.whx2 = nn.Linear(k, 1)

        self.Wx3 = nn.Linear(d, k)
        self.Wg3 = nn.Linear(d, k)
        self.whx3 = nn.Linear(k, 1)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, V):
        """
        Inputs:
            Q: question feature in a shape of BxTxd
            V: image feature in a shape of BxNxd
        Outputs:
            shat: attended question feature in a shape of Bxk
            vhat: attended image feature in a shape of Bxk
        """
        B = Q.shape[0]
        # print("Q shape: ", Q.shape, "V shape: ", V.shape)
        # 1st step
        H = torch.tanh(self.Wx1(Q))
        H = self.dropout(H)
        ax = F.softmax(self.whx1(H), dim=1)
        shat = torch.sum(Q * ax, dim=1, keepdim=True)

        # 2nd step
        H = torch.tanh(self.Wx2(V) + self.Wg2(shat))
        H = self.dropout(H)
        ax = F.softmax(self.whx2(H), dim=1)
        vhat = torch.sum(V * ax, dim=1, keepdim=True)

        # 3rd step
        H = torch.tanh(self.Wx3(Q) + self.Wg3(vhat))
        H = self.dropout(H)
        ax = F.softmax(self.whx3(H), dim=1)
        shat2 = torch.sum(Q * ax, dim=1, keepdim=True)

        return shat2.squeeze(), vhat.squeeze()

class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, word_inp_size, embeddind_size, answers_list_length):
        super().__init__()
        ############ 3.3 TODO
<<<<<<< HEAD
        self.ques_feat_layer = QuestionFeatureExtractor(word_inp_size, embeddind_size)

        self.word_attention_layer = AlternatingCoAttention(d=embeddind_size, k=embeddind_size)
        # self.word_attention_layer = AlternatingCoAttention(d=embeddind_size)
        # self.word_attention_layer = AlternatingCoAttention(d=embeddind_size)

        self.Ww = nn.Sequential(
            nn.Linear(embeddind_size, embeddind_size),
            nn.Tanh()
        )
        self.Wp = nn.Sequential(
            nn.Linear(2*embeddind_size, embeddind_size),
            nn.Tanh()
        )
        self.Ws = nn.Sequential(
            nn.Linear(2*embeddind_size, embeddind_size),
            nn.Tanh()
        )

        self.dropout = nn.Dropout(0.5) # please refer to the paper about when you should use dropout

        self.classifier = nn.Linear(embeddind_size, answers_list_length)
=======
        self.ques_feat_layer = None

        self.word_attention_layer = None
        self.phrase_attention = None
        self.question_attention = None

        self.Ww = None
        self.Wp = None
        self.Ws = None

        self.dropout = None # please refer to the paper about when you should use dropout

        self.classifier = None
>>>>>>> 392fef898293fd8844e295ed0c8b4d8ff8751e21
        ############ 

    def forward(self, image_feat, question_encoding):
        ############ 3.3 TODO
        # 1. extract hierarchical question
        Qw, Qp, Qs = self.ques_feat_layer(question_encoding)
    
        # 2. Perform attention between image feature and question feature in each hierarchical layer
        # pass
        image_feat = image_feat.reshape(image_feat.shape[0], image_feat.shape[1], -1).permute((0,2,1))
        qhatw, vhatw = self.word_attention_layer(Qw, image_feat)
        qhatp, vhatp = self.word_attention_layer(Qp, image_feat)
        qhats, vhats = self.word_attention_layer(Qs, image_feat)
        
        # 3. fuse the attended features
        # pass
        hw = self.dropout(self.Ww(qhatw + vhatw))
        # print("hw size should be (B x k (512))is: ", hw.shape)
        concat1 = torch.cat([qhatp+vhatp, hw], dim=1)
        # print("after concat input to Wp layer is size: ", concat1.shape)
        hp = self.dropout(self.Wp(concat1))
        # print("hp size should be (B x k (512))is: ", hw.shape)
        hs = self.dropout(self.Wp(torch.cat([qhats+vhats, hp], dim=1)))
        
        # 4. predict the final answer using the fused feature
        # pass
        p = self.classifier(hs)
        ############ 
        # raise NotImplementedError()
        return p


class CoattentionNetSentenceOnly(CoattentionNet):
    def __init__(self, word_inp_size, embedding_size, dropout=0.5):

        super().__init__(word_inp_size, embedding_size, dropout)
    
    def forward(self, image_feat, question_encoding):
        # 1. extract hierarchical question
        # print("YOU ARE ONLY USING SENTENCES")
        Qw, Qp, Qs = self.ques_feat_layer(question_encoding)
    
        # 2. Perform attention between image feature and question feature in each hierarchical layer
        # pass
        image_feat = image_feat.reshape(image_feat.shape[0], image_feat.shape[1], -1).permute((0,2,1))
        # qhatw, vhatw = self.word_attention_layer(Qw, image_feat)
        # qhatp, vhatp = self.word_attention_layer(Qp, image_feat)
        qhats, vhats = self.word_attention_layer(Qs, image_feat)
        
        # 3. fuse the attended features
        # pass
        # hw = self.dropout(self.Ww(qhatw + vhatw))
        hs = self.dropout(self.Ww(qhats + vhats))
        # # print("hw size should be (B x k (512))is: ", hw.shape)
        # concat1 = torch.cat([qhatp+vhatp, hw], dim=1)
        # # print("after concat input to Wp layer is size: ", concat1.shape)
        # hp = self.dropout(self.Wp(concat1))
        # # print("hp size should be (B x k (512))is: ", hw.shape)
        # hs = self.dropout(self.Wp(torch.cat([qhats+vhats, hp], dim=1)))
        
        # 4. predict the final answer using the fused feature
        # pass
        p = self.classifier(hs)
        ############ 
        # raise NotImplementedError()
        return p
