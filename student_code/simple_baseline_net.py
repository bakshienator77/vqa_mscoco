from sympy import im
import torch.nn as nn
import torch
from external.googlenet.googlenet import googlenet


class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self, question_word_list_length, answer_list_length, word_feature_size=1024): # 2.2 TODO: add arguments needed
        super().__init__()
	    ############ 2.2 TODO
        self.image_feature_extractor = googlenet(pretrained=True, transform_input=False)
        for param in self.image_feature_extractor.parameters():
            param.requires_grad = False
        self.word_feature_extractor = nn.Sequential(
            nn.Linear(question_word_list_length, word_feature_size),
            nn.ReLU(),
            nn.Linear(word_feature_size, word_feature_size),
        )
        self.classifier = nn.Linear(word_feature_size+1024, answer_list_length)
	    ############

    def forward(self, image, question_encoding):
	    ############ 2.2 TODO
        self.image_feature_extractor.eval()
        image_enc = self.image_feature_extractor(image)
        # print("image_enc shape should be (Bx1024): ", image_enc.shape)
        # print("I am assuming that the question_enc shape is (B x 26 x dict)", question_encoding.shape)
        question_encoding = torch.sum(question_encoding, dim=1)
        # print("The question_enc shape is now (B x dict)", question_encoding.shape)
        word_enc = self.word_feature_extractor(question_encoding)
        x = torch.concat([image_enc, word_enc], dim=1)
        # print("After concatenation shape should be: (Bx(1024+word_emb_dim))", x.shape)
        x = self.classifier(x)
        return x
	    ############
        # raise NotImplementedError()
