import os
from re import L
import torch
import torch.nn as nn

from student_code.coattention_net import CoattentionNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset
import torchvision
import torch.nn.functional as F

class CoattentionNetExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Co-Attention model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path,test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers, cache_location, lr, log_validation):

        ############ 3.1 TODO: set up transform
        transform = torchvision.transforms.Compose([
                                torchvision.transforms.Resize((448,448)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),]
        )
        ############ 
        res18 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        image_encoder = nn.Sequential(*list(res18.children())[:-2])
        image_encoder.eval()
        for param in image_encoder.parameters():
            param.requires_grad = False

        question_word_list_length = 5746
        answer_list_length = 1000

        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{0:012d}.jpg",
                                   transform=transform,
                                   question_word_list_length=question_word_list_length,
                                   answer_list_length=answer_list_length,
                                   cache_location=os.path.join(cache_location, "tmp_train"),
                                   ############ 3.1 TODO: fill in the arguments
                                   question_word_to_id_map=None,
                                   answer_to_id_map=None,
                                   ############
                                   pre_encoder=image_encoder)
        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{0:012d}.jpg",
                                 transform=transform,
                                 question_word_list_length=question_word_list_length,
                                 answer_list_length=answer_list_length,
                                 cache_location=os.path.join(cache_location, "tmp_val"),
                                 ############ 3.1 TODO: fill in the arguments
                                 question_word_to_id_map=train_dataset.question_word_to_id_map ,#'change this argument',
                                 answer_to_id_map= train_dataset.answer_to_id_map, #'change this argument',
                                 ############
                                 pre_encoder=image_encoder)

        self._model = CoattentionNet(train_dataset.question_word_list_length, 512, train_dataset.answer_list_length)

        super().__init__(train_dataset, val_dataset, self._model, batch_size, num_epochs,
                         num_data_loader_workers=num_data_loader_workers, log_validation=log_validation)

        ############ 3.4 TODO: set up optimizer
        self.model_type = "coattention"
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, weight_decay=1e-8)
        ############ 

    def _optimize(self, predicted_answers, true_answer_ids):
        ############ 3.4 TODO: implement the optimization step
        self.optimizer.zero_grad() 
        # print("I expect true answer IDs to be of shape: (B x 10 x 5217): ",  true_answer_ids.shape)
        prod_scores = torch.sum(true_answer_ids, dim=1)/true_answer_ids.shape[1]
        # print("then I convert to prod scores of shape: (B x 5217): ",  prod_scores.shape)
        loss = F.cross_entropy(predicted_answers, prod_scores)
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self._model.parameters(), 20)
        self.optimizer.step()
        # for layer in self._model.word_feature_extractor:
            # if hasattr(layer, "weight"):
                # layer.weight.data.clamp_(-1500, 1500)
        # self._model.classifier.weight.data.clamp_(-20,20)
        ############
        # raise NotImplementedError()
        return loss
        
