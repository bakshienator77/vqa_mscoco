from math import prod
from sympy import true
from student_code.simple_baseline_net import SimpleBaselineNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset
import torchvision
import torch
import torch.nn.functional as F


class SimpleBaselineExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Simple Baseline model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path,test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers, cache_location, lr, log_validation):

        ############ 2.3 TODO: set up transform
        print("Instantiation of ExperimentRunner")
        transform = torchvision.transforms.Compose([
                                torchvision.transforms.Resize((224,224)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),]
        )

        ############
        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{0:012d}.jpg",
                                   transform=transform,
                                   ############ 2.4 TODO: fill in the arguments
                                   question_word_to_id_map=None, #'change this argument',
                                   answer_to_id_map=None, #'change this argument',
                                   ############
                                   )
        print("train_dataset_loaded")

        
        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{0:012d}.jpg",
                                 transform=transform,
                                 ############ 2.4 TODO: fill in the arguments
                                 question_word_to_id_map = train_dataset.question_word_to_id_map, #'change this argument',
                                 answer_to_id_map = train_dataset.answer_to_id_map, #'change this argument',
                                 ############
                                 )
        print("test_dataset_loaded")

        model = SimpleBaselineNet(train_dataset.question_word_list_length, train_dataset.answer_list_length)

        super().__init__(train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers)

        ############ 2.5 TODO: set up optimizer
        self.optimizer = torch.optim.SGD([{"params": model.word_feature_extractor.parameters(), "lr": 0.8}, 
                                          {"params": model.classifier.parameters()}],
                                          lr = 0.01)
        ############
        self.model_type = "baseline"
        print("Simple baseline model instatiated")


    def _optimize(self, predicted_answers, true_answer_ids):
        ############ 2.7 TODO: compute the loss, run back propagation, take optimization step.
        self.optimizer.zero_grad() 
        # print("I expect true answer IDs to be of shape: (B x 10 x 5217): ",  true_answer_ids.shape)
        prod_scores = torch.sum(true_answer_ids, dim=1)/true_answer_ids.shape[1]
        # print("then I convert to prod scores of shape: (B x 5217): ",  prod_scores.shape)
        loss = F.cross_entropy(predicted_answers, prod_scores)
        loss.backward()
        torch.nn.utils.clip_grad_value_(self._model.parameters(), 20)
        self.optimizer.step()
        for layer in self._model.word_feature_extractor:
            if hasattr(layer, "weight"):
                layer.weight.data.clamp_(-1500, 1500)
        self._model.classifier.weight.data.clamp_(-20,20)
        ############
        # raise NotImplementedError()
        return loss
