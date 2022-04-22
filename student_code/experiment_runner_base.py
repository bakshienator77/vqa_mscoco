from signal import valid_signals
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import os

from PIL import Image

def save_checkpoint(epoch, model, val_acc, model_type, best=False):
    if best:
        path = os.path.join("./checkpoints", 'best_model_{}.pt'.format(model_type))
    else:
        path = os.path.join("./checkpoints", 'model_epoch_{}_{}_{}.pt'.format(epoch, val_acc, model_type))
    torch.save(model.state_dict(), path)

class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=10, log_validation=False):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 10  # Steps
        self._test_freq = 250  # Steps

        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()

        if self._cuda:
            self._model = self._model.cuda()

        self._log_validation = log_validation

        self.writer = SummaryWriter()

        self.getQuesIds = self._val_dataset_loader.dataset._vqa.getQuesIds()

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self, epoch, full=False):
        ############ 2.8 TODO
        # Should return your validation accuracy
        self._model.eval()
        losses = []
        accuracy = []
        test_idx = 0
        # print("VALIDATION LOOP TIMING: ")
        for batch_id, batch_data in tqdm(enumerate(self._val_dataset_loader)):
            if self._cuda:
                predicted_answer = self._model(batch_data["image"].cuda(), batch_data["question"].cuda()) #None # TODO
                ground_truth_answer = batch_data["answers"].cuda() # TODO
            else:
                predicted_answer = self._model(batch_data["image"], batch_data["question"]) #None # TODO
                ground_truth_answer = batch_data["answers"] # TODO
            prod_scores = torch.sum(ground_truth_answer, dim=1)/ground_truth_answer.shape[1]
            loss = F.cross_entropy(predicted_answer, prod_scores)
            losses.append(loss.item())
            be_binary = (torch.argmax(prod_scores, dim=-1) == torch.argmax(predicted_answer, dim=-1)).reshape(-1)
            # print("VALIDATE FUNCTION, be binary shape should be batch_size: ", be_binary.shape)
            accuracy.extend(be_binary.cpu().detach().tolist())
            if batch_id == 5:
                invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                                    std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                                        std = [ 1., 1., 1. ]),
                               ])
                # print("IMAGE ID IS: ", )
                image_id = self._val_dataset_loader.dataset._vqa.qqa[self.getQuesIds[batch_data["idx"][test_idx]]]["image_id"]
                image_path = os.path.join(self._val_dataset_loader.dataset._image_dir, 
                            self._val_dataset_loader.dataset._image_filename_pattern.format(image_id))
                image = Image.open(image_path).convert('RGB')
                tf = transforms.ToTensor()
                image = tf(image)
                # image = 
                question_text = self._val_dataset_loader.dataset._vqa.qqa[self.getQuesIds[batch_data["idx"][test_idx]]]["question"]
                # self.writer.add_text("Question/test", 
                # self._val_dataset_loader.dataset._vqa.qqa[self.getQuesIds[batch_data["idx"][test_idx]]]["question"],
                # epoch)
                should_have_the_answers = self._val_dataset_loader.dataset._vqa.qa[self.getQuesIds[batch_data["idx"][test_idx]]]
                answers = should_have_the_answers["answers"]
                # print("answers are: ", answers)
                class_winner = torch.argmax(predicted_answer[test_idx])
                # idx_of_interest = torch.argmax(ground_truth_answer[test_idx, :, class_winner])
                pred_ans_text = list(self._val_dataset_loader.dataset.answer_to_id_map.keys())[list(self._val_dataset_loader.dataset.answer_to_id_map.values()).index(class_winner)] 
                # print("Index of interest is: ", idx_of_interest)
                # answer = answers[idx_of_interest]["answer"]
                # self.writer.add_text("Answer/predicted", answer, epoch)
                class_winner = torch.argmax(torch.sum(ground_truth_answer[test_idx], dim=0))
                idx_of_interest = torch.argmax(ground_truth_answer[test_idx, :, class_winner])
                answer = answers[idx_of_interest]["answer"]
                # self.writer.add_text("Answer/ground_truth", answer, epoch)
                self.writer.add_image("Image/Question: {}, GT: {}, Pred: {}".format(question_text, answer, pred_ans_text), image, epoch)

            if batch_id == 10 and not full:
                break

        if self._log_validation:
            ############ 2.9 TODO
            # you probably want to plot something here
            # pass
            self.writer.add_scalar('Loss/test', sum(losses)/len(losses), epoch)
            ############
        return float(sum(accuracy))/len(accuracy)

    def train(self):
        best = -1
        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)

            for batch_id, batch_data in tqdm(enumerate(self._train_dataset_loader)):
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                ############ 2.6 TODO
                # Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                if self._cuda:
                    predicted_answer = self._model(batch_data["image"].cuda(), batch_data["question"].cuda()) #None # TODO
                    ground_truth_answer = batch_data["answers"].cuda() # TODO
                else:
                    predicted_answer = self._model(batch_data["image"], batch_data["question"]) #None # TODO
                    ground_truth_answer = batch_data["answers"] # TODO

                ############

                # Optimize the model according to the predictions
                loss = self._optimize(predicted_answer, ground_truth_answer)

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss))
                    ############ 2.9 TODO
                    # you probably want to plot something here
                    self.writer.add_scalar("Loss/train", loss.item(), current_step)
                    ############

                if current_step % self._test_freq == 0:
                    self._model.eval()
                    val_accuracy = self.validate(current_step)
                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                    if val_accuracy > best:
                        print("Saving Best Model")
                        best = val_accuracy
                        save_checkpoint(current_step, self._model, val_accuracy, self.model_type, best=True)
                    # else: 
                    save_checkpoint(current_step, self._model, val_accuracy, self.model_type)
                    ############ 2.9 TODO
                    # you probably want to plot something here
                    self.writer.add_scalar('Accuracy/test', val_accuracy, current_step)

                    ############
