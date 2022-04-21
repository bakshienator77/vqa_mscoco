from itertools import count
from multiprocessing.connection import answer_challenge
import os
from sympy import count_ops, im
import torch
import torchvision
import re

from PIL import Image
from torch.utils.data import Dataset
from external.vqa.vqa import VQA
from torch.nn.functional import one_hot


class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern,
                 transform=None, question_word_to_id_map=None, answer_to_id_map=None, question_word_list_length=5746, answer_list_length=5216,
                 pre_encoder=None, cache_location=None):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        self._vqa = VQA(annotation_file=annotation_json_file_path, question_file=question_json_file_path)
        self._image_dir = image_dir
        self._image_filename_pattern = image_filename_pattern
        self._transform = transform
        self._max_question_length = 26

        # Publicly accessible dataset parameters
        self.question_word_list_length = question_word_list_length + 1
        self.unknown_question_word_index = question_word_list_length
        self.answer_list_length = answer_list_length + 1
        self.unknown_answer_index = answer_list_length
        self._pre_encoder = pre_encoder
        self._cache_location = cache_location
        self.getQuesIds = self._vqa.getQuesIds()

        if self._cache_location is not None:
            try:
                os.makedirs(self._cache_location)
            except OSError:
                pass

        # Create the question map if necessary
        if question_word_to_id_map is None:
            print("Creating question_word_to_id_map")
            ############ 1.6 TODO
            sentences = [self._vqa.qqa[id]["question"] for id in self.getQuesIds]
            self.question_word_to_id_map = self._create_id_map(self._create_word_list(sentences), self.question_word_list_length)
            ############
            # raise NotImplementedError()
        else:
            self.question_word_to_id_map = question_word_to_id_map

        # Create the answer map if necessary
        if answer_to_id_map is None:
            ############ 1.7 TODO
            print("Creating answer_to_id_map")
            answers = []
            [answers.extend(self._vqa.qa[id]["answers"]) for id in self.getQuesIds]
            sentences = [ans["answer"] for ans in answers]
            self.answer_to_id_map = self._create_id_map(sentences, self.answer_list_length)
            ############
            # raise NotImplementedError()
        else:
            self.answer_to_id_map = answer_to_id_map
        print("done with init of vqadataset")


    def _create_word_list(self, sentences):
        """
        Turn a list of sentences into a list of processed words (no punctuation, lowercase, etc)
        Args:
            sentences: a list of str, sentences to be splitted into words
        Return:
            A list of str, words from the split, order remained.
        """

        ############ 1.4 TODO
        print("creating word list")
        out = []
        for sentence in sentences:
            # print("Sentence before the split and depunc: ", sentence)
            sentence = re.sub(r'[^\w\s]', '', sentence).lower().split(" ")
            # print("Sentence after split, depunc and lower: ", sentence)
            out.extend(sentence)
        ############
        print("done creating word list")

        return out


    def _create_id_map(self, word_list, max_list_length):
        """
        Find the most common str in a list, then create a map from str to id (its rank in the frequency)
        Args:
            word_list: a list of str, where the most frequent elements are picked out
            max_list_length: the number of strs picked
        Return:
            A map (dict) from str to id (rank)
        """
        ############ 1.5 TODO
        unique_list = list(set(word_list))
        counts = {}
        print("Number of unique words: ", len(unique_list))
        print("creating count list")
        for word in word_list:
            if word in counts.keys():           
               counts[word] += 1
            else:
               counts[word] = 1

        print("done creating count list")
        # print("counts list: ", counts)
        ref = [x for x, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)][:max_list_length]
        # print("descedning order list: " , ref[:10])
        # print("descedning order list: " , ref[-10:])
        map_dict = {}
        for i,x in enumerate(ref):
            map_dict[x] = i
        ############
        return map_dict


    def __len__(self):
        ############ 1.8 TODO
        return len(self.getQuesIds)
        ###########
        # raise NotImplementedError()

    def __getitem__(self, idx):
        """
        Load an item of the dataset
        Args:
            idx: index of the data item
        Return:
            A dict containing multiple torch tensors for image, question and answers.
        """

        ############ 1.9 TODO
        # figure out the idx-th item of dataset from the VQA API
        image_id = self._vqa.qqa[self.getQuesIds[idx]]["image_id"]
        ############

        if self._cache_location is not None and self._pre_encoder is not None:
            # the caching and loading logic here
            feat_path = os.path.join(self._cache_location, f'{image_id}.pt')
            try:
                image = torch.load(feat_path)
            except:
                image_path = os.path.join(
                    self._image_dir, self._image_filename_pattern.format(image_id))
                image = Image.open(image_path).convert('RGB')
                image = self._transform(image).unsqueeze(0)
                image = self._pre_encoder(image)[0]
                torch.save(image, feat_path)
        else:
            ############ 1.9 TODO
            # load the image from disk, apply self._transform (if not None)
            image_path = os.path.join(
                    self._image_dir, self._image_filename_pattern.format(image_id))
            image = Image.open(image_path).convert('RGB')
            if self._transform is not None:
                image = self._transform(image)
            else: 
                tf = torchvision.transforms.ToTensor()
                image = tf(image).unsqueeze(0)
                print(torch.max(image), torch.min(image))
            ############
            # raise NotImplementedError()

        ############ 1.9 TODO
        # load and encode the question and answers, convert to torch tensors
        question = self._vqa.qqa[self.getQuesIds[idx]]["question"]
        question = re.sub(r'[^\w\s]', '', question).lower().split(" ")

        # last_words = []
        # for key in self.question_word_to_id_map:
        #     print(key, self.question_word_to_id_map[key])
        #     if self.question_word_to_id_map[key] > self.question_word_list_length - 3:
        #         print(key, self.question_word_to_id_map[key])
        #         last_words.append(key)
        # print("Pay attention to the above words, they're value should NOT overlap with the UNK token: ", self.question_word_list_length, "\n it should also be only the next value after these not too far away")
        
        question = [self.question_word_to_id_map[word]  if word in self.question_word_to_id_map else self.unknown_question_word_index for word in question]
        # print(question)
        question_tensor = one_hot(torch.Tensor(question).long(), num_classes = self.question_word_list_length)
        if question_tensor.shape[0] < self._max_question_length:
            question_tensor = torch.cat([question_tensor, torch.zeros([self._max_question_length-question_tensor.shape[0], question_tensor.shape[1]])])
        question_tensor = question_tensor[:self._max_question_length, :]
        # print("Question Tensor size is: ", question_tensor.shape)

        answers = [self.answer_to_id_map[ans["answer"]] if ans["answer"] in self.answer_to_id_map else self.unknown_answer_index for ans in self._vqa.qa[self.getQuesIds[idx]]["answers"]]
        # print(answers, self._vqa.qa[self.getQuesIds[idx]]["answers"])
        answers_tensor = one_hot(torch.Tensor(answers).long(), num_classes = self.answer_list_length)
        # answers_tensor = one_hot(torch.argmax(torch.sum(answers_tensor, dim=0)).long(), num_classes = self.answer_list_length).unsqueeze(0)
        # print(torch.argmax(torch.sum(answers_tensor, dim=0)), "This should be 5k something")
        # print(torch.argmax(torch.Tensor([0, 1,1,1,0])), "This should be 5k something")

        # print("Answer Tensor size is: ", answers_tensor.shape)

        ############
        return {
            'idx': idx,
            'image': image,
            'question': question_tensor,
            'answers': answers_tensor
        }
