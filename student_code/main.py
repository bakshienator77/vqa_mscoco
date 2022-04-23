import argparse

from cv2 import cuda_BufferPool
from student_code.simple_baseline_experiment_runner import SimpleBaselineExperimentRunner
from student_code.coattention_experiment_runner import CoattentionNetExperimentRunnerSentence, CoattentionNetExperimentRunner
import os
import torch

if __name__ == "__main__":
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='Load VQA.')
    parser.add_argument('--model', type=str, choices=['simple', 'coattention', 'coattentionsentence'], default='simple')
    parser.add_argument('--train_image_dir', type=str)
    parser.add_argument('--train_question_path', type=str)
    parser.add_argument('--train_annotation_path', type=str)
    parser.add_argument('--datadir', type=str, default="/home/nabakshi/Studies/VLR_spring22/vlr-hw4/data/")
    parser.add_argument('--test_image_dir', type=str)
    parser.add_argument('--test_question_path', type=str)
    parser.add_argument('--test_annotation_path', type=str)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_data_loader_workers', type=int, default=10)
    parser.add_argument('--cache_location', type=str, default="")
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--log_validation', action='store_true')
    parser.add_argument('--eval_mode', action='store_true')
    parser.add_argument('--sentence_only', action='store_true')
    parser.add_argument('--load_checkpoint', type=str, default='')

    args = parser.parse_args()

    if args.model == "simple":
        experiment_runner_class = SimpleBaselineExperimentRunner
    elif "coattention" in args.model:
        experiment_runner_class = CoattentionNetExperimentRunner
    else:
        raise ModuleNotFoundError()

    args.train_image_dir = os.path.join(args.datadir, "train2014")
    args.train_question_path = os.path.join(args.datadir, "OpenEnded_mscoco_train2014_questions.json")
    args.train_annotation_path = os.path.join(args.datadir, "mscoco_train2014_annotations.json")
    args.test_image_dir = os.path.join(args.datadir, "val2014")
    args.test_question_path = os.path.join(args.datadir, "OpenEnded_mscoco_val2014_questions.json")
    args.test_annotation_path = os.path.join(args.datadir, "mscoco_val2014_annotations.json")

    experiment_runner = experiment_runner_class(train_image_dir=args.train_image_dir,
                                                train_question_path=args.train_question_path,
                                                train_annotation_path=args.train_annotation_path,
                                                test_image_dir=args.test_image_dir,
                                                test_question_path=args.test_question_path,
                                                test_annotation_path=args.test_annotation_path,
                                                batch_size=args.batch_size,
                                                num_epochs=args.num_epochs,
                                                num_data_loader_workers=args.num_data_loader_workers,
                                                cache_location=args.cache_location,
                                                lr=args.lr,
                                                log_validation=args.log_validation,
                                                sentence_only=args.sentence_only)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.load_checkpoint != "":
        model_path = './checkpoints/{}.pt'.format(args.load_checkpoint)
        with open(model_path, 'rb') as f:
            state_dict = torch.load(f, map_location=device)
            experiment_runner._model.load_state_dict(state_dict)
        print("successfully loaded checkpoint from {}".format(model_path))

    if not args.eval_mode:
        experiment_runner.train()
    else:
        final_val_accuracy = experiment_runner.validate(999, full=True)
        print("Validation accuracy is: ", final_val_accuracy)
