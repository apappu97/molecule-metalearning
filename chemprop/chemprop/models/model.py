import torch.nn as nn

from .mpn import MPN
from chemprop.args import TrainArgs
from chemprop.nn_utils import get_activation_function, initialize_weights

import learn2learn as l2l 
import pdb

class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs, featurizer: bool = False):
        """
        Initializes the MoleculeModel.

        :param args: Arguments.
        :param featurizer: Whether the model should act as a featurizer, i.e. outputting
                           learned features in the final layer before prediction.
        """
        super(MoleculeModel, self).__init__()

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.featurizer = featurizer

        # Check to see if we are meta learning. If we are, then only single output node is required
        if args.meta_learning:
            print("Meta learning, so output of FFN is 1")
            self.output_size = 1
        else:
            self.output_size = args.num_tasks

        if self.multiclass:
            self.output_size *= args.multiclass_num_classes

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        self.create_encoder(args)
        self.create_ffn(args)

        initialize_weights(self, args.kaiming)

    def create_encoder(self, args: TrainArgs):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args: TrainArgs):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_size

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, self.output_size),
            ])

        # Remove first dropout if dealing with features only
        if args.features_only:
            assert ffn[0] is dropout
            ffn = ffn[1:]
            assert ffn[0] is not dropout

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def featurize(self, *input):
        """
        Computes feature vectors of the input by leaving out the last layer.
        :param input: Input.
        :return: The feature vectors computed by the MoleculeModel.
        """
        return self.ffn[:-1](self.encoder(*input))

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Molecular input.
        :return: The output of the MoleculeModel. Either property predictions
                 or molecule features if self.featurizer is True.
        """
        if self.featurizer:
            return self.featurize(*input)

        output = self.ffn(self.encoder(*input))

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))  # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output

class ANILMoleculeModel(nn.Module):
    """ Wrapper class just used for setting up separation of featurizer and classifier in the molecule model """

    def __init__(self, molecule_model: MoleculeModel, fast_lr = None, classifier = None):
        """
        Inits the ANIL model by storing the gnn featurizer and classifier separately and MAML wrapping the head
        :param molecule_model: MoleculeModel
        :param fast_lr: adaptation learning rage
        """
        super(ANILMoleculeModel, self).__init__()

        if fast_lr and classifier:
            raise ValueError("There can only be one of fast lr and classifier, as the classifier, if passed in, is pre loaded with a fast lr")
        if not fast_lr and not classifier:
            raise ValueError("At least one of fast lr and classifier must be passed into the ANIL constructor")

        self.molecule_model = molecule_model
        self.gnn_featurizer = molecule_model.encoder

        # Just wrap the classifier with the MAML wrapper
        if classifier:
            self.classifier = classifier
        else:
            self.classifier = l2l.algorithms.MAML(molecule_model.ffn, lr=fast_lr)

    def clone(self):
        """
        Return a new ANILMoleculeModel object that creates a clone of the classifier, but points to the original molecule model, 
        so that the featurizer is not cloned.
        """
        return ANILMoleculeModel(molecule_model=self.molecule_model, classifier = self.classifier.clone())

    def adapt(self, loss, first_order=False):
        """
        Simply call the underlying MAML-wrapped head's adapt method for fast adaptation
        """
        self.classifier.adapt(loss, first_order=first_order)

    def parameters(self):
        """
        Wrapper which returns the parameters of the maml wrapped classifier and the gnn featurizer
        """
        all_parameters = list(self.classifier.parameters()) + list(self.gnn_featurizer.parameters())
        return all_parameters

    def forward(self, *input):
        """
        Run the ANIL-wrapped molecule model on input
        :param input: Molecular input.
        :return: The output of the MoleculeModel. Either property predictions
                 or molecule features if self.featurizer is True.
        """

        if self.molecule_model.featurizer:
            return self.molecule_model.featurize(*input)

        output = self.classifier(self.gnn_featurizer(*input))

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.molecule_model.classification and not self.training:
            output = self.molecule_model.sigmoid(output)
        if self.molecule_model.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))  # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output 
