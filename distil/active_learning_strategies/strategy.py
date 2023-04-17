import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import gc

class Strategy:
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}): #
        
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.model = net
        self.target_classes = nclasses
        self.args = args

        self.executor = args['executor']
        self.accelerator = args['accelerator']
        
        if 'batch_size' not in args:
            args['batch_size'] = 1
        
        if 'device' not in args:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = args['device']
            
        if 'loss' not in args:
            self.loss = F.cross_entropy
        else:
            self.loss = args['loss']

    def select(self, budget):
        pass

    def update_data(self, labeled_dataset, unlabeled_dataset): #
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        
    def update_queries(self, query_dataset):
        self.query_dataset= query_dataset

    def update_privates(self, private_dataset):
        self.private_dataset= private_dataset

    def update_model(self, clf):
        self.model = clf

    def predict(self, to_predict_dataset):

        dataloader = self.executor.get_dataloader_indices(to_predict_dataset, batch_size=self.args['batch_size'])

        predictions = torch.as_tensor(self.executor.predict(self.model, dataloader, self.accelerator, predictions_only=True))

        return predictions

    def predict_prob(self, to_predict_dataset):

        # Ensure model is on right device and is in eval. mode
        dataloader = self.executor.get_dataloader_indices(to_predict_dataset, batch_size=self.args['batch_size'])

        probs = self.executor.predict_prob(self.model, dataloader, self.accelerator)
        indices = [obj[1] for obj in probs]
        probs = torch.as_tensor([obj[0] for obj in probs])

        return indices,probs

    def predict_prob_dropout(self, to_predict_dataset, n_drop):

        # Ensure model is on right device and is in TRAIN mode.
        # Train mode is needed to activate randomness in dropout modules.
        self.model.train()
        self.model = self.model.to(self.device)
        
        # Create a tensor to hold probabilities
        probs = torch.zeros([len(to_predict_dataset), self.target_classes]).to(self.device)
        
        # Create a dataloader object to load the dataset
        to_predict_dataloader = self.executor.get_dataloader_indices(to_predict_dataset, batch_size = self.args['batch_size'], shuffle = False)

        with torch.no_grad():
            # Repeat n_drop number of times to obtain n_drop dropout samples per data instance
            for i in range(n_drop):
                
                evaluated_instances = 0
                for batch_idx, elements_to_predict in enumerate(to_predict_dataloader):
                
                    # Calculate softmax (probabilities) of predictions
                    elements_to_predict = elements_to_predict.to(self.device)
                    out = self.model(elements_to_predict)
                    pred = F.softmax(out, dim=1)
                
                    # Accumulate the calculated batch of probabilities into the tensor to return
                    start_slice = evaluated_instances
                    end_slice = start_slice + elements_to_predict.shape[0]
                    probs[start_slice:end_slice] += pred
                    evaluated_instances = end_slice

        # Divide through by n_drop to get average prob.
        probs /= n_drop

        return probs         

    def predict_prob_dropout_split(self, to_predict_dataset, n_drop):
        
        # Ensure model is on right device and is in TRAIN mode.
        # Train mode is needed to activate randomness in dropout modules.
        self.model.train()
        self.model = self.model.to(self.device)
        
        # Create a tensor to hold probabilities
        probs = torch.zeros([n_drop, len(to_predict_dataset), self.target_classes]).to(self.device)
        
        # Create a dataloader object to load the dataset
        to_predict_dataloader = self.executor.get_dataloader_indices(to_predict_dataset, batch_size = self.args['batch_size'], shuffle = False)

        with torch.no_grad():
            # Repeat n_drop number of times to obtain n_drop dropout samples per data instance
            for i in range(n_drop):
                
                evaluated_instances = 0
                for batch_idx, elements_to_predict in enumerate(to_predict_dataloader):
                
                    # Calculate softmax (probabilities) of predictions
                    elements_to_predict = elements_to_predict.to(self.device)
                    out = self.model(elements_to_predict)
                    pred = F.softmax(out, dim=1)
                
                    # Accumulate the calculated batch of probabilities into the tensor to return
                    start_slice = evaluated_instances
                    end_slice = start_slice + elements_to_predict.shape[0]
                    probs[i][start_slice:end_slice] = pred
                    evaluated_instances = end_slice

        return probs 

    def get_embedding(self, to_predict_dataset):

        # Ensure model is on right device and is in eval. mode
        dataloader = self.executor.get_dataloader_indices(to_predict_dataset, batch_size=self.args['batch_size'])

        embeddings = self.executor.get_embeddings(self.model, dataloader, self.accelerator)

        return embeddings

    # gradient embedding (assumes cross-entropy loss)
    #calculating hypothesised labels within
    def get_grad_embedding(self, dataset, predict_labels, grad_embedding_type="bias_linear"):

        embDim = self.executor.get_embed_dim()
        
        # Create the tensor to return depending on the grad_embedding_type, which can have bias only, 
        # linear only, or bias and linear
        if grad_embedding_type == "bias":
            grad_embedding = torch.zeros([len(dataset), self.target_classes])
        elif grad_embedding_type == "linear":
            grad_embedding = torch.zeros([len(dataset), embDim * self.target_classes])
        elif grad_embedding_type == "bias_linear":
            grad_embedding = torch.zeros([len(dataset), (embDim + 1) * self.target_classes])
        else:
            raise ValueError("Grad embedding type not supported: Pick one of 'bias', 'linear', or 'bias_linear'")
          
        # Create a dataloader object to load the dataset
        dataloader = self.executor.get_dataloader_indices(dataset, batch_size = self.args['batch_size'], shuffle = False)
        accelerator = self.accelerator

        dataloader = accelerator.prepare(dataloader)

        self.model.eval()
          
        evaluated_instances = 0
        ids_added = set()
        indices_list = []
        
        # If labels need to be predicted, then do so. Calculate output as normal.
        for batch_idx, batch in enumerate(dataloader):
            indices = batch.pop('index')
            start_slice = evaluated_instances

            out = self.model(**batch)
            l1 = torch.mean(out.hidden_states[-1], dim=1).squeeze()
            if predict_labels:
                targets = torch.argmax(out.logits, dim=-1)
                out, l1, targets, indices = accelerator.gather((out.logits, l1, targets, indices))
            else:
                out, l1, targets, indices = accelerator.gather((out.logits, l1, batch['labels'], indices))

            # Calculate loss as a sum, allowing for the calculation of the gradients using autograd wprt the outputs (bias gradients)
            loss = self.loss(out, targets, reduction="sum")
            l0_grads = torch.autograd.grad(loss, out)[0]
            del loss, out, targets

            l0_grads, l1 = l0_grads.tolist(), l1.tolist()

            l0_grads_new, new_indices, l1_new = [], [], []
            for i in range(len(l0_grads)):
                index = indices[i]
                if index not in ids_added:
                    l0_grads_new.append(l0_grads[i])
                    l1_new.append(l1[i])
                    ids_added.add(index)
                    new_indices.append(index)

            l0_grads_new, l1_new = torch.as_tensor(l0_grads_new), torch.as_tensor(l1_new)
            indices_list.extend(new_indices)

            del l1, l0_grads

            end_slice = start_slice + len(new_indices)

            # Calculate the linear layer gradients as well if needed
            if grad_embedding_type != "bias":
                l0_expand = torch.repeat_interleave(l0_grads_new, embDim, dim=1).cpu()
                l1_grads = l0_expand * l1_new.repeat(1, self.target_classes).cpu()

            # Populate embedding tensor according to the supplied argument.
            if grad_embedding_type == "bias":
                grad_embedding[start_slice:end_slice] = l0_grads_new
            elif grad_embedding_type == "linear":
                grad_embedding[start_slice:end_slice] = l1_grads
                del l1_grads
            else:
                grad_embedding[start_slice:end_slice] = torch.cat([l0_grads_new, l1_grads], dim=1)
                del l1_grads

            evaluated_instances = end_slice

            del l0_expand, l0_grads_new, new_indices
            gc.collect()

            # Empty the cache as the gradient embeddings could be very large
            torch.cuda.empty_cache()
        
        # Return final gradient embedding
        return grad_embedding, indices_list

    def feature_extraction(self, inp, layer_name):
        feature = {}
        model = self.model
        def get_features(name):
            def hook(model, inp, output):
                feature[name] = output.detach()
            return hook
        for name, layer in self.model._modules.items():
            if name == layer_name:
                layer.register_forward_hook(get_features(layer_name))
        output = self.model(inp)
        return torch.squeeze(feature[layer_name])

    def get_feature_embedding(self, dataset, unlabeled, layer_name='avgpool'):
        dataloader = DataLoader(dataset, batch_size = self.args['batch_size'], shuffle = False)
        features = []
        if(unlabeled):
            for batch_idx, inputs in enumerate(dataloader):
                inputs = inputs.to(self.device)
                batch_features = self.feature_extraction(inputs, layer_name)
                features.append(batch_features)
        else:
            for batch_idx, (inputs,_) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                batch_features = self.feature_extraction(inputs, layer_name)
                features.append(batch_features)
        return torch.vstack(features)