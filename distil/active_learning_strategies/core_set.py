import torch
import gc

from .strategy import Strategy

class CoreSet(Strategy):
    
    """
    Implementation of CoreSet :footcite:`sener2018active` Strategy. A diversity-based 
    approach using coreset selection. The embedding of each example is computed by the network’s 
    penultimate layer and the samples at each round are selected using a greedy furthest-first 
    traversal conditioned on all labeled examples.
    
    Parameters
    ----------
    labeled_dataset: torch.utils.data.Dataset
        The labeled training dataset
    unlabeled_dataset: torch.utils.data.Dataset
        The unlabeled pool dataset
    net: torch.nn.Module
        The deep model to use
    nclasses: int
        Number of unique values for the target
    args: dict
        Specify additional parameters
        
        - **batch_size**: The batch size used internally for torch.utils.data.DataLoader objects. (int, optional)
        - **device**: The device to be used for computation. PyTorch constructs are transferred to this device. Usually is one of 'cuda' or 'cpu'. (string, optional)
        - **loss**: The loss function to be used in computations. (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional)
    """
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        super(CoreSet, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
  
    def furthest_first(self, unlabeled_embeddings, labeled_embeddings, n):

        unlabeled_idx = [obj[1] for obj in unlabeled_embeddings]
        unlabeled_embeddings = torch.as_tensor([obj[0] for obj in unlabeled_embeddings])

        labeled_embeddings = torch.as_tensor([obj[0] for obj in labeled_embeddings])
        
        unlabeled_embeddings = unlabeled_embeddings.to(self.device)
        labeled_embeddings = labeled_embeddings.to(self.device)
        
        m = unlabeled_embeddings.shape[0]
        if m == 0:
            min_dist = torch.tile(torch.tensor(float("inf")), (m,))
        else:
            dist_ctr = torch.cdist(unlabeled_embeddings, labeled_embeddings, p=2)
            # per unlabeled embedding, keep the minimum distance between them and the labeled embeddings
            min_dist = torch.min(dist_ctr, dim=1)[0]

        
        idxs = []
        #self.accelerator.print(len(unlabeled_idx))


        while len(idxs) < n:
            # we search for the row corresponding to the unlabeled embeddings with the highest distance
            idx = torch.argmax(min_dist).item()

            if unlabeled_idx[idx] in idxs:
                old_idx = idx
                self.accelerator.print(f"error for index {unlabeled_idx[idx]}")
                min_dist[idx] = -100.0
                idx = torch.argmax(min_dist).item()
                if idx == old_idx:
                    self.accelerator.print(unlabeled_idx.count(unlabeled[idx]))
                    exit()

                
            idxs.append(unlabeled_idx[idx])

            # we calculate the new distance between the unlabeled embeddings and the
            # unlabeled embeddings with the highest distance and keep the minimum between those distance
            # and the ones previously calculated
            dist_new_ctr = torch.cdist(unlabeled_embeddings, unlabeled_embeddings[[idx],:])
            min_dist = torch.minimum(min_dist, dist_new_ctr[:,0])

        del unlabeled_embeddings, labeled_embeddings
        gc.collect()
        torch.cuda.empty_cache()
                
        return list(idxs)
  
    def select(self, budget):
        
        """
        Selects next set of points
        
        Parameters
        ----------
        budget: int
            Number of data points to select for labeling
            
        Returns
        ----------
        idxs: list
            List of selected data point indices
        """	
        
        self.model.eval()
        embedding_unlabeled = self.get_embedding(self.unlabeled_dataset)
        embedding_labeled = self.get_embedding(self.labeled_dataset)
        chosen = self.furthest_first(embedding_unlabeled, embedding_labeled, budget)

        return chosen
