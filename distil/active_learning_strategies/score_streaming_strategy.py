from .strategy import Strategy
from torch.utils.data import Subset

def merge(list_1, list_2, key, merged_list_size_cap):
    
    list_1_index = 0
    list_2_index = 0
    
    merged_list = list()
    
    while list_1_index < len(list_1) and list_2_index < len(list_2) and len(merged_list) < merged_list_size_cap:
        
        list_1_elem = list_1[list_1_index]
        list_2_elem = list_2[list_2_index]
        
        if key(list_1_elem) >= key(list_2_elem):
            merged_list.append(list_1_elem)
            list_1_index += 1
        else:
            merged_list.append(list_2_elem)
            list_2_index += 1
            
    while list_1_index < len(list_1) and len(merged_list) < merged_list_size_cap:
        
        list_1_elem = list_1[list_1_index]
        merged_list.append(list_1_elem)
        list_1_index += 1
        
    while list_2_index < len(list_2) and len(merged_list) < merged_list_size_cap:
        
        list_2_elem = list_2[list_2_index]
        merged_list.append(list_2_elem)
        list_2_index += 1
            
    return merged_list

class ScoreStreamingStrategy(Strategy):
    
    """
    Provides a framework for AL strategies wherein each data point in the unlabeled set is 
    attributed a 'score' in a streaming manner. The largest score is then selected.
    
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
        - **stream_buffer_size**: The buffer size of the stream used in calculating scores (int, optional)
    """
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        super(ScoreStreamingStrategy, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        
        if 'stream_buffer_size' not in args:
            self.stream_buffer_size = 10000
        else:
            self.stream_buffer_size = args['stream_buffer_size']
        
    def acquire_scores(self, unlabeled_batch):
        pass
    
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

        scores, indices = self.acquire_scores(self.unlabeled_dataset)
        scores = [(x, i) for i, x in zip(indices, scores)]
        scores = sorted(scores, key=lambda x: x[0], reverse=True)
            
        return [i for (_,i) in scores[:budget]]
            
            