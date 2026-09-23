import numpy as np, torch
from utilities.hyperbolic import dist_matrix_efficient
class Metric:
    def __init__(self,k,opt=None):
        self.k=k; self.c=float(getattr(opt,'hyp_c',.1)); self.requires=['hyp_features','target_labels']; self.name='h_recall@%d'%k
        self.device=getattr(opt,'device','cpu'); self.chunk_size=512
    def __call__(self, target_labels, hyp_features):
        # A full [N,N,D] Mobius broadcast requires ~18 GiB already for CUB.
        # Chunk queries and keep only a [chunk,N] distance matrix instead.
        x=torch.as_tensor(hyp_features,device=self.device)
        y=np.asarray(target_labels).reshape(-1); hits=[]
        with torch.no_grad():
            for start in range(0,len(x),self.chunk_size):
                end=min(start+self.chunk_size,len(x))
                d=dist_matrix_efficient(x[start:end],x,self.c)
                rows=torch.arange(end-start,device=x.device)
                d[rows,torch.arange(start,end,device=x.device)]=float('inf')
                idx=d.topk(self.k,largest=False).indices.cpu().numpy()
                hits.append(np.any(y[idx]==y[start:end,None],axis=1))
        return np.concatenate(hits).mean()
