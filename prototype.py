import torch
from torch.nn import Module
import math
import random

class Prototype(Module):
    def __init__(self, feat_dim=512, queue_size=16384, scale=30.0, margin=0.0, loss_type='softmax'):
        super(Prototype, self).__init__()
        self.queue_size = queue_size
        self.feat_dim = feat_dim
        self.scale = scale
        self.margin = margin
        self.loss_type = loss_type
        
        # initialize the prototype queue
        self.register_buffer('queue', torch.rand(feat_dim,queue_size).uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5))
        self.index = 0

    def add_margin(self, cos_theta, label, batch_size):

        cos_theta = cos_theta.clamp(-1, 1) 

        # additive cosine margin
        if self.loss_type == 'am_softmax':
            cos_theta_m = cos_theta[torch.arange(0, batch_size), label].view(-1, 1) - self.margin
            cos_theta.scatter_(1, label.data.view(-1, 1), cos_theta_m)
        # additive angurlar margin
        elif self.loss_type == 'arc_softmax':
            gt = cos_theta[torch.arange(0, batch_size), label].view(-1, 1)
            sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
            cos_theta_m = gt * math.cos(self.margin) - sin_theta * math.sin(self.margin) 
            cos_theta.scatter_(1, label.data.view(-1, 1), cos_theta_m)

        return cos_theta

    def compute_theta(self, p, g, label, batch_size):

        queue = self.queue.clone()
        queue[:,self.index:self.index+batch_size] = g.transpose(0,1)
        cos_theta = torch.mm(p, queue.detach())
        cos_theta = self.add_margin(cos_theta, label,batch_size)

        return cos_theta

    def update_queue(self, g, batch_size):

        with torch.no_grad():
            self.queue[:,self.index:self.index+batch_size] = g.transpose(0,1)
            self.index = (self.index + batch_size) % self.queue_size

    def forward(self, p1, g2, p2, g1, label):
        
        batch_size = p1.shape[0]
        g1 = g1.detach()
        g2 = g2.detach()

        output1 = self.compute_theta(p1, g2, label, batch_size)
        output2 = self.compute_theta(p2, g1, label, batch_size)
        output1 *= self.scale
        output2 *= self.scale
        
        if random.random() > 0.5:
            self.update_queue(g1, batch_size)
        else:
            self.update_queue(g2, batch_size) 
        
        return output1,output2