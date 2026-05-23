import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch

from detr.main import build_ACT_model_and_optimizer
import IPython
e = IPython.embed


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')


    def __call__(self, qpos, image, actions=None, is_pad=None, device=None, tactile=None, tactile_next=None, epoch=0):
        env_state = None
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            a_hat, is_pad_hat, (mu, logvar), tac_hat = self.model(qpos, image, env_state, tactile, actions, is_pad, tactile_next, epoch=epoch)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight

            B, T, D = actions.shape

            if tac_hat is not None:
                is_pad_tac = is_pad[:, :tac_hat.shape[1]]
                all_l1_tac = F.l1_loss(tactile_next, tac_hat, reduction='none')
                l1_tac = (all_l1_tac * ~is_pad_tac.unsqueeze(-1)).mean()
                loss_dict['l1_tac'] = l1_tac
                loss_dict['loss'] = loss_dict['loss'] + loss_dict['l1_tac']

            return loss_dict
        else: # inference time
            # epoch>=75 disables teacher forcing so the model uses its own predicted tactile_next
            a_hat, _, (_, _), _ = self.model(qpos, image, env_state, tactile, epoch=999) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer
    
def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

