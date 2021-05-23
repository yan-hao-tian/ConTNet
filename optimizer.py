import torch

def build_optimizer(model, optim, lr, wd, momentum):

    def _no_bias_decay(model):
        has_decay = []
        no_decay = []
        skip_list = ['relative_position_bias_table', 'pe']

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue 
            if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list):
                no_decay.append(param)
            else:
                has_decay.append(param)
        
        assert len(list(model.parameters())) == len(has_decay) + len(no_decay), '{} vs. {}'.format(
            len(list(model.parameters())), len(has_decay) + len(no_decay))
        
        return [{'params': has_decay},
                {'params': no_decay, 'weight_decay': 0.}]
    
    parameters = _no_bias_decay(model)
    kwargs = dict(lr=lr, weight_decay=wd)
    if optim.lower() == 'SGD':
        kwargs['momentum'] = momentum
    
    optimizer = getattr(torch.optim, optim)(params=parameters, **kwargs)
   
    return optimizer
