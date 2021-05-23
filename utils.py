import torch
import torch.distributed as dist
import os

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def resume_model(resume_path, model, optimizer, scheduler):
    print(f"=> loading checkpoint '{resume_path}'")
    checkpoint = torch.load(resume_path)
    start_epoch = checkpoint['epoch']
    best_acc1 = checkpoint['best_acc1']
    best_epoch = checkpoint['best_epoch']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    print(f"=> loaded checkpoint successfully '{resume_path}' (epoch {start_epoch})")

    return model, optimizer, scheduler, start_epoch, best_acc1, best_epoch

def save_model(save_path, model, optimizer, scheduler, best_acc1, epoch, is_best):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict(),
                  'best_acc1': best_acc1,
                  'epoch': epoch}

    os.makedirs(save_path, exist_ok=True)
    checkpoint_name = f'checkpoint_bestTop1.pth' if is_best else f'checkpoint_{epoch}.pth'
    save_path = os.path.join(save_path, checkpoint_name)
    torch.save(save_state, save_path)
    print(f'=> Saved checkpoint of epoch {epoch} to {save_path}')