import torch

from model.kg_completion_gnn import KGCompletionGNN


def save_model(model: KGCompletionGNN, path: str):
    save_obj = {
        'model_type': type(model),
        'instantiation_args': model.instantiation_args,
        'arg_signature': model.arg_signature,
        'state_dict': model.state_dict()
    }
    torch.save(save_obj, path)


def load_model(path: str, ignore_state_dict=False):
    save_obj = torch.load(path, map_location="cpu")
    inst_sig = save_obj['arg_signature']
    inst_val = save_obj['instantiation_args']
    print("Loading model with instantiation signature:")
    for i in range(len(inst_sig)):
        print('\t' + str(inst_sig[i]) + '=' + str(inst_val[i]))
    model: KGCompletionGNN = save_obj['model_type'](*inst_val)
    if not ignore_state_dict:
        print("Loading state dict")
        model.load_state_dict(save_obj['state_dict'])
    return model


def save_checkpoint(model: KGCompletionGNN, next_epoch, opt: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.MultiStepLR, path: str):
    save_obj = {
        'model_type': type(model),
        'instantiation_args': model.instantiation_args,
        'arg_signature': model.arg_signature,
        'state_dict': model.state_dict()
    }
    save_obj.update({'next_epoch': next_epoch, 'opt_state_dict': opt.state_dict(), 'scheduler_state_dict': scheduler.state_dict()})
    torch.save(save_obj, path)


def load_opt_checkpoint(path: str, opt: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.MultiStepLR):
    save_obj = torch.load(path, map_location="cpu")
    opt.load_state_dict(save_obj['opt_state_dict'])
    scheduler.load_state_dict(save_obj['scheduler_state_dict'])
    epoch: int = save_obj['next_epoch']
    return epoch

