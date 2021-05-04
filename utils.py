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


def load_model(path: str):
    save_obj = torch.load(path)
    inst_sig = save_obj['arg_signature']
    inst_val = save_obj['instantiation_args']
    print("Loading model with instantiation signature:")
    for i in range(len(inst_sig)):
        print('\t' + str(inst_sig[i]) + '=' + str(inst_val[i]))
    model: KGCompletionGNN = save_obj['model_type'](*inst_val)
    model.load_state_dict(save_obj['state_dict'])
    return model


def save_checkpoint(model: KGCompletionGNN, epoch, opt: torch.optim.Optimizer, path: str):
    save_obj = {
        'model_type': type(model),
        'instantiation_args': model.instantiation_args,
        'arg_signature': model.arg_signature,
        'state_dict': model.state_dict()
    }
    save_obj.update({'epoch': epoch, 'opt_state_dict': opt.state_dict()})
    torch.save(save_obj, path)


def load_opt_checkpoint(path: str, opt: torch.optim.Optimizer):
    save_obj = torch.load(path)
    opt.load_state_dict(save_obj['opt_state_dict'])
    epoch: int = save_obj['epoch']
    return opt, epoch

