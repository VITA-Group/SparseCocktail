
import copy 
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


__all__  = ['MultiMaskWrapper', 'check_sparsity','check_specific_sparsity']


def _apply_mask(module, input):
    if module.mask_mode[0]==0:
        module.weight = module.weight_dense_mask * module.weight_orig
    elif module.mask_mode[0] == 1:
        module.weight = module.weight_element_mask * module.weight_orig
    elif module.mask_mode[0]==2:
        module.weight = module.weight_channel_mask * module.weight_orig
    elif module.mask_mode[0]==3:
        module.weight = module.weight_nm_mask * module.weight_orig
    # module.weight = module.mask[module.mask_mode[0]] * module.weight_orig



def NM_prune(weight, N, M):

    length = weight.numel()
    group = int(length / M)

    weight_temp = weight.detach().abs().permute(0, 2, 3, 1).reshape(group, M)
    index = torch.argsort(weight_temp, dim=1)[:, :int(M - N)]

    w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
    w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.permute(0, 2, 3, 1).shape)
    w_b = w_b.permute(0, 3, 1, 2)

    return w_b


class MultiMaskWrapper(nn.Module):


    def __init__(self,model,refill_factor=0.8):
        super().__init__()
        self.mask_mode=[0]#0:dense, 1: element-wise, 2: channel-wise
        self.model=model
        self.refill_factor=refill_factor
        for name,module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if name=='conv1':
                    continue
                module.mask_mode=self.mask_mode
                module.register_parameter('weight_orig',nn.Parameter(module.weight.detach()))
                module.register_parameter('weight_dense_mask',nn.Parameter(torch.ones_like(module.weight),requires_grad=False))
                module.register_parameter('weight_element_mask',nn.Parameter(torch.ones_like(module.weight),requires_grad=False))
                module.register_parameter('weight_channel_mask',nn.Parameter(torch.ones_like(module.weight),requires_grad=False))
                module.register_parameter('weight_nm_mask',nn.Parameter(torch.ones_like(module.weight),requires_grad=False))
                # module.mask=[module.weight_dense_mask,module.weight_element_mask,module.weight_channel_mask, module.weight_nm_mask]
                del module.weight
                module.weight=module.weight_orig*module.weight_dense_mask
                module.register_forward_pre_hook(_apply_mask)


    def forward(self,x):
        return self.model(x)

    def mask_dict(self,sparse_mode):
        return_dict={}
        for name, param in self.model.named_parameters():
            if (sparse_mode=='element' and 'element_mask' in name) or \
                (sparse_mode=='channel' and 'channel_mask' in name) or \
                    (sparse_mode == 'nm' and 'nm_mask' in name):
                return_dict[name]=param
        return return_dict

    def param_dict(self):
        state_dict=self.model.state_dict()
        return_dict={}
        for key in state_dict:
            if 'mask' not in key:
                return_dict[key]=state_dict[key]
        return return_dict


    def change_mask_mode(self,mode):
        if mode=='dense':
            self.mask_mode[0]=0
        elif mode=='element':
            self.mask_mode[0]=1
        elif mode == 'channel':
            self.mask_mode[0]=2
        elif mode == 'nm':
            self.mask_mode[0] = 3
        else:
            raise NotImplementedError

    def inde_prune(self,prune_mode,prune_rate):
        element_sparsity,channel_sparsity,nm_sparsity=check_sparsity(self)
        if prune_mode=='element':
            remain_ratio=(1-element_sparsity)*(1-prune_rate)
            self._global_unstructured_prune(remain_ratio)
        elif prune_mode=='channel':
            #TODO: fix
            # remain_ratio=(1-channel_sparsity)*(1-prune_rate)
            # self._global_prune(remain_ratio)
            pass
        elif prune_mode=='nm':
            if prune_rate is not None:
                assert prune_rate in ["1:2", "2:4", "4:8"]
                self._layerwise_NM_prune(prune_rate)


    def UMG_prune(self,element_prune_rate,channel_prune_rate,nm_prune_rate=None):
        element_sparsity,channel_sparsity,nm_sparsity=check_sparsity(self)
        element_remain_ratio = (1 - element_sparsity) * (1 - element_prune_rate)
        self._global_unstructured_prune(element_remain_ratio)

        if nm_prune_rate is not None:
            assert nm_prune_rate in ["1:2", "2:4", "4:8"]
            self._layerwise_NM_prune(nm_prune_rate)

        channel_remain_ratio = (1 - channel_sparsity) * (1 - channel_prune_rate)
        self._layerwise_refill_prune(channel_remain_ratio)

    def get_masked_param_prefix(self):
        ret=[]
        for name,param in self.model.named_parameters():
            if name.endswith('_orig'):
                ret.append(name[:-5])
        return ret

    def _layerwise_refill_prune(self,remain_ratio):
        state_dic=self.model.state_dict()
        element_mask=self.mask_dict('element')
        channel_mask=self.mask_dict('channel')
        nm_mask=self.mask_dict('nm')
        prefix=self.get_masked_param_prefix()
        with torch.no_grad():
            for name in prefix:
                refill_sum=(self.refill_factor*element_mask[name+'_element_mask'] + (1-self.refill_factor)* nm_mask[name+'_nm_mask'])*torch.abs(state_dic[name+'_orig'])
                refill_sum=torch.sum(refill_sum,axis=[1,2,3])
                num_params_to_keep = int(len(refill_sum) * remain_ratio)
                threshold, _ = torch.topk(refill_sum,num_params_to_keep,sorted=True)
                acceptable_score=threshold[-1]
                channel_mask[name+'_channel_mask'].mul_(0).add_((refill_sum>acceptable_score).float().view(-1,1,1,1))


    def _layerwise_NM_prune(self, prune_rate):
        N,M=None,None
        if prune_rate == '1:2':
            N, M = 1, 2
        elif prune_rate == '2:4':
            N, M = 2, 4
        elif prune_rate == '4:8':
            N, M = 4, 8
        else:
            raise NotImplementedError
        state_dic=self.model.state_dict()
        nm_mask=self.mask_dict('nm')
        prefix=self.get_masked_param_prefix()
        for name in prefix:
            mask=NM_prune(state_dic[name+'_orig'],N,M)
            nm_mask[name+'_nm_mask'].mul_(0).add_(mask)

    def _global_unstructured_prune(self,remain_ratio):
        state_dic = self.model.state_dict()
        element_mask = self.mask_dict('element')
        prefix = self.get_masked_param_prefix()
        masked_weight_abs={}
        with torch.no_grad():
            for name in prefix:
                masked_weight_abs[name]=torch.abs(state_dic[name+'_orig'])*element_mask[name+'_element_mask']

            all_scores = torch.cat([torch.flatten(x) for x in masked_weight_abs.values()])
            num_params_to_keep = int(len(all_scores) * remain_ratio)

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for name in prefix:
                element_mask[name+'_element_mask'].mul_(0).add_((masked_weight_abs[name]>acceptable_score).float())







# Mask statistic function
def check_sparsity(masked_model,print_result=False):

    element_mask=torch.cat([x.view(-1) for x in masked_model.mask_dict('element').values()])

    channel_mask=torch.cat([x.view(-1) for x in masked_model.mask_dict('channel').values()])

    nm_mask=torch.cat([x.view(-1) for x in masked_model.mask_dict('nm').values()])

    element_sparsity=((element_mask==0).sum()/element_mask.nelement()).item()

    channel_sparsity=((channel_mask==0).sum()/channel_mask.nelement()).item()

    nm_sparsity=((nm_mask==0).sum()/nm_mask.nelement()).item()

    if print_result:
        print('*' * 20 + 'sparsity check' + '*' * 20)
        print('element sparsity:%.3f' % element_sparsity, '\t channel sparsity:%.3f' % channel_sparsity, '\t NM sparsity:%.3f' % nm_sparsity)
        print('*' * 20 + '**************' + '*' * 20)

    return element_sparsity,channel_sparsity,nm_sparsity


def check_specific_sparsity(mask_dict,print_result=False):

    mask=torch.cat([x.view(-1) for x in mask_dict.values()])

    sparsity=((mask==0).sum()/mask.nelement()).item()

    if print_result:
        print('sparsity:%.3f' % sparsity)

    return sparsity