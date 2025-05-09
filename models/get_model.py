from .m3ddcnn import m3ddcnn
from .cnn3d import cnn3d
from .rssan import rssan
from .ablstm import ablstm
from .dffn import dffn
from .speformer import speformer
from .ssftt import ssftt
from .proposed import proposed
from .asf_group_former import asf_group, asf_group_dual_stream
from .cnn2d import cnn2d
from .ssrn import ssrn
from .posfree_vit import posfree_vit
from .rs_mamba_ss import vmamba, rsm_group, asf_rsm_group, semamba
from .msrnn import msrnn
from .msrt import msrt
from .ssmamba import ssmamba

def get_model(model_config, model_name, dataset_name, patch_size):
    # example: model_name='cnn3d', dataset_name='pu'
    if model_name == 'm3ddcnn':
        model = m3ddcnn(dataset_name, patch_size)

    elif model_name == 'cnn3d':
        model = cnn3d(dataset_name, patch_size)

    elif model_name == 'cnn2d':
        model = cnn2d(dataset_name, patch_size)
    
    elif model_name == 'ssrn':
        model = ssrn(dataset_name, patch_size)
    
    elif model_name == 'rssan':
        model = rssan(dataset_name, patch_size)
    
    elif model_name == 'ablstm':
        model = ablstm(model_config)

    elif model_name == 'dffn':
        model = dffn(dataset_name, patch_size)    
    
    elif model_name == 'speformer':
        model = speformer(dataset_name, patch_size) 

    elif model_name == 'proposed':
        model = proposed(model_config, dataset_name, patch_size)

    elif model_name == 'ssftt':
        model = ssftt(dataset_name, patch_size)

    elif model_name == 'asf_group':
        model = asf_group(model_config, dataset_name, patch_size)

    elif model_name == 'asf_group_dual_stream':
        model = asf_group_dual_stream(dataset_name, patch_size)
    
    elif model_name == 'posfree_vit':
        model = posfree_vit(dataset_name, patch_size)

    elif model_name == 'vmamba':
        model = vmamba(model_config, dataset_name, patch_size)
        
    elif model_name == 'rsm_group':
        model = rsm_group(model_config)
    
    elif model_name == 'asf_rsm_group':
        model = asf_rsm_group(model_config)

    elif model_name == 'msrnn':
        model = msrnn(model_config)
    
    elif model_name == 'msrt':
        model = msrt(model_config)
    
    elif model_name == "ssmamba":
        model = ssmamba(model_config)

    elif model_name == 'semamba':
        model = semamba(model_config)
    else:
        raise KeyError("{} model is not supported yet".format(model_name))

    return model


