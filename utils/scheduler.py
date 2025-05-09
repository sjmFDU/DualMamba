import torch.optim as optim


def load_scheduler(sch_config, model_name, model):
    optimizer, scheduler = None, None
    if not sch_config:
        if model_name == 'm3ddcnn':
            optimizer = optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.01)
            scheduler = None

        elif model_name == 'cnn3d' or model_name == 'cnn2d':
            # MaxEpoch in the paper is unknown, so 300 is set as MaxEpoch
            # and paper said: for each (Max Epoch / 3) iteration, the learning rate is divided by 10
            optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200], gamma=0.1)

        elif model_name == 'rssan':
            optimizer = optim.RMSprop(model.parameters(), lr=0.0003, weight_decay=0.0, momentum=0.0)
            scheduler = None

        elif model_name == 'ablstm':
            optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0005)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [150], gamma=0.1)

        elif model_name == 'dffn':
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200], gamma=0.1)

        elif model_name == 'speformer':
            optimizer = optim.Adam(model.parameters(), lr=5e-4)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90, 120, 150, 180, 210, 240, 270], gamma=0.9)

        elif model_name == 'ssftt':
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = None
            
        elif model_name == 'ssrn':
            optimizer = optim.Adam(model.parameters(), lr=0.002)
            scheduler = None

        elif model_name == 'proposed' or model_name == 'posfree_vit':
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.00005)
            scheduler = None

        elif model_name == 'asf_group_dual_stream':
            optimizer = optim.AdamW(model.parameters(), lr=3e-4)
            #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90, 120, 150, 180, 210, 240, 270], gamma=0.9)

        elif model_name == 'asf_group':
            #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
            optimizer = optim.AdamW(model.parameters(), lr=0.0002)
            #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90, 120, 150, 180, 210, 240, 270], gamma=0.9)
        
    else:
        opti_mode = sch_config['optimizer'].get('name', None)
        lr_mode = sch_config['lr_decay'].get('name', None)
        if opti_mode == 'Adam':
            optimizer = optim.Adam(model.parameters(), **sch_config['optimizer']['params'])
        elif opti_mode == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), **sch_config['optimizer']['params'])
        elif opti_mode == 'SGD':
            optimizer = optim.SGD(model.parameters(), **sch_config['optimizer']['params'])
        if lr_mode == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, **sch_config['lr_decay']['params'])
        elif lr_mode == 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **sch_config['lr_decay']['params'])
        elif lr_mode == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **sch_config['lr_decay']['params'])
        elif lr_mode == 'CosineAnnealingWarmRestarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **sch_config['lr_decay']['params'])
        elif lr_mode == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, **sch_config['lr_decay']['params'])
        else:
            scheduler = None
    return optimizer, scheduler


