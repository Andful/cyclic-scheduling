mapping = {
    '/conv1/Conv': {'core_allocation': (0,)}, #Layer 0
    '/maxpool/MaxPool': {'core_allocation': (0,)}, #Layer 2
    '/layer1/layer1.0/conv1/Conv': {'core_allocation': (0,)}, #Layer 3
    '/layer1/layer1.0/conv2/Conv': {'core_allocation': (0,)}, #Layer 5
    '/layer1/layer1.0/Add': {'core_allocation': (0,)}, #Layer 6
    '/layer1/layer1.1/conv1/Conv': {'core_allocation': (0,)}, #Layer 8 
    '/layer1/layer1.1/conv2/Conv': {'core_allocation': (1,)}, #Layer 10 
    '/layer1/layer1.1/Add': {'core_allocation': (1,)}, #Layer 11 
    
    '/layer2/layer2.0/conv1/Conv': {'core_allocation': (1,)}, #Layer 13
    '/layer2/layer2.0/conv2/Conv': {'core_allocation': (1,)}, #Layer 15 
    '/layer2/layer2.0/downsample/downsample.0/Conv': {'core_allocation': (1,)}, #Layer 16
    '/layer2/layer2.0/Add': {'core_allocation': (1,)}, #Layer 17 
    '/layer2/layer2.1/conv1/Conv': {'core_allocation': (1,)}, #Layer 19 
    '/layer2/layer2.1/conv2/Conv': {'core_allocation': (1,)}, #Layer 21 
    '/layer2/layer2.1/Add': {'core_allocation': (1,)}, #Layer 22
    
    '/layer3/layer3.0/conv1/Conv': {'core_allocation': (1,)}, #Layer 24 
    '/layer3/layer3.0/conv2/Conv': {'core_allocation': (1,)}, #Layer 26 
    '/layer3/layer3.0/downsample/downsample.0/Conv': {'core_allocation': (1,)}, #Layer 27 
    '/layer3/layer3.0/Add': {'core_allocation': (1,)}, #Layer 28 
    '/layer3/layer3.1/conv1/Conv': {'core_allocation': (1,)}, #Layer 30 
    '/layer3/layer3.1/conv2/Conv': {'core_allocation': (0,)}, #Layer 32 
    '/layer3/layer3.1/Add': {'core_allocation': (0,)}, #Layer 33
    
    '/layer4/layer4.0/conv1/Conv': {'core_allocation': (0, 1)}, #Layer 35 
    '/layer4/layer4.0/downsample/downsample.0/Conv': {'core_allocation': (0, 1)}, #Layer 38 
    '/layer4/layer4.0/conv2/Conv': {'core_allocation': (0, 1)}, #Layer 37 
    '/layer4/layer4.0/Add': {'core_allocation': (0, 1)}, #Layer 39 
    '/layer4/layer4.1/conv1/Conv': {'core_allocation': (0, 1)}, #Layer 41 
    '/layer4/layer4.1/conv2/Conv': {'core_allocation': (0, 1)}, #Layer 43 
    '/layer4/layer4.1/Add': {'core_allocation': (0, 1)}, #Layer 44 
    '/avgpool/GlobalAveragePool': {'core_allocation': (0, 1)}, #Layer 46 
    '/fc/Gemm': {'core_allocation': (0, 1)} #Layer 48
}
