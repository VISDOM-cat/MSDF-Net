def build_prototype(model, train_loader, num_classes):
    device = next(model.parameters()).device 
    num_patches = model.total_patches 
    feature_dim = model.feature_dim  

    patch_features = [[[] for _ in range(num_patches)] for _ in range(num_classes)]

    model.eval() 
    with torch.no_grad():  
        for images, labels in tqdm(train_loader, desc="build_prototype"):
            images = images.to(device)
            batch_size = images.shape[0]
            patch_feats = model(images)  

            for i in range(batch_size):
                cls = labels[i].item()  
                if cls < 0 or cls >= num_classes:
                    continue
                for k in range(num_patches):
                    feat = patch_feats[i, k, :].cpu().numpy()
                    patch_features[cls][k].append(feat)

    prototype = torch.zeros(num_classes, num_patches, feature_dim, device=device)
    for cls in range(num_classes):
        for k in range(num_patches):
            if len(patch_features[cls][k]) > 0:
                prototype[cls, k] = torch.tensor(np.mean(patch_features[cls][k], axis=0), device=device)
            else:
                prototype[cls, k] = torch.randn(feature_dim, device=device) * 0.01
    return prototype