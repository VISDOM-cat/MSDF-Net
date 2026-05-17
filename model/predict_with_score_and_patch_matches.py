def predict_with_score_and_patch_matches(test_image, model1, proto1,
                                         model2, proto2, transform,
                                         weight1=0.4):
    device1 = next(model1.parameters()).device
    device2 = next(model2.parameters()).device
    
    if isinstance(test_image, Image.Image):
        test_image = preprocess_image(test_image) 
        test_tensor = transform(test_image).unsqueeze(0) 
    else:
        test_tensor = test_image.unsqueeze(0) 

    model1.eval() 
    with torch.no_grad():  
        feats1 = model1(test_tensor.to(device1)).squeeze(0) 
        num_cls = proto1.shape[0]  
        dist1 = torch.zeros(num_cls, device=device1)  
        patch_dists1 = torch.zeros(num_cls, model1.total_patches, device=device1) 

        for cls in range(num_cls):
            total_dist = 0.0
            for k in range(model1.total_patches):
                dist = torch.norm(feats1[k] - proto1[cls, k], p=2)
                patch_dists1[cls, k] = dist
                total_dist += dist
            dist1[cls] = total_dist / model1.total_patches 

        pred_cls1 = torch.argmin(dist1).item()
        patch_matches1 = []
        for k in range(model1.total_patches):
            min_dist_k = torch.min(patch_dists1[:, k])  
            max_dist_k = torch.max(patch_dists1[:, k]) 
            pred_dist_k = patch_dists1[pred_cls1, k] 
            norm_dist = (pred_dist_k - min_dist_k) / (max_dist_k - min_dist_k + 1e-8)
            patch_matches1.append(100 * (1 - norm_dist).cpu().item())

    model2.eval()
    with torch.no_grad():
        feats2 = model2(test_tensor.to(device2)).squeeze(0)
        dist2 = torch.zeros(num_cls, device=device2)
        patch_dists2 = torch.zeros(num_cls, model2.total_patches, device=device2)
        
        for cls in range(num_cls):
            total_dist = 0.0
            for k in range(model2.total_patches):
                dist = torch.norm(feats2[k] - proto2[cls, k], p=2)
                patch_dists2[cls, k] = dist
                total_dist += dist
            dist2[cls] = total_dist / model2.total_patches
        
        pred_cls2 = torch.argmin(dist2).item()
        patch_matches2 = []
        for k in range(model2.total_patches):
            min_dist_k = torch.min(patch_dists2[:, k])
            max_dist_k = torch.max(patch_dists2[:, k])
            pred_dist_k = patch_dists2[pred_cls2, k]
            norm_dist = (pred_dist_k - min_dist_k) / (max_dist_k - min_dist_k + 1e-8)
            patch_matches2.append(100 * (1 - norm_dist).cpu().item())

    dist1_norm = dist1 / torch.max(dist1)  
    dist2_norm = dist2 / torch.max(dist2)  
    
    w1 = weight1  
    w2 = 1.0 - weight1  
    fused_dist = w1 * dist1_norm.cpu() + w2 * dist2_norm.cpu()  
    pred_label = torch.argmin(fused_dist).item() 

    min_dist = torch.min(fused_dist)
    max_dist = torch.max(fused_dist)
    match_score = 100 * (1 - (min_dist / (max_dist + 1e-8)))

    return pred_label, match_score.item(), patch_matches1, patch_matches2