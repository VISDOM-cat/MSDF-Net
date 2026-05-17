import os
import time
import numpy as np
import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import KFold
from tqdm import tqdm
from copy import deepcopy

def test_fusion_weights(test_loader, model1, model2, proto1, proto2, transform, 
                        alpha_list=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        save_dir="fusion_test_results"):

    all_results = []
    total_start_time = time.time()
    
    for alpha in alpha_list:
        eval_results = evaluate_model(
            model1=model1,
            proto1=proto1,
            model2=model2,
            proto2=proto2,
            test_loader=test_loader,
            transform=transform,
            weight1=alpha
        )
        all_results.append({
            'alpha': alpha,
            'eval_results': eval_results
        })
        
        metrics = eval_results['metrics']
        print(f"{metrics['accuracy']:.4f}")
        print(f"{metrics['macro_f1']:.4f}")
        print(f"{metrics['micro_f1']:.4f}")
    
    total_time = time.time() - total_start_time
    
    os.makedirs(save_dir, exist_ok=True)
    

def cross_validation(dataset_root, n_splits=5, feature_dim=256, seed=42,
                     fusion_alpha_list=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    set_seed(seed)
    
    transform = base_transform
    full_dataset_raw = TibetanDataset(root_dir=dataset_root, transform=transform)
    
    valid_samples = [
        (img, lbl, path)
        for img, lbl, path in full_dataset_raw
        if lbl != -1 and 0 <= lbl < full_dataset_raw.num_classes
    ]
    
    valid_images = torch.stack([s[0] for s in valid_samples])
    valid_labels = torch.tensor([s[1] for s in valid_samples])
    full_dataset = TensorDataset(valid_images, valid_labels)
    
    full_sample_list = [(s[2], s[1]) for s in valid_samples]  
    
    num_classes = full_dataset_raw.num_classes
    print(f"{num_classes}")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []
    total_time_stats = {
        'round1_train': [],
        'round2_train': [],
        'fusion_test': [],
        'total_train': []
    }
    
    main_save_dir = f"MODELS_CV_seed{seed}"
    os.makedirs(main_save_dir, exist_ok=True)
    
    dataset_info = {
        'dataset_root': dataset_root,
        'total_valid_samples': len(valid_samples),
        'num_classes': num_classes,
        'feature_dim': feature_dim,
        'n_splits': n_splits,
        'seed': seed,
        'fusion_alpha_list': fusion_alpha_list,
        'class_names': full_dataset_raw.classes,
        'class_to_idx': full_dataset_raw.class_to_idx
    }
    with open(os.path.join(main_save_dir, 'dataset_info.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=4)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(full_dataset)):
        print(f"\n{fold+1}/{n_splits}")

        train_dataset = Subset(full_dataset, train_idx)
        test_dataset = Subset(full_dataset, test_idx)
        
        train_sample_list = [full_sample_list[i] for i in train_idx]
        print(f"{len(train_sample_list)}")
        print(f"{len(test_idx)}")
        
        train_triplet_dataset = TibetanDataset(
            root_dir=dataset_root, 
            transform=transform,
            train_samples=train_sample_list 
        )

        train_loader = DataLoader(
            train_dataset, batch_size=8, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        
        fold_save_dir = os.path.join(main_save_dir, f"fold_{fold+1}")
        os.makedirs(fold_save_dir, exist_ok=True)
        
        model_coarse = SemanticPatchFeatureExtractor(
            feature_dim=feature_dim,
            num_vertical_patches=2,
            num_horizontal_patches=2,
        )
        
        total_params, trainable_params = count_model_parameters(model_coarse)
        
        model_coarse, round1_time = train_round(
            model=model_coarse,
            train_dataset=train_triplet_dataset,
            epochs=12,
            margin=0.3,
            lr=0.002,
            max_neg_per_batch=7,
            writer=None,
        )
        proto_coarse = build_prototype(model_coarse, train_loader, num_classes)
        
        torch.save(model_coarse.state_dict(), os.path.join(fold_save_dir, 'coarse_model.pth'))
        torch.save(proto_coarse, os.path.join(fold_save_dir, 'coarse_prototype.pth'))
        print(f"{os.path.join(fold_save_dir, 'coarse_model.pth')}")
        print(f"{os.path.join(fold_save_dir, 'coarse_prototype.pth')}")
    
        model_fine = SemanticPatchFeatureExtractor(
            feature_dim=feature_dim,
            num_horizontal_patches=3,
            num_vertical_patches=3,
        )
        coarse_weights = model_coarse.state_dict()
        fine_weights = model_fine.state_dict()
        for k in coarse_weights:
            if "patch_extractor" in k:
                fine_weights[k] = coarse_weights[k]
        model_fine.load_state_dict(fine_weights)
        
        total_params2, trainable_params2 = count_model_parameters(model_fine)
        
        model_fine, round2_time = train_round(
            model=model_fine,
            train_dataset=train_triplet_dataset,
            epochs=18,
            margin=0.5,
            lr=0.005,
            max_neg_per_batch=7,
            writer=None,
        )
        proto_fine = build_prototype(model_fine, train_loader, num_classes)
        
        torch.save(model_fine.state_dict(), os.path.join(fold_save_dir, 'fine_model.pth'))
        torch.save(proto_fine, os.path.join(fold_save_dir, 'fine_prototype.pth'))
        print(f"{os.path.join(fold_save_dir, 'fine_model.pth')}")
        print(f"{os.path.join(fold_save_dir, 'fine_prototype.pth')}")
        
        fusion_test_dir = os.path.join(fold_save_dir, 'fusion_test_results')
        fusion_results, fusion_summary = test_fusion_weights(
            test_loader=test_loader,
            model1=model_coarse,
            model2=model_fine,
            proto1=proto_coarse,
            proto2=proto_fine,
            transform=transform,
            alpha_list=fusion_alpha_list,
            save_dir=fusion_test_dir
        )
        
        best_alpha = fusion_summary['best_alpha']
        final_eval_results = evaluate_model(
            model1=model_coarse,
            proto1=proto_coarse,
            model2=model_fine,
            proto2=proto_fine,
            test_loader=test_loader,
            transform=transform,
            weight1=best_alpha
        )

if __name__ == "__main__":
    dataset_root = r"" 
    n_splits = 4 
    feature_dim = 256 
    random_seed = 42 
    fusion_alpha_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 

    cross_validation(
        dataset_root=dataset_root,
        n_splits=n_splits,
        feature_dim=feature_dim,
        seed=random_seed,
        fusion_alpha_list=fusion_alpha_list
    )