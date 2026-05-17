def get_triplet_all_neg(self, max_neg_per_batch=5, hard_neg_ratio=0.5):
    valid_train_classes = [cls for cls in self.train_class_samples if len(self.train_class_samples[cls]) >= 2]
    anchor_cls = random.choice(valid_train_classes)
    anchor_samples = self.train_class_samples[anchor_cls]
    
    anchor_path = random.choice(anchor_samples)
    positive_path = random.choice([p for p in anchor_samples if p != anchor_path])

    all_neg_cls = [c for c in self.train_class_samples if c != anchor_cls and len(self.train_class_samples[c]) >= 1]

    if len(all_neg_cls) > max_neg_per_batch:
        candidate_neg_cls = random.sample(all_neg_cls, 2 * max_neg_per_batch)
        anchor_feat = self._get_simple_feat(anchor_path)
        neg_distances = []
        for neg_cls in candidate_neg_cls:
            neg_path = random.choice(self.train_class_samples[neg_cls])
            neg_feat = self._get_simple_feat(neg_path)
            dist = np.linalg.norm(anchor_feat - neg_feat)
            neg_distances.append((neg_cls, dist))
        neg_distances.sort(key=lambda x: x[1])
        num_hard = int(max_neg_per_batch * hard_neg_ratio)
        selected_neg_cls = [cls for cls, _ in neg_distances[:num_hard]]
        selected_neg_cls += [cls for cls, _ in neg_distances[num_hard:max_neg_per_batch]]
    else:
        selected_neg_cls = all_neg_cls

    def load_img(path):
        img = Image.open(path).convert("L")
        img = preprocess_image(img)
        return self.transform(img) if self.transform else img

    anchor = load_img(anchor_path)
    positive = load_img(positive_path)
    negatives = [load_img(random.choice(self.train_class_samples[neg_cls])) for neg_cls in selected_neg_cls]

    return anchor, positive, negatives, selected_neg_cls