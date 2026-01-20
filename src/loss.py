import torch

# [b, c, h, w]
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    gram = torch.mm(features, features.t())
    return gram / (c * h * w) # [c, c]

def content_loss(gen, content):
    return torch.mean((gen - content) ** 2)

def style_loss(gen_feats, style_grams, weights):
    loss = 0
    for layer in weights:
        gen_gram = gram_matrix(gen_feats[layer])
        loss += weights[layer] * torch.mean(
            (gen_gram - style_grams[layer]) ** 2
        )
    return loss
