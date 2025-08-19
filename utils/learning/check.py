import torch

if __name__ == "__main__":

    checkpoint = torch.load("/home/junsoo/result/test_Varnet/final/swinunet/checkpoints/best_model.pt", map_location='cpu', weights_only=False)
    print(checkpoint['epoch'])
    print(checkpoint['best_val_loss'])
