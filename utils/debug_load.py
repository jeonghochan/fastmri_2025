import torch

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load("/mnt/c/Users/jhch0/Desktop/result/fastmri/second_cont/unet_2/checkpoints/best_model.pt", map_location=device, weights_only=False)
    print(checkpoint['val_loss'])
    print(checkpoint['best_val_loss'])
    print(checkpoint['epoch'])
    # print(checkpoint['optimizer'])
