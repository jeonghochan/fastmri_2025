import torch

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load("/home/junsoo/result/test_Varnet/unet/checkpoints/model_epoch_33.pt", map_location=device, weights_only=False)

    breakpoint()
