print("Download the file and place it in: endoscopy-app/models/res_18_model_layers(4,fc).pth")
import torch
export_path = "/content/drive/MyDrive/res_18_model_layers(4,fc).pth"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "hidden_size": 347,
        "dropout": 0.5,
        "architecture": "resnet18",
        "positive_class": 1,
        "negative_class": 0,
        "img_size": 224,
    },
    export_path,
)

print(f"Saved model to {export_path}")
print("Download the file and place it in: endoscopy-app/models/resnet_model.pth")
