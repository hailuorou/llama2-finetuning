import torch
from datasets import load_dataset
from awq.utils.eval_mmbench import MMBenchDataset,ImageEvalProcessor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from torchvision import transforms  
import torchvision.datasets as datasets
import math


def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    if data == "pileval":
        # dataset = load_dataset("json", data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst", split="train")
        dataset = load_dataset("json", data_files="./awq/utils/val.json", split="train")
        # dataset = load_dataset("json", data_files="/zeng_chao/code/llm-awq/awq/utils/output.json", split="train")
        print(dataset)
    else:
        raise NotImplementedError
    # dataset = MMBenchDataset("/zhang_miao/zeng_chao/codes/JiuTian/mmbench_test_20230712.tsv", vis_processors=ImageEvalProcessor())
    # dataloader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=False)
    
    dataset = dataset.shuffle(seed=42)
    
    samples = []
    n_run = 0
                
    for data in dataset:
        line = data["text"]
        line = line.strip()
        # print(line)
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    # print(n_samples)
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]

def get_vit_calib_dataset(n_samples=512, block_size=512):
    
    val_transform = build_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), crop_pct=0.875)
    
    val_dataset = datasets.ImageFolder("/zeng_chao/dataset/ImageNet/", val_transform)
    
    # dataset = val_dataset.shuffle(seed=42)
    
    samples = []
    n_run = 0
                
    for data in val_dataset:
        sample = data[0].unsqueeze(0).expand(1, -1, -1, -1).half()  # 在第0维度添加批次维度
        # if sample.numel() == 0:
        #     continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    cat_samples = torch.cat(samples, dim=0).to(torch.float16)
    # n_split = cat_samples.shape[1] // block_size
    # print(f" * Split into {n_split} blocks")
    print(cat_samples.shape)
    return cat_samples


def build_transform(input_size=224,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    crop_pct=0.875):

    t = []
    t.append(
        transforms.Resize((224, 224))  # to maintain same ratio w.r.t. 224 images
    )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


if __name__ == '__main__':
    get_vit_calib_dataset()
