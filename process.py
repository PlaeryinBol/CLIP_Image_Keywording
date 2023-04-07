import argparse
import json
import math
import os
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


def compute_text_features(texts_batch):
    """Compute text features for batch of keyword templates.
        Args:
            texts_batch (list): list of templates.
        Returns:
            text_features (numpy.array) CLIP-embeddings array. 
    """
    text = clip.tokenize(texts_batch).to(opt.device)
    texts_preprocessed = torch.stack([t for t in text]).to(opt.device)

    with torch.no_grad():
        text_features = opt.model.encode_text(texts_preprocessed)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy()


def get_text_embeddings(batches, keywords_list):
    """Getting text embeddings for every vocabularu sample.
        Args:
            batches (int): batches count.
            keywords_list (list): vocabulary keywords list.
        Returns:
            features (numpy.array) common features array for all keywords.
            keyword_templates (pandas.DataFrame) dataframe with templates.
    """
    # load data from disk if it has already been saved
    if (os.path.exists(os.path.join(opt.features_dir, 'features.npy'))) \
            and (os.path.exists(os.path.join(opt.features_dir, 'keyword_templates.csv'))):
        features = np.load(os.path.join(opt.features_dir, 'features.npy'))
        keyword_templates = pd.read_csv(os.path.join(opt.features_dir, 'keyword_templates.csv'))
        print('Features loaded from disk')
        return features, keyword_templates

    # process each batch
    for i in tqdm(range(batches), desc="Text features saving"):
        batch_ids_path = opt.features_dir / 'csv' / f"{i:010d}.csv"
        batch_data_path = opt.features_dir / 'npy' / f"{i:010d}.npy"
        # only do the processing if the batch wasn't processed yet
        if not batch_data_path.exists():
            batch_files = keywords_list[i * opt.batch_size: (i + 1) * opt.batch_size]
            batch_files = [opt.template.replace('<keyword>', text) for text in batch_files]
            # compute the features and save to a numpy file
            batch_features = compute_text_features(batch_files)
            np.save(batch_data_path, batch_features)
            # save the templates to a CSV file
            keyword_templates = [keyword for keyword in batch_files]
            keyword_templates_data = pd.DataFrame(keyword_templates, columns=['text_id'])
            keyword_templates_data.to_csv(batch_ids_path, index=False)

    # load all numpy files
    features_list = [np.load(features_file) for features_file in sorted(opt.npy_dir.glob("*.npy"))]
    features = np.concatenate(features_list)
    np.save(os.path.join(opt.features_dir, 'features.npy'), features)

    # load all templates
    keyword_templates = pd.concat([pd.read_csv(ids_file) for ids_file in sorted(opt.csvs_dir.glob("*.csv"))])
    keyword_templates.to_csv(os.path.join(opt.features_dir, 'keyword_templates.csv'), index=False)
    print('Features save to disk')
    return features, keyword_templates


def image_keywording(text_features, images, keywords_list):
    """Create dataframe with the best vocabulary keywords for given images.
        Args:
            text_features (numpy.array) common features array for all keywords.
            images (list): image paths list.
            keywords_list (list): vocabulary keywords list.
        Returns:
            df (pandas.DataFrame) dataframe with best keywords.
    """
    text_features = torch.from_numpy(text_features).to(opt.device)
    data_dict = {'images': [os.path.basename(img) for img in images]}
    data_dict.update({i: None for i in range(opt.top_k)})
    df = pd.DataFrame(data=data_dict)
    # find best keywords for every image
    for k, img in enumerate(tqdm(images, desc="Find best image keywords")):
        image = opt.preprocess(Image.open(img)).unsqueeze(0).to(opt.device)

        with torch.no_grad():
            image_features = opt.model.encode_image(image).to(opt.device)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            scores, indices = torch.topk(image_features @ text_features.T, opt.top_k)

        best_keywords = list(zip(scores.flatten(), indices.flatten()))
        # save top_k keywords to dataframe
        for i in range(opt.top_k):
            k_idx = best_keywords[i][1]
            df[i][k] = keywords_list[k_idx]

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for text embeddings computing')
    parser.add_argument('--template', type=str, default='photo for an article on the topic of <keyword>.',
                        help='template for keyword substitution')
    parser.add_argument('--top_k', type=int, default=10, help='top k best keywords from vocabulary')
    opt = parser.parse_args()

    with open('./vocabulary.json') as f:
        keywords_list = json.load(f)

    opt.features_dir = Path('./text_features')
    opt.features_dir.mkdir(parents=True, exist_ok=True)
    opt.csvs_dir = opt.features_dir / 'csv'
    opt.csvs_dir.mkdir(parents=True, exist_ok=True)
    opt.npy_dir = opt.features_dir / 'npy'
    opt.npy_dir.mkdir(parents=True, exist_ok=True)
    images = list(Path('./images').glob("*.jpg"))
    batches = math.ceil(len(keywords_list) / opt.batch_size)

    opt.device = "cuda" if torch.cuda.is_available() else "cpu"
    opt.model, opt.preprocess = clip.load("ViT-B/32", device=opt.device)
    print('Model loaded')

    features, keyword_templates = get_text_embeddings(batches, keywords_list)
    keywords_df = image_keywording(features, images, keywords_list)
    keywords_df.to_csv('./keywords_df.tsv', sep='\t', index=False)
