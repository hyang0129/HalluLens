"""
activation_parser.py
Handles parsing of metadata from JSON files and looking up corresponding activations from LMDB.
"""
import json
import hashlib
from typing import Dict, Any, List, Optional, Literal
from pathlib import Path
from loguru import logger
import pandas as pd 
import json 
from loguru import logger
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import random

from .activations_logger import ActivationsLogger

class ActivationDataset(Dataset):
    """PyTorch Dataset for loading activation data."""
    
    def __init__(self, parser: 'ActivationParser', split: Literal['train', 'test'], relevant_layers: List[int] = None):
        """
        Initialize the dataset.
        
        Args:
            parser: ActivationParser instance containing the data
            split: Which split to use ('train' or 'test')
        """
        self.parser = parser
        self.split = split
        self.df = parser.df[parser.df['split'] == split].reset_index(drop=True)

        self.relevant_layers = relevant_layers if relevant_layers is not None else list(range(16,30))
        self.pad_length = 63 


    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single data point.
        
        Args:
            idx: Index of the data point
            
        Returns:
            Dictionary containing:
            - hashkey: The prompt hash
            - halu: Whether this is a hallucination
            - all_activations: All padded activations
            - layer1_activations: Activations from first randomly selected layer
            - layer2_activations: Activations from second randomly selected layer
            - layer1_idx: Index of first selected layer
            - layer2_idx: Index of second selected layer
            - input_length: Length of the input prompt
        """
        row, result, activations, input_length = self.parser.get_activations(idx)
        
        # Filter to relevant layers and pad if necessary
        padded_activations = []
        for layer_idx in self.relevant_layers:
            act = activations[layer_idx]
            seq_len = act.shape[1]
            
            if seq_len < self.pad_length:
                # Generate random noise with same shape as activations
                noise = torch.randn(act.shape[0], self.pad_length - seq_len, act.shape[2], 
                                  device=act.device, dtype=act.dtype)
                # Concatenate original activations with noise
                act = torch.cat([act, noise], dim=1)
            elif seq_len > self.pad_length:
                # Truncate if longer than pad_length
                act = act[:, :self.pad_length, :]
                
            padded_activations.append(act)
            
        # Randomly select two different layers
        layer1_idx, layer2_idx = random.sample(range(len(padded_activations)), 2)
        layer1_activations = padded_activations[layer1_idx]
        layer2_activations = padded_activations[layer2_idx]
            
        return {
            'hashkey': row['prompt_hash'],
            'halu': torch.tensor(row['halu'], dtype=torch.float32),
            'all_activations': padded_activations,
            'layer1_activations': layer1_activations,
            'layer2_activations': layer2_activations,
            'layer1_idx': layer1_idx,
            'layer2_idx': layer2_idx,
            'input_length': input_length
        }
    



class ActivationParser:
    def __init__(self, inference_json: str, eval_json: str, lmdb_path: str):
        """
        Initialize the ActivationParser.
        
        Args:
            json_path: Path to the JSON file containing metadata
            lmdb_path: Path to the LMDB file containing activations
        """
        self.inference_json = Path(inference_json)
        if not self.inference_json.exists():
            raise FileNotFoundError(f"JSON file not found: {inference_json}")
            
        self.eval_json = Path(eval_json)
        if not self.eval_json.exists():
            raise FileNotFoundError(f"JSON file not found: {eval_json}")

        self.lmdb_path = lmdb_path
        self.activation_logger = ActivationsLogger(lmdb_path=lmdb_path, read_only=True)
        
        # Load metadata from JSON
        self.df = self._load_metadata()


        
    def _load_metadata(self) -> Dict[str, Any]:


        gendf = pd.read_json(self.inference_json, lines=True)

        with open(self.eval_json, 'r') as f:
            data = json.loads(f.read())
            
        gendf['abstain'] = data['abstantion']
        gendf['halu'] = data['halu_test_res']



        gendf['prompt_hash'] = gendf['prompt'].apply(lambda x : 
                                                    hashlib.sha256(('user: ' + 
                                                                    x).encode("utf-8")).hexdigest())

        gendf = gendf[~gendf['prompt_hash'].duplicated(keep=False)]

        keys = self.activation_logger.list_entries()

        gendf = gendf[gendf['prompt_hash'].isin(keys)]
        gendf = gendf[~gendf['abstain']]

        # Apply train/test split
        train_df, test_df = train_test_split(gendf, test_size=0.2, 
                                           stratify=gendf['halu'], random_state=42)

        gendf['split'] = 'unassigned'
        gendf.loc[train_df.index, 'split'] = 'train'
        gendf.loc[test_df.index, 'split'] = 'test'

        logger.info(f"Found {len(gendf)} prompts with activations")
        logger.info(f"Found {len(gendf[gendf['halu']])} hallucinations")
        logger.info(f"Found {len(gendf[~gendf['halu']])} non-hallucinations")
        logger.info(f"Found {gendf['halu'].sum()/len(gendf)}% hallucinations")
        logger.info(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")

        # gen df contains these columns ['index', 'title', 'h_score_cat', 'pageid', 'revid', 'description',
        # 'categories', 'reference', 'prompt', 'answer', 'generation', 'abstain',
        # 'halu', 'prompt_hash', 'split'] 

        return gendf

            
    def _hash_prompt(self, prompt: str) -> str:
        """
        Hash a prompt string to match the format used in ActivationsLogger.
        
        Args:
            prompt: The prompt string to hash
            
        Returns:
            SHA-256 hash of the prompt
        """
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        
    def get_activations(self, idx) -> Optional[Dict[str, Any]]:

        row = self.df.iloc[idx]
        result = self.activation_logger.get_entry(row['prompt_hash'])
        input_length = result['input_length']
        
        # Trim activations to only include generated tokens
        activations = [act[:, input_length:, :] for act in result['all_layers_activations']]
    
        return row, result, activations, input_length

    def get_dataset(self, split: Literal['train', 'test']) -> ActivationDataset:
        """
        Get a PyTorch Dataset for the specified split.
        
        Args:
            split: Which split to use ('train' or 'test')
            
        Returns:
            ActivationDataset instance for the specified split
        """
        return ActivationDataset(self, split)

    def close(self):
        """Close the LMDB connection."""
        self.logger.close() 


