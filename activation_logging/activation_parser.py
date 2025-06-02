"""
activation_parser.py
Handles parsing of metadata from JSON files and looking up corresponding activations from LMDB.
"""
import json
import hashlib
from typing import Dict, Any, List, Optional
from pathlib import Path
from loguru import logger
import pandas as pd 
import json 
from loguru import logger
from sklearn.model_selection import train_test_split



from .activations_logger import ActivationsLogger

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
        activations = self.activation_logger.get_entry(row['prompt_hash'])
        return row, activations

            
    def close(self):
        """Close the LMDB connection."""
        self.logger.close() 