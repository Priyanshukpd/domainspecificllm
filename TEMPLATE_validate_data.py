#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TEMPLATE: validate_data.py - JSONL Data Validation System
Quick Start for implementing data quality checks
"""

import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DataValidator:
    """Validate and analyze JSONL data quality"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.stats = {
            'total_samples': 0,
            'valid_samples': 0,
            'invalid_samples': 0,
            'format_distribution': defaultdict(int),
            'output_lengths': [],
            'input_lengths': [],
            'issues': []
        }
    
    def validate(self) -> Dict[str, Any]:
        """Run full validation"""
        logger.info(f"📊 Validating JSONL file: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    self._check_sample(sample, line_num)
                    self.stats['valid_samples'] += 1
                except json.JSONDecodeError as e:
                    self.stats['issues'].append(f"Line {line_num}: Invalid JSON - {str(e)[:50]}")
                    self.stats['invalid_samples'] += 1
                except Exception as e:
                    self.stats['issues'].append(f"Line {line_num}: {str(e)[:50]}")
                    self.stats['invalid_samples'] += 1
                
                self.stats['total_samples'] += 1
                if line_num % 100000 == 0:
                    logger.info(f"   ✓ Processed {line_num:,} samples")
        
        return self.stats
    
    def _check_sample(self, sample: Dict, line_num: int):
        """Check individual sample"""
        # Detect format
        if 'messages' in sample:
            self.stats['format_distribution']['chat'] += 1
            self._validate_chat_format(sample, line_num)
        elif 'instruction' in sample:
            self.stats['format_distribution']['instruction'] += 1
            self._validate_instruction_format(sample, line_num)
        else:
            self.stats['issues'].append(f"Line {line_num}: Unknown format (missing 'instruction' or 'messages')")
    
    def _validate_chat_format(self, sample: Dict, line_num: int):
        """Validate chat format"""
        messages = sample.get('messages', [])
        if not isinstance(messages, list):
            self.stats['issues'].append(f"Line {line_num}: 'messages' should be a list")
            return
        
        for msg in messages:
            if 'role' not in msg or 'content' not in msg:
                self.stats['issues'].append(f"Line {line_num}: Message missing 'role' or 'content'")
    
    def _validate_instruction_format(self, sample: Dict, line_num: int):
        """Validate instruction format"""
        instruction = sample.get('instruction', '')
        output = sample.get('output', '')
        
        if not instruction:
            self.stats['issues'].append(f"Line {line_num}: Empty 'instruction'")
        if not output:
            self.stats['issues'].append(f"Line {line_num}: Empty 'output'")
        
        self.stats['output_lengths'].append(len(output.split()))
        self.stats['input_lengths'].append(len(instruction.split()))
    
    def print_report(self):
        """Print validation report"""
        print("\n" + "="*60)
        print("📊 DATA VALIDATION REPORT")
        print("="*60)
        print(f"Total samples: {self.stats['total_samples']:,}")
        print(f"Valid samples: {self.stats['valid_samples']:,}")
        print(f"Invalid samples: {self.stats['invalid_samples']:,}")
        print(f"Valid rate: {self.stats['valid_samples']/max(self.stats['total_samples'], 1)*100:.1f}%")
        print(f"\nFormat distribution:")
        for fmt, count in self.stats['format_distribution'].items():
            print(f"  {fmt}: {count:,}")
        
        if self.stats['issues']:
            print(f"\n⚠️  Issues found: {len(self.stats['issues'])}")
            for issue in self.stats['issues'][:10]:  # Show first 10
                print(f"  • {issue}")
            if len(self.stats['issues']) > 10:
                print(f"  ... and {len(self.stats['issues']) - 10} more")
        
        print("="*60 + "\n")

if __name__ == "__main__":
    import sys
    
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data.jsonl"
    
    validator = DataValidator(data_path)
    stats = validator.validate()
    validator.print_report()
