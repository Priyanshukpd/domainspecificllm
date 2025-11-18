"""
Utility Functions
"""
from pathlib import Path
import re


def get_latest_checkpoint(output_dir):
    """Find the latest checkpoint automatically."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    
    checkpoints = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            match = re.match(r'checkpoint-(\d+)', item.name)
            if match:
                step_num = int(match.group(1))
                checkpoints.append((step_num, str(item)))
    
    if not checkpoints:
        return None
    
    checkpoints.sort(reverse=True)
    return checkpoints[0][1]


def print_training_summary(trainer):
    """Print final training summary."""
    print("\n" + "="*70)
    print("✅ Training complete!")
    print("="*70)
    print("\n📊 Training Summary:")
    print(f"   Total steps: {trainer.state.global_step}")
    
    # Safe formatting for best metric
    if trainer.state.best_metric is not None:
        print(f"   Best eval loss: {trainer.state.best_metric:.4f}")
    else:
        print(f"   Best eval loss: N/A (insufficient evaluation steps)")
    
    # Safe formatting for final loss
    if trainer.state.log_history:
        last_log = trainer.state.log_history[-1]
        final_loss = last_log.get('loss')
        if final_loss is not None:
            print(f"   Final training loss: {final_loss:.4f}")
        else:
            eval_loss = last_log.get('eval_loss')
            if eval_loss is not None:
                print(f"   Final eval loss: {eval_loss:.4f}")
            else:
                print(f"   Final loss: N/A")
    else:
        print(f"   Final loss: N/A (no logs available)")
    
    print("="*70)
