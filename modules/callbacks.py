"""
Training Callbacks for Logging, Plotting, and Early Stopping
"""
from transformers import TrainerCallback
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import json
import torch


class StepLoggingCallback(TrainerCallback):
    """Prints training metrics every N steps in a clean format."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called whenever logging happens (controlled by logging_steps)."""
        if logs and state.global_step > 0:
            # Extract relevant metrics
            loss = logs.get('loss')
            grad_norm = logs.get('grad_norm')
            learning_rate = logs.get('learning_rate')
            epoch = logs.get('epoch')
            
            # Only print if we have the core metrics
            if loss is not None:
                output = f"Step {state.global_step:>5d} | "
                output += f"Loss: {loss:.4f} | "
                
                if grad_norm is not None:
                    output += f"Grad Norm: {grad_norm:.4f} | "
                
                if learning_rate is not None:
                    output += f"LR: {learning_rate:.2e} | "
                
                if epoch is not None:
                    output += f"Epoch: {epoch:.2f}"
                
                print(output)


class SafeEarlyStoppingCallback(TrainerCallback):
    """
    Early stopping that only activates after minimum training steps.
    Prevents stopping before the model has seen the full dataset at least once.
    """
    
    def __init__(self, min_steps=None, patience=3):
        """
        Args:
            min_steps: Minimum steps before early stopping can trigger
                      If None, will be calculated as steps_per_epoch (1 full epoch)
            patience: Number of evaluations without improvement before stopping
        """
        self.min_steps = min_steps
        self.patience = patience
        self.patience_counter = 0
        self.best_metric = None
        self.best_step = None
        self.early_stopping_triggered = False
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Calculate minimum steps if not provided and display protection info."""
        if self.min_steps is None:
            # Calculate steps per epoch from training dataset
            train_dataloader = kwargs.get('train_dataloader')
            if train_dataloader:
                steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
                self.min_steps = steps_per_epoch
            else:
                # Fallback: use eval_steps as minimum
                self.min_steps = args.eval_steps
        
        print(f"🛡️  Early Stopping Protection:")
        print(f"   Minimum training steps: {self.min_steps} (≥1 full epoch)")
        print(f"   Patience after minimum: {self.patience} evaluations")
        print(f"   Early stopping will be active after step {self.min_steps}")
        print()
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Check if early stopping should trigger."""
        if not metrics or 'eval_loss' not in metrics:
            return
        
        current_step = state.global_step
        current_metric = metrics['eval_loss']
        
        # SAFETY: Don't allow early stopping before minimum steps
        if current_step < self.min_steps:
            if self.best_metric is None or current_metric < self.best_metric:
                self.best_metric = current_metric
                self.best_step = current_step
            print(f"   🛡️  Early stopping protected (step {current_step}/{self.min_steps})")
            return
        
        # Standard early stopping logic (after minimum steps)
        if self.best_metric is None or current_metric < self.best_metric:
            self.best_metric = current_metric
            self.best_step = current_step
            self.patience_counter = 0
            print(f"   ✅ New best! Validation loss: {current_metric:.4f}")
        else:
            self.patience_counter += 1
            print(f"   ⚠️  No improvement for {self.patience_counter} eval(s). Patience: {self.patience_counter}/{self.patience}")
            
            if self.patience_counter >= self.patience:
                print(f"\n{'='*70}")
                print(f"🛑 EARLY STOPPING TRIGGERED at step {current_step}")
                print(f"   Best validation loss: {self.best_metric:.4f} (step {self.best_step})")
                print(f"   Current validation loss: {current_metric:.4f}")
                print(f"   No improvement for {self.patience} consecutive evaluations")
                print(f"   Loading best model from step {self.best_step}...")
                print(f"{'='*70}\n")
                control.should_training_stop = True
                self.early_stopping_triggered = True


class PlottingCallback(TrainerCallback):
    """Plots training metrics in real-time and saves figures."""
    
    def __init__(self, output_dir="./output/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store metrics
        self.train_losses = []
        self.train_steps = []
        self.val_losses = []
        self.val_steps = []
        self.learning_rates = []
        self.grad_norms = []
        self.perplexities = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Collect training metrics."""
        if logs:
            step = state.global_step
            
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
                self.train_steps.append(step)
            
            if 'learning_rate' in logs:
                self.learning_rates.append(logs['learning_rate'])
            
            if 'grad_norm' in logs:
                self.grad_norms.append(logs['grad_norm'])
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Collect validation metrics and generate plots."""
        if metrics:
            step = state.global_step
            
            if 'eval_loss' in metrics:
                self.val_losses.append(metrics['eval_loss'])
                self.val_steps.append(step)
            
            if 'eval_perplexity' in metrics:
                self.perplexities.append(metrics['eval_perplexity'])
            
            # Generate plots after each evaluation
            self._generate_plots(step)
    
    def _generate_plots(self, current_step):
        """Generate comprehensive training plots."""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Metrics (Step {current_step})', fontsize=16, fontweight='bold')
        
        # ============= PLOT 1: Train vs Val Loss (MOST IMPORTANT) =============
        ax1 = axes[0, 0]
        if self.train_losses and self.val_losses:
            ax1.plot(self.train_steps, self.train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.7)
            ax1.plot(self.val_steps, self.val_losses, 'r-', label='Val Loss', linewidth=2, marker='o', markersize=4)
            
            # Highlight the gap
            if len(self.val_steps) > 1:
                gap = self.train_losses[-1] - self.val_losses[-1]
                # Calculate overlapping region for fill_between
                min_len = min(len(self.train_steps), len(self.val_losses))
                if min_len > 0:
                    ax1.fill_between(
                        self.train_steps[-min_len:],
                        self.train_losses[-min_len:],
                        self.val_losses[-min_len:],
                        alpha=0.2,
                        color='yellow' if abs(gap) < 0.3 else 'orange' if abs(gap) < 0.5 else 'red',
                        label=f'Gap: {gap:.3f}'
                    )
            
            ax1.set_xlabel('Steps', fontsize=11)
            ax1.set_ylabel('Loss', fontsize=11)
            ax1.set_title('📉 Training vs Validation Loss', fontsize=12, fontweight='bold')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center', transform=ax1.transAxes)
        
        # ============= PLOT 2: Perplexity =============
        ax2 = axes[0, 1]
        if self.perplexities:
            ax2.plot(self.val_steps, self.perplexities, 'g-', label='Val Perplexity', linewidth=2, marker='s', markersize=4)
            ax2.set_xlabel('Steps', fontsize=11)
            ax2.set_ylabel('Perplexity', fontsize=11)
            ax2.set_title('📊 Validation Perplexity', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add horizontal line for "good" perplexity threshold
            good_threshold = 15.0
            ax2.axhline(y=good_threshold, color='r', linestyle='--', alpha=0.5, label=f'Target: <{good_threshold}')
            ax2.legend(loc='upper right')
        else:
            ax2.text(0.5, 0.5, 'Waiting for evaluation...', ha='center', va='center', transform=ax2.transAxes)
        
        # ============= PLOT 3: Learning Rate Schedule =============
        ax3 = axes[1, 0]
        if self.learning_rates:
            ax3.plot(self.train_steps, self.learning_rates, 'm-', label='Learning Rate', linewidth=2)
            ax3.set_xlabel('Steps', fontsize=11)
            ax3.set_ylabel('Learning Rate', fontsize=11)
            ax3.set_title('📈 Learning Rate Schedule', fontsize=12, fontweight='bold')
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)
            ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        else:
            ax3.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center', transform=ax3.transAxes)
        
        # ============= PLOT 4: Gradient Norm (Stability) =============
        ax4 = axes[1, 1]
        if self.grad_norms:
            ax4.plot(self.train_steps, self.grad_norms, 'c-', label='Gradient Norm', linewidth=2, alpha=0.7)
            ax4.set_xlabel('Steps', fontsize=11)
            ax4.set_ylabel('Gradient Norm', fontsize=11)
            ax4.set_title('⚡ Gradient Norm (Stability)', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Add warning line for gradient explosion
            max_norm = 1.0  # From training_args.max_grad_norm
            ax4.axhline(y=max_norm, color='r', linestyle='--', alpha=0.5, label=f'Clip Threshold: {max_norm}')
            ax4.legend(loc='upper right')
        else:
            ax4.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"training_metrics_step_{current_step}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save as "latest" for easy viewing
        self._save_latest_plot(current_step)
        
        print(f"📊 Plots saved: {plot_path}")
    
    def _save_latest_plot(self, current_step):
        """Save current plots as 'latest' version."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Metrics (Step {current_step})', fontsize=16, fontweight='bold')
        
        # Re-create plots for latest (same code as _generate_plots)
        ax1 = axes[0, 0]
        if self.train_losses and self.val_losses:
            ax1.plot(self.train_steps, self.train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.7)
            ax1.plot(self.val_steps, self.val_losses, 'r-', label='Val Loss', linewidth=2, marker='o', markersize=4)
            if len(self.val_steps) > 1:
                gap = self.train_losses[-1] - self.val_losses[-1]
                min_len = min(len(self.train_steps), len(self.val_losses))
                if min_len > 0:
                    ax1.fill_between(
                        self.train_steps[-min_len:],
                        self.train_losses[-min_len:],
                        self.val_losses[-min_len:],
                        alpha=0.2,
                        color='yellow' if abs(gap) < 0.3 else 'orange' if abs(gap) < 0.5 else 'red',
                        label=f'Gap: {gap:.3f}'
                    )
            ax1.set_xlabel('Steps', fontsize=11)
            ax1.set_ylabel('Loss', fontsize=11)
            ax1.set_title('📉 Training vs Validation Loss', fontsize=12, fontweight='bold')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        if self.perplexities:
            ax2.plot(self.val_steps, self.perplexities, 'g-', label='Val Perplexity', linewidth=2, marker='s', markersize=4)
            ax2.set_xlabel('Steps', fontsize=11)
            ax2.set_ylabel('Perplexity', fontsize=11)
            ax2.set_title('📊 Validation Perplexity', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            good_threshold = 15.0
            ax2.axhline(y=good_threshold, color='r', linestyle='--', alpha=0.5, label=f'Target: <{good_threshold}')
            ax2.legend(loc='upper right')
        
        ax3 = axes[1, 0]
        if self.learning_rates:
            ax3.plot(self.train_steps, self.learning_rates, 'm-', label='Learning Rate', linewidth=2)
            ax3.set_xlabel('Steps', fontsize=11)
            ax3.set_ylabel('Learning Rate', fontsize=11)
            ax3.set_title('📈 Learning Rate Schedule', fontsize=12, fontweight='bold')
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)
            ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        ax4 = axes[1, 1]
        if self.grad_norms:
            ax4.plot(self.train_steps, self.grad_norms, 'c-', label='Gradient Norm', linewidth=2, alpha=0.7)
            ax4.set_xlabel('Steps', fontsize=11)
            ax4.set_ylabel('Gradient Norm', fontsize=11)
            ax4.set_title('⚡ Gradient Norm (Stability)', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            max_norm = 1.0
            ax4.axhline(y=max_norm, color='r', linestyle='--', alpha=0.5, label=f'Clip Threshold: {max_norm}')
            ax4.legend(loc='upper right')
        
        plt.tight_layout()
        latest_path = self.output_dir / "training_metrics_latest.png"
        plt.savefig(latest_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def on_train_end(self, args, state, control, **kwargs):
        """Generate final comprehensive plot and save metrics."""
        self._generate_plots(state.global_step)
        
        # Also save metrics as JSON
        metrics_file = self.output_dir / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                'train_steps': self.train_steps,
                'train_losses': self.train_losses,
                'val_steps': self.val_steps,
                'val_losses': self.val_losses,
                'learning_rates': self.learning_rates,
                'grad_norms': self.grad_norms,
                'perplexities': self.perplexities
            }, f, indent=2)
        
        print(f"✅ Final metrics saved: {metrics_file}")


class EnhancedEvalCallback(TrainerCallback):
    """Combined perplexity tracking + sample generation callback."""
    
    def __init__(self, tokenizer, prompts):
        self.tokenizer = tokenizer
        self.prompts = prompts
    
    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            perplexity = np.exp(metrics['eval_loss'])
            metrics['eval_perplexity'] = perplexity
            
            print(f"\n{'='*70}")
            print(f"📊 Evaluation at Step {state.global_step}:")
            print(f"{'='*70}")
            print(f"   Eval Loss: {metrics['eval_loss']:.4f}")
            print(f"   Perplexity: {perplexity:.2f}")
            
            # Generate samples every 1000 steps
            if state.global_step % 1000 == 0 and model is not None:
                print(f"\n📝 Sample Generations:")
                print(f"{'-'*70}")
                model.eval()
                
                for i, prompt in enumerate(self.prompts[:3], 1):
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=150,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                    
                    generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    continuation = generated[len(prompt):].strip()
                    
                    print(f"\n{i}. Prompt: \"{prompt}\"")
                    print(f"   Model: \"{continuation[:200]}...\"" if len(continuation) > 200 else f"   Model: \"{continuation}\"")
                
                print(f"\n{'-'*70}")
                model.train()
            
            print(f"{'='*70}\n")
