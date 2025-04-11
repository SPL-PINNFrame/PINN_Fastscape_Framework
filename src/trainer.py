import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Import losses
from .losses import (
    compute_pde_residual, 
    compute_pde_residual_grid_focused,
    compute_pde_residual_dual_output,
    compute_total_loss
)

class PINNTrainer:
    """
    Trainer for Physics-Informed Neural Networks with improvements
    for landscape evolution models.

    This version includes specialized loss calculations and adaptive
    optimization strategies focused on geomorphic processes.
    """

    def __init__(
        self,
        model,
        optimizer=None,
        lr=1e-3,
        weight_decay=1e-5,
        device=None,
        save_path=None,
        scheduler=None,
        use_adaptive_weights=False,
        pde_residual_mode="dual_output", # 默认使用dual_output模式
        log_level=logging.INFO,
    ):
        """
        Initialize the PINN trainer.

        Args:
            model: The neural network model to train.
            optimizer: Optional custom optimizer. If None, Adam will be used.
            lr: Learning rate (only used if optimizer is None).
            weight_decay: Weight decay factor for regularization.
            device: Device to use for training ('cuda', 'cpu', etc.)
            save_path: Directory to save model checkpoints.
            scheduler: Optional learning rate scheduler.
            use_adaptive_weights: Whether to use adaptive weighting for loss components.
            pde_residual_mode: Method to calculate PDE residual ('classic', 'adaptive', 'grid_focused', 'dual_output')
            log_level: Logging verbosity level.
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(save_path, "training.log") if save_path else "training.log")
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.logger.info(f"Using device: {self.device}")
        
        # Store model and move to device
        self.model = model.to(self.device)
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer
            
        # Store other settings
        self.save_path = save_path
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
            
        self.scheduler = scheduler
        self.use_adaptive_weights = use_adaptive_weights
        
        # Validate and set PDE residual mode
        valid_modes = ["classic", "adaptive", "grid_focused", "dual_output"]
        if pde_residual_mode not in valid_modes:
            self.logger.warning(f"Invalid pde_residual_mode '{pde_residual_mode}'. Using 'dual_output'")
            pde_residual_mode = "dual_output"  # 确保强制fallback到dual_output
            
        self.pde_residual_mode = pde_residual_mode
        
        # 添加安全检查和提示，引导用户使用dual_output模式
        if pde_residual_mode in ["classic", "adaptive", "grid_focused"]:
            self.logger.warning(
                f"PDE residual mode '{pde_residual_mode}' uses autograd.grad for temporal derivatives, "
                f"which may lead to gradient flow issues. Consider using 'dual_output' mode instead."
            )
            
            # 如果用户选择了grid_focused模式，提供额外警告
            if pde_residual_mode == "grid_focused":
                self.logger.warning(
                    "grid_focused mode with 'allow_unused=False' will raise errors if time doesn't influence predictions. "
                    "Make sure your model properly connects time to outputs."
                )
        else:
            self.logger.info(
                "Using 'dual_output' mode for PDE residual calculation, which avoids autograd.grad "
                "issues by directly using model-predicted derivatives."
            )
        
        # Initialize training statistics
        self.train_history = {
            'total_loss': [],
            'data_loss': [],
            'physics_loss': [],
            'epochs': [],
            'lr': []
        }

    def _compute_physics_loss(self, model_outputs, collocation_coords, physics_params):
        """
        Compute the physics loss based on selected residual mode.
        
        Args:
            model_outputs: Dictionary of model outputs.
            collocation_coords: Dictionary of collocation coordinates.
            physics_params: Dictionary of physics parameters.
            
        Returns:
            Tensor of physics loss.
        """
        # Safety check - handle None outputs
        if model_outputs is None:
            self.logger.error("Model outputs is None in _compute_physics_loss")
            # Return a zero tensor with gradient connection
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        # Extract predictions based on what the model returns
        if self.pde_residual_mode == "dual_output":
            # 确保模型输出包含必要的键
            required_keys = ['state', 'derivative']
            if not all(key in model_outputs for key in required_keys):
                self.logger.error(
                    f"Model outputs for dual_output mode must contain keys: {required_keys}. "
                    f"Available keys: {list(model_outputs.keys())}. This is a model implementation issue."
                )
                # Provide a reasonable fallback
                state_key = next((k for k in model_outputs.keys() if 'state' in k or 'pred' in k or 'h' in k), None)
                if state_key:
                    dummy_state = model_outputs[state_key]
                    # Return zero tensor with correct shape and device, connected to state output
                    return dummy_state.sum() * 0.0
                else:
                    # Return simple zero tensor if no suitable state found
                    return torch.tensor(0.0, device=self.device, requires_grad=True)
                
            try:
                # Calculate residual directly from model's state and derivative outputs
                return compute_pde_residual_dual_output(model_outputs, physics_params)
            except Exception as e:
                self.logger.error(f"Error in compute_pde_residual_dual_output: {e}")
                # Fallback return
                return model_outputs['state'].sum() * 0.0
                
        elif self.pde_residual_mode == "grid_focused":
            # WARNING: This method uses autograd.grad for dh/dt internally, potential gradient flow issues.
            # Grid-focused method works on the model's predicted grid
            h_pred_grid = model_outputs.get('pred', None)
            t_grid = collocation_coords.get('t', None)
            
            # Safety checks
            if h_pred_grid is None:
                self.logger.error("Missing 'pred' in model_outputs for grid_focused mode")
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            if t_grid is None:
                self.logger.error("Missing 't' in collocation_coords for grid_focused mode")
                return torch.tensor(0.0, device=self.device, requires_grad=True)
                
            try:
                return compute_pde_residual_grid_focused(h_pred_grid, t_grid, physics_params)
            except Exception as e:
                self.logger.error(f"Error in compute_pde_residual_grid_focused: {e}")
                return h_pred_grid.sum() * 0.0
                
        else:  # "classic" or "adaptive"
            # WARNING: These methods use autograd.grad for dh/dt internally, potential gradient flow issues.
            # For classic/adaptive methods, we need collocation point predictions
            coll_pred = model_outputs.get('collocation_pred', None)
            
            # Safety check
            if coll_pred is None: 
                self.logger.error(f"Missing 'collocation_pred' in model_outputs for {self.pde_residual_mode} mode")
                return torch.tensor(0.0, device=self.device, requires_grad=True)
                
            try:
                # Use classic method by default (adaptive may be removed in future)
                return compute_pde_residual(coll_pred, collocation_coords, physics_params)
            except Exception as e:
                self.logger.error(f"Error in compute_pde_residual: {e}")
                return coll_pred.sum() * 0.0

    def train_epoch(self, data_batch, physics_batch, physics_params, loss_weights):
        """
        Train for a single epoch.
        
        Args:
            data_batch: Batch of data points with inputs and targets.
            physics_batch: Batch of physics-informed points (collocation points).
            physics_params: Dictionary of physics parameters.
            loss_weights: Dictionary of weights for different loss components.
            
        Returns:
            Dictionary of loss values.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Extract data inputs and targets
        data_inputs = data_batch.get('inputs', None)
        data_targets = data_batch.get('targets', None)
        
        # Safety check for data inputs/targets
        if data_inputs is None or data_targets is None:
            logging.error("Missing 'inputs' or 'targets' in data_batch. Skipping batch.")
            return {'total_loss': float('nan')}
        
        # Move data to device
        data_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in data_inputs.items()}
        data_targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in data_targets.items()}
        
        # Get collocation points
        collocation_coords = physics_batch.get('coords', None)
        if collocation_coords is None:
            logging.error("Missing 'coords' in physics_batch. Skipping batch.")
            return {'total_loss': float('nan')}
            
        # Move collocation points to device
        collocation_coords = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                             for k, v in collocation_coords.items()}
        
        # Forward pass through model
        try:
            model_outputs = self.model(data_inputs, collocation_coords)
        except Exception as e:
            self.logger.error(f"Error during model forward pass: {e}")
            # Return a dictionary indicating error
            return {'total_loss': float('nan'), 'error': str(e)}
        
        # Compute physics loss
        physics_loss = self._compute_physics_loss(
            model_outputs, 
            collocation_coords, 
            physics_params
        )
        
        # Get data predictions for data loss 
        data_pred = model_outputs.get('pred', None)
        target_topo = data_targets.get('h', None)  # 'h' typically holds topography
        
        # Safety check for predictions
        if data_pred is None:
            logging.error("Missing 'pred' in model outputs. Cannot compute data loss.")
            data_pred = torch.zeros_like(target_topo, device=self.device, requires_grad=True)
        
        if target_topo is None:
            logging.error("Missing 'h' in data targets. Cannot compute data loss.")
            target_topo = torch.zeros_like(data_pred, device=self.device)
            
        # Calculate total loss and get weighted components
        total_loss, loss_dict = compute_total_loss(
            data_pred=data_pred,
            final_topo=target_topo,
            physics_loss_value=physics_loss,
            physics_params=physics_params,
            loss_weights=loss_weights,
            collocation_pred=model_outputs.get('collocation_pred', None),
            collocation_coords=collocation_coords
        )
        
        # Backward pass and optimizer step
        if torch.isfinite(total_loss):
            try:
                total_loss.backward()
                self.optimizer.step()
            except RuntimeError as e:
                self.logger.error(f"Error during backward pass: {e}")
                # Add error info to loss dict
                loss_dict['backward_error'] = str(e)
                
        # Return loss values for logging
        return loss_dict
    
    def train(self, 
              train_data_loader, 
              physics_params, 
              loss_weights, 
              num_epochs, 
              val_data_loader=None,
              checkpoint_freq=10,
              early_stop_patience=None):
        """
        Train the model for multiple epochs.
        
        Args:
            train_data_loader: DataLoader providing training data and physics points.
            physics_params: Dictionary of physics parameters.
            loss_weights: Dictionary of weights for different loss components.
            num_epochs: Number of epochs to train for.
            val_data_loader: Optional DataLoader for validation.
            checkpoint_freq: How often to save model checkpoints.
            early_stop_patience: Number of epochs to wait for improvement before stopping.
            
        Returns:
            Dictionary of training history.
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Loss weights: {loss_weights}")
        self.logger.info(f"PDE residual mode: {self.pde_residual_mode}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            epoch_losses = []
            
            # Train on batches
            self.model.train()
            progress_bar = tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                # Extract data and physics batches
                data_batch = batch.get('data', None)
                physics_batch = batch.get('physics', None)
                
                if data_batch is None or physics_batch is None:
                    self.logger.warning("Missing 'data' or 'physics' in batch. Skipping batch.")
                    continue
                
                # Train on this batch
                batch_losses = self.train_epoch(
                    data_batch, 
                    physics_batch, 
                    physics_params,
                    loss_weights
                )
                
                # Update progress bar with loss info
                if 'total_loss' in batch_losses and np.isfinite(batch_losses['total_loss']):
                    progress_bar.set_postfix(loss=f"{batch_losses['total_loss']:.6f}")
                    epoch_losses.append(batch_losses)
            
            # Calculate average losses for this epoch
            if epoch_losses:
                avg_total_loss = np.mean([l.get('total_loss', float('nan')) for l in epoch_losses 
                                         if np.isfinite(l.get('total_loss', float('nan')))])
                avg_data_loss = np.mean([l.get('data_loss', float('nan')) for l in epoch_losses
                                        if np.isfinite(l.get('data_loss', float('nan')))])
                avg_physics_loss = np.mean([l.get('physics_loss', float('nan')) for l in epoch_losses
                                           if np.isfinite(l.get('physics_loss', float('nan')))])
                
                # Log epoch results
                self.logger.info(f"Epoch {epoch+1} - "
                                 f"Loss: {avg_total_loss:.6f}, "
                                 f"Data: {avg_data_loss:.6f}, "
                                 f"Physics: {avg_physics_loss:.6f}, "
                                 f"Time: {time.time() - epoch_start_time:.2f}s")
                
                # Update training history
                self.train_history['total_loss'].append(avg_total_loss)
                self.train_history['data_loss'].append(avg_data_loss)
                self.train_history['physics_loss'].append(avg_physics_loss)
                self.train_history['epochs'].append(epoch + 1)
                self.train_history['lr'].append(self.optimizer.param_groups[0]['lr'])
                
                # Validate if requested
                if val_data_loader is not None:
                    val_loss = self.validate(val_data_loader, physics_params, loss_weights)
                    self.logger.info(f"Validation loss: {val_loss:.6f}")
                    
                    # Check early stopping
                    if early_stop_patience is not None:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                            # Save best model
                            if self.save_path:
                                self.save_checkpoint(os.path.join(self.save_path, 'best_model.pth'))
                        else:
                            patience_counter += 1
                            if patience_counter >= early_stop_patience:
                                self.logger.info(f"Early stopping after {epoch+1} epochs")
                                break
                
                # Save checkpoint if needed
                if self.save_path and (epoch + 1) % checkpoint_freq == 0:
                    self.save_checkpoint(os.path.join(self.save_path, f'model_epoch_{epoch+1}.pth'))
                
                # Update learning rate if scheduler exists
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(avg_total_loss)
                    else:
                        self.scheduler.step()
            else:
                self.logger.warning(f"No valid losses for epoch {epoch+1}")
        
        # Save final model
        if self.save_path:
            self.save_checkpoint(os.path.join(self.save_path, 'final_model.pth'))
            
        self.logger.info("Training complete")
        return self.train_history
    
    def validate(self, val_data_loader, physics_params, loss_weights):
        """
        Validate the model on a validation dataset.
        
        Args:
            val_data_loader: DataLoader providing validation data.
            physics_params: Dictionary of physics parameters.
            loss_weights: Dictionary of weights for different loss components.
            
        Returns:
            Average validation loss.
        """
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_data_loader:
                # Extract data and physics batches
                data_batch = batch.get('data', None)
                physics_batch = batch.get('physics', None)
                
                if data_batch is None or physics_batch is None:
                    continue
                
                # Move data to device
                data_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                              for k, v in data_batch.get('inputs', {}).items()}
                data_targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                               for k, v in data_batch.get('targets', {}).items()}
                
                collocation_coords = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                     for k, v in physics_batch.get('coords', {}).items()}
                
                # Forward pass
                try:
                    model_outputs = self.model(data_inputs, collocation_coords)
                except Exception as e:
                    self.logger.error(f"Error during validation forward pass: {e}")
                    continue
                
                # Compute physics loss
                physics_loss = self._compute_physics_loss(
                    model_outputs, 
                    collocation_coords, 
                    physics_params
                )
                
                # Get data predictions
                data_pred = model_outputs.get('pred', None)
                target_topo = data_targets.get('h', None)
                
                # Safety checks
                if data_pred is None or target_topo is None:
                    continue
                
                # Calculate total loss
                total_loss, _ = compute_total_loss(
                    data_pred=data_pred,
                    final_topo=target_topo,
                    physics_loss_value=physics_loss,
                    physics_params=physics_params,
                    loss_weights=loss_weights,
                    collocation_pred=model_outputs.get('collocation_pred', None),
                    collocation_coords=collocation_coords
                )
                
                if torch.isfinite(total_loss):
                    val_losses.append(total_loss.item())
        
        # Return average validation loss
        return np.mean(val_losses) if val_losses else float('inf')
    
    def save_checkpoint(self, filepath):
        """
        Save a model checkpoint.
        
        Args:
            filepath: Path to save the checkpoint to.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
        
    def load_checkpoint(self, filepath):
        """
        Load a model checkpoint.
        
        Args:
            filepath: Path to the checkpoint file.
        """
        if not os.path.exists(filepath):
            self.logger.error(f"Checkpoint file {filepath} does not exist")
            return False
            
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            if 'train_history' in checkpoint:
                self.train_history = checkpoint['train_history']
                
            self.logger.info(f"Checkpoint loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return False