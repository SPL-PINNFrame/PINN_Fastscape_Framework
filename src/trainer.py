import os
import logging
import torch
import numpy as np # Added for isnan check
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import random # For collocation point sampling

# Import necessary components from the project
# Import specific model types and loss functions
from .models import FastscapePINN, AdaptiveFastscapePINN, MLP_PINN # Add AdaptiveFastscapePINN
# Import all relevant loss functions
from .losses import (
    compute_total_loss,
    compute_pde_residual,
    compute_pde_residual_adaptive,
    compute_pde_residual_grid_focused, # Import the new grid-focused loss
    compute_pde_residual_dual_output # Import the dual output loss function
)
from .utils import set_seed, get_device # Assuming utils.py has these

# --- DynamicWeightScheduler and LossScaler remain the same ---
class DynamicWeightScheduler:
    def __init__(self, config):
        # config here is the main config dictionary
        self.weights_config = config.get('training', {}).get('loss_weights', {}) # Get weights from training section
        # Add logic for warmup/scheduling if needed
        logging.info(f"Initializing DynamicWeightScheduler with weights: {self.weights_config}")

    def get_weights(self, epoch):
        # Placeholder: return static weights for now
        # TODO: Implement dynamic scheduling based on epoch or other metrics if needed
        return self.weights_config

class LossScaler:
     def __init__(self, enabled=False):
         self.enabled = enabled
         if self.enabled:
             logging.info("Loss scaling enabled (placeholder implementation).")
         else:
             logging.info("Loss scaling disabled.")
         # Add logic for tracking loss history and calculating scales

     def scale_losses(self, loss_dict):
         # Placeholder: return unscaled losses
         return loss_dict
# --- End of unchanged classes ---


class PINNTrainer:
    """Handles the training and validation loops for the PINN model."""
    def __init__(self, model, config, train_loader, val_loader):
        self.config = config
        self.train_config = config.get('training', {}) # Store training config section
        self.device = get_device(config) # Use utility function
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader # Validation might only use data loss

        # Determine model type for conditional logic later
        if isinstance(self.model, AdaptiveFastscapePINN):
            self.model_type = 'adaptive'
            logging.info("Trainer initialized with AdaptiveFastscapePINN model.")
        elif isinstance(self.model, FastscapePINN):
             self.model_type = 'original' # Refers to the CNN+MLP model
             logging.info("Trainer initialized with FastscapePINN model.")
        elif isinstance(self.model, MLP_PINN):
             self.model_type = 'mlp_only' # Handle pure MLP case if needed
             logging.info("Trainer initialized with MLP_PINN model.")
        else:
             self.model_type = 'unknown'
             logging.warning(f"Trainer initialized with unknown model type: {type(self.model)}")

        # Optimizer
        optimizer_name = self.train_config.get('optimizer', 'Adam').lower()
        lr = self.train_config.get('learning_rate', 1e-3)
        weight_decay = self.train_config.get('weight_decay', 0)
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
             self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'lbfgs':
             self.optimizer = optim.LBFGS(self.model.parameters(), lr=lr, line_search_fn="strong_wolfe") # Add line search for stability
             logging.warning("LBFGS optimizer selected, requires closure for step.")
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        logging.info(f"Optimizer: {optimizer_name}, LR: {lr}, Weight Decay: {weight_decay}")


        # Learning Rate Scheduler
        scheduler_config = self.train_config.get('lr_scheduler', None)
        self.scheduler = None
        if scheduler_config:
            scheduler_name = scheduler_config.get('name', 'StepLR').lower()
            if scheduler_name == 'steplr':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config.get('step_size', 30),
                    gamma=scheduler_config.get('gamma', 0.1)
                )
            elif scheduler_name == 'reducelronplateau':
                 self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                     self.optimizer,
                     mode=scheduler_config.get('mode', 'min'),
                     factor=scheduler_config.get('factor', 0.1),
                     patience=scheduler_config.get('patience', 10)
                 )
            # Add other schedulers like CosineAnnealingLR if needed
            elif scheduler_name == 'cosineannealinglr':
                 self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                      self.optimizer,
                      T_max=scheduler_config.get('T_max', config.get('epochs', 100)), # Usually total epochs
                      eta_min=scheduler_config.get('eta_min', 0)
                 )
            logging.info(f"LR Scheduler: {scheduler_name} configured.")


        # Loss components
        self.physics_params = config.get('physics_params', {})
        # --- Domain info for collocation points (still needed for interpolation/adaptive methods) ---
        self.domain_x = self.physics_params.get('domain_x', [0.0, 100.0])
        self.domain_y = self.physics_params.get('domain_y', [0.0, 100.0])
        self.total_time = self.physics_params.get('total_time', 1000.0)
        self.n_collocation = self.train_config.get('n_collocation_points', 10000)
        # --- PDE Loss Method Selection ---
        self.pde_loss_method = self.train_config.get('pde_loss_method', 'grid_focused').lower()
        if self.pde_loss_method not in ['grid_focused', 'interpolation', 'adaptive']:
             logging.warning(f"Invalid pde_loss_method '{self.pde_loss_method}'. Defaulting to 'grid_focused'.")
             self.pde_loss_method = 'grid_focused'
        logging.info(f"Using PDE loss method: {self.pde_loss_method}")
        # ---------------------------------
        self.loss_weight_scheduler = DynamicWeightScheduler(config) # Pass full config
        self.loss_scaler = LossScaler(enabled=self.train_config.get('use_loss_scaling', False))

        # Logging and Checkpointing
        self.run_name = config.get('run_name', f'pinn_run_{int(time.time())}')
        self.output_dir = os.path.join(config.get('output_dir', 'results'), self.run_name)
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        logging.info(f"Run Name: {self.run_name}")
        logging.info(f"Outputs will be saved to: {self.output_dir}")


        # Mixed Precision
        self.use_amp = self.train_config.get('use_mixed_precision', False)
        # Use the updated torch.amp API
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.use_amp) # Match device type
        if self.use_amp:
             logging.info("Mixed precision training enabled.")


        self.start_epoch = 0
        self.best_val_loss = float('inf')

        # Load checkpoint if specified
        load_path = config.get('load_checkpoint', None)
        if load_path:
            self.load_checkpoint(load_path)

    def _generate_collocation_points(self, n_points):
        """Generates random collocation points within the defined domain."""
        x_min, x_max = self.domain_x
        y_min, y_max = self.domain_y
        t_min, t_max = 0.0, self.total_time # Assuming time starts at 0

        try:
            model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            logging.warning("Could not determine model dtype in _generate_collocation_points. Defaulting to float32.")
            model_dtype = torch.float32

        x_collocation = torch.rand(n_points, 1, device=self.device, dtype=model_dtype) * (x_max - x_min) + x_min
        y_collocation = torch.rand(n_points, 1, device=self.device, dtype=model_dtype) * (y_max - y_min) + y_min
        t_collocation = torch.rand(n_points, 1, device=self.device, dtype=model_dtype) * (t_max - t_min) + t_min

        # Coordinates need gradients for PDE loss calculation
        x_collocation.requires_grad_(True)
        y_collocation.requires_grad_(True)
        t_collocation.requires_grad_(True)

        coords = {'x': x_collocation, 'y': y_collocation, 't': t_collocation}
        return coords


    def _run_epoch(self, epoch, is_training):
        """Runs a single epoch of training or validation."""
        epoch_loss = 0.0
        epoch_loss_components = {} # To store average loss components
        batch_count = 0
        finite_batch_count = 0

        self.model.train(is_training)
        loader = self.train_loader if is_training else self.val_loader
        use_physics_loss = is_training or self.train_config.get('validate_with_physics', False)

        # progress_bar = tqdm(loader, desc=f"Epoch {epoch} {'Train' if is_training else 'Val'}", leave=False) # Temporarily remove tqdm

        for batch_idx, batch_data in enumerate(loader): # Iterate directly over loader
            if batch_data is None:
                logging.warning(f"Skipping batch {batch_idx} due to loading error.")
                continue

            batch_count += 1

            # --- Data Handling (Moved outside closure) ---
            try:
                # Check batch_data type before processing
                if not isinstance(batch_data, dict):
                    logging.error(f"Epoch {epoch}, Batch {batch_idx}: Unexpected batch_data type: {type(batch_data)}. Content: {str(batch_data)[:200]}. Skipping batch.")
                    continue # Skip this problematic batch

                data_inputs = {}
                final_topo = None
                for k, v in batch_data.items():
                    if isinstance(v, torch.Tensor):
                        if k == 'final_topo': final_topo = v.to(self.device)
                        else: data_inputs[k] = v.to(self.device)
                if final_topo is None: raise ValueError("'final_topo' not found in batch_data.")
            except Exception as e:
                 logging.error(f"Error processing data batch {batch_idx} before closure: {e}. Skipping batch.", exc_info=True)
                 continue # Skip batch if initial processing fails

            # --- Define Closure for LBFGS ---
            def closure():
                if is_training:
                    self.optimizer.zero_grad()

                # data_inputs and final_topo are now accessible from the outer scope

                # --- Forward Pass & Loss Calculation ---
                with torch.set_grad_enabled(is_training):
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        # 1. Prediction for Data Loss (Grid Prediction)
                        initial_state = data_inputs.get('initial_topo')
                        params_dict = {
                            'K': data_inputs.get('k_f'), 'D': data_inputs.get('k_d'),
                            'U': data_inputs.get('uplift_rate'), 'm': data_inputs.get('m'),
                            'n': data_inputs.get('n')
                        }
                        t_target = data_inputs.get('run_time')
                        if t_target is None:
                             t_target = torch.tensor(self.total_time, device=self.device, dtype=initial_state.dtype if initial_state is not None else torch.float32)

                        if initial_state is None:
                             logging.error("Missing 'initial_topo' for predict_state mode in trainer.")
                             return torch.tensor(float('nan'), device=self.device)

                        model_input_state = {'initial_state': initial_state, 'params': params_dict, 't_target': t_target}
                        try:
                            # Model now returns a dict {'state': ..., 'derivative': ...} or single tensor
                            model_outputs_state = self.model(model_input_state, mode='predict_state')
                            # Extract state prediction for data loss
                            if isinstance(model_outputs_state, dict):
                                data_pred = model_outputs_state.get('state')
                                if data_pred is None:
                                     raise ValueError("Model output dictionary missing 'state' key in predict_state mode.")
                            else:
                                # Assume single output is state if not a dict
                                data_pred = model_outputs_state
                        except Exception as e:
                            logging.error(f"Error during model forward pass (predict_state): {e}", exc_info=True)
                            return torch.tensor(float('nan'), device=self.device)

                        # 2. Calculate Physics Loss (PDE Residual)
                        physics_loss = None
                        collocation_pred = None
                        collocation_coords = None

                        if use_physics_loss:
                            try:
                                # --- Physics Loss Calculation using Dual Output Model ---
                                # 2.1 Generate collocation points
                                collocation_coords = self._generate_collocation_points(self.n_collocation)

                                # 2.2 Add parameter grids (if available in data) for sampling
                                model_input_coords = collocation_coords.copy()
                                k_grid = data_inputs.get('k_f') # Use k_f from data as k_grid
                                u_grid = data_inputs.get('uplift_rate') # Use uplift_rate as u_grid
                                model_input_coords['k_grid'] = k_grid # Pass even if None, handled in model
                                model_input_coords['u_grid'] = u_grid

                                # 2.3 Get model predictions (state and derivative) at collocation points
                                # Ensure model outputs both state and derivative for physics loss
                                self.model.set_output_mode(state=True, derivative=True)
                                collocation_outputs = self.model(model_input_coords, mode='predict_coords')
                                # Reset model output mode if needed (e.g., if only state needed elsewhere)
                                # self.model.set_output_mode(state=True, derivative=False) # Example

                                if not isinstance(collocation_outputs, dict) or 'state' not in collocation_outputs or 'derivative' not in collocation_outputs:
                                     logging.error(f"Model did not return expected dictionary with 'state' and 'derivative' in predict_coords mode for physics loss.")
                                     physics_loss = None
                                else:
                                     # 2.4 Calculate PDE residual using the dual output loss function
                                     # Pass collocation_coords in physics_params if needed by compute_local_physics
                                     physics_params_with_coords = self.physics_params.copy()
                                     physics_params_with_coords['coords'] = collocation_coords
                                     physics_params_with_coords['k_grid'] = k_grid # Pass grids too
                                     physics_params_with_coords['u_grid'] = u_grid

                                     physics_loss = compute_pde_residual_dual_output(
                                         outputs=collocation_outputs,
                                         physics_params=physics_params_with_coords
                                     )

                            except Exception as e:
                                 logging.error(f"Error calculating physics loss (dual output method): {e}", exc_info=True)
                                 physics_loss = None # Ensure it's None if calculation fails

                        # 3. Compute Total Loss
                        current_loss_weights = self.loss_weight_scheduler.get_weights(epoch)
                        total_loss, loss_components = compute_total_loss(
                            data_pred=data_pred,
                            final_topo=final_topo,
                            physics_loss_value=physics_loss, # Pass the calculated physics loss
                            physics_params=self.physics_params,
                            loss_weights=current_loss_weights,
                            # collocation_pred is no longer needed separately if physics loss uses dual output
                        )

                # --- Backward Pass (if training) ---
                if is_training:
                    if not isinstance(total_loss, torch.Tensor) or not torch.isfinite(total_loss):
                        loss_val_repr = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
                        logging.warning(f"Epoch {epoch}, Batch {batch_idx}: Invalid or non-finite total loss ({loss_val_repr}) in closure. Skipping backward pass.")
                        # For LBFGS, returning NaN might signal an issue
                        return torch.tensor(float('nan'), device=self.device)
                    else:
                        # Use AMP scaler for backward pass
                        self.scaler.scale(total_loss).backward()
                        # Return loss for LBFGS
                        return total_loss
                else:
                    # For validation, just return the loss
                    return total_loss if isinstance(total_loss, torch.Tensor) and torch.isfinite(total_loss) else torch.tensor(float('nan'), device=self.device)

            # --- Optimization Step ---
            if is_training:
                if isinstance(self.optimizer, optim.LBFGS):
                    # Pass closure to LBFGS step
                    # Note: LBFGS requires re-evaluation, so data_inputs needs to be accessible inside closure
                    loss = self.optimizer.step(closure) # LBFGS step returns the loss
                    total_loss = loss # Assign loss for accumulation later
                    # AMP scaler update might need adjustment for LBFGS if used together
                else:
                    # For Adam/AdamW, call closure once to compute loss and grads
                    total_loss = closure() # This now uses data_inputs from outer scope
                    # Check loss validity before stepping
                    if isinstance(total_loss, torch.Tensor) and torch.isfinite(total_loss):
                        # Gradient clipping (optional)
                        if self.train_config.get('clip_grad_norm', None):
                            self.scaler.unscale_(self.optimizer) # Unscale grads before clipping
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config['clip_grad_norm'])
                        # Optimizer step using AMP scaler
                        self.scaler.step(self.optimizer)
                    else:
                         logging.warning(f"Epoch {epoch}, Batch {batch_idx}: Skipping optimizer step due to invalid loss.")

                # Update AMP scaler
                self.scaler.update()
            else:
                 # For validation, just run closure to get the loss
                 total_loss = closure()


            # --- Accumulate Loss (retrieve from closure's last run if needed) ---
            # Re-fetch loss components calculated inside closure if LBFGS was used
            # Note: This part is tricky with LBFGS as closure might run multiple times.
            # For simplicity, we'll rely on the loss returned by the last closure call.
            # A more robust way might involve storing loss_components within the closure's scope.
            current_loss_weights = self.loss_weight_scheduler.get_weights(epoch)
            # Recompute loss components outside closure for logging consistency (might be slightly different if LBFGS ran multiple evals)
            # This is inefficient but ensures logged components match the final state.
            with torch.no_grad(): # Recompute loss components without grad tracking
                 # Recompute loss components using data_inputs and final_topo from outer scope
                 with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                      # Reconstruct model_input_state for the no_grad prediction
                      initial_state_no_grad = data_inputs.get('initial_topo')
                      params_dict_no_grad = {
                          'K': data_inputs.get('k_f'), 'D': data_inputs.get('k_d'),
                          'U': data_inputs.get('uplift_rate'), 'm': data_inputs.get('m'),
                          'n': data_inputs.get('n')
                      }
                      t_target_no_grad = data_inputs.get('run_time')
                      if t_target_no_grad is None:
                           t_target_no_grad = torch.tensor(self.total_time, device=self.device, dtype=initial_state_no_grad.dtype if initial_state_no_grad is not None else torch.float32)

                      if initial_state_no_grad is None:
                           logging.error("Missing 'initial_topo' for no_grad loss component calculation.")
                           loss_components_log = {}
                           data_pred_no_grad_state = None
                      else:
                           model_input_state_no_grad = {'initial_state': initial_state_no_grad, 'params': params_dict_no_grad, 't_target': t_target_no_grad}
                           data_pred_no_grad = self.model(model_input_state_no_grad, mode='predict_state')
                           if isinstance(data_pred_no_grad, dict):
                                data_pred_no_grad_state = data_pred_no_grad.get('state')
                           else:
                                data_pred_no_grad_state = data_pred_no_grad

                      # Need physics_loss from closure scope or recompute
                      # Assuming physics_loss variable from closure scope is accessible
                      # If not, it needs to be returned/stored by closure.
                      # For simplicity, assume it's available. A better approach might be needed for LBFGS.
                      # physics_loss_log = physics_loss # Use the value computed in closure
                      # Let's recompute physics loss here for consistency, although less efficient
                      physics_loss_log = None
                      if use_physics_loss:
                           try:
                                # Re-generate points (or reuse if stored)
                                collocation_coords_log = self._generate_collocation_points(self.n_collocation)
                                model_input_coords_log = collocation_coords_log.copy()
                                k_grid_log = data_inputs.get('k_f')
                                u_grid_log = data_inputs.get('uplift_rate')
                                model_input_coords_log['k_grid'] = k_grid_log
                                model_input_coords_log['u_grid'] = u_grid_log
                                self.model.set_output_mode(state=True, derivative=True)
                                collocation_outputs_log = self.model(model_input_coords_log, mode='predict_coords')
                                if isinstance(collocation_outputs_log, dict) and 'state' in collocation_outputs_log and 'derivative' in collocation_outputs_log:
                                     physics_params_with_coords_log = self.physics_params.copy()
                                     physics_params_with_coords_log['coords'] = collocation_coords_log
                                     physics_params_with_coords_log['k_grid'] = k_grid_log
                                     physics_params_with_coords_log['u_grid'] = u_grid_log
                                     physics_loss_log = compute_pde_residual_dual_output(
                                         outputs=collocation_outputs_log,
                                         physics_params=physics_params_with_coords_log
                                     )
                           except Exception as e_log:
                                logging.warning(f"Error recomputing physics loss for logging: {e_log}")
                                physics_loss_log = None # Set to None if recomputation fails

                      if data_pred_no_grad_state is not None:
                           _, loss_components_log = compute_total_loss(
                                data_pred=data_pred_no_grad_state, final_topo=final_topo, # Use final_topo from outer scope
                                physics_loss_value=physics_loss_log, # Use recomputed physics loss
                                physics_params=self.physics_params, loss_weights=current_loss_weights
                           )
                      else:
                           loss_components_log = {}


            if isinstance(total_loss, torch.Tensor) and torch.isfinite(total_loss):
                 finite_batch_count += 1
                 loss_item = total_loss.item()
                 epoch_loss += loss_item
                 # Accumulate components from the no_grad calculation
                 for key, value in loss_components_log.items():
                     if isinstance(value, (int, float)) and not isinstance(value, bool) and not np.isnan(value):
                          epoch_loss_components[key] = epoch_loss_components.get(key, 0.0) + value
                 loss_display = f"{loss_item:.4f}"
            else:
                 loss_display = "NaN"

            # Update progress bar (Removed tqdm)
            # progress_bar.set_postfix({'loss': loss_display})
            # Simple print for now if not using tqdm
            if batch_idx % 10 == 0: # Log every 10 batches
                 logging.debug(f"Epoch {epoch} Batch {batch_idx} Loss: {loss_display}")


        # Calculate average loss for the epoch
        if batch_count == 0:
             logging.warning(f"Epoch {epoch} {'Train' if is_training else 'Val'}: No batches were processed.")
             return 0.0, {}

        avg_loss = epoch_loss / batch_count # Average over all processed batches
        # Average components over batches with finite loss
        avg_loss_components = {key: value / finite_batch_count for key, value in epoch_loss_components.items()} if finite_batch_count > 0 else {}

        return avg_loss, avg_loss_components


    # --- train, save_checkpoint, load_checkpoint methods remain largely the same ---
    def train(self):
        """Main training loop."""
        epochs = self.train_config.get('epochs', 100)
        log_interval = self.train_config.get('log_interval', 10) # Currently logs per epoch
        val_interval = self.train_config.get('val_interval', 1) # Validate every N epochs
        save_best_only = self.train_config.get('save_best_only', True)
        save_interval = self.train_config.get('save_interval', 10) # Save every N epochs if not save_best_only

        logging.info(f"Starting training from epoch {self.start_epoch} for {epochs} epochs.")

        for epoch in range(self.start_epoch, epochs):
            epoch_start_time = time.time()
            # Training epoch
            train_loss, train_loss_components = self._run_epoch(epoch, is_training=True)
            logging.info(f"Epoch {epoch} Train Loss: {train_loss:.6f}")
            # Log training losses to TensorBoard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            for name, value in train_loss_components.items():
                self.writer.add_scalar(f'LossComponents/Train/{name}', value, epoch)


            # Validation epoch
            val_loss = float('inf') # Default if no validation
            if (epoch + 1) % val_interval == 0 and self.val_loader:
                val_loss, val_loss_components = self._run_epoch(epoch, is_training=False)
                logging.info(f"Epoch {epoch} Val Loss: {val_loss:.6f}")
                # Log validation losses to TensorBoard
                self.writer.add_scalar('Loss/Val', val_loss, epoch)
                for name, value in val_loss_components.items():
                    self.writer.add_scalar(f'LossComponents/Val/{name}', value, epoch)


                # LR Scheduler step (if based on validation loss)
                if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('LearningRate', current_lr, epoch)
                    logging.info(f"Epoch {epoch} LR (on plateau): {current_lr:.6e}")


                # Checkpoint saving based on validation loss
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    logging.info(f"New best validation loss: {self.best_val_loss:.6f}")
                    self.save_checkpoint(epoch, 'best_model.pth', is_best=True)

            # Save checkpoint periodically if not saving best only
            if not save_best_only and (epoch + 1) % save_interval == 0:
                 self.save_checkpoint(epoch, f'epoch_{epoch:04d}.pth')


            # LR Scheduler step (if epoch-based)
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('LearningRate', current_lr, epoch)
                if (epoch + 1) % 10 == 0: # Log LR less frequently for epoch-based schedulers
                     logging.info(f"Epoch {epoch} LR: {current_lr:.6e}")

            epoch_duration = time.time() - epoch_start_time
            logging.info(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds.")


        self.writer.close()
        logging.info("Training finished.")
        logging.info(f"Best Validation Loss: {self.best_val_loss:.6f}")


    def save_checkpoint(self, epoch, filename, is_best=False):
        """Saves model checkpoint."""
        state = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config # Save config for reproducibility
        }
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.use_amp:
             state['amp_scaler_state_dict'] = self.scaler.state_dict()

        filepath = os.path.join(self.checkpoint_dir, filename)
        try:
            torch.save(state, filepath)
            logging.info(f"Checkpoint saved to {filepath}")
            if is_best:
                 # Ensure the best model is always saved with a consistent name
                 best_filepath = os.path.join(self.checkpoint_dir, 'best_model.pth')
                 torch.save(state, best_filepath)
                 logging.info(f"Best model checkpoint updated: {best_filepath}")
        except Exception as e:
             logging.error(f"Failed to save checkpoint {filepath}: {e}")


    def load_checkpoint(self, filepath):
        """Loads model checkpoint."""
        if not os.path.exists(filepath):
            logging.error(f"Checkpoint file not found: {filepath}")
            return

        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint.get('epoch', 0) # Use get with default
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf')) # Use get with default
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                     logging.warning(f"Could not load scheduler state_dict: {e}. Scheduler state might be reset.")
            if self.use_amp and 'amp_scaler_state_dict' in checkpoint:
                 try:
                      self.scaler.load_state_dict(checkpoint['amp_scaler_state_dict'])
                 except Exception as e:
                      logging.warning(f"Could not load AMP scaler state_dict: {e}. Scaler state might be reset.")

            logging.info(f"Checkpoint loaded from {filepath}. Resuming from epoch {self.start_epoch}.")
        except Exception as e:
            logging.error(f"Error loading checkpoint from {filepath}: {e}", exc_info=True)
            # Reset start epoch and best loss if loading fails
            self.start_epoch = 0
            self.best_val_loss = float('inf')


if __name__ == '__main__':
    # Example usage (requires dummy model, data, config - more complex to set up here)
    print("PINNTrainer class defined. Run from a main script (e.g., scripts/train.py).")
    # Setup basic logging for standalone run info
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("To test PINNTrainer, create dummy data, model, config and run its train() method.")
