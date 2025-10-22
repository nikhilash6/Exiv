# import torch
# import contextlib
# import logging

# # Get your logger
# app_logger = logging.getLogger(__name__) 

# class AsyncStreamManager:
#     """
#     Manages CUDA streams and tensor lifecycle for asynchronous, offloaded execution.
#     An instance of this class is owned by a ModelMixin instance.
#     """
#     def __init__(self, device: torch.device):
#         self.device = device
#         self.stash = {}

#         if self.device.type == 'cuda':
#             # The stream for data transfers (CPU -> GPU)
#             self.mover_stream = torch.cuda.Stream()
#             # The stream for computation (F.linear, F.conv2d, etc.)
#             self.compute_stream = torch.cuda.current_stream()
#             app_logger.info("AsyncStreamManager initialized in CUDA mode.")
#         else:
#             self.mover_stream = None
#             self.compute_stream = None
#             app_logger.info("AsyncStreamManager initialized in CPU mode.")

#     def async_load_parameters(self, module: torch.nn.Module, x: torch.Tensor):
#         """
#         Asynchronously loads a module's parameters to match the input tensor's device/dtype.
        
#         Returns:
#             (weight_gpu, bias_gpu, copy_done_signal)
#         """
#         # 1. Handle CPU-only mode
#         if self.mover_stream is None:
#             weight = getattr(module, 'weight', None)
#             bias = getattr(module, 'bias', None)
#             return weight, bias, None

#         # 2. Get target device/dtype from the input tensor
#         target_device = x.device
#         target_dtype = x.dtype
        
#         # 3. Get master parameters (which are on CPU)
#         weight = getattr(module, 'weight', None)
#         bias = getattr(module, 'bias', None)
        
#         # 4. Asynchronously copy on the mover stream
#         with torch.cuda.stream(self.mover_stream):
#             if weight is not None:
#                 weight = weight.to(device=target_device, dtype=target_dtype, non_blocking=True)
            
#             if bias is not None:
#                 bias = bias.to(device=target_device, dtype=target_dtype, non_blocking=True)
            
#             # 5. Record an event to signal when the copy is finished
#             copy_done_signal = self.mover_stream.record_event()
            
#         return weight, bias, copy_done_signal

#     @contextlib.contextmanager
#     def stream_worker(self, tensors_to_stash: list, copy_signal: torch.cuda.Event):
#         """
#         A context manager that:
#         1. Waits for the async copy to finish.
#         2. Yields to let the computation run.
#         3. Stashes the temporary tensors for cleanup.
#         4. Cleans up *old* stashed tensors.
#         """
#         # 1. Handle CPU-only mode
#         if copy_signal is None:
#             yield
#             return

#         # 2. Run on the compute stream
#         with torch.cuda.stream(self.compute_stream):
#             # 3. Wait for the copy (on mover_stream) to finish
#             self.compute_stream.wait_event(copy_signal)
            
#             # 4. YIELD: The computation (F.linear, etc.) runs here
#             yield
            
#             # 5. After computation is queued, record when it's *actually* done
#             compute_done_signal = self.compute_stream.record_event()
            
#             # 6. Stash the temporary tensors to keep them alive
#             stashed_tensors = [t for t in tensors_to_stash if t is not None]
#             if stashed_tensors:
#                 # We key the stash by the *signal* that tells us when compute is done
#                 self.stash[id(compute_done_signal)] = (stashed_tensors, compute_done_signal)
        
#         # 7. Clean up old, finished tensors from *previous* steps
#         self._cleanup_stash(full=False)

#     def _cleanup_stash(self, full: bool = False):
#         """
#         Cleans the stash.
#         - If full=False: Cleans only tensors whose computation is finished.
#         - If full=True: Cleans everything and synchronizes streams.
#         """
#         if self.mover_stream is None:
#             return

#         garbage = []
#         for k, (tensors, signal) in self.stash.items():
#             # .query() is True if the event has been recorded and completed
#             if full or signal.query():
#                 garbage.append(k)
        
#         for k in garbage:
#             del self.stash[k] # This releases the tensors for garbage collection
        
#         if full:
#             # Used for a final cleanup after the whole model pass
#             app_logger.debug("Performing full stream cleanup and synchronization.")
#             self.mover_stream.synchronize()
#             self.compute_stream.synchronize()
#             self.stash.clear()