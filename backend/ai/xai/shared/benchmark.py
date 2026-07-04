import time
import torch
from typing import Callable, Any, Tuple
from ai.xai.shared.report import BenchmarkInfo

class XAIBenchmarker:
    @staticmethod
    def run(func: Callable, *args, **kwargs) -> Tuple[Any, BenchmarkInfo]:
        """
        Executes a function and measures its execution time and memory usage.
        
        Args:
            func (Callable): The explanation generation function.
            
        Returns:
            Tuple[Any, BenchmarkInfo]: The function's result and the benchmark information.
        """
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
        start_time = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        
        generation_time_ms = (end_time - start_time) * 1000
        peak_memory_mb = None
        
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
        return result, BenchmarkInfo(generation_time_ms=generation_time_ms, peak_memory_mb=peak_memory_mb)
