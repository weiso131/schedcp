#!/usr/bin/env python3
"""
Scheduler Manager Module

This module provides functionality to start and stop schedulers based on
the configuration defined in schedulers.json.
"""

import json
import os
import subprocess
import signal
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class SchedulerManager:
    """Manages scheduler processes based on JSON configuration."""
    
    def __init__(self, config_path: str = "/home/yunwei37/ai-os/scheduler/schedulers.json", 
                 bin_path: str = "/home/yunwei37/ai-os/scheduler/sche_bin"):
        """
        Initialize the SchedulerManager.
        
        Args:
            config_path: Path to the schedulers.json configuration file
            bin_path: Path to the directory containing scheduler binaries
        """
        self.config_path = Path(config_path)
        self.bin_path = Path(bin_path)
        self.running_processes: Dict[str, subprocess.Popen] = {}
        self.config: Dict = {}
        
        # Load configuration
        self._load_config()
    
    def _load_config(self) -> None:
        """Load scheduler configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    def get_available_schedulers(self) -> List[str]:
        """
        Get list of available schedulers from configuration.
        
        Returns:
            List of scheduler names
        """
        return list(self.config.get("schedulers", {}).keys())
    
    def get_scheduler_info(self, scheduler_name: str) -> Dict:
        """
        Get information about a specific scheduler.
        
        Args:
            scheduler_name: Name of the scheduler
            
        Returns:
            Dictionary containing scheduler information
            
        Raises:
            KeyError: If scheduler not found in configuration
        """
        schedulers = self.config.get("schedulers", {})
        if scheduler_name not in schedulers:
            raise KeyError(f"Scheduler '{scheduler_name}' not found in configuration")
        
        return schedulers[scheduler_name]
    
    def _get_scheduler_binary_path(self, scheduler_name: str) -> Path:
        """
        Get the path to a scheduler binary.
        
        Args:
            scheduler_name: Name of the scheduler
            
        Returns:
            Path to the scheduler binary
            
        Raises:
            FileNotFoundError: If scheduler binary not found
        """
        binary_path = self.bin_path / scheduler_name
        
        if not binary_path.exists():
            raise FileNotFoundError(f"Scheduler binary not found: {binary_path}")
        
        if not binary_path.is_file():
            raise FileNotFoundError(f"Scheduler path is not a file: {binary_path}")
        
        if not os.access(binary_path, os.X_OK):
            raise PermissionError(f"Scheduler binary is not executable: {binary_path}")
        
        return binary_path
    
    def start_scheduler(self, scheduler_name: str, args: List[str] = None) -> bool:
        """
        Start a scheduler process.
        
        Args:
            scheduler_name: Name of the scheduler to start
            args: Additional command-line arguments for the scheduler
            
        Returns:
            True if scheduler started successfully, False otherwise
        """
        if scheduler_name in self.running_processes:
            print(f"Scheduler '{scheduler_name}' is already running")
            return False
        
        try:
            # Validate scheduler exists in configuration
            self.get_scheduler_info(scheduler_name)
            
            # Get binary path
            binary_path = self._get_scheduler_binary_path(scheduler_name)
            
            # Prepare command
            cmd = [str(binary_path)]
            if args:
                cmd.extend(args)
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give process a moment to start
            time.sleep(0.1)
            
            # Check if process is still running
            if process.poll() is None:
                self.running_processes[scheduler_name] = process
                print(f"Started scheduler '{scheduler_name}' with PID {process.pid}")
                return True
            else:
                # Process terminated immediately
                stdout, stderr = process.communicate()
                print(f"Failed to start scheduler '{scheduler_name}'")
                if stderr:
                    print(f"Error: {stderr}")
                return False
                
        except (KeyError, FileNotFoundError, PermissionError) as e:
            print(f"Error starting scheduler '{scheduler_name}': {e}")
            return False
        except Exception as e:
            print(f"Unexpected error starting scheduler '{scheduler_name}': {e}")
            return False
    
    def stop_scheduler(self, scheduler_name: str, timeout: int = 10) -> bool:
        """
        Stop a running scheduler process.
        
        Args:
            scheduler_name: Name of the scheduler to stop
            timeout: Maximum time to wait for graceful shutdown
            
        Returns:
            True if scheduler stopped successfully, False otherwise
        """
        if scheduler_name not in self.running_processes:
            print(f"Scheduler '{scheduler_name}' is not running")
            return False
        
        process = self.running_processes[scheduler_name]
        
        try:
            # Try graceful shutdown first
            process.terminate()
            
            # Wait for process to terminate
            try:
                process.wait(timeout=timeout)
                print(f"Stopped scheduler '{scheduler_name}' gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                process.kill()
                process.wait()
                print(f"Force killed scheduler '{scheduler_name}'")
            
            # Remove from running processes
            del self.running_processes[scheduler_name]
            return True
            
        except Exception as e:
            print(f"Error stopping scheduler '{scheduler_name}': {e}")
            return False
    
    def stop_all_schedulers(self, timeout: int = 10) -> bool:
        """
        Stop all running scheduler processes.
        
        Args:
            timeout: Maximum time to wait for each scheduler to stop
            
        Returns:
            True if all schedulers stopped successfully, False otherwise
        """
        if not self.running_processes:
            print("No schedulers are currently running")
            return True
        
        success = True
        scheduler_names = list(self.running_processes.keys())
        
        for scheduler_name in scheduler_names:
            if not self.stop_scheduler(scheduler_name, timeout):
                success = False
        
        return success
    
    def get_running_schedulers(self) -> List[Tuple[str, int]]:
        """
        Get list of currently running schedulers.
        
        Returns:
            List of tuples containing (scheduler_name, pid)
        """
        running = []
        
        # Clean up terminated processes
        terminated = []
        for name, process in self.running_processes.items():
            if process.poll() is not None:
                terminated.append(name)
            else:
                running.append((name, process.pid))
        
        # Remove terminated processes from tracking
        for name in terminated:
            del self.running_processes[name]
        
        return running
    
    def is_scheduler_running(self, scheduler_name: str) -> bool:
        """
        Check if a specific scheduler is running.
        
        Args:
            scheduler_name: Name of the scheduler to check
            
        Returns:
            True if scheduler is running, False otherwise
        """
        if scheduler_name not in self.running_processes:
            return False
        
        process = self.running_processes[scheduler_name]
        if process.poll() is not None:
            # Process has terminated, remove from tracking
            del self.running_processes[scheduler_name]
            return False
        
        return True
    
    def get_scheduler_output(self, scheduler_name: str) -> Tuple[str, str]:
        """
        Get stdout and stderr output from a running scheduler.
        
        Args:
            scheduler_name: Name of the scheduler
            
        Returns:
            Tuple of (stdout, stderr) strings
            
        Raises:
            KeyError: If scheduler is not running
        """
        if scheduler_name not in self.running_processes:
            raise KeyError(f"Scheduler '{scheduler_name}' is not running")
        
        process = self.running_processes[scheduler_name]
        
        # Non-blocking read
        try:
            stdout = process.stdout.read() if process.stdout else ""
            stderr = process.stderr.read() if process.stderr else ""
            return stdout, stderr
        except Exception:
            return "", ""


def main():
    """Example usage of the SchedulerManager."""
    manager = SchedulerManager()
    
    print("Available schedulers:")
    for scheduler in manager.get_available_schedulers():
        info = manager.get_scheduler_info(scheduler)
        status = "PRODUCTION" if info.get("production_ready", False) else "EXPERIMENTAL"
        print(f"  - {scheduler} ({status}): {info.get('description', 'No description')[:60]}...")
    
    print(f"\nCurrently running schedulers: {len(manager.get_running_schedulers())}")
    for name, pid in manager.get_running_schedulers():
        print(f"  - {name} (PID: {pid})")


if __name__ == "__main__":
    main()