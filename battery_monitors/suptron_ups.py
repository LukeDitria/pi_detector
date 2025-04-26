#!/usr/bin/env python3
import struct
import smbus
import gpiozero
from datetime import datetime
import csv

import logging
from typing import Union, Generator, List, Optional

class SupTronicsBatteryMonitor:
    def __init__(self, log_file_path, low_battery_voltage, log_remote, project_id, device_name):

        self.log_file_path = log_file_path
        self.low_battery_voltage = low_battery_voltage

        self.log_remote = log_remote
        self.project_id = project_id
        self.device_name = device_name

        self.battery_monitor_available = True
        self.address = 0x36

        # Try to initialize the battery monitor
        try:
            self.bus = smbus.SMBus(1)
            # Test if we can read from the device
            self.read_voltage()
        except Exception as e:
            logging.info(f"Battery monitor initialization failed: {e}")
            logging.info("Continuing without battery monitoring")
            self.battery_monitor_available = False

        self.ensure_log_file_exists()
        logging.info(f"Battery monitoring started. Logging to {self.log_file_path}")

        if self.log_remote:
            from firestore_logger import FirestoreLogger
            logging.info(f"Firestore remote logging")
            try:
                self.fire_logger = FirestoreLogger(project_id=self.project_id,
                                                   firestore_collection=self.device_name,
                                                   logger_type="battery")
                logging.info(f"Firestore logging initialized")
            except Exception as e:
                logging.info(f"Firestore initialization failed: {e}")
                logging.info("Continuing without remote logging")
                self.fire_logger = None
        else:
            self.fire_logger = None

    def read_voltage(self) -> Optional[float]:
        """Read battery voltage"""
        if not self.battery_monitor_available:
            return None

        try:
            read = self.bus.read_word_data(self.address, 2)
            swapped = struct.unpack("<H", struct.pack(">H", read))[0]
            voltage = swapped * 1.25 / 1000 / 16
            return voltage
        except Exception as e:
            logging.info(f"Error reading voltage: {e}")
            logging.info("Continuing without battery monitoring")
            self.battery_monitor_available = False
            return None

    def read_capacity(self) -> Optional[float]:
        """Read battery capacity"""
        if not self.battery_monitor_available:
            return None
        try:
            read = self.bus.read_word_data(self.address, 4)
            swapped = struct.unpack("<H", struct.pack(">H", read))[0]
            capacity = swapped / 256
            return capacity
        except Exception as e:
            logging.info(f"Error reading capacity: {e}")
            logging.info("Continuing without battery monitoring")
            self.battery_monitor_available = False
            return None

    def is_battery_low(self):
        current_voltage = self.read_voltage()
        return current_voltage < self.low_battery_voltage

    def create_log_dict(self, battery_voltage: float, battery_capacity: float, status: str) -> dict:
        doc_data = {
            "type": "battery_status",
            "battery_voltage": battery_voltage,
            "battery_capacity": battery_capacity,
            "status": status
        }
        return doc_data

    def ensure_log_file_exists(self):
        """Create the log file with headers if it doesn't exist"""
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Voltage', 'Capacity', 'Shutdown_Reason'])

    def log_data(self, timestamp: datetime, shutdown_reason: Optional[str] = None):
        """Log the current battery voltage to CSV file"""
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        voltage = self.read_voltage()
        capacity = self.read_capacity()

        # Always try to log to local file first
        try:
            with open(self.log_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if voltage is not None:
                    writer.writerow([timestamp_str, f"{voltage:.2f}", f"{capacity:.2f}", shutdown_reason or "on"])
                    logging.info(f"Logged: Time: {timestamp_str}, Voltage: {voltage:.2f}V, Capacity: {capacity:.2f}%")
                else:
                    writer.writerow([timestamp_str, "N/A", shutdown_reason or "on (no battery data)"])
                    logging.info(f"Logged: Time: {timestamp_str}, Voltage: N/A (battery monitor unavailable)")
        except Exception as e:
            logging.info(f"Error logging to file: {e}")

        # Then try to log to Firestore if enabled
        if self.log_remote and self.fire_logger:
            try:
                if voltage is not None:
                    doc_dict = self.create_log_dict(voltage, capacity, shutdown_reason or "on")
                else:
                    doc_dict = self.create_log_dict(None, None, shutdown_reason or "on (no battery data)")

                self.fire_logger.log_data_to_firestore(doc_dict,
                                                       doc_type="battery",
                                                       timestamp=timestamp,
                                                       add_time_to_dict=True)
            except Exception as e:
                logging.info(f"Error during remote logging: {e}")

